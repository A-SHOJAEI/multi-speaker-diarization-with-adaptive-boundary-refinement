"""Training loop implementation with learning rate scheduling and early stopping."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import BoundaryConsistencyLoss, TemporalConsistencyLoss
from ..models.model import MultiSpeakerDiarizationModel

logger = logging.getLogger(__name__)


class DiarizationTrainer:
    """Trainer class for multi-speaker diarization model with adaptive boundary refinement."""

    def __init__(
        self,
        model: MultiSpeakerDiarizationModel,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            model: Diarization model to train.
            config: Training configuration dictionary.
            device: Device to train on (cuda/cpu).
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training configuration
        train_config = config["training"]
        self.num_epochs = train_config["num_epochs"]
        self.gradient_clip_norm = train_config["gradient_clip_norm"]
        self.early_stopping_patience = train_config["early_stopping_patience"]
        self.checkpoint_dir = Path(train_config["checkpoint_dir"])
        self.best_model_path = Path(train_config["best_model_path"])

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
        )

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler(config["scheduler"])

        # Initialize loss functions
        loss_config = config["loss"]
        self.diarization_weight = loss_config["diarization_weight"]
        self.boundary_weight = loss_config["boundary_consistency_weight"]
        self.temporal_weight = loss_config["temporal_consistency_weight"]
        self.confusion_penalty = loss_config["speaker_confusion_penalty"]

        self.boundary_loss = BoundaryConsistencyLoss()
        self.temporal_loss = TemporalConsistencyLoss()

        # Mixed precision training
        self.use_amp = train_config.get("mixed_precision", False)
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {"train_loss": [], "val_loss": []}

        logger.info(f"Initialized trainer with device: {device}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mlflow_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            mlflow_logger: Optional MLflow logger for tracking.

        Returns:
            Dictionary containing training history and final metrics.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate_epoch(val_loader)

            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            else:
                self.scheduler.step()

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics, mlflow_logger)

            # Save checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(is_best=True)
                self.patience_counter = 0
                logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(is_best=False, suffix=f"_epoch{epoch+1}")

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )
                break

        logger.info("Training completed")
        return self.training_history

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_diar_loss = 0.0
        total_boundary_loss = 0.0
        total_temporal_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        for batch in pbar:
            # Move data to device
            mel_spec = batch["mel_spectrogram"].to(self.device)
            spectral_flux = batch["spectral_flux"].to(self.device)
            labels = batch["labels"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss, loss_components = self._compute_loss(
                        mel_spec, spectral_flux, labels, boundaries
                    )
            else:
                loss, loss_components = self._compute_loss(
                    mel_spec, spectral_flux, labels, boundaries
                )

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_diar_loss += loss_components["diarization"].item()
            total_boundary_loss += loss_components["boundary"].item()
            total_temporal_loss += loss_components["temporal"].item()

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        num_batches = len(train_loader)
        metrics = {
            "loss": total_loss / num_batches,
            "diarization_loss": total_diar_loss / num_batches,
            "boundary_loss": total_boundary_loss / num_batches,
            "temporal_loss": total_temporal_loss / num_batches,
        }

        self.training_history["train_loss"].append(metrics["loss"])
        return metrics

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_diar_loss = 0.0
        total_boundary_loss = 0.0
        total_temporal_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                mel_spec = batch["mel_spectrogram"].to(self.device)
                spectral_flux = batch["spectral_flux"].to(self.device)
                labels = batch["labels"].to(self.device)
                boundaries = batch["boundaries"].to(self.device)

                # Forward pass
                loss, loss_components = self._compute_loss(
                    mel_spec, spectral_flux, labels, boundaries
                )

                # Update metrics
                total_loss += loss.item()
                total_diar_loss += loss_components["diarization"].item()
                total_boundary_loss += loss_components["boundary"].item()
                total_temporal_loss += loss_components["temporal"].item()

        num_batches = len(val_loader)
        metrics = {
            "loss": total_loss / num_batches,
            "diarization_loss": total_diar_loss / num_batches,
            "boundary_loss": total_boundary_loss / num_batches,
            "temporal_loss": total_temporal_loss / num_batches,
        }

        self.training_history["val_loss"].append(metrics["loss"])
        return metrics

    def _compute_loss(
        self,
        mel_spec: torch.Tensor,
        spectral_flux: torch.Tensor,
        labels: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss function.

        Args:
            mel_spec: Mel spectrogram features.
            spectral_flux: Spectral flux features.
            labels: Speaker labels.
            boundaries: Boundary labels.

        Returns:
            Tuple of (total_loss, loss_components).
        """
        # Forward pass
        outputs = self.model(mel_spec, spectral_flux)

        speaker_logits = outputs["speaker_logits"]
        boundary_probs = outputs["refined_boundaries"]

        # Diarization loss (cross entropy for speaker classification)
        diarization_loss = F.cross_entropy(
            speaker_logits.transpose(1, 2).reshape(-1, speaker_logits.shape[1]),
            labels.reshape(-1),
        )

        # Boundary consistency loss - disable autocast for binary_cross_entropy
        with torch.cuda.amp.autocast(enabled=False):
            boundary_loss = self.boundary_loss(
                boundary_probs.float(), spectral_flux.float(), boundaries.float()
            )

        # Temporal consistency loss
        temporal_loss = self.temporal_loss(speaker_logits, spectral_flux)

        # Combined loss
        total_loss = (
            self.diarization_weight * diarization_loss
            + self.boundary_weight * boundary_loss
            + self.temporal_weight * temporal_loss
        )

        loss_components = {
            "diarization": diarization_loss,
            "boundary": boundary_loss,
            "temporal": temporal_loss,
        }

        return total_loss, loss_components

    def _create_scheduler(self, scheduler_config: Dict[str, Any]) -> Any:
        """Create learning rate scheduler.

        Args:
            scheduler_config: Scheduler configuration.

        Returns:
            Learning rate scheduler instance.
        """
        scheduler_type = scheduler_config["type"]

        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs - scheduler_config.get("warmup_epochs", 0),
                eta_min=scheduler_config.get("min_lr", 1e-6),
            )
        elif scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 5),
                min_lr=scheduler_config.get("min_lr", 1e-6),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _save_checkpoint(self, is_best: bool = False, suffix: str = "") -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far.
            suffix: Optional suffix for checkpoint filename.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": self.config,
        }

        if is_best:
            torch.save(checkpoint, self.best_model_path)
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint{suffix}.pt"
            torch.save(checkpoint, checkpoint_path)

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        mlflow_logger: Optional[Any] = None,
    ) -> None:
        """Log training metrics.

        Args:
            epoch: Current epoch number.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            mlflow_logger: Optional MLflow logger.
        """
        # Console logging
        logger.info(
            f"Epoch {epoch + 1}/{self.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )

        # MLflow logging
        if mlflow_logger is not None:
            try:
                mlflow_logger.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train_diarization_loss": train_metrics["diarization_loss"],
                        "train_boundary_loss": train_metrics["boundary_loss"],
                        "train_temporal_loss": train_metrics["temporal_loss"],
                        "val_diarization_loss": val_metrics["diarization_loss"],
                        "val_boundary_loss": val_metrics["boundary_loss"],
                        "val_temporal_loss": val_metrics["temporal_loss"],
                    },
                    step=epoch,
                )
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
