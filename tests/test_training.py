"""Tests for training pipeline."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_speaker_diarization_with_adaptive_boundary_refinement.data import (
    create_dataloaders,
)
from multi_speaker_diarization_with_adaptive_boundary_refinement.evaluation import (
    compute_all_metrics,
)
from multi_speaker_diarization_with_adaptive_boundary_refinement.models import (
    MultiSpeakerDiarizationModel,
)
from multi_speaker_diarization_with_adaptive_boundary_refinement.training import (
    DiarizationTrainer,
)


class TestDiarizationTrainer:
    """Tests for trainer class."""

    def test_trainer_initialization(self, sample_config, device):
        """Test trainer initialization."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )

        trainer = DiarizationTrainer(model, sample_config, device)

        assert trainer.device == device
        assert trainer.num_epochs == sample_config["training"]["num_epochs"]
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_trainer_train_epoch(self, sample_config, device):
        """Test training for one epoch."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )

        trainer = DiarizationTrainer(model, sample_config, device)

        # Create small dataloader
        train_loader, _, _ = create_dataloaders(sample_config, use_synthetic=True)

        # Train for one epoch
        metrics = trainer._train_epoch(train_loader)

        assert "loss" in metrics
        assert "diarization_loss" in metrics
        assert "boundary_loss" in metrics
        assert "temporal_loss" in metrics

        assert metrics["loss"] > 0.0

    def test_trainer_validate_epoch(self, sample_config, device):
        """Test validation for one epoch."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )

        trainer = DiarizationTrainer(model, sample_config, device)

        # Create small dataloader
        _, val_loader, _ = create_dataloaders(sample_config, use_synthetic=True)

        # Validate
        metrics = trainer._validate_epoch(val_loader)

        assert "loss" in metrics
        assert metrics["loss"] > 0.0

    def test_trainer_checkpoint_save_load(self, sample_config, device, tmp_path):
        """Test checkpoint saving and loading."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )

        # Update config to use temp directory
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
        sample_config["training"]["best_model_path"] = str(tmp_path / "best_model.pt")

        trainer = DiarizationTrainer(model, sample_config, device)

        # Save checkpoint
        trainer._save_checkpoint(is_best=True)

        assert Path(trainer.best_model_path).exists()

        # Load checkpoint
        new_model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )
        new_trainer = DiarizationTrainer(new_model, sample_config, device)

        new_trainer.load_checkpoint(str(trainer.best_model_path))

        assert new_trainer.current_epoch == trainer.current_epoch


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""

    def test_compute_all_metrics(self, sample_labels, sample_boundaries):
        """Test computing all metrics."""
        # Create predictions (slightly different from true labels)
        predicted_labels = sample_labels.clone()
        predicted_labels[0, 50:100] = (predicted_labels[0, 50:100] + 1) % 4

        predicted_boundaries = sample_boundaries.clone()

        metrics = compute_all_metrics(
            predicted_labels,
            sample_labels,
            predicted_boundaries,
            sample_boundaries,
            collar=0.25,
        )

        assert "diarization_error_rate" in metrics
        assert "jaccard_error_rate" in metrics
        assert "boundary_f1_score" in metrics
        assert "speaker_confusion" in metrics
        assert "frame_accuracy" in metrics

        # Check value ranges
        assert 0.0 <= metrics["diarization_error_rate"] <= 1.0
        assert 0.0 <= metrics["jaccard_error_rate"] <= 1.0
        assert 0.0 <= metrics["boundary_f1_score"] <= 1.0
        assert 0.0 <= metrics["speaker_confusion"] <= 1.0
        assert 0.0 <= metrics["frame_accuracy"] <= 1.0

    def test_metrics_perfect_prediction(self, sample_labels, sample_boundaries):
        """Test metrics with perfect predictions."""
        metrics = compute_all_metrics(
            sample_labels,
            sample_labels,
            sample_boundaries,
            sample_boundaries,
            collar=0.25,
        )

        # With perfect predictions
        assert metrics["diarization_error_rate"] == 0.0
        assert metrics["frame_accuracy"] == 1.0


class TestEndToEndTraining:
    """End-to-end training tests."""

    def test_full_training_loop(self, sample_config, device):
        """Test complete training loop with minimal data."""
        # Use very small config for fast testing
        sample_config["training"]["num_epochs"] = 2
        sample_config["data"]["synthetic_data_size"] = 10

        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )

        trainer = DiarizationTrainer(model, sample_config, device)

        train_loader, val_loader, _ = create_dataloaders(
            sample_config, use_synthetic=True
        )

        # Train
        history = trainer.train(train_loader, val_loader, mlflow_logger=None)

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) <= sample_config["training"]["num_epochs"]
