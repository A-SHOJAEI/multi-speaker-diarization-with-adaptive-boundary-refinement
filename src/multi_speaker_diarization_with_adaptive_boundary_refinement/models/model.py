"""Core model architecture for multi-speaker diarization with boundary refinement."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import AdaptiveBoundaryRefinement

logger = logging.getLogger(__name__)


class MultiSpeakerDiarizationModel(nn.Module):
    """Multi-speaker diarization model with adaptive boundary refinement.

    This model combines a CNN-based feature extractor with LSTM for temporal
    modeling, followed by speaker classification and optional adaptive boundary
    refinement for improved accuracy at speaker change points.
    """

    def __init__(
        self,
        input_dim: int = 80,
        embedding_dim: int = 512,
        num_speakers: int = 4,
        dropout: float = 0.3,
        use_adaptive_refinement: bool = True,
        refinement_config: Optional[Dict] = None,
    ):
        """Initialize multi-speaker diarization model.

        Args:
            input_dim: Input feature dimension (mel bins).
            embedding_dim: Embedding dimension.
            num_speakers: Number of speaker classes.
            dropout: Dropout probability.
            use_adaptive_refinement: Whether to use adaptive boundary refinement.
            refinement_config: Configuration for boundary refinement module.
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.use_adaptive_refinement = use_adaptive_refinement

        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=embedding_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Speaker classification head
        self.speaker_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_speakers),
        )

        # Boundary detection head
        self.boundary_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Adaptive boundary refinement module
        if self.use_adaptive_refinement:
            refinement_config = refinement_config or {}
            self.boundary_refinement = AdaptiveBoundaryRefinement(
                window_size=refinement_config.get("window_size", 5),
                overlap=refinement_config.get("overlap", 0.5),
                gradient_threshold=refinement_config.get("gradient_threshold", 0.15),
                max_iterations=refinement_config.get("max_iterations", 5),
            )

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"Initialized model with {self.count_parameters():,} parameters"
        )

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        spectral_flux: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            mel_spectrogram: Mel spectrogram features of shape (batch, mel_bins, time).
            spectral_flux: Spectral flux features of shape (batch, time).

        Returns:
            Dictionary containing:
                - speaker_logits: Speaker predictions (batch, num_speakers, time)
                - boundary_probs: Boundary probabilities (batch, time)
                - embeddings: Frame-level embeddings (batch, embedding_dim, time)
                - refined_boundaries: Refined boundaries if refinement is enabled
        """
        batch_size, mel_bins, time_frames = mel_spectrogram.shape

        # Extract CNN features
        conv_features = self.conv_layers(mel_spectrogram)  # (batch, 512, time)

        # Transpose for LSTM (batch, time, features)
        conv_features = conv_features.transpose(1, 2)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(conv_features)  # (batch, time, embedding_dim)
        embeddings = lstm_out.transpose(1, 2)  # (batch, embedding_dim, time)

        # Speaker classification
        speaker_logits = self.speaker_classifier(lstm_out)  # (batch, time, num_speakers)
        speaker_logits = speaker_logits.transpose(1, 2)  # (batch, num_speakers, time)

        # Boundary detection
        boundary_probs = self.boundary_detector(lstm_out)  # (batch, time, 1)
        boundary_probs = boundary_probs.squeeze(-1)  # (batch, time)

        output = {
            "speaker_logits": speaker_logits,
            "boundary_probs": boundary_probs,
            "embeddings": embeddings,
        }

        # Apply adaptive boundary refinement if enabled
        if self.use_adaptive_refinement and spectral_flux is not None:
            refined_boundaries, debug_info = self.boundary_refinement(
                mel_spectrogram, spectral_flux, boundary_probs
            )
            output["refined_boundaries"] = refined_boundaries
            output["refinement_debug"] = debug_info
        else:
            output["refined_boundaries"] = boundary_probs

        return output

    def predict_speakers(
        self, mel_spectrogram: torch.Tensor, spectral_flux: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict speaker labels and boundaries for input audio.

        Args:
            mel_spectrogram: Mel spectrogram features.
            spectral_flux: Spectral flux features.

        Returns:
            Tuple of (speaker_labels, boundary_points).
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(mel_spectrogram, spectral_flux)

            # Get speaker predictions
            speaker_probs = F.softmax(output["speaker_logits"], dim=1)
            speaker_labels = torch.argmax(speaker_probs, dim=1)

            # Get boundary points using adaptive threshold based on distribution
            # The refinement module may shift probabilities, so use a relative
            # threshold: mean + 0.5 * std to detect outlier peaks as boundaries
            refined = output["refined_boundaries"]
            mean_val = refined.mean(dim=-1, keepdim=True)
            std_val = refined.std(dim=-1, keepdim=True)
            adaptive_threshold = mean_val + 0.5 * std_val
            # Clamp the threshold so it stays in a reasonable range
            adaptive_threshold = torch.clamp(adaptive_threshold, min=0.1, max=0.9)
            boundary_points = (refined > adaptive_threshold).long()

        return speaker_labels, boundary_points

    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def count_parameters(self) -> int:
        """Count the number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Embedding dimension.
        """
        return self.embedding_dim
