"""Custom components for adaptive boundary refinement in speaker diarization.

This module implements novel components for improving speaker boundary detection:
1. AdaptiveBoundaryRefinement: Iteratively adjusts speaker change points using
   local acoustic consistency scoring.
2. BoundaryConsistencyLoss: Custom loss that penalizes inconsistent boundaries.
3. TemporalConsistencyLoss: Encourages temporal smoothness in speaker predictions.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AdaptiveBoundaryRefinement(nn.Module):
    """Adaptive boundary refinement module using local acoustic consistency.

    This is the core novel component that iteratively refines speaker boundaries
    by analyzing spectral similarity gradients in overlapping windows.
    """

    def __init__(
        self,
        window_size: int = 5,
        overlap: float = 0.5,
        gradient_threshold: float = 0.15,
        max_iterations: int = 5,
    ):
        """Initialize adaptive boundary refinement module.

        Args:
            window_size: Size of local analysis window (in frames).
            overlap: Overlap ratio between windows.
            gradient_threshold: Threshold for detecting significant changes.
            max_iterations: Maximum refinement iterations.
        """
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.gradient_threshold = gradient_threshold
        self.max_iterations = max_iterations

        # Learnable weights for similarity scoring
        self.similarity_weights = nn.Parameter(torch.ones(3) / 3)  # mel, spectral, temporal

    def forward(
        self,
        mel_features: torch.Tensor,
        spectral_features: torch.Tensor,
        initial_boundaries: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Refine speaker boundaries using adaptive local consistency.

        Args:
            mel_features: Mel spectrogram features of shape (batch, mel_bins, time).
            spectral_features: Spectral flux features of shape (batch, time).
            initial_boundaries: Initial boundary predictions of shape (batch, time).

        Returns:
            Tuple of (refined_boundaries, debug_info).
        """
        batch_size, _, time_frames = mel_features.shape
        refined_boundaries = initial_boundaries.clone()

        # Normalize weights
        weights = F.softmax(self.similarity_weights, dim=0)

        # Track refinement statistics
        total_adjustments = 0

        for iteration in range(self.max_iterations):
            # Compute local acoustic consistency scores
            consistency_scores = self._compute_consistency_scores(
                mel_features, spectral_features, weights
            )

            # Detect candidate boundary points
            boundary_candidates = self._detect_boundary_candidates(
                consistency_scores, refined_boundaries
            )

            # Refine boundaries based on local gradients
            adjustments = self._adjust_boundaries(
                boundary_candidates, consistency_scores, refined_boundaries
            )

            refined_boundaries = refined_boundaries + adjustments
            refined_boundaries = torch.clamp(refined_boundaries, 0, 1)

            total_adjustments += adjustments.abs().sum().item()

            # Early stopping if no significant changes
            if adjustments.abs().sum() < 0.01:
                break

        debug_info = {
            "iterations": iteration + 1,
            "total_adjustments": total_adjustments,
            "final_consistency": consistency_scores.mean().item(),
        }

        return refined_boundaries, debug_info

    def _compute_consistency_scores(
        self,
        mel_features: torch.Tensor,
        spectral_features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local acoustic consistency scores.

        Args:
            mel_features: Mel spectrogram features.
            spectral_features: Spectral flux features.
            weights: Normalized feature weights.

        Returns:
            Consistency scores of shape (batch, time).
        """
        batch_size, mel_bins, time_frames = mel_features.shape

        # Compute frame-to-frame similarity in mel space
        mel_similarity = torch.zeros(batch_size, time_frames, device=mel_features.device)
        for t in range(1, time_frames):
            mel_diff = F.cosine_similarity(
                mel_features[:, :, t - 1],
                mel_features[:, :, t],
                dim=1,
            )
            mel_similarity[:, t] = mel_diff

        # Spectral flux similarity (inverse of change)
        spectral_diff = torch.abs(spectral_features[:, 1:] - spectral_features[:, :-1])
        spectral_similarity = 1.0 / (1.0 + spectral_diff)
        spectral_similarity = F.pad(spectral_similarity, (1, 0), value=1.0)

        # Temporal smoothness (local variance)
        kernel_size = min(self.window_size, time_frames)
        if kernel_size > 1:
            temporal_smooth = F.avg_pool1d(
                mel_features.mean(dim=1, keepdim=True),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ).squeeze(1)
            temporal_consistency = 1.0 - torch.std(temporal_smooth, dim=1, keepdim=True).expand(
                -1, time_frames
            )
        else:
            temporal_consistency = torch.ones_like(mel_similarity)

        # Weighted combination
        consistency = (
            weights[0] * mel_similarity
            + weights[1] * spectral_similarity
            + weights[2] * temporal_consistency
        )

        return consistency

    def _detect_boundary_candidates(
        self, consistency_scores: torch.Tensor, current_boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Detect candidate boundary points based on consistency drops.

        Args:
            consistency_scores: Local consistency scores.
            current_boundaries: Current boundary predictions.

        Returns:
            Binary tensor indicating candidate locations.
        """
        # Compute gradient of consistency scores
        gradients = torch.abs(consistency_scores[:, 1:] - consistency_scores[:, :-1])
        gradients = F.pad(gradients, (1, 0), value=0)

        # Threshold to find significant drops
        candidates = (gradients > self.gradient_threshold).float()

        # Combine with current boundaries
        candidates = torch.maximum(candidates, current_boundaries)

        return candidates

    def _adjust_boundaries(
        self,
        candidates: torch.Tensor,
        consistency_scores: torch.Tensor,
        current_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust boundary locations based on local consistency.

        Args:
            candidates: Candidate boundary locations.
            consistency_scores: Local consistency scores.
            current_boundaries: Current boundary predictions.

        Returns:
            Boundary adjustments (delta values).
        """
        batch_size, time_frames = candidates.shape
        adjustments = torch.zeros_like(current_boundaries)

        # For each candidate, check if it should be strengthened or weakened
        for b in range(batch_size):
            for t in range(1, time_frames - 1):
                if candidates[b, t] > 0.5:
                    # Check local consistency
                    local_consistency = consistency_scores[b, max(0, t - 2) : min(time_frames, t + 3)].mean()

                    # If consistency is high, weaken boundary
                    if local_consistency > 0.7:
                        adjustments[b, t] = -0.1
                    # If consistency is low, strengthen boundary
                    elif local_consistency < 0.4:
                        adjustments[b, t] = 0.1

        return adjustments


class BoundaryConsistencyLoss(nn.Module):
    """Custom loss function that enforces boundary consistency.

    This loss penalizes boundaries that don't align with acoustic discontinuities,
    encouraging the model to place boundaries at true speaker change points.
    """

    def __init__(self, margin: float = 0.1):
        """Initialize boundary consistency loss.

        Args:
            margin: Margin for boundary detection consistency.
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        predicted_boundaries: torch.Tensor,
        spectral_features: torch.Tensor,
        target_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary consistency loss.

        Args:
            predicted_boundaries: Predicted boundary probabilities (batch, time).
            spectral_features: Spectral flux features (batch, time).
            target_boundaries: Ground truth boundaries (batch, time).

        Returns:
            Scalar loss value.
        """
        # Compute spectral discontinuity
        spectral_diff = torch.abs(spectral_features[:, 1:] - spectral_features[:, :-1])
        spectral_diff = F.pad(spectral_diff, (1, 0), value=0)

        # Normalize to [0, 1]
        spectral_diff = spectral_diff / (spectral_diff.max() + 1e-8)

        # Penalize boundaries not aligned with spectral changes
        # Clamp predicted_boundaries to avoid numerical issues
        predicted_boundaries = torch.clamp(predicted_boundaries, min=1e-7, max=1.0 - 1e-7)
        alignment_loss = F.binary_cross_entropy(
            predicted_boundaries,
            target_boundaries,
            reduction="none",
        )

        # Weight by spectral discontinuity (higher weight where spectral change is low)
        consistency_weight = 1.0 - spectral_diff
        weighted_loss = alignment_loss * consistency_weight

        return weighted_loss.mean()


class TemporalConsistencyLoss(nn.Module):
    """Custom loss that encourages temporal smoothness in speaker predictions.

    This loss penalizes rapid speaker changes that are unlikely in natural speech,
    reducing false positives in boundary detection.
    """

    def __init__(self, window_size: int = 5, smoothness_weight: float = 0.1):
        """Initialize temporal consistency loss.

        Args:
            window_size: Size of temporal window for consistency check.
            smoothness_weight: Weight for smoothness penalty.
        """
        super().__init__()
        self.window_size = window_size
        self.smoothness_weight = smoothness_weight

    def forward(
        self, speaker_predictions: torch.Tensor, temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal consistency loss.

        Args:
            speaker_predictions: Speaker logits of shape (batch, num_speakers, time).
            temporal_features: Temporal features for consistency (batch, time).

        Returns:
            Scalar loss value.
        """
        batch_size, num_speakers, time_frames = speaker_predictions.shape

        # Compute speaker probabilities
        speaker_probs = F.softmax(speaker_predictions, dim=1)

        # Compute temporal variation (how much predictions change over time)
        temporal_variation = torch.zeros(batch_size, device=speaker_predictions.device)

        for t in range(1, time_frames):
            # KL divergence between consecutive frames
            kl_div = F.kl_div(
                torch.log(speaker_probs[:, :, t] + 1e-8),
                speaker_probs[:, :, t - 1],
                reduction="none",
            ).sum(dim=1)
            temporal_variation += kl_div

        temporal_variation = temporal_variation / (time_frames - 1)

        # Penalize excessive variation
        smoothness_penalty = self.smoothness_weight * temporal_variation.mean()

        # Encourage consistency in local windows
        window_consistency = 0.0
        num_windows = max(1, time_frames - self.window_size + 1)

        for start in range(0, time_frames - self.window_size + 1, self.window_size // 2):
            end = start + self.window_size
            window_probs = speaker_probs[:, :, start:end]

            # Variance within window (should be low for consistent segments)
            window_var = window_probs.var(dim=2).mean()
            window_consistency += window_var

        window_consistency = window_consistency / num_windows

        total_loss = smoothness_penalty + self.smoothness_weight * window_consistency

        return total_loss
