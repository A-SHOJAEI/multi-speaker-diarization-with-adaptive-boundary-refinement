"""Evaluation metrics for speaker diarization."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class DiarizationMetrics:
    """Metrics computation for speaker diarization evaluation."""

    def __init__(self, collar: float = 0.25, num_speakers: int = 4):
        """Initialize metrics calculator.

        Args:
            collar: Collar size in seconds for boundary evaluation.
            num_speakers: Number of speaker classes.
        """
        self.collar = collar
        self.num_speakers = num_speakers

    def diarization_error_rate(
        self,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray,
        predicted_boundaries: np.ndarray,
        true_boundaries: np.ndarray,
    ) -> float:
        """Compute Diarization Error Rate (DER).

        DER = (False Alarm + Missed Speech + Speaker Error) / Total Speech Time

        Args:
            predicted_labels: Predicted speaker labels.
            true_labels: Ground truth speaker labels.
            predicted_boundaries: Predicted boundary points.
            true_boundaries: Ground truth boundary points.

        Returns:
            DER score (lower is better).
        """
        total_frames = len(true_labels)

        # Compute frame-level errors
        speaker_errors = np.sum(predicted_labels != true_labels)

        # Normalize by total frames
        der = speaker_errors / total_frames

        return float(der)

    def jaccard_error_rate(
        self,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """Compute Jaccard Error Rate (JER).

        JER measures overlap between predicted and true segments.

        Args:
            predicted_labels: Predicted speaker labels.
            true_labels: Ground truth speaker labels.

        Returns:
            JER score (lower is better).
        """
        # Convert to segments
        pred_segments = self._labels_to_segments(predicted_labels)
        true_segments = self._labels_to_segments(true_labels)

        # Compute Jaccard index
        total_union = 0
        total_intersection = 0

        for true_start, true_end, true_spk in true_segments:
            for pred_start, pred_end, pred_spk in pred_segments:
                if true_spk == pred_spk:
                    # Compute intersection
                    intersection_start = max(true_start, pred_start)
                    intersection_end = min(true_end, pred_end)
                    intersection = max(0, intersection_end - intersection_start)

                    # Compute union
                    union_start = min(true_start, pred_start)
                    union_end = max(true_end, pred_end)
                    union = union_end - union_start

                    total_intersection += intersection
                    total_union += union

        if total_union == 0:
            return 1.0

        jaccard = total_intersection / total_union
        jer = 1.0 - jaccard

        return float(jer)

    def boundary_f1_score(
        self,
        predicted_boundaries: np.ndarray,
        true_boundaries: np.ndarray,
        collar_frames: int = 5,
    ) -> float:
        """Compute F1 score for boundary detection with collar.

        Args:
            predicted_boundaries: Predicted boundary points (binary).
            true_boundaries: Ground truth boundary points (binary).
            collar_frames: Collar size in frames.

        Returns:
            F1 score for boundaries (higher is better).
        """
        # Find boundary indices
        pred_boundary_indices = np.where(predicted_boundaries > 0.5)[0]
        true_boundary_indices = np.where(true_boundaries > 0.5)[0]

        if len(true_boundary_indices) == 0:
            return 1.0 if len(pred_boundary_indices) == 0 else 0.0

        # Match predicted boundaries to true boundaries within collar
        true_positives = 0
        matched_true = set()

        for pred_idx in pred_boundary_indices:
            # Find closest true boundary
            distances = np.abs(true_boundary_indices - pred_idx)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist <= collar_frames and min_dist_idx not in matched_true:
                true_positives += 1
                matched_true.add(min_dist_idx)

        # Compute precision and recall
        precision = true_positives / len(pred_boundary_indices) if len(pred_boundary_indices) > 0 else 0.0
        recall = true_positives / len(true_boundary_indices)

        # Compute F1
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)

        return float(f1)

    def speaker_confusion(
        self,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """Compute speaker confusion rate.

        Measures the rate at which speakers are confused with each other.

        Args:
            predicted_labels: Predicted speaker labels.
            true_labels: Ground truth speaker labels.

        Returns:
            Confusion rate (lower is better).
        """
        # Get unique speakers
        unique_speakers = np.unique(true_labels)

        # Compute confusion matrix
        confusion = np.zeros((len(unique_speakers), len(unique_speakers)))

        for i, true_spk in enumerate(unique_speakers):
            true_mask = true_labels == true_spk
            for j, pred_spk in enumerate(unique_speakers):
                pred_mask = predicted_labels == pred_spk
                confusion[i, j] = np.sum(true_mask & pred_mask)

        # Normalize by row (true speaker)
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_confusion = confusion / row_sums

        # Confusion is the average off-diagonal elements
        off_diagonal = normalized_confusion * (1 - np.eye(len(unique_speakers)))
        confusion_rate = off_diagonal.sum() / len(unique_speakers)

        return float(confusion_rate)

    def _labels_to_segments(
        self, labels: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Convert frame-level labels to segments.

        Args:
            labels: Frame-level speaker labels.

        Returns:
            List of (start_frame, end_frame, speaker_id) tuples.
        """
        segments = []
        if len(labels) == 0:
            return segments

        current_speaker = labels[0]
        start_frame = 0

        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                segments.append((start_frame, i, int(current_speaker)))
                current_speaker = labels[i]
                start_frame = i

        # Add final segment
        segments.append((start_frame, len(labels), int(current_speaker)))

        return segments

    def compute_frame_accuracy(
        self, predicted_labels: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """Compute frame-level accuracy.

        Args:
            predicted_labels: Predicted speaker labels.
            true_labels: Ground truth speaker labels.

        Returns:
            Frame accuracy (higher is better).
        """
        return float(accuracy_score(true_labels, predicted_labels))

    def compute_precision_recall_f1(
        self, predicted_labels: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1 score.

        Args:
            predicted_labels: Predicted speaker labels.
            true_labels: Ground truth speaker labels.

        Returns:
            Dictionary with precision, recall, and f1 scores.
        """
        precision = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


def compute_all_metrics(
    predicted_labels: torch.Tensor,
    true_labels: torch.Tensor,
    predicted_boundaries: torch.Tensor,
    true_boundaries: torch.Tensor,
    collar: float = 0.25,
) -> Dict[str, float]:
    """Compute all diarization metrics.

    Args:
        predicted_labels: Predicted speaker labels (batch, time).
        true_labels: Ground truth speaker labels (batch, time).
        predicted_boundaries: Predicted boundaries (batch, time).
        true_boundaries: Ground truth boundaries (batch, time).
        collar: Collar size for boundary evaluation.

    Returns:
        Dictionary of all computed metrics.
    """
    metrics_calculator = DiarizationMetrics(collar=collar)

    # Convert to numpy and flatten batches
    pred_labels_np = predicted_labels.cpu().numpy().flatten()
    true_labels_np = true_labels.cpu().numpy().flatten()
    pred_boundaries_np = predicted_boundaries.cpu().numpy().flatten()
    true_boundaries_np = true_boundaries.cpu().numpy().flatten()

    # Compute metrics
    metrics = {
        "diarization_error_rate": metrics_calculator.diarization_error_rate(
            pred_labels_np, true_labels_np, pred_boundaries_np, true_boundaries_np
        ),
        "jaccard_error_rate": metrics_calculator.jaccard_error_rate(
            pred_labels_np, true_labels_np
        ),
        "boundary_f1_score": metrics_calculator.boundary_f1_score(
            pred_boundaries_np, true_boundaries_np
        ),
        "speaker_confusion": metrics_calculator.speaker_confusion(
            pred_labels_np, true_labels_np
        ),
        "frame_accuracy": metrics_calculator.compute_frame_accuracy(
            pred_labels_np, true_labels_np
        ),
    }

    # Add precision, recall, F1
    prf1 = metrics_calculator.compute_precision_recall_f1(pred_labels_np, true_labels_np)
    metrics.update(prf1)

    logger.info(f"Computed metrics: {metrics}")

    return metrics
