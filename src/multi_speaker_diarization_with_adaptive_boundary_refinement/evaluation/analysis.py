"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def analyze_results(
    predictions: List[Dict[str, Any]],
    save_dir: str,
) -> Dict[str, Any]:
    """Analyze and save evaluation results.

    Args:
        predictions: List of prediction dictionaries containing metrics.
        save_dir: Directory to save analysis results.

    Returns:
        Dictionary containing aggregated analysis.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics across all samples
    all_metrics = {}
    for pred in predictions:
        for key, value in pred.get("metrics", {}).items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)

    # Compute statistics
    analysis = {
        "mean_metrics": {},
        "std_metrics": {},
        "min_metrics": {},
        "max_metrics": {},
    }

    for key, values in all_metrics.items():
        analysis["mean_metrics"][key] = float(np.mean(values))
        analysis["std_metrics"][key] = float(np.std(values))
        analysis["min_metrics"][key] = float(np.min(values))
        analysis["max_metrics"][key] = float(np.max(values))

    # Save analysis to JSON
    analysis_path = save_path / "analysis_summary.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Saved analysis summary to {analysis_path}")

    # Create detailed CSV
    df = pd.DataFrame(all_metrics)
    csv_path = save_path / "detailed_metrics.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved detailed metrics to {csv_path}")

    return analysis


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    save_path: str,
    num_speakers: int = 4,
) -> None:
    """Plot and save confusion matrix for speaker classification.

    Args:
        true_labels: Ground truth speaker labels.
        predicted_labels: Predicted speaker labels.
        save_path: Path to save the plot.
        num_speakers: Number of speaker classes.
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"Speaker {i}" for i in range(num_speakers)],
        yticklabels=[f"Speaker {i}" for i in range(num_speakers)],
    )
    plt.title("Speaker Confusion Matrix")
    plt.ylabel("True Speaker")
    plt.xlabel("Predicted Speaker")
    plt.tight_layout()

    # Save
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {save_file}")


def plot_training_curves(
    training_history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Plot and save training curves.

    Args:
        training_history: Dictionary containing training metrics history.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(training_history["train_loss"]) + 1)

    axes.plot(epochs, training_history["train_loss"], label="Train Loss", marker="o")
    axes.plot(epochs, training_history["val_loss"], label="Validation Loss", marker="s")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_title("Training and Validation Loss")
    axes.legend()
    axes.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curves to {save_file}")


def plot_boundary_analysis(
    true_boundaries: np.ndarray,
    predicted_boundaries: np.ndarray,
    save_path: str,
    max_samples: int = 5,
) -> None:
    """Plot boundary detection visualization.

    Args:
        true_boundaries: Ground truth boundaries.
        predicted_boundaries: Predicted boundaries.
        save_path: Path to save the plot.
        max_samples: Maximum number of samples to visualize.
    """
    num_samples = min(max_samples, len(true_boundaries))

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]

        time_frames = np.arange(len(true_boundaries[i]))

        ax.plot(time_frames, true_boundaries[i], label="True Boundaries", linewidth=2)
        ax.plot(
            time_frames,
            predicted_boundaries[i],
            label="Predicted Boundaries",
            alpha=0.7,
            linewidth=2,
        )

        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Boundary Probability")
        ax.set_title(f"Sample {i + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved boundary analysis to {save_file}")


def plot_metric_comparison(
    baseline_metrics: Dict[str, float],
    refined_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Plot comparison between baseline and refined model metrics.

    Args:
        baseline_metrics: Metrics from baseline model.
        refined_metrics: Metrics from model with refinement.
        save_path: Path to save the plot.
    """
    # Common metrics to compare
    metrics_to_plot = [
        "diarization_error_rate",
        "jaccard_error_rate",
        "boundary_f1_score",
        "speaker_confusion",
    ]

    baseline_values = [baseline_metrics.get(m, 0) for m in metrics_to_plot]
    refined_values = [refined_metrics.get(m, 0) for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, baseline_values, width, label="Baseline", alpha=0.8)
    ax.bar(x + width / 2, refined_values, width, label="With Refinement", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Adaptive Boundary Refinement")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved metric comparison to {save_file}")
