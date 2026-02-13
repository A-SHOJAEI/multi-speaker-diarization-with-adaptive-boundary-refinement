#!/usr/bin/env python
"""Evaluation script for multi-speaker diarization model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from multi_speaker_diarization_with_adaptive_boundary_refinement.data import create_dataloaders
from multi_speaker_diarization_with_adaptive_boundary_refinement.evaluation import (
    analyze_results,
    compute_all_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)
from multi_speaker_diarization_with_adaptive_boundary_refinement.models import (
    MultiSpeakerDiarizationModel,
)
from multi_speaker_diarization_with_adaptive_boundary_refinement.utils import (
    load_config,
    set_random_seeds,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate speaker diarization model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data for evaluation",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> MultiSpeakerDiarizationModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    model_config = config["model"]
    refinement_config = config.get("boundary_refinement", {})

    model = MultiSpeakerDiarizationModel(
        input_dim=config["data"].get("n_mels", 80),
        embedding_dim=model_config["embedding_dim"],
        num_speakers=model_config["num_speakers"],
        dropout=model_config["dropout"],
        use_adaptive_refinement=model_config["use_adaptive_refinement"],
        refinement_config=refinement_config,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


def evaluate_model(
    model: MultiSpeakerDiarizationModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
) -> dict:
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate.
        data_loader: Data loader.
        device: Device for computation.
        config: Configuration dictionary.

    Returns:
        Dictionary containing evaluation results.
    """
    all_predictions = []
    all_true_labels = []
    all_pred_labels = []
    all_true_boundaries = []
    all_pred_boundaries = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            mel_spec = batch["mel_spectrogram"].to(device)
            spectral_flux = batch["spectral_flux"].to(device)
            true_labels = batch["labels"].to(device)
            true_boundaries = batch["boundaries"].to(device)

            # Forward pass
            speaker_labels, boundary_points = model.predict_speakers(mel_spec, spectral_flux)

            # Collect predictions
            all_true_labels.append(true_labels)
            all_pred_labels.append(speaker_labels)
            all_true_boundaries.append(true_boundaries)
            all_pred_boundaries.append(boundary_points.float())

    # Concatenate all batches
    all_true_labels = torch.cat(all_true_labels, dim=0)
    all_pred_labels = torch.cat(all_pred_labels, dim=0)
    all_true_boundaries = torch.cat(all_true_boundaries, dim=0)
    all_pred_boundaries = torch.cat(all_pred_boundaries, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(
        all_pred_labels,
        all_true_labels,
        all_pred_boundaries,
        all_true_boundaries,
        collar=config["evaluation"]["collar"],
    )

    return {
        "metrics": metrics,
        "predictions": {
            "true_labels": all_true_labels.cpu().numpy(),
            "pred_labels": all_pred_labels.cpu().numpy(),
            "true_boundaries": all_true_boundaries.cpu().numpy(),
            "pred_boundaries": all_pred_boundaries.cpu().numpy(),
        },
    }


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seeds
    set_random_seeds(config.get("random_seed", 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(config, use_synthetic=args.use_synthetic)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)

    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, device, config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics to JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in results["metrics"].items():
        print(f"{metric:30s}: {value:.4f}")
    print("=" * 60 + "\n")

    # Generate visualizations
    logger.info("Generating visualizations...")

    # Confusion matrix
    plot_confusion_matrix(
        results["predictions"]["true_labels"].flatten(),
        results["predictions"]["pred_labels"].flatten(),
        str(output_dir / "confusion_matrix.png"),
        num_speakers=config["model"]["num_speakers"],
    )

    # Load and plot training curves if available
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            training_history = json.load(f)
        plot_training_curves(training_history, str(output_dir / "training_curves.png"))

    # Save detailed results
    detailed_results_path = output_dir / "detailed_results.npz"
    np.savez(
        detailed_results_path,
        true_labels=results["predictions"]["true_labels"],
        pred_labels=results["predictions"]["pred_labels"],
        true_boundaries=results["predictions"]["true_boundaries"],
        pred_boundaries=results["predictions"]["pred_boundaries"],
    )
    logger.info(f"Saved detailed results to {detailed_results_path}")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
