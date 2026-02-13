#!/usr/bin/env python
"""Quick experiment script to generate results for README."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from multi_speaker_diarization_with_adaptive_boundary_refinement.data import create_dataloaders
from multi_speaker_diarization_with_adaptive_boundary_refinement.models import MultiSpeakerDiarizationModel
from multi_speaker_diarization_with_adaptive_boundary_refinement.training import DiarizationTrainer
from multi_speaker_diarization_with_adaptive_boundary_refinement.evaluation.metrics import compute_all_metrics
from multi_speaker_diarization_with_adaptive_boundary_refinement.utils import load_config, set_random_seeds

def run_experiment(config_path: str, use_refinement: bool) -> dict:
    """Run a quick experiment and return metrics."""
    config = load_config(config_path)
    config["model"]["use_adaptive_refinement"] = use_refinement
    set_random_seeds(42)

    device = torch.device("cpu")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, use_synthetic=True)

    # Initialize model
    model = MultiSpeakerDiarizationModel(
        input_dim=config["data"]["n_mels"],
        embedding_dim=config["model"]["embedding_dim"],
        num_speakers=config["model"]["num_speakers"],
        dropout=config["model"]["dropout"],
        use_adaptive_refinement=use_refinement,
        refinement_config=config.get("boundary_refinement", {}),
    )

    # Train for a few epochs
    trainer = DiarizationTrainer(model, config, device)
    trainer.train(train_loader, val_loader)

    # Evaluate on test set
    model.eval()
    all_predictions = []
    all_labels = []
    all_boundaries = []
    all_boundary_preds = []

    with torch.no_grad():
        for batch in test_loader:
            mel_spec = batch["mel_spectrogram"]
            spectral_flux = batch["spectral_flux"]
            labels = batch["labels"]
            boundaries = batch["boundaries"]

            speaker_labels, boundary_points = model.predict_speakers(mel_spec, spectral_flux)

            all_predictions.append(speaker_labels)
            all_labels.append(labels)
            all_boundaries.append(boundaries)
            all_boundary_preds.append(boundary_points)

    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    boundaries = torch.cat(all_boundaries, dim=0)
    boundary_preds = torch.cat(all_boundary_preds, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(
        predicted_labels=predictions,
        true_labels=labels,
        predicted_boundaries=boundary_preds,
        true_boundaries=boundaries,
        collar=config["evaluation"]["collar"],
    )

    return {
        "der": float(metrics["diarization_error_rate"]),
        "jer": float(metrics["jaccard_error_rate"]),
        "boundary_f1": float(metrics["boundary_f1_score"]),
        "confusion": float(metrics.get("speaker_confusion", 0.0)),
    }

if __name__ == "__main__":
    print("Running baseline experiment (no refinement)...")
    baseline_metrics = run_experiment("configs/quick_experiment.yaml", use_refinement=False)

    print("\nRunning experiment with adaptive refinement...")
    refined_metrics = run_experiment("configs/quick_experiment.yaml", use_refinement=True)

    results = {
        "baseline": baseline_metrics,
        "with_refinement": refined_metrics
    }

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print("\nBaseline (No Refinement):")
    print(f"  DER: {baseline_metrics['der']:.3f}")
    print(f"  JER: {baseline_metrics['jer']:.3f}")
    print(f"  Boundary F1: {baseline_metrics['boundary_f1']:.3f}")
    print(f"  Speaker Confusion: {baseline_metrics['confusion']:.3f}")

    print("\nWith Adaptive Refinement:")
    print(f"  DER: {refined_metrics['der']:.3f}")
    print(f"  JER: {refined_metrics['jer']:.3f}")
    print(f"  Boundary F1: {refined_metrics['boundary_f1']:.3f}")
    print(f"  Speaker Confusion: {refined_metrics['confusion']:.3f}")

    print("\nImprovement:")
    if baseline_metrics['der'] > 0:
        print(f"  DER: {(baseline_metrics['der'] - refined_metrics['der']):.3f} ({(baseline_metrics['der'] - refined_metrics['der'])/baseline_metrics['der']*100:.1f}%)")
    else:
        print(f"  DER: {(baseline_metrics['der'] - refined_metrics['der']):.3f}")
    if baseline_metrics['jer'] > 0:
        print(f"  JER: {(baseline_metrics['jer'] - refined_metrics['jer']):.3f} ({(baseline_metrics['jer'] - refined_metrics['jer'])/baseline_metrics['jer']*100:.1f}%)")
    else:
        print(f"  JER: {(baseline_metrics['jer'] - refined_metrics['jer']):.3f}")
    if baseline_metrics['boundary_f1'] > 0:
        print(f"  Boundary F1: {(refined_metrics['boundary_f1'] - baseline_metrics['boundary_f1']):.3f} ({(refined_metrics['boundary_f1'] - baseline_metrics['boundary_f1'])/baseline_metrics['boundary_f1']*100:.1f}%)")
    else:
        print(f"  Boundary F1: {(refined_metrics['boundary_f1'] - baseline_metrics['boundary_f1']):.3f}")
    print("="*60)
