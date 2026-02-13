#!/usr/bin/env python
"""Prediction script for speaker diarization on new audio files."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from multi_speaker_diarization_with_adaptive_boundary_refinement.data import AudioPreprocessor
from multi_speaker_diarization_with_adaptive_boundary_refinement.models import (
    MultiSpeakerDiarizationModel,
)
from multi_speaker_diarization_with_adaptive_boundary_refinement.utils import load_config

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
    parser = argparse.ArgumentParser(description="Predict speaker diarization for audio file")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file or directory of audio files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of predictions",
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str, config: dict, device: torch.device
) -> MultiSpeakerDiarizationModel:
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

    return model


def predict_audio(
    audio_path: str,
    model: MultiSpeakerDiarizationModel,
    preprocessor: AudioPreprocessor,
    device: torch.device,
) -> dict:
    """Predict speaker diarization for audio file.

    Args:
        audio_path: Path to audio file.
        model: Trained model.
        preprocessor: Audio preprocessor.
        device: Device for computation.

    Returns:
        Dictionary containing predictions and metadata.
    """
    logger.info(f"Processing audio: {audio_path}")

    # Load and preprocess audio
    waveform = preprocessor.load_audio(audio_path)
    mel_spec = preprocessor.extract_mel_spectrogram(waveform)
    spectral_features = preprocessor.compute_spectral_features(waveform)

    # Add batch dimension and move to device
    mel_spec = mel_spec.unsqueeze(0).to(device)
    spectral_flux = spectral_features["spectral_flux"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        speaker_labels, boundary_points = model.predict_speakers(mel_spec, spectral_flux)

    # Convert to numpy
    speaker_labels = speaker_labels.cpu().numpy()[0]
    boundary_points = boundary_points.cpu().numpy()[0]

    # Extract segments
    segments = extract_segments(speaker_labels, boundary_points, frame_duration=0.01)

    # Compute confidence scores (based on boundary probabilities)
    outputs = model(mel_spec, spectral_flux)
    speaker_probs = torch.softmax(outputs["speaker_logits"], dim=1)
    confidence_scores = speaker_probs.max(dim=1)[0].cpu().numpy()[0]

    return {
        "audio_path": audio_path,
        "segments": segments,
        "speaker_labels": speaker_labels.tolist(),
        "boundary_points": boundary_points.tolist(),
        "confidence_scores": confidence_scores.tolist(),
        "num_speakers": len(set(speaker_labels)),
    }


def extract_segments(
    labels: np.ndarray, boundaries: np.ndarray, frame_duration: float = 0.01
) -> List[dict]:
    """Extract speaker segments from predictions.

    Args:
        labels: Speaker labels per frame.
        boundaries: Boundary probabilities per frame.
        frame_duration: Duration of each frame in seconds.

    Returns:
        List of segment dictionaries.
    """
    segments = []
    current_speaker = labels[0]
    start_frame = 0

    for i in range(1, len(labels)):
        if labels[i] != current_speaker or boundaries[i] > 0.5:
            # End of segment
            end_frame = i
            segments.append(
                {
                    "speaker_id": int(current_speaker),
                    "start_time": start_frame * frame_duration,
                    "end_time": end_frame * frame_duration,
                    "duration": (end_frame - start_frame) * frame_duration,
                }
            )
            current_speaker = labels[i]
            start_frame = i

    # Add final segment
    segments.append(
        {
            "speaker_id": int(current_speaker),
            "start_time": start_frame * frame_duration,
            "end_time": len(labels) * frame_duration,
            "duration": (len(labels) - start_frame) * frame_duration,
        }
    )

    return segments


def visualize_predictions(predictions: dict, output_path: str) -> None:
    """Visualize predictions.

    Args:
        predictions: Prediction dictionary.
        output_path: Path to save visualization.
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot speaker labels
        frames = np.arange(len(predictions["speaker_labels"]))
        ax1.plot(frames, predictions["speaker_labels"], drawstyle="steps-post")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Speaker ID")
        ax1.set_title(f"Speaker Diarization - {Path(predictions['audio_path']).name}")
        ax1.grid(True, alpha=0.3)

        # Plot boundaries
        ax2.plot(frames, predictions["boundary_points"])
        ax2.axhline(y=0.5, color="r", linestyle="--", label="Threshold")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Boundary Probability")
        ax2.set_title("Speaker Boundaries")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved visualization to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sample_rate=config["data"]["sample_rate"])

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)

    # Get audio files
    audio_path = Path(args.audio)
    if audio_path.is_dir():
        audio_files = list(audio_path.glob("*.wav")) + list(audio_path.glob("*.mp3"))
    else:
        audio_files = [audio_path]

    logger.info(f"Found {len(audio_files)} audio file(s) to process")

    # Process each audio file
    all_predictions = []
    for audio_file in audio_files:
        try:
            predictions = predict_audio(str(audio_file), model, preprocessor, device)
            all_predictions.append(predictions)

            # Print summary
            print(f"\n{audio_file.name}:")
            print(f"  Number of speakers: {predictions['num_speakers']}")
            print(f"  Number of segments: {len(predictions['segments'])}")
            for i, segment in enumerate(predictions["segments"][:5]):  # Show first 5
                print(
                    f"    Segment {i+1}: Speaker {segment['speaker_id']} "
                    f"({segment['start_time']:.2f}s - {segment['end_time']:.2f}s)"
                )
            if len(predictions["segments"]) > 5:
                print(f"    ... and {len(predictions['segments']) - 5} more segments")

            # Generate visualization if requested
            if args.visualize:
                vis_path = Path(args.output).parent / f"{audio_file.stem}_visualization.png"
                visualize_predictions(predictions, str(vis_path))

        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            continue

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)

    logger.info(f"Saved predictions to {output_path}")
    logger.info("Prediction completed successfully!")


if __name__ == "__main__":
    main()
