#!/usr/bin/env python
"""Training script for multi-speaker diarization with adaptive boundary refinement."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from multi_speaker_diarization_with_adaptive_boundary_refinement.data import create_dataloaders
from multi_speaker_diarization_with_adaptive_boundary_refinement.models import (
    MultiSpeakerDiarizationModel,
)
from multi_speaker_diarization_with_adaptive_boundary_refinement.training import DiarizationTrainer
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
    parser = argparse.ArgumentParser(
        description="Train multi-speaker diarization model with adaptive boundary refinement"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data for training",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seeds for reproducibility
    seed = config.get("random_seed", 42)
    set_random_seeds(seed)
    logger.info(f"Set random seed to {seed}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            config, use_synthetic=args.use_synthetic
        )
        logger.info(
            f"Created dataloaders: {len(train_loader)} train, "
            f"{len(val_loader)} val, {len(test_loader)} test batches"
        )
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise

    # Initialize model
    logger.info("Initializing model...")
    try:
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

        logger.info(f"Model initialized with {model.count_parameters():,} parameters")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    # Initialize trainer
    logger.info("Initializing trainer...")
    try:
        trainer = DiarizationTrainer(model, config, device)
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Initialize MLflow logger
    mlflow_logger = None
    if config.get("mlflow", {}).get("enabled", False):
        try:
            import mlflow

            mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", "mlruns"))
            mlflow.set_experiment(config["mlflow"]["experiment_name"])
            mlflow.start_run()

            # Log configuration
            mlflow.log_params(
                {
                    "embedding_dim": model_config["embedding_dim"],
                    "num_speakers": model_config["num_speakers"],
                    "learning_rate": config["training"]["learning_rate"],
                    "batch_size": config["data"]["batch_size"],
                    "use_adaptive_refinement": model_config["use_adaptive_refinement"],
                }
            )

            mlflow_logger = mlflow
            logger.info("MLflow logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}. Continuing without MLflow.")
            mlflow_logger = None

    # Train model
    logger.info("Starting training...")
    try:
        training_history = trainer.train(train_loader, val_loader, mlflow_logger)

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Best model saved to: {trainer.best_model_path}")

        # Save training history
        results_dir = Path(config["evaluation"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        import json

        history_path = results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # End MLflow run
        if mlflow_logger is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
