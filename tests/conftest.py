"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data": {
            "sample_rate": 16000,
            "segment_duration": 3.0,
            "hop_duration": 0.5,
            "batch_size": 4,
            "num_workers": 0,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "synthetic_data_size": 20,
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": 80,
        },
        "model": {
            "embedding_dim": 128,
            "num_speakers": 4,
            "dropout": 0.3,
            "use_adaptive_refinement": True,
        },
        "boundary_refinement": {
            "enabled": True,
            "window_size": 5,
            "overlap": 0.5,
            "gradient_threshold": 0.15,
            "max_iterations": 3,
        },
        "training": {
            "num_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 5,
            "checkpoint_dir": "test_checkpoints",
            "best_model_path": "test_models/best_model.pt",
            "mixed_precision": False,
        },
        "scheduler": {
            "type": "cosine",
            "warmup_epochs": 1,
            "min_lr": 0.00001,
        },
        "loss": {
            "diarization_weight": 1.0,
            "boundary_consistency_weight": 0.5,
            "temporal_consistency_weight": 0.3,
            "speaker_confusion_penalty": 0.2,
        },
        "evaluation": {
            "collar": 0.25,
            "metrics": ["diarization_error_rate", "jaccard_error_rate"],
            "save_predictions": False,
            "results_dir": "test_results",
        },
        "mlflow": {
            "enabled": False,
        },
        "random_seed": 42,
    }


@pytest.fixture
def device():
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_audio():
    """Generate sample audio tensor."""
    # 3 seconds at 16kHz
    return torch.randn(1, 48000)


@pytest.fixture
def sample_mel_spec():
    """Generate sample mel spectrogram."""
    # (batch=2, mel_bins=80, time=300)
    return torch.randn(2, 80, 300)


@pytest.fixture
def sample_spectral_flux():
    """Generate sample spectral flux features."""
    # (batch=2, time=300)
    return torch.randn(2, 300)


@pytest.fixture
def sample_labels():
    """Generate sample speaker labels."""
    # (batch=2, time=300)
    return torch.randint(0, 4, (2, 300))


@pytest.fixture
def sample_boundaries():
    """Generate sample boundary labels."""
    # (batch=2, time=300)
    boundaries = torch.zeros(2, 300)
    # Add some boundaries
    boundaries[0, 50] = 1.0
    boundaries[0, 150] = 1.0
    boundaries[1, 100] = 1.0
    boundaries[1, 200] = 1.0
    return boundaries
