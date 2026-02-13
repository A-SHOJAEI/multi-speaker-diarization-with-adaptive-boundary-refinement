"""Tests for data loading and preprocessing."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_speaker_diarization_with_adaptive_boundary_refinement.data import (
    AudioPreprocessor,
    DiarizationDataset,
    create_dataloaders,
    generate_synthetic_data,
)


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AudioPreprocessor(sample_rate=16000)
        assert preprocessor.sample_rate == 16000
        assert preprocessor.n_mels == 80

    def test_extract_mel_spectrogram(self, sample_audio):
        """Test mel spectrogram extraction."""
        preprocessor = AudioPreprocessor(sample_rate=16000)
        mel_spec = preprocessor.extract_mel_spectrogram(sample_audio)

        assert mel_spec.dim() == 2  # (mel_bins, time)
        assert mel_spec.shape[0] == 80  # mel_bins

    def test_compute_mfcc(self, sample_audio):
        """Test MFCC computation."""
        preprocessor = AudioPreprocessor(sample_rate=16000)
        mfcc = preprocessor.compute_mfcc(sample_audio, n_mfcc=13)

        assert mfcc.dim() == 2  # (n_mfcc, time)
        assert mfcc.shape[0] == 13

    def test_compute_spectral_features(self, sample_audio):
        """Test spectral feature extraction."""
        preprocessor = AudioPreprocessor(sample_rate=16000)
        features = preprocessor.compute_spectral_features(sample_audio)

        assert "spectral_centroid" in features
        assert "spectral_rolloff" in features
        assert "spectral_flux" in features
        assert "zero_crossing_rate" in features

        # Check shapes
        for key, value in features.items():
            assert value.dim() == 1  # Time dimension only


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(
            num_samples=10, sample_rate=16000, duration=2.0, num_speakers=3, seed=42
        )

        assert len(data) == 10

        for waveform, labels, boundaries in data:
            # Check waveform shape
            assert waveform.shape[0] == 1  # Mono
            assert waveform.shape[1] == 32000  # 2 seconds at 16kHz

            # Check labels
            assert labels.dim() == 1
            assert labels.max() < 3  # num_speakers

            # Check boundaries
            assert len(boundaries) > 0
            for start, end, speaker_id in boundaries:
                assert 0 <= start < end <= 2.0
                assert 0 <= speaker_id < 3

    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible."""
        data1 = generate_synthetic_data(num_samples=5, seed=42)
        data2 = generate_synthetic_data(num_samples=5, seed=42)

        for (w1, l1, b1), (w2, l2, b2) in zip(data1, data2):
            assert torch.allclose(w1, w2)
            assert torch.equal(l1, l2)
            assert b1 == b2


class TestDiarizationDataset:
    """Tests for DiarizationDataset class."""

    def test_dataset_creation(self, sample_config):
        """Test dataset creation."""
        data = generate_synthetic_data(num_samples=5, seed=42)
        preprocessor = AudioPreprocessor()

        dataset = DiarizationDataset(data, preprocessor, augment=False)

        assert len(dataset) == 5

    def test_dataset_getitem(self, sample_config):
        """Test getting items from dataset."""
        data = generate_synthetic_data(num_samples=5, seed=42)
        preprocessor = AudioPreprocessor()

        dataset = DiarizationDataset(data, preprocessor, augment=False)

        sample = dataset[0]

        assert "waveform" in sample
        assert "mel_spectrogram" in sample
        assert "spectral_flux" in sample
        assert "labels" in sample
        assert "boundaries" in sample
        assert "num_speakers" in sample

        # Check shapes
        assert sample["mel_spectrogram"].dim() == 2
        assert sample["spectral_flux"].dim() == 1
        assert sample["labels"].dim() == 1
        assert sample["boundaries"].dim() == 1

    def test_dataset_augmentation(self, sample_config):
        """Test data augmentation."""
        data = generate_synthetic_data(num_samples=2, seed=42)
        preprocessor = AudioPreprocessor()

        dataset_no_aug = DiarizationDataset(data, preprocessor, augment=False)
        dataset_with_aug = DiarizationDataset(data, preprocessor, augment=True)

        # Get same sample multiple times with augmentation
        sample1 = dataset_with_aug[0]
        sample2 = dataset_with_aug[0]

        # Augmentation may produce different results
        # Just check that shapes are preserved
        assert sample1["waveform"].shape == sample2["waveform"].shape


class TestDataLoaders:
    """Tests for dataloader creation."""

    def test_create_dataloaders(self, sample_config):
        """Test dataloader creation."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_config, use_synthetic=True
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        # Test batch retrieval
        batch = next(iter(train_loader))

        assert "mel_spectrogram" in batch
        assert "labels" in batch
        assert "boundaries" in batch

        # Check batch size
        assert batch["mel_spectrogram"].shape[0] <= sample_config["data"]["batch_size"]
