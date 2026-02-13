"""Tests for model components."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_speaker_diarization_with_adaptive_boundary_refinement.models import (
    AdaptiveBoundaryRefinement,
    BoundaryConsistencyLoss,
    MultiSpeakerDiarizationModel,
    TemporalConsistencyLoss,
)


class TestMultiSpeakerDiarizationModel:
    """Tests for main diarization model."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80,
            embedding_dim=128,
            num_speakers=4,
            dropout=0.3,
            use_adaptive_refinement=True,
        )

        assert model.embedding_dim == 128
        assert model.num_speakers == 4
        assert model.use_adaptive_refinement is True

    def test_model_forward(self, sample_mel_spec, sample_spectral_flux):
        """Test forward pass."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4, use_adaptive_refinement=True
        )

        outputs = model(sample_mel_spec, sample_spectral_flux)

        assert "speaker_logits" in outputs
        assert "boundary_probs" in outputs
        assert "embeddings" in outputs
        assert "refined_boundaries" in outputs

        # Check shapes
        batch_size, mel_bins, time_frames = sample_mel_spec.shape
        assert outputs["speaker_logits"].shape == (batch_size, 4, time_frames)
        assert outputs["boundary_probs"].shape == (batch_size, time_frames)
        assert outputs["refined_boundaries"].shape == (batch_size, time_frames)

    def test_model_forward_no_refinement(self, sample_mel_spec, sample_spectral_flux):
        """Test forward pass without refinement."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4, use_adaptive_refinement=False
        )

        outputs = model(sample_mel_spec, sample_spectral_flux)

        # Refined boundaries should equal boundary probs when refinement is off
        assert torch.equal(outputs["refined_boundaries"], outputs["boundary_probs"])

    def test_model_predict_speakers(self, sample_mel_spec, sample_spectral_flux):
        """Test speaker prediction."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )
        model.eval()

        speaker_labels, boundary_points = model.predict_speakers(
            sample_mel_spec, sample_spectral_flux
        )

        batch_size, _, time_frames = sample_mel_spec.shape
        assert speaker_labels.shape == (batch_size, time_frames)
        assert boundary_points.shape == (batch_size, time_frames)

        # Check value ranges
        assert speaker_labels.min() >= 0
        assert speaker_labels.max() < 4
        assert boundary_points.min() >= 0
        assert boundary_points.max() <= 1

    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = MultiSpeakerDiarizationModel(
            input_dim=80, embedding_dim=128, num_speakers=4
        )

        param_count = model.count_parameters()
        assert param_count > 0


class TestAdaptiveBoundaryRefinement:
    """Tests for adaptive boundary refinement module."""

    def test_refinement_initialization(self):
        """Test refinement module initialization."""
        refinement = AdaptiveBoundaryRefinement(
            window_size=5, overlap=0.5, gradient_threshold=0.15, max_iterations=3
        )

        assert refinement.window_size == 5
        assert refinement.max_iterations == 3

    def test_refinement_forward(
        self, sample_mel_spec, sample_spectral_flux, sample_boundaries
    ):
        """Test refinement forward pass."""
        refinement = AdaptiveBoundaryRefinement(max_iterations=2)

        refined_boundaries, debug_info = refinement(
            sample_mel_spec, sample_spectral_flux, sample_boundaries
        )

        # Check output shape
        assert refined_boundaries.shape == sample_boundaries.shape

        # Check debug info
        assert "iterations" in debug_info
        assert "total_adjustments" in debug_info
        assert debug_info["iterations"] <= 2

    def test_refinement_values_in_range(
        self, sample_mel_spec, sample_spectral_flux, sample_boundaries
    ):
        """Test that refined boundaries stay in valid range."""
        refinement = AdaptiveBoundaryRefinement()

        refined_boundaries, _ = refinement(
            sample_mel_spec, sample_spectral_flux, sample_boundaries
        )

        assert refined_boundaries.min() >= 0.0
        assert refined_boundaries.max() <= 1.0


class TestBoundaryConsistencyLoss:
    """Tests for boundary consistency loss."""

    def test_loss_computation(self, sample_boundaries, sample_spectral_flux):
        """Test loss computation."""
        loss_fn = BoundaryConsistencyLoss()

        predicted = torch.rand_like(sample_boundaries)
        loss = loss_fn(predicted, sample_spectral_flux, sample_boundaries)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0.0

    def test_loss_gradient_flow(self, sample_boundaries, sample_spectral_flux):
        """Test that gradients flow through loss."""
        loss_fn = BoundaryConsistencyLoss()

        predicted = torch.rand_like(sample_boundaries, requires_grad=True)
        loss = loss_fn(predicted, sample_spectral_flux, sample_boundaries)

        loss.backward()

        assert predicted.grad is not None
        assert not torch.isnan(predicted.grad).any()


class TestTemporalConsistencyLoss:
    """Tests for temporal consistency loss."""

    def test_loss_computation(self, sample_spectral_flux):
        """Test loss computation."""
        loss_fn = TemporalConsistencyLoss()

        # Create sample speaker predictions
        batch_size, time_frames = sample_spectral_flux.shape
        speaker_predictions = torch.randn(batch_size, 4, time_frames)

        loss = loss_fn(speaker_predictions, sample_spectral_flux)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0.0

    def test_loss_gradient_flow(self, sample_spectral_flux):
        """Test gradient flow."""
        loss_fn = TemporalConsistencyLoss()

        batch_size, time_frames = sample_spectral_flux.shape
        speaker_predictions = torch.randn(
            batch_size, 4, time_frames, requires_grad=True
        )

        loss = loss_fn(speaker_predictions, sample_spectral_flux)
        loss.backward()

        assert speaker_predictions.grad is not None
        assert not torch.isnan(speaker_predictions.grad).any()
