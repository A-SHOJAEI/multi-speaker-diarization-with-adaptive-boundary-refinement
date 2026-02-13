# Multi-Speaker Diarization with Adaptive Boundary Refinement

A speaker diarization system combining CNN-LSTM architecture with adaptive boundary refinement. The system iteratively adjusts speaker change points using learned acoustic consistency scoring.

## Features

- Adaptive boundary refinement using spectral similarity gradients
- Custom temporal and boundary consistency loss functions
- Real-time inference with configurable refinement iterations
- Comprehensive evaluation metrics (DER, JER, boundary F1)

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Training

```bash
python scripts/train.py
python scripts/train.py --config configs/ablation.yaml  # baseline without refinement
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

### Inference

```bash
python scripts/predict.py --audio path/to/audio.wav --checkpoint models/best_model.pt
```

## Architecture

1. **Feature Extraction**: CNN layers extract hierarchical features from mel spectrograms
2. **Temporal Modeling**: Bidirectional LSTM captures long-range dependencies
3. **Adaptive Refinement**: Iteratively refines boundaries using:
   - Local acoustic consistency scores (mel similarity, spectral flux, temporal smoothness)
   - Gradient-based boundary detection
   - Learned similarity weights

## Key Innovation

The adaptive boundary refinement module addresses boundary precision issues in clustering-based diarization by:

- Computing multi-scale consistency using three acoustic signals: mel-spectrogram cosine similarity, spectral flux inverse, and temporal smoothness
- Combining signals via learned weights optimized during training
- Iteratively adjusting boundaries: weakening in high-consistency regions, strengthening in low-consistency regions
- Running 3-5 refinement iterations with early stopping

Custom loss functions enforce alignment:
- **BoundaryConsistencyLoss**: Penalizes boundaries misaligned with spectral discontinuities
- **TemporalConsistencyLoss**: Uses KL divergence to prevent unrealistic speaker changes

## Configuration

Key parameters in `configs/default.yaml`:

- `model.use_adaptive_refinement`: Enable/disable refinement (default: true)
- `boundary_refinement.max_iterations`: Refinement iterations (default: 5)
- `loss.boundary_consistency_weight`: Boundary loss weight (default: 0.5)
- `loss.temporal_consistency_weight`: Temporal loss weight (default: 0.3)

## Results

Experiments on synthetic data (run `python scripts/train.py` to reproduce):

| Metric | Baseline | With Refinement |
|--------|----------|-----------------|
| Diarization Error Rate | TBD | TBD |
| Jaccard Error Rate | TBD | TBD |
| Boundary F1 Score | TBD | TBD |

Real dataset support: Implement AMI Meeting Corpus data loader in `src/data/loader.py` and set `use_synthetic=False`.

## Project Structure

```
src/multi_speaker_diarization_with_adaptive_boundary_refinement/
├── data/               # Data loading and preprocessing
├── models/             # Model architecture and custom components
├── training/           # Training loop with LR scheduling
├── evaluation/         # Metrics and analysis
└── utils/              # Configuration and helpers

scripts/
├── train.py           # Training pipeline
├── evaluate.py        # Evaluation script
└── predict.py         # Inference script
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei
