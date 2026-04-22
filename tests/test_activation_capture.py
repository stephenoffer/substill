"""Tests for activation capture engine and covariance accumulation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from asd.profiling.activation_capture import (
    ActivationCaptureEngine,
    CovarianceAccumulator,
)


def test_covariance_accumulator_shape():
    """Covariance matrix should be (C, C)."""
    acc = CovarianceAccumulator(num_channels=64)
    batch = torch.randn(8, 64, 16, 16)
    acc.update(batch)
    cov = acc.finalize()
    assert cov.shape == (64, 64)


def test_covariance_accumulator_symmetry():
    """Covariance matrix should be symmetric."""
    acc = CovarianceAccumulator(num_channels=32)
    for _ in range(5):
        acc.update(torch.randn(4, 32, 8, 8))
    cov = acc.finalize()
    assert torch.allclose(cov, cov.T, atol=1e-5)


def test_covariance_trace_matches_variance_gap():
    """In GAP mode, trace should equal sum of per-channel variance of GAP features."""
    acc = CovarianceAccumulator(num_channels=16, mode="gap")
    all_pooled = []
    for _ in range(50):
        batch = torch.randn(32, 16, 4, 4)
        acc.update(batch)
        all_pooled.append(batch.mean(dim=(2, 3)))

    cov = acc.finalize()
    all_pooled = torch.cat(all_pooled)
    expected_var = all_pooled.var(dim=0, correction=0).sum()
    trace = cov.diagonal().sum()
    assert abs(trace.item() - expected_var.item()) < 0.1, \
        f"Trace {trace.item():.4f} != expected variance {expected_var.item():.4f}"


def test_covariance_trace_matches_variance_per_pixel():
    """In per-pixel mode, trace should equal sum of per-channel variance of all pixels."""
    acc = CovarianceAccumulator(num_channels=16, mode="per_pixel")
    all_pixels = []
    for _ in range(50):
        batch = torch.randn(32, 16, 4, 4)
        acc.update(batch)
        all_pixels.append(batch.permute(0, 2, 3, 1).reshape(-1, 16))

    cov = acc.finalize()
    all_pixels = torch.cat(all_pixels)
    expected_var = all_pixels.var(dim=0, correction=0).sum()
    trace = cov.diagonal().sum()
    assert abs(trace.item() - expected_var.item()) < 0.1, \
        f"Trace {trace.item():.4f} != expected variance {expected_var.item():.4f}"


def test_sparsity_tracking():
    """Sparsity ratio should be correct for known input."""
    acc = CovarianceAccumulator(num_channels=8)
    # Create tensor with known sparsity (50% zeros via ReLU-like)
    x = torch.randn(10, 8, 4, 4)
    x[x < 0] = 0  # ~50% zeros
    acc.update(x)
    ratio = acc.sparsity_ratio
    assert 0.3 < ratio < 0.7, f"Expected ~50% sparsity, got {ratio:.2f}"


def test_capture_engine_hooks():
    """Engine should capture activations from specified layers."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
    )
    engine = ActivationCaptureEngine(model, layer_names=["1", "3"])

    dataset = TensorDataset(torch.randn(16, 3, 8, 8), torch.zeros(16, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=4)

    accumulators = engine.run(loader, device="cpu")
    assert "1" in accumulators
    assert "3" in accumulators
    assert accumulators["1"].num_channels == 16
    assert accumulators["3"].num_channels == 32


def test_histogram_budget_tracking():
    """Histogram budget should stop storing after limit."""
    acc = CovarianceAccumulator(num_channels=8)
    acc._hist_budget = 500  # Small budget for testing

    for _ in range(100):
        acc.update(torch.randn(32, 8, 4, 4))

    sample = acc.get_activation_sample()
    # Should have stopped before storing all 100 batches worth
    assert len(sample) <= acc._hist_budget + 1000  # some slack for last batch
