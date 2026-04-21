"""Tests for SVD analysis and effective rank computation."""

import torch

from asd.profiling.svd_analysis import SVDAnalyzer, profiles_to_stage_widths
from asd.profiling.sparsity_analysis import SparsityStats


def _make_dummy_sparsity():
    return SparsityStats(
        sparsity_ratio=0.5,
        activation_histogram=torch.ones(64) / 64,
        bin_edges=torch.linspace(0, 1, 65),
        entropy=4.0,
        mean_activation=0.5,
        std_activation=0.3,
    )


def test_effective_rank_low_rank():
    """Low-rank covariance should have small effective rank."""
    analyzer = SVDAnalyzer(variance_threshold=0.95)

    # Create a rank-5 covariance in 64-dim space
    A = torch.randn(5, 64)
    cov = A.T @ A  # rank 5

    profile = analyzer.analyze("test_layer", cov, _make_dummy_sparsity())
    assert profile.effective_rank <= 10, f"Expected rank ~5, got {profile.effective_rank}"
    assert profile.effective_rank >= 4, f"Expected rank ~5, got {profile.effective_rank}"


def test_effective_rank_full_rank():
    """Identity covariance (full rank) should need all components at 95%."""
    analyzer = SVDAnalyzer(variance_threshold=0.95)
    cov = torch.eye(32)
    profile = analyzer.analyze("test_layer", cov, _make_dummy_sparsity())
    # For identity, cumsum reaches 0.95 at 95% of components
    assert profile.effective_rank >= 28  # should be ~31 for 95% of 32


def test_eigenvalues_sum_to_trace():
    """Sum of eigenvalues should equal trace of covariance."""
    analyzer = SVDAnalyzer()
    cov = torch.randn(20, 20)
    cov = cov @ cov.T  # Make PSD

    profile = analyzer.analyze("test", cov, _make_dummy_sparsity())
    sv_sum = profile.eigenvalues.sum().item()
    trace = cov.diagonal().sum().item()
    assert abs(sv_sum - trace) < 0.1, f"SV sum {sv_sum:.4f} != trace {trace:.4f}"


def test_principal_components_shape():
    """Principal components should be (C, k) where k = effective_rank."""
    analyzer = SVDAnalyzer(variance_threshold=0.95)
    A = torch.randn(3, 32)
    cov = A.T @ A

    profile = analyzer.analyze("test", cov, _make_dummy_sparsity())
    assert profile.principal_components.shape == (32, profile.effective_rank)


def test_sparsity_analyzer_identical_values():
    """Histogram should handle the edge case where all values are identical."""
    from asd.profiling.sparsity_analysis import SparsityAnalyzer

    analyzer = SparsityAnalyzer(num_bins=64)
    # All non-zero values are identical
    sample = torch.ones(1000) * 5.0
    stats = analyzer.analyze(sparsity_ratio=0.0, activation_sample=sample)

    assert torch.isfinite(stats.activation_histogram).all()
    assert stats.bin_edges[0] < stats.bin_edges[-1], "Bin edges must span a range"


def test_profiles_to_stage_widths():
    """Stage widths should be rounded to multiples of width_multiple."""
    from asd.profiling.svd_analysis import LayerProfile

    profiles = [
        LayerProfile("l1.0", torch.ones(10), torch.randn(256, 5), 45, 256, 0.18, _make_dummy_sparsity()),
        LayerProfile("l2.0", torch.ones(10), torch.randn(512, 5), 89, 512, 0.17, _make_dummy_sparsity()),
        LayerProfile("l3.0", torch.ones(10), torch.randn(1024, 5), 156, 1024, 0.15, _make_dummy_sparsity()),
        LayerProfile("l4.0", torch.ones(10), torch.randn(2048, 5), 312, 2048, 0.15, _make_dummy_sparsity()),
    ]

    widths = profiles_to_stage_widths(profiles, min_width=16, width_multiple=8)
    assert len(widths) == 4
    for w in widths:
        assert w % 8 == 0, f"Width {w} not a multiple of 8"
        assert w >= 16, f"Width {w} below minimum 16"


def test_profiles_save_load_roundtrip():
    """Verify save → load preserves all profile data exactly."""
    import tempfile
    from asd.profiling.svd_analysis import LayerProfile, save_profiles, load_profiles

    profiles = [
        LayerProfile("layer1.2", torch.randn(256), torch.randn(256, 48), 48, 256, 0.19, _make_dummy_sparsity()),
        LayerProfile("layer2.3", torch.randn(512), torch.randn(512, 96), 96, 512, 0.19, _make_dummy_sparsity()),
        LayerProfile("layer3.5", torch.randn(1024), torch.randn(1024, 160), 160, 1024, 0.16, _make_dummy_sparsity()),
        LayerProfile("layer4.2", torch.randn(2048), torch.randn(2048, 320), 320, 2048, 0.16, _make_dummy_sparsity()),
    ]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_profiles(profiles, f.name)
        loaded = load_profiles(f.name)

    assert len(loaded) == len(profiles)
    for orig, ld in zip(profiles, loaded):
        assert orig.name == ld.name
        assert orig.effective_rank == ld.effective_rank
        assert orig.total_channels == ld.total_channels
        assert torch.allclose(orig.eigenvalues, ld.eigenvalues)
        assert torch.allclose(orig.principal_components, ld.principal_components)
        assert abs(orig.sparsity_stats.sparsity_ratio - ld.sparsity_stats.sparsity_ratio) < 1e-6
        assert torch.allclose(orig.sparsity_stats.activation_histogram, ld.sparsity_stats.activation_histogram)
        assert torch.allclose(orig.sparsity_stats.bin_edges, ld.sparsity_stats.bin_edges)
