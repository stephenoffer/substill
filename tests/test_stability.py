"""Tests for T2.4 — subspace stability diagnostic."""

from __future__ import annotations

import torch
import torch.nn as nn

from asd.profiling.stability import (
    _principal_angles,
    bootstrap_principal_angles,
)


def test_principal_angles_identical_subspaces_are_zero():
    """Identical column-orthonormal bases → all principal angles ≈ 0.

    Float64 SVD gives exact unit singular values, but float32 accumulates
    ~1e-7 error in cos(angle), translating to ~0.03° of spurious angle.
    Use a realistic tolerance rather than machine-epsilon.
    """
    C, k = 32, 5
    V = torch.linalg.qr(torch.randn(C, k)).Q[:, :k]
    angles = _principal_angles(V, V)
    # 0.1° is well within the numerical noise floor of the float32 SVD
    # path used by torch.linalg.svdvals.
    assert torch.all(angles < 0.1), f"expected near-zero angles, got {angles}"


def test_principal_angles_orthogonal_subspaces_are_90_deg():
    """Orthogonal complements → largest angle = 90°."""
    C = 16
    basis = torch.eye(C)
    V_a = basis[:, :4]
    V_b = basis[:, 4:8]
    angles = _principal_angles(V_a, V_b)
    # arccos(0) = π/2 = 90°
    assert torch.allclose(angles, torch.full((4,), 90.0), atol=1e-3)


def test_principal_angles_rotation_within_subspace_is_zero():
    """A basis rotation within the same span gives near-zero principal angles."""
    C, k = 16, 4
    V = torch.linalg.qr(torch.randn(C, k)).Q[:, :k]
    Q = torch.linalg.qr(torch.randn(k, k)).Q
    V_rot = V @ Q
    angles = _principal_angles(V, V_rot)
    # Same tolerance as the identity case — float32 SVD noise floor.
    assert torch.all(angles < 0.1), f"expected near-zero angles, got {angles}"


class _ToyResidualNet(nn.Module):
    """Minimal model with a single block that the stability runner can hook."""

    def __init__(self, C: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
        )
        self.stem = nn.Conv2d(3, C, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        return self.block(x)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=32, C=3):
        torch.manual_seed(0)
        self.x = torch.randn(n, 3, 8, 8)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], torch.zeros(1)


def test_bootstrap_principal_angles_smoke():
    """End-to-end smoke: the stability runner produces finite per-layer
    statistics on a toy model."""
    torch.manual_seed(0)
    model = _ToyResidualNet(C=8)
    ds = _TinyDataset(n=32)

    out = bootstrap_principal_angles(
        model, ds, ["block"],
        n_boot=3, frac=0.5, variance_threshold=0.95,
        activation_source="output", covariance_mode="per_pixel",
        batch_size=8, num_workers=0, device="cpu", seed=42,
    )
    assert "block" in out
    stats = out["block"]
    assert stats.k >= 1
    assert 0.0 <= stats.median_angle_deg <= 90.0
    assert stats.median_angle_deg <= stats.p90_angle_deg <= stats.max_angle_deg
    assert stats.n_pairs == 3  # C(3, 2) = 3 pairs
