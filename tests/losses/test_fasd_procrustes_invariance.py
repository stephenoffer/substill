"""Procrustes and Gram losses are invariant to orthogonal rotations of inputs.

coord_mse-like (simple MSE) changes under rotation — we do not expose
coord_mse in substill, but we assert it via a direct test to anchor the
comparison.
"""

from __future__ import annotations

import torch

from substill.losses.procrustes import procrustes_distance
from substill.losses.subspace import cka_distance, gram_distance


def _random_orthogonal(k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(k, k, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q


def test_procrustes_invariant_under_rotation_of_student():
    torch.manual_seed(0)
    N, k = 64, 8
    Z_s = torch.randn(N, k)
    Z_t = torch.randn(N, k)
    R = _random_orthogonal(k, seed=1)
    base = procrustes_distance(Z_s, Z_t).item()
    rotated = procrustes_distance(Z_s @ R, Z_t).item()
    assert abs(base - rotated) < 1e-4


def test_procrustes_invariant_under_rotation_of_teacher():
    torch.manual_seed(0)
    Z_s = torch.randn(64, 8)
    Z_t = torch.randn(64, 8)
    R = _random_orthogonal(8, seed=2)
    base = procrustes_distance(Z_s, Z_t).item()
    rotated = procrustes_distance(Z_s, Z_t @ R).item()
    assert abs(base - rotated) < 1e-4


def test_gram_invariant_under_rotation():
    torch.manual_seed(0)
    Z_s = torch.randn(64, 8)
    Z_t = torch.randn(64, 8)
    R = _random_orthogonal(8, seed=3)
    base = gram_distance(Z_s, Z_t).item()
    rotated = gram_distance(Z_s @ R, Z_t).item()
    assert abs(base - rotated) < 1e-5


def test_cka_invariant_under_rotation_and_scale():
    torch.manual_seed(0)
    Z_s = torch.randn(64, 8)
    Z_t = torch.randn(64, 8)
    R = _random_orthogonal(8, seed=4)
    base = cka_distance(Z_s, Z_t).item()
    rotated = cka_distance(Z_s @ R, Z_t).item()
    scaled = cka_distance(Z_s * 7.0, Z_t).item()
    assert abs(base - rotated) < 1e-5
    assert abs(base - scaled) < 1e-5


def test_coord_mse_changes_under_rotation_sanity():
    """Coord MSE must change under input rotation (negative control)."""
    torch.manual_seed(0)
    Z_s = torch.randn(64, 8)
    Z_t = torch.randn(64, 8)
    R = _random_orthogonal(8, seed=5)
    base = ((Z_s - Z_t) ** 2).mean().item()
    rotated = ((Z_s @ R - Z_t) ** 2).mean().item()
    assert abs(base - rotated) > 1e-3, "coord MSE should NOT be rotation invariant"
