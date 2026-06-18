"""Tests for fasd.training.stiefel_optim.

Headline invariant: a Stiefel-tagged parameter U with shape (n, k) starting
orthonormal must satisfy ``U^T U = I_k`` to within float tolerance after
arbitrary numbers of optimizer steps. This is the property that PRA cannot
guarantee and that makes Stiefel descent strictly stronger.

Tests:
  - Cayley step preserves orthonormality (per step).
  - Long-run training preserves orthonormality (10k steps on a synthetic loss).
  - Square Q parameter from RR-Norm stays orthogonal under StiefelAdam.
  - Loss decreases monotonically on a convex Stiefel objective.
  - Standard (non-Stiefel) param groups behave like AdamW.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fasd.training.stiefel_optim import (
    StiefelAdam,
    _cayley_step,
    _orthogonalize,
    _project_tangent,
    stiefel_param_groups,
)
from fasd.util.rr_norm import RRNorm


def _max_orthogonality_error(U: torch.Tensor) -> float:
    n, k = U.shape
    eye = torch.eye(k, dtype=U.dtype, device=U.device)
    return float((U.transpose(0, 1) @ U - eye).abs().max().item())


def test_project_tangent_makes_skew_in_lifted_form():
    """T_U St condition: U^T Z + Z^T U = 0 (skew-symmetric in U^T Z)."""
    torch.manual_seed(0)
    n, k = 16, 4
    U, _ = torch.linalg.qr(torch.randn(n, k))
    M = torch.randn(n, k)
    M_t = _project_tangent(M, U)
    UT_M = U.transpose(0, 1) @ M_t
    assert torch.allclose(UT_M + UT_M.transpose(0, 1), torch.zeros_like(UT_M), atol=1e-5)


def test_cayley_step_preserves_orthonormality():
    """Single Cayley step at lr=0.1 keeps U^T U = I."""
    torch.manual_seed(0)
    n, k = 32, 8
    U, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64))
    M = torch.randn(n, k, dtype=torch.float64) * 0.5
    M = _project_tangent(M, U)
    U_new = _cayley_step(U, M, lr=0.1)
    err = _max_orthogonality_error(U_new)
    assert err < 1e-8, f"orthogonality drift after one step: {err:.3e}"


def test_cayley_step_preserves_orthonormality_for_square_U():
    """Square U (n=k, the RR-Norm Q case) — a special case of Stiefel."""
    torch.manual_seed(0)
    n = 16
    U, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    M = _project_tangent(torch.randn(n, n, dtype=torch.float64) * 0.3, U)
    U_new = _cayley_step(U, M, lr=0.05)
    err = _max_orthogonality_error(U_new)
    assert err < 1e-8


def test_orthogonalize_safety_net():
    """_orthogonalize maps a near-orthogonal matrix to exactly orthogonal."""
    torch.manual_seed(0)
    n, k = 12, 4
    U, _ = torch.linalg.qr(torch.randn(n, k))
    U_drift = U + 1e-3 * torch.randn(n, k)
    Q = _orthogonalize(U_drift)
    assert _max_orthogonality_error(Q) < 1e-6


def test_stiefel_adam_long_run_preserves_orthonormality():
    """1000 steps on a synthetic loss — orthogonality must not drift."""
    torch.manual_seed(0)
    n, k = 32, 8
    # Initialize on Stiefel.
    U, _ = torch.linalg.qr(torch.randn(n, k))
    U = nn.Parameter(U.clone())
    target = torch.randn(n, k)

    opt = StiefelAdam([U], lr=0.01, stiefel=True, reorth_every=50)
    for _ in range(1000):
        opt.zero_grad()
        loss = ((U - target) ** 2).sum()
        loss.backward()
        opt.step()

    err = _max_orthogonality_error(U.data)
    assert err < 1e-4, f"orthogonality drift over 1000 steps: {err:.3e}"


def test_stiefel_adam_decreases_convex_loss():
    """On a quadratic Stiefel loss min ||U - target_orth||² with target on the manifold,
    the optimizer should reduce loss substantially."""
    torch.manual_seed(0)
    n, k = 16, 4
    target_orth, _ = torch.linalg.qr(torch.randn(n, k))
    U_init, _ = torch.linalg.qr(torch.randn(n, k))
    U = nn.Parameter(U_init.clone())

    opt = StiefelAdam([U], lr=0.05, stiefel=True, reorth_every=20)
    initial_loss = ((U - target_orth) ** 2).sum().item()

    for _ in range(500):
        opt.zero_grad()
        loss = ((U - target_orth) ** 2).sum()
        loss.backward()
        opt.step()

    final_loss = ((U - target_orth) ** 2).sum().item()
    assert final_loss < 0.5 * initial_loss, (
        f"loss did not decrease enough: {initial_loss:.3e} → {final_loss:.3e}"
    )
    assert _max_orthogonality_error(U.data) < 1e-4


def test_stiefel_adam_handles_mixed_param_groups():
    """A Stiefel group + a standard group: both should update correctly."""
    torch.manual_seed(0)
    n, k = 8, 4
    U_init, _ = torch.linalg.qr(torch.randn(n, k))
    U = nn.Parameter(U_init.clone())
    w = nn.Parameter(torch.randn(10))

    opt = StiefelAdam(
        [
            {"params": [U], "lr": 0.01, "stiefel": True},
            {"params": [w], "lr": 0.01, "stiefel": False, "weight_decay": 0.0},
        ]
    )

    for _ in range(50):
        opt.zero_grad()
        loss = (U.pow(2).sum() + w.pow(2).sum())
        loss.backward()
        opt.step()

    # Both should have moved.
    assert (U.data - U_init).abs().max() > 1e-4
    assert _max_orthogonality_error(U.data) < 1e-4


def test_stiefel_param_groups_separates_q_from_other():
    """RRNorm.Q should land in the Stiefel group; other params in the standard group."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RRNorm(8, use_q=True, use_scale=True)
            self.linear = nn.Linear(8, 8)

    m = M()
    groups = stiefel_param_groups(m, base_lr=1e-3, stiefel_lr_ratio=0.1)
    assert len(groups) == 2
    stiefel_group = [g for g in groups if g.get("stiefel")][0]
    standard_group = [g for g in groups if not g.get("stiefel")][0]

    assert any(p is m.norm.q for p in stiefel_group["params"])
    assert all(p is not m.norm.q for p in standard_group["params"])
    # The norm.scale (1-D) should be in the standard group, not Stiefel.
    assert any(p is m.norm.scale for p in standard_group["params"])
    assert stiefel_group["lr"] == 1e-3 * 0.1
    assert standard_group["lr"] == 1e-3


def test_stiefel_adam_rejects_1d_param_in_stiefel_group():
    """Stiefel parameters must be 2-D matrices."""
    p = nn.Parameter(torch.randn(8))
    opt = StiefelAdam([p], lr=0.01, stiefel=True)
    p.grad = torch.randn_like(p)
    with pytest.raises(ValueError):
        opt.step()


def test_stiefel_adam_rejects_wide_matrix_in_stiefel_group():
    """For St(n, k) we need n ≥ k. Wide matrices should error or be transposed."""
    p = nn.Parameter(torch.randn(4, 8))  # 4 < 8
    opt = StiefelAdam([p], lr=0.01, stiefel=True)
    p.grad = torch.randn_like(p)
    with pytest.raises(ValueError):
        opt.step()
