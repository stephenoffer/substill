"""Tests for fasd.compression.factored_linear.FactoredLinear (TODO #4).

Headline invariants:
  - U_in and U_out are orthonormal at init (Stiefel).
  - effective_weight() reconstructs U_out @ B @ U_in.T (+ S) bit-for-bit.
  - from_teacher init reproduces the absorbed-init formula
    W_S = V_out^T W_T V_in (i.e., the projection of W_T onto the retained subspace).
  - Stiefel registration: stiefel_parameters() returns U_in, U_out only.
  - Forward pass agrees with materialised effective_weight @ x.T + b.
  - Long-run training under StiefelAdam keeps U^T U = I.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fasd.compression.factored_linear import (
    FactoredLinear,
    stiefel_parameters_of,
)
from fasd.training.stiefel_optim import StiefelAdam, stiefel_param_groups


def _max_orthog_err(U: torch.Tensor) -> float:
    n, k = U.shape
    return float((U.T @ U - torch.eye(k, dtype=U.dtype, device=U.device)).abs().max().item())


def test_init_stiefel_orthonormality():
    torch.manual_seed(0)
    fl = FactoredLinear(d_in=16, d_out=8, k_in=8, k_out=4)
    assert _max_orthog_err(fl.U_in) < 1e-5
    assert _max_orthog_err(fl.U_out) < 1e-5


def test_init_validates_rank_bounds():
    with pytest.raises(ValueError):
        FactoredLinear(d_in=8, d_out=8, k_in=10, k_out=4)
    with pytest.raises(ValueError):
        FactoredLinear(d_in=8, d_out=8, k_in=4, k_out=10)


def test_forward_matches_effective_weight():
    torch.manual_seed(0)
    fl = FactoredLinear(d_in=16, d_out=8, k_in=8, k_out=4, bias=True)
    x = torch.randn(3, 5, 16)
    y_fwd = fl(x)

    W = fl.effective_weight()  # (d_out, d_in)
    y_via_weight = x @ W.T + fl.bias
    assert torch.allclose(y_fwd, y_via_weight, atol=1e-5)


def test_effective_weight_reconstructs_factorization():
    torch.manual_seed(0)
    fl = FactoredLinear(d_in=8, d_out=8, k_in=4, k_out=4, bias=False)
    W = fl.effective_weight()
    expected = fl.U_out @ fl.B @ fl.U_in.T
    assert torch.allclose(W, expected, atol=1e-6)


def test_stiefel_parameters_returns_only_U():
    fl = FactoredLinear(d_in=8, d_out=8, k_in=4, k_out=4, bias=True)
    sp = fl.stiefel_parameters()
    assert len(sp) == 2
    ids = {id(p) for p in sp}
    assert id(fl.U_in) in ids
    assert id(fl.U_out) in ids
    assert id(fl.B) not in ids
    assert id(fl.bias) not in ids


def test_from_teacher_full_rank_reproduces_teacher():
    """When V_in=I and V_out=I (no compression), effective_weight equals the teacher."""
    torch.manual_seed(0)
    d_in, d_out = 8, 6
    teacher = nn.Linear(d_in, d_out, bias=True)

    V_in = torch.eye(d_in)
    V_out = torch.eye(d_out)
    fl = FactoredLinear.from_teacher(teacher, V_in=V_in, V_out=V_out)

    W_recon = fl.effective_weight()
    assert torch.allclose(W_recon, teacher.weight, atol=1e-5)
    assert torch.allclose(fl.bias, teacher.bias, atol=1e-5)

    # And forward outputs match.
    x = torch.randn(2, d_in)
    y_t = teacher(x)
    y_f = fl(x)
    assert torch.allclose(y_t, y_f, atol=1e-5)


def test_from_teacher_compressed_reproduces_projection():
    """V_in, V_out non-trivial → effective_weight = V_out V_out^T W_T V_in V_in^T."""
    torch.manual_seed(0)
    d_in, d_out = 16, 8
    k_in, k_out = 6, 4
    teacher = nn.Linear(d_in, d_out, bias=False)

    V_in_raw = torch.randn(d_in, k_in)
    V_in, _ = torch.linalg.qr(V_in_raw)
    V_out_raw = torch.randn(d_out, k_out)
    V_out, _ = torch.linalg.qr(V_out_raw)

    fl = FactoredLinear.from_teacher(teacher, V_in=V_in, V_out=V_out)

    expected = V_out @ V_out.T @ teacher.weight @ V_in @ V_in.T
    actual = fl.effective_weight()
    assert torch.allclose(actual, expected, atol=1e-5), (
        f"max diff = {(expected - actual).abs().max().item():.3e}"
    )


def test_from_teacher_validates_shapes():
    teacher = nn.Linear(8, 4)
    V_in = torch.randn(7, 3)  # wrong d_in
    V_out = torch.randn(4, 2)
    with pytest.raises(ValueError):
        FactoredLinear.from_teacher(teacher, V_in=V_in, V_out=V_out)


def test_with_sparse_block_correction_zero_init_preserves_factorization():
    """At zero correction, effective_weight should equal U_out @ B @ U_in^T."""
    torch.manual_seed(0)
    fl = FactoredLinear(
        d_in=16, d_out=16, k_in=8, k_out=8,
        use_sparse_block=True, num_heads=4,
    )
    W = fl.effective_weight()
    expected = fl.U_out @ fl.B @ fl.U_in.T
    assert torch.allclose(W, expected, atol=1e-6)


def test_sparse_block_correction_appears_in_effective_weight():
    """After setting a non-zero correction, effective_weight should reflect it."""
    torch.manual_seed(0)
    fl = FactoredLinear(
        d_in=8, d_out=8, k_in=4, k_out=4,
        use_sparse_block=True, num_heads=2,
    )
    with torch.no_grad():
        # Set head 0's correction to identity * 3.
        fl.correction.weight[0] = torch.eye(4) * 3.0
    W = fl.effective_weight()
    base = fl.U_out @ fl.B @ fl.U_in.T
    diff = W - base
    # diff[0:4, 0:4] should be 3*I, rest zero.
    assert torch.allclose(diff[:4, :4], torch.eye(4) * 3.0, atol=1e-5)
    assert torch.allclose(diff[4:, 4:], torch.zeros(4, 4), atol=1e-5)
    assert torch.allclose(diff[:4, 4:], torch.zeros(4, 4), atol=1e-5)


def test_sparse_block_validates_square():
    with pytest.raises(ValueError):
        FactoredLinear(d_in=8, d_out=16, k_in=4, k_out=8,
                       use_sparse_block=True, num_heads=2)


def test_stiefel_param_groups_finds_factored_linear_U_matrices():
    """The trainer's `stiefel_param_groups` should pick up U_in, U_out from
    a model containing FactoredLinears."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fl1 = FactoredLinear(8, 8, 4, 4)
            self.fl2 = FactoredLinear(16, 16, 8, 8)
            self.regular = nn.Linear(8, 8)

    m = M()
    groups = stiefel_param_groups(m, base_lr=1e-3, stiefel_lr_ratio=0.1)
    stiefel_g = [g for g in groups if g.get("stiefel")][0]
    standard_g = [g for g in groups if not g.get("stiefel")][0]

    stiefel_ids = {id(p) for p in stiefel_g["params"]}
    standard_ids = {id(p) for p in standard_g["params"]}

    # All four U matrices should be in the Stiefel group.
    assert id(m.fl1.U_in) in stiefel_ids
    assert id(m.fl1.U_out) in stiefel_ids
    assert id(m.fl2.U_in) in stiefel_ids
    assert id(m.fl2.U_out) in stiefel_ids

    # B, bias, and the regular linear's params should be in the standard group.
    assert id(m.fl1.B) in standard_ids
    assert id(m.fl2.B) in standard_ids
    assert id(m.regular.weight) in standard_ids


def test_long_run_training_preserves_orthogonality():
    """500 steps of StiefelAdam on a synthetic loss — U^T U = I to within tolerance."""
    torch.manual_seed(0)
    d_in, d_out = 16, 8
    k_in, k_out = 8, 4
    fl = FactoredLinear(d_in, d_out, k_in, k_out)
    target_W = torch.randn(d_out, d_in) * 0.5

    groups = stiefel_param_groups(fl, base_lr=0.01, stiefel_lr_ratio=0.1, reorth_every=20)
    opt = StiefelAdam(groups)

    for _ in range(500):
        opt.zero_grad()
        loss = (fl.effective_weight() - target_W).pow(2).sum()
        loss.backward()
        opt.step()

    assert _max_orthog_err(fl.U_in) < 1e-3, (
        f"U_in drift: {_max_orthog_err(fl.U_in):.3e}"
    )
    assert _max_orthog_err(fl.U_out) < 1e-3, (
        f"U_out drift: {_max_orthog_err(fl.U_out):.3e}"
    )


def test_stiefel_parameters_of_walks_arbitrary_module_tree():
    """`stiefel_parameters_of` finds is_stiefel-tagged params anywhere in a tree."""
    torch.manual_seed(0)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fl = FactoredLinear(8, 8, 4, 4)
            self.linear = nn.Linear(8, 8)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Inner()
            self.b = Inner()

    m = Outer()
    sp = stiefel_parameters_of(m)
    # Two FactoredLinears, two U matrices each = 4 Stiefel params.
    assert len(sp) == 4
    ids = {id(p) for p in sp}
    assert id(m.a.fl.U_in) in ids
    assert id(m.b.fl.U_out) in ids


def test_factored_linear_loss_decreases_to_target():
    """End-to-end: FactoredLinear under StiefelAdam should fit a target weight."""
    torch.manual_seed(0)
    d_in, d_out = 12, 8
    k_in, k_out = 8, 6
    fl = FactoredLinear(d_in, d_out, k_in, k_out, bias=False)
    target_W = torch.randn(d_out, d_in)
    # Project target onto a (k_out, k_in)-rank subspace so it's reachable.
    U_target_in, _ = torch.linalg.qr(torch.randn(d_in, k_in))
    U_target_out, _ = torch.linalg.qr(torch.randn(d_out, k_out))
    target_W_proj = U_target_out @ U_target_out.T @ target_W @ U_target_in @ U_target_in.T

    groups = stiefel_param_groups(fl, base_lr=0.05, stiefel_lr_ratio=0.5, reorth_every=20)
    opt = StiefelAdam(groups)

    initial_loss = (fl.effective_weight() - target_W_proj).pow(2).sum().item()
    for _ in range(2000):
        opt.zero_grad()
        loss = (fl.effective_weight() - target_W_proj).pow(2).sum()
        loss.backward()
        opt.step()

    final_loss = (fl.effective_weight() - target_W_proj).pow(2).sum().item()
    assert final_loss < 0.5 * initial_loss, (
        f"loss did not decrease enough: {initial_loss:.3e} → {final_loss:.3e}"
    )
