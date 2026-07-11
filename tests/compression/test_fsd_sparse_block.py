"""Tests for substill.compression.sparse_block."""

from __future__ import annotations

import pytest
import torch

from substill.compression.sparse_block import BlockDiagonalCorrection, CorrectedLinear


def test_zero_init_correction_is_identity_at_zero():
    """At init (zero correction), the module's output is the zero vector."""
    torch.manual_seed(0)
    bd = BlockDiagonalCorrection(num_heads=4, d_head=8, init="zero")
    x = torch.randn(3, 32)  # 4 heads * 8 d_head
    y = bd(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_correction_acts_per_head_independently():
    """Setting block h to a known matrix should affect only head h's output channels."""
    torch.manual_seed(0)
    H, D = 3, 4
    bd = BlockDiagonalCorrection(H, D, init="zero")
    # Set head 1 to identity * 2.0; others remain zero.
    with torch.no_grad():
        bd.weight[1] = torch.eye(D) * 2.0
    x = torch.randn(2, H * D)
    y = bd(x)
    # Reshape outputs per head.
    y_h = y.view(2, H, D)
    x_h = x.view(2, H, D)
    # Head 0, 2: should be zero.
    assert torch.allclose(y_h[:, 0], torch.zeros(2, D), atol=1e-6)
    assert torch.allclose(y_h[:, 2], torch.zeros(2, D), atol=1e-6)
    # Head 1: should equal 2 * x_h[:, 1].
    assert torch.allclose(y_h[:, 1], 2.0 * x_h[:, 1], atol=1e-5)


def test_correction_param_count():
    bd = BlockDiagonalCorrection(num_heads=8, d_head=16)
    assert bd.num_extra_params() == 8 * 16 * 16


def test_correction_dim_mismatch_raises():
    bd = BlockDiagonalCorrection(4, 8)
    with pytest.raises(ValueError):
        bd(torch.randn(2, 30))  # 30 ≠ 4*8=32


def test_correction_random_init_nonzero():
    torch.manual_seed(0)
    bd = BlockDiagonalCorrection(2, 4, init="random")
    assert bd.weight.abs().max() > 0


def test_corrected_linear_matches_linear_at_zero_correction():
    """CorrectedLinear with zero correction should equal a plain Linear."""
    torch.manual_seed(0)
    H, D = 4, 8
    cl = CorrectedLinear(
        in_features=H * D, out_features=H * D, num_heads=H, d_head=D,
        correction_init="zero",
    )
    x = torch.randn(3, H * D)
    y = cl(x)
    expected = cl.linear(x)
    assert torch.allclose(y, expected, atol=1e-6)


def test_corrected_linear_with_active_correction():
    torch.manual_seed(0)
    H, D = 2, 3
    cl = CorrectedLinear(
        in_features=H * D, out_features=H * D, num_heads=H, d_head=D,
        correction_init="zero",
    )
    with torch.no_grad():
        cl.correction.weight[0] = torch.eye(D)
    x = torch.randn(1, H * D)
    y = cl(x)
    # y = linear(x) + correction(x)
    # correction(x) on head 0 = x_h[0], on head 1 = 0
    expected = cl.linear(x).clone()
    expected_corr = torch.zeros_like(x)
    expected_corr[:, :D] = x[:, :D]
    assert torch.allclose(y, expected + expected_corr, atol=1e-5)


def test_corrected_linear_non_square_raises():
    """CorrectedLinear requires square (in_features == out_features) for now."""
    with pytest.raises(ValueError):
        CorrectedLinear(in_features=16, out_features=32, num_heads=4, d_head=4)


def test_correction_grad_flows_to_per_head_block():
    """Backward through the correction should produce non-zero grads on its weight."""
    torch.manual_seed(0)
    H, D = 2, 4
    bd = BlockDiagonalCorrection(H, D, init="zero")
    bd.weight.requires_grad_(True)
    x = torch.randn(3, H * D, requires_grad=True)
    y = bd(x)
    loss = y.pow(2).sum()
    loss.backward()
    # At zero init, output is zero, so gradient w.r.t. correction.weight should be
    # zero from y.pow(2).sum() (chain rule: dL/dW = 2y * dy/dW = 0 when y=0).
    # Make a non-trivial perturbation: add a nonzero correction first.
    bd.weight.grad = None
    with torch.no_grad():
        bd.weight[0] = torch.eye(D) * 0.5
    y = bd(x)
    loss = y.pow(2).sum()
    loss.backward()
    assert bd.weight.grad is not None
    # Gradient on head 0 should be nonzero (was active); head 1 should be near zero.
    assert bd.weight.grad[0].abs().max() > 0
    assert bd.weight.grad[1].abs().max() < 1e-5
