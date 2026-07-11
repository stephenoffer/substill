"""Tests for substill.util.rr_norm.RRNorm and replace_layernorm_with_rrnorm.

Key invariants:
  - With Q=I, scale=1, no centering: RRNorm equals isotropic RMSNorm.
  - With Q=I, scale=1, centering=True: RRNorm equals LayerNorm with γ=1, β=0.
  - Q stays orthogonal under standard PyTorch optimizers only by accident; it
    *will* stay orthogonal under the Stiefel optimizer. This file
    only verifies forward semantics; orthogonality is tested in the Stiefel
    optimizer tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from substill.util.rr_norm import RRNorm, replace_layernorm_with_rrnorm


def test_rr_norm_default_matches_isotropic_rms():
    torch.manual_seed(0)
    d = 16
    x = torch.randn(3, 5, d)
    rr = RRNorm(d, eps=1e-6, use_scale=False, use_q=False, center=False)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    out = rr(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rr_norm_with_centering_matches_layernorm_gamma1_beta0():
    torch.manual_seed(0)
    d = 12
    x = torch.randn(2, 4, d)
    rr = RRNorm(d, eps=1e-6, use_scale=False, use_q=False, center=True)
    out = rr(x)
    centered = x - x.mean(-1, keepdim=True)
    expected = centered * torch.rsqrt(centered.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out, expected, atol=1e-5)


def test_rr_norm_q_eye_is_identity_at_init():
    """Q is initialised to I, so RRNorm with use_q=True should equal use_q=False at init."""
    torch.manual_seed(0)
    d = 8
    x = torch.randn(2, d)
    rr_no_q = RRNorm(d, use_scale=False, use_q=False)
    rr_with_q = RRNorm(d, use_scale=False, use_q=True)
    assert torch.allclose(rr_no_q(x), rr_with_q(x), atol=1e-6)


def test_rr_norm_scale_acts_as_multiplier():
    torch.manual_seed(0)
    d = 8
    x = torch.randn(2, d)
    rr = RRNorm(d, use_scale=True, use_q=False, init_scale=2.5)
    expected = 2.5 * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(rr(x), expected, atol=1e-5)


def test_rr_norm_q_random_orthogonal_preserves_norm():
    """Output magnitude under random Q ∈ O(d) equals output magnitude under Q=I."""
    torch.manual_seed(0)
    d = 16
    x = torch.randn(3, d)
    rr = RRNorm(d, use_scale=False, use_q=True)
    # Inject a random orthogonal matrix.
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    rr.q.data.copy_(Q)
    y = rr(x)
    # ||y[..., :]||² should be invariant to Q (right-multiply by orthogonal).
    # Compute reference manually.
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y.norm(dim=-1), ref.norm(dim=-1), atol=1e-5)


def test_calibrate_scale_sets_scalar():
    rr = RRNorm(8, use_scale=True, use_q=False)
    rr.calibrate_scale(rms_t=1.5, rms_s=0.5)
    assert pytest.approx(float(rr.scale.item()), abs=1e-6) == 3.0


def test_replace_layernorm_with_rrnorm_swaps_in_place():
    """Walk a small model and verify LN modules are swapped to RRNorm."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(16)
            self.ln2 = nn.LayerNorm(16)
            self.linear = nn.Linear(16, 16)

        def forward(self, x):
            return self.linear(self.ln2(self.ln1(x)))

    m = M()
    n = replace_layernorm_with_rrnorm(m, d_model=16, use_q=True, use_scale=True)
    assert n == 2
    assert isinstance(m.ln1, RRNorm)
    assert isinstance(m.ln2, RRNorm)
    # Forward still runs.
    out = m(torch.randn(2, 16))
    assert out.shape == (2, 16)


def test_replace_layernorm_skips_wrong_dim():
    """LN modules of a different dim must be left alone."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_residual = nn.LayerNorm(64)
            self.ln_other = nn.LayerNorm(8)

        def forward(self, x):
            return x

    m = M()
    n = replace_layernorm_with_rrnorm(m, d_model=64)
    assert n == 1
    assert isinstance(m.ln_residual, RRNorm)
    assert isinstance(m.ln_other, nn.LayerNorm)


def test_replace_rmsnorm_when_present():
    """Llama-style RMSNorm should be swapped too."""

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps

        def forward(self, x):
            return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(32)

    m = M()
    n = replace_layernorm_with_rrnorm(m, d_model=32, use_q=True)
    assert n == 1
    assert isinstance(m.norm, RRNorm)


def test_stiefel_parameters_returns_q():
    rr = RRNorm(8, use_q=True)
    sp = rr.stiefel_parameters()
    assert len(sp) == 1
    assert sp[0] is rr.q


def test_stiefel_parameters_empty_when_no_q():
    rr = RRNorm(8, use_q=False)
    assert rr.stiefel_parameters() == []
