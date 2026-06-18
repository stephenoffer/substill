"""Tests for the adaptive-objective additions to fasd.losses.generative_kd."""

from __future__ import annotations

import pytest
import torch

from fasd.losses.generative_kd import (
    PlateauDetector,
    adaptive_skew_kl,
    skew_kl,
    unified_token_weights,
)


def test_adaptive_skew_kl_returns_scalar():
    torch.manual_seed(0)
    s = torch.randn(2, 4, 16)
    t = torch.randn(2, 4, 16)
    loss = adaptive_skew_kl(s, t)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_adaptive_skew_kl_alpha_high_when_teacher_more_uncertain():
    """If teacher is uniform (high entropy) and student is peaked (low entropy),
    adaptive alpha should approach alpha_max (lean teacher)."""
    torch.manual_seed(0)
    V = 32
    # Teacher: uniform → high entropy.
    t = torch.zeros(1, 1, V)
    # Student: peaked at index 0 → very low entropy.
    s = torch.zeros(1, 1, V)
    s[..., 0] = 50.0

    _, alpha = adaptive_skew_kl(s, t, tau=2.0, return_alpha=True)
    # Alpha should be near alpha_max (default 0.9).
    assert alpha.item() > 0.85, f"alpha = {alpha.item()}"


def test_adaptive_skew_kl_alpha_low_when_student_more_uncertain():
    """Reverse: peaked teacher, uniform student → alpha → alpha_min."""
    V = 32
    s = torch.zeros(1, 1, V)
    t = torch.zeros(1, 1, V)
    t[..., 0] = 50.0
    _, alpha = adaptive_skew_kl(s, t, tau=2.0, return_alpha=True)
    assert alpha.item() < 0.15, f"alpha = {alpha.item()}"


def test_adaptive_skew_kl_clamps_to_bounds():
    V = 16
    s = torch.zeros(1, 1, V)
    t = torch.zeros(1, 1, V)
    t[..., 0] = 1000.0  # extreme teacher peak
    _, alpha = adaptive_skew_kl(s, t, tau=10.0, alpha_min=0.2, alpha_max=0.8, return_alpha=True)
    assert alpha.item() >= 0.2 - 1e-6
    assert alpha.item() <= 0.8 + 1e-6


def test_adaptive_skew_kl_invalid_bounds():
    s = torch.randn(1, 4, 8)
    t = torch.randn(1, 4, 8)
    with pytest.raises(ValueError):
        adaptive_skew_kl(s, t, alpha_min=0.5, alpha_max=0.5)
    with pytest.raises(ValueError):
        adaptive_skew_kl(s, t, alpha_min=0.0, alpha_max=0.5)


def test_adaptive_skew_kl_with_mask():
    V = 8
    s = torch.randn(2, 5, V)
    t = torch.randn(2, 5, V)
    mask = torch.ones(2, 5)
    mask[:, -1] = 0  # ignore last position
    out_full = adaptive_skew_kl(s, t)
    out_masked = adaptive_skew_kl(s, t, mask=mask)
    assert torch.isfinite(out_full)
    assert torch.isfinite(out_masked)


def test_unified_token_weights_shape_and_average():
    torch.manual_seed(0)
    s = torch.randn(3, 4, 16)
    t = torch.randn(3, 4, 16)
    w = unified_token_weights(s, t)
    assert w.shape == (3, 4)
    # Normalised: mean is 1.
    assert pytest.approx(w.mean().item(), abs=1e-5) == 1.0


def test_unified_token_weights_zero_when_distributions_equal():
    """Identical distributions → zero TV distance → zero weights."""
    V = 8
    logits = torch.randn(1, 3, V)
    w = unified_token_weights(logits, logits, normalise=False)
    assert torch.allclose(w, torch.zeros(1, 3), atol=1e-6)


def test_unified_token_weights_high_when_disagreement_large():
    """High weights when both teacher entropy is non-trivial AND TV is high.

    With near-uniform teacher (high H_t) and a confident student that disagrees,
    we expect a large weight. With agreeing distributions, weight = 0.
    """
    V = 8
    # Teacher: moderately mixed (entropy ≈ log(2) for two modes).
    t = torch.zeros(1, 2, V)
    t[..., 0] = 1.5
    t[..., V - 1] = 1.5
    # Student: confidently peaked at 0 → strong disagreement on the V-1 mass.
    s = torch.zeros(1, 2, V)
    s[..., 0] = 5.0

    w_disagree = unified_token_weights(s, t, normalise=False)
    w_agree = unified_token_weights(t, t, normalise=False)
    assert w_disagree.mean() > w_agree.mean() + 1e-3, (
        f"disagree weight {w_disagree.mean().item():.3e} vs agree {w_agree.mean().item():.3e}"
    )


def test_plateau_detector_does_not_fire_during_warmup():
    pd = PlateauDetector(min_step=100, window=10, tolerance=1e-3, patience=3)
    fired = False
    for _ in range(50):
        if pd.update(loss=1.0):
            fired = True
    assert not fired
    assert not pd.triggered()


def test_plateau_detector_fires_on_flat_loss():
    # Use a low decay so the EMA settles within the test window.
    pd = PlateauDetector(min_step=10, window=5, tolerance=1e-3, patience=2, decay=0.5)
    # First feed decreasing loss, then a long plateau at 8.0.
    for i in range(20):
        pd.update(loss=10.0 - i * 0.1)
    fired = False
    for _ in range(50):
        if pd.update(loss=8.0):
            fired = True
    assert fired
    assert pd.triggered()


def test_plateau_detector_does_not_fire_on_decreasing_loss():
    pd = PlateauDetector(min_step=10, window=5, tolerance=1e-3, patience=2, decay=0.5)
    for i in range(50):
        # Steady decrease — slope per step ~0.05, slope per window ~0.05.
        pd.update(loss=10.0 - i * 0.05)
    assert not pd.triggered()


def test_plateau_detector_state_returns_dict():
    pd = PlateauDetector()
    pd.update(loss=1.0)
    s = pd.state()
    assert isinstance(s, dict)
    assert "step" in s and "ema" in s


def test_adaptive_skew_kl_recovers_skew_kl_at_constant_alpha():
    """When teacher and student have equal entropy, adaptive alpha = sigmoid(0) = 0.5.
    Compare to plain skew_kl at alpha=0.5 — they should be close."""
    torch.manual_seed(0)
    V = 16
    # Use random logits scaled identically so entropies match exactly.
    base = torch.randn(2, 4, V)
    s = base
    t = base + 1e-6  # near-identical
    out_adaptive = adaptive_skew_kl(s, t, tau=1.0, alpha_min=0.49, alpha_max=0.51)
    out_static = skew_kl(s, t, alpha=0.5)
    assert torch.allclose(out_adaptive, out_static, atol=1e-3)
