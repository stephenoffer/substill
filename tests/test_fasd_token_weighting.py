"""Token-weighting utilities."""

from __future__ import annotations

import torch

from fasd.profiling.token_weighting import (
    completion_mask,
    compute_weights,
    disagreement_weights,
    entropy_weights,
    uniform_weights,
)


def test_entropy_weights_higher_for_uniform_token():
    B, T, V = 1, 4, 8
    logits = torch.zeros(B, T, V)
    # Make the last token very peaked; first token uniform.
    logits[0, -1, 0] = 100.0
    w = entropy_weights(logits)
    assert w.shape == (B, T)
    assert w[0, 0] > w[0, -1], "uniform token should have higher entropy than peaked token"


def test_disagreement_weights_zero_when_equal():
    logits = torch.randn(2, 3, 4)
    w = disagreement_weights(logits, logits)
    assert torch.allclose(w, torch.zeros_like(w), atol=1e-6)


def test_completion_mask_zeros_prompt():
    B, T = 2, 6
    plen = torch.tensor([3, 4])
    m = completion_mask((B, T), plen)
    assert m.shape == (B, T)
    assert m[0, :3].sum() == 0
    assert m[0, 3:].sum() == 3
    assert m[1, :4].sum() == 0
    assert m[1, 4:].sum() == 2


def test_compute_weights_dispatch_normalize():
    logits = torch.randn(1, 4, 5)
    w = compute_weights("entropy", teacher_logits=logits)
    # After normalization, the mean of non-zero entries should be ~1.
    assert abs(float(w.mean().item()) - 1.0) < 1e-4
