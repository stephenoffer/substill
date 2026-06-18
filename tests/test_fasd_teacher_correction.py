"""Teacher correction reduces LM loss on a tiny corpus."""

from __future__ import annotations

import pytest
import torch

from fasd.training.teacher_correction import correct_teacher


def _toy_gpt2():
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(vocab_size=40, n_positions=16, n_embd=16, n_layer=2, n_head=2, n_inner=32)
    return GPT2LMHeadModel(cfg)


def test_correct_teacher_reduces_loss():
    model = _toy_gpt2()
    if model is None:
        pytest.skip("transformers not installed")
    torch.manual_seed(0)
    B = 2
    T = 8
    # Same tokens repeated — easy to memorize.
    tokens = torch.randint(5, 30, (B, T))
    loader = [{"input_ids": tokens, "labels": tokens}] * 20
    before_params = [p.detach().clone() for p in model.parameters()]
    stats = correct_teacher(model, loader, steps=10, lr=1e-3)
    assert stats["steps"] > 0
    assert stats["final_loss"] <= stats["initial_loss"] + 1e-3
    # Teacher weights changed in place.
    changed = any(
        not torch.equal(b, a) for b, a in zip(before_params, model.parameters(), strict=False)
    )
    assert changed
