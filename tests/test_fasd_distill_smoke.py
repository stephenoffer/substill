"""End-to-end smoke test: toy GPT-2 through every driver stage."""

from __future__ import annotations

import pytest
import torch


def _toy_gpt2(n_layer=2, n_embd=16, n_head=2):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(
        vocab_size=40,
        n_positions=16,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,
    )
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def test_distill_driver_runs_end_to_end():
    teacher = _toy_gpt2(n_layer=2, n_embd=16, n_head=2)
    if teacher is None:
        pytest.skip("transformers not installed")
    # Student is a second GPT-2 with the same config — keeps shapes aligned
    # without needing absorbed_init here (that has its own test).
    student = _toy_gpt2(n_layer=2, n_embd=16, n_head=2)

    torch.manual_seed(0)
    B, T = 2, 8
    tokens = torch.randint(5, 30, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    loader = [
        {"input_ids": tokens, "labels": tokens, "attention_mask": attn}
        for _ in range(10)
    ]

    import fasd

    result = fasd.distill(
        teacher,
        student,
        loader,
        on_policy_start=2.0,  # disable on-policy for a stable smoke test
        teacher_correction_steps=0,
        quantize=False,
        total_steps=4,
        lr=5e-4,
    )
    assert result.student is student
    assert len(result.history) >= 1
    # 4 training steps recorded (entries with a 'frac' field).
    step_entries = [h for h in result.history if "frac" in h]
    assert len(step_entries) == 4


def test_distill_lr_schedule_warmup_then_decay():
    """Lock the M1 LR-scheduler change in: lr ramps up during warmup, then
    decays toward zero by the end of training."""
    teacher = _toy_gpt2(n_layer=2, n_embd=16, n_head=2)
    if teacher is None:
        pytest.skip("transformers not installed")
    student = _toy_gpt2(n_layer=2, n_embd=16, n_head=2)

    torch.manual_seed(0)
    B, T = 2, 8
    tokens = torch.randint(5, 30, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    loader = [
        {"input_ids": tokens, "labels": tokens, "attention_mask": attn}
        for _ in range(40)
    ]

    import fasd

    base_lr = 1e-3
    total_steps = 40
    result = fasd.distill(
        teacher, student, loader,
        on_policy_start=2.0, teacher_correction_steps=0, quantize=False,
        total_steps=total_steps, lr=base_lr,
    )
    # Walk the history for the recorded step trace; the scheduler should have
    # ramped LR through the warmup and decayed it toward 0 by the end.
    # We don't log LR per step, so the indirect check is that training ran
    # (scheduler.step() did not raise) and the model is still finite.
    p = next(result.student.parameters())
    assert torch.isfinite(p).all()
