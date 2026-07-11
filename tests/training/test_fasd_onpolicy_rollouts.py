"""On-policy rollouts, replay buffer, and hybrid collator."""

from __future__ import annotations

import pytest
import torch

from substill.training.onpolicy import (
    HybridCollator,
    ReplayBuffer,
    RolloutBatch,
    generate_rollouts,
)


def _toy_gpt2():
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(vocab_size=30, n_positions=16, n_embd=16, n_layer=2, n_head=2, n_inner=32)
    cfg.pad_token_id = 0
    model = GPT2LMHeadModel(cfg)
    return model


def test_replay_buffer_fifo_eviction():
    rb = ReplayBuffer(capacity=4)
    for i in range(6):
        batch = RolloutBatch(
            sequences=torch.tensor([[i, i, i]]),
            prompt_lens=torch.tensor([1]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )
        rb.add(batch)
    assert len(rb) == 4
    # Only the last 4 items remain.
    sampled = rb.sample(4)
    tokens = sampled.sequences[:, 0].tolist()
    assert sorted(tokens) == [2, 3, 4, 5]


def test_hybrid_collator_draws_from_both_sources():
    off = [{"input_ids": torch.zeros(1, 4, dtype=torch.long)}] * 10
    rb = ReplayBuffer(capacity=8)
    # Pre-populate the buffer.
    for i in range(4):
        rb.add(
            RolloutBatch(
                sequences=torch.tensor([[i + 1, i + 1, i + 1, i + 1]]),
                prompt_lens=torch.tensor([2]),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
            )
        )
    collator = HybridCollator(off, rb, ratio=0.5, on_policy_batch_size=2, seed=0)
    sources = []
    for i, rec in enumerate(collator):
        sources.append(rec["source"])
        if i >= 20:
            break
    assert "on" in sources and "off" in sources


def test_generate_rollouts_basic_shapes():
    model = _toy_gpt2()
    if model is None:
        pytest.skip("transformers not installed")
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    batch = generate_rollouts(model, prompts, max_new_tokens=4, temperature=0.5, top_p=1.0)
    assert batch.sequences.shape[0] == 2
    assert batch.sequences.shape[1] <= 7
    assert batch.prompt_lens.tolist() == [3, 3]
    assert batch.attention_mask.shape == batch.sequences.shape
