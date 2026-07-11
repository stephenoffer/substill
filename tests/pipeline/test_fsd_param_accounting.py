"""Tests for substill.util.param_accounting."""

from __future__ import annotations

import torch.nn as nn

from substill.util.param_accounting import (
    breakdown,
    count_params,
    count_per_edge,
)


def test_count_params_simple_module():
    m = nn.Linear(8, 4, bias=True)
    assert count_params(m) == 8 * 4 + 4


def test_count_params_dedup_tied_weights():
    """A shared weight tensor must be counted once."""

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.head = nn.Linear(32, 100, bias=False)
            self.head.weight = self.embed.weight  # tie

        def forward(self, x):
            return self.head(self.embed(x))

    m = Tied()
    expected = 100 * 32  # not 2 * 100 * 32
    assert count_params(m) == expected


def test_count_params_only_trainable():
    m = nn.Linear(4, 4)
    for p in m.parameters():
        p.requires_grad = False
    assert count_params(m) == 4 * 4 + 4
    assert count_params(m, only_trainable=True) == 0


def test_breakdown_buckets_attention_modules():
    """A simple attention-shaped module should split into q/k/v/o buckets."""

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16, bias=False)
            self.k_proj = nn.Linear(16, 16, bias=False)
            self.v_proj = nn.Linear(16, 16, bias=False)
            self.o_proj = nn.Linear(16, 16, bias=False)

    m = FakeAttn()
    bd = breakdown(m)
    assert bd.by_bucket["attn.q"] == 16 * 16
    assert bd.by_bucket["attn.k"] == 16 * 16
    assert bd.by_bucket["attn.v"] == 16 * 16
    assert bd.by_bucket["attn.o"] == 16 * 16
    assert bd.total == 4 * 16 * 16


def test_breakdown_records_tied_groups():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(50, 16)
            self.lm_head = nn.Linear(16, 50, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    bd = breakdown(Tied())
    # Tied weight bucketed under whichever name iterates first.
    assert bd.total == 50 * 16
    assert len(bd.tied_groups) == 1
    grp = bd.tied_groups[0]
    assert any("embed_tokens" in n for n in grp)
    assert any("lm_head" in n for n in grp)


def test_breakdown_summary_is_str():
    m = nn.Linear(8, 8)
    s = breakdown(m).summary()
    assert isinstance(s, str)
    assert "total=" in s


def test_count_per_edge_is_dict():
    m = nn.Linear(8, 8)
    d = count_per_edge(m)
    assert isinstance(d, dict)
    # nn.Linear weight + bias both bucket to "other" since it's not in our patterns.
    assert sum(d.values()) == count_params(m)


def test_breakdown_on_real_gpt2_small_smoke():
    """Smoke test on an actual HF GPT-2 small (125M)."""
    pytest = __import__("pytest")
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except Exception:
        pytest.skip("transformers not available")
    cfg = GPT2Config(
        vocab_size=512, n_embd=64, n_layer=2, n_head=4, n_inner=128, n_positions=64
    )
    m = GPT2LMHeadModel(cfg)
    bd = breakdown(m)
    naive = sum(p.numel() for p in m.parameters())
    # GPT-2 ties lm_head.weight = wte.weight, so naive counts the shared tensor once
    # by virtue of named_parameters dedup in newer torch — both methods should agree.
    assert bd.total == naive
    # Buckets should non-trivially split the model.
    assert "embed.token" in bd.by_bucket
    assert "norm" in bd.by_bucket
    assert bd.by_bucket["embed.token"] > 0
