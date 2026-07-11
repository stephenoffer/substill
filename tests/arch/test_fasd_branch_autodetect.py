"""Branch-level autodetection on GPT-2 and Llama."""

from __future__ import annotations

import pytest
import torch.nn as nn

from substill.autodetect import autodetect_branches


def _try_gpt2_tiny():
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(
        vocab_size=50, n_positions=16, n_embd=24, n_layer=2, n_head=3, n_inner=48
    )
    return GPT2LMHeadModel(cfg)


def _try_llama_tiny():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return None
    cfg = LlamaConfig(
        vocab_size=50,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=3,
        max_position_embeddings=16,
    )
    return LlamaForCausalLM(cfg)


def test_autodetect_gpt2_branches():
    model = _try_gpt2_tiny()
    if model is None:
        pytest.skip("transformers not installed")
    branches = autodetect_branches(model, mode="branch")
    names = [b.name for b in branches]
    # 6 branches per block (q,k,v,o,up,down) × 2 blocks = 12.
    assert len(branches) == 12
    assert "transformer.h.0.attn.q" in names
    assert "transformer.h.1.ffn.down" in names
    # q/k/v share module_path but differ by slice.
    q = next(b for b in branches if b.name == "transformer.h.0.attn.q")
    k = next(b for b in branches if b.name == "transformer.h.0.attn.k")
    assert q.module_path == k.module_path
    assert q.slice != k.slice


def test_autodetect_gpt2_residual_mode():
    model = _try_gpt2_tiny()
    if model is None:
        pytest.skip("transformers not installed")
    branches = autodetect_branches(model, mode="residual")
    assert len(branches) == 2  # one per block
    for b in branches:
        assert b.kind == "block.residual"
        assert b.slice is None


def test_autodetect_llama_branches():
    model = _try_llama_tiny()
    if model is None:
        pytest.skip("transformers not installed")
    branches = autodetect_branches(model, mode="branch")
    names = [b.name for b in branches]
    # 7 branches per block × 2 = 14 (q,k,v,o,gate,up,down)
    assert len(branches) == 14
    assert "model.layers.0.attn.q" in names
    assert "model.layers.0.ffn.gate" in names
    assert "model.layers.1.ffn.down" in names
    for b in branches:
        # Llama uses separate modules, so no slicing.
        assert b.slice is None


def test_autodetect_unknown_raises():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    with pytest.raises(NotImplementedError):
        autodetect_branches(Dummy())
