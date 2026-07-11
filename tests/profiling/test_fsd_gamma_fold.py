"""Tests for substill.profiling.gamma_fold.

Core invariant: γ-fold is a *parameter rewrite* that does not change the
function the model computes. Pre-norm `LN(x; γ, β) → W·LN(x) + b` equals
`W' · LN0(x) + b'` where `W' = W·diag(γ)`, `b' = b + W·β`, and
`LN0(x) = (x - μ̄)/σ` (no γ, no β).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from substill.profiling.gamma_fold import (
    FoldEdge,
    fold_gpt2,
    fold_llama,
    fold_pair,
    fold_shared_norm,
    make_folded_copy,
)


def test_fold_pair_layernorm_to_linear_preserves_output():
    """LN → Linear: folded version should produce identical output."""
    torch.manual_seed(0)
    d = 16
    ln = nn.LayerNorm(d)
    # Random non-trivial γ, β.
    with torch.no_grad():
        ln.weight.copy_(torch.randn(d) * 0.5 + 1.0)
        ln.bias.copy_(torch.randn(d) * 0.3)
    linear = nn.Linear(d, 8, bias=True)
    with torch.no_grad():
        linear.bias.copy_(torch.randn(8) * 0.2)

    x = torch.randn(4, 5, d)
    y_orig = linear(ln(x))

    folded_ln = copy.deepcopy(ln)
    folded_linear = copy.deepcopy(linear)
    fold_pair(folded_ln, folded_linear)
    y_folded = folded_linear(folded_ln(x))

    assert torch.allclose(y_orig, y_folded, atol=1e-5), (
        f"max diff = {(y_orig - y_folded).abs().max().item():.3e}"
    )

    # γ should be ones, β should be zeros after fold.
    assert torch.allclose(folded_ln.weight, torch.ones(d), atol=1e-6)
    assert torch.allclose(folded_ln.bias, torch.zeros(d), atol=1e-6)


def test_fold_pair_rmsnorm_like_to_linear_preserves_output():
    """RMSNorm-style (γ only, no β) → Linear: parity test."""

    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = 1e-6

        def forward(self, x):
            return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    torch.manual_seed(0)
    d = 16
    rms = RMSNorm(d)
    with torch.no_grad():
        rms.weight.copy_(torch.randn(d) * 0.5 + 1.0)
    linear = nn.Linear(d, 8, bias=False)

    x = torch.randn(4, 5, d)
    y_orig = linear(rms(x))

    folded_rms = copy.deepcopy(rms)
    folded_linear = copy.deepcopy(linear)
    fold_pair(folded_rms, folded_linear)
    y_folded = folded_linear(folded_rms(x))

    assert torch.allclose(y_orig, y_folded, atol=1e-5), (
        f"max diff = {(y_orig - y_folded).abs().max().item():.3e}"
    )
    assert torch.allclose(folded_rms.weight, torch.ones(d), atol=1e-6)


def test_fold_pair_synthesizes_bias_when_linear_has_no_bias():
    """If linear has no bias but LN has β, fold must add a bias parameter."""
    torch.manual_seed(0)
    d = 8
    ln = nn.LayerNorm(d)
    with torch.no_grad():
        ln.bias.copy_(torch.randn(d) * 0.5)
    linear = nn.Linear(d, 4, bias=False)

    x = torch.randn(2, d)
    y_orig = linear(ln(x))

    folded_ln = copy.deepcopy(ln)
    folded_linear = copy.deepcopy(linear)
    fold_pair(folded_ln, folded_linear)

    assert folded_linear.bias is not None, "fold must synthesize bias when β ≠ 0"
    y_folded = folded_linear(folded_ln(x))
    assert torch.allclose(y_orig, y_folded, atol=1e-5)


def test_fold_shared_norm_three_consumers_preserves_each():
    """Llama pattern: one RMSNorm feeds three linears (q_proj, k_proj, v_proj)."""

    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = 1e-6

        def forward(self, x):
            return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = RMSNorm(d)
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)

        def forward(self, x):
            n = self.norm(x)
            return self.q(n), self.k(n), self.v(n)

    torch.manual_seed(0)
    d = 12
    blk = Block(d)
    with torch.no_grad():
        blk.norm.weight.copy_(torch.randn(d) * 0.4 + 1.0)
    x = torch.randn(3, 4, d)
    q0, k0, v0 = blk(x)

    folded = copy.deepcopy(blk)
    fold_shared_norm(folded, "norm", ["q", "k", "v"])
    q1, k1, v1 = folded(x)
    assert torch.allclose(q0, q1, atol=1e-5)
    assert torch.allclose(k0, k1, atol=1e-5)
    assert torch.allclose(v0, v1, atol=1e-5)
    assert torch.allclose(folded.norm.weight, torch.ones(d), atol=1e-6)


def test_make_folded_copy_does_not_modify_original():
    torch.manual_seed(0)
    d = 8
    ln = nn.LayerNorm(d)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(d) + 1.0)
    linear = nn.Linear(d, 4)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = ln
            self.linear = linear

        def forward(self, x):
            return self.linear(self.ln(x))

    m = Wrap()
    gamma_before = m.ln.weight.detach().clone()

    edges = [FoldEdge("ln", "linear")]
    folded = make_folded_copy(m, edges)
    # Original untouched.
    assert torch.allclose(m.ln.weight, gamma_before)
    # Copy was folded.
    assert torch.allclose(folded.ln.weight, torch.ones(d), atol=1e-6)


def test_fold_gpt2_full_model_preserves_output():
    """End-to-end on a tiny HF GPT-2: folding must not change logits."""
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except Exception:
        pytest.skip("transformers not available")

    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=128, n_embd=32, n_layer=2, n_head=4, n_inner=64, n_positions=32
    )
    m = GPT2LMHeadModel(cfg).eval()

    # Inject non-trivial γ/β so the fold has something real to do.
    with torch.no_grad():
        for blk in m.transformer.h:
            blk.ln_1.weight.copy_(torch.randn_like(blk.ln_1.weight) * 0.3 + 1.0)
            blk.ln_1.bias.copy_(torch.randn_like(blk.ln_1.bias) * 0.2)
            blk.ln_2.weight.copy_(torch.randn_like(blk.ln_2.weight) * 0.3 + 1.0)
            blk.ln_2.bias.copy_(torch.randn_like(blk.ln_2.bias) * 0.2)

    input_ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        logits_before = m(input_ids).logits.clone()

    fold_gpt2(m)

    with torch.no_grad():
        logits_after = m(input_ids).logits

    assert torch.allclose(logits_before, logits_after, atol=1e-3), (
        f"GPT-2 γ-fold changed logits; max diff = "
        f"{(logits_before - logits_after).abs().max().item():.3e}"
    )

    # Verify all touched γ are ones / β are zeros (ln_f untouched).
    for blk in m.transformer.h:
        assert torch.allclose(blk.ln_1.weight, torch.ones_like(blk.ln_1.weight), atol=1e-6)
        assert torch.allclose(blk.ln_1.bias, torch.zeros_like(blk.ln_1.bias), atol=1e-6)
        assert torch.allclose(blk.ln_2.weight, torch.ones_like(blk.ln_2.weight), atol=1e-6)
        assert torch.allclose(blk.ln_2.bias, torch.zeros_like(blk.ln_2.bias), atol=1e-6)


def test_fold_llama_full_model_preserves_output():
    """End-to-end on a tiny HF Llama: folding must not change logits."""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception:
        pytest.skip("transformers not available")

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    m = LlamaForCausalLM(cfg).eval()

    with torch.no_grad():
        for layer in m.model.layers:
            layer.input_layernorm.weight.copy_(
                torch.randn_like(layer.input_layernorm.weight) * 0.3 + 1.0
            )
            layer.post_attention_layernorm.weight.copy_(
                torch.randn_like(layer.post_attention_layernorm.weight) * 0.3 + 1.0
            )

    input_ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        logits_before = m(input_ids).logits.clone()

    fold_llama(m)

    with torch.no_grad():
        logits_after = m(input_ids).logits

    assert torch.allclose(logits_before, logits_after, atol=1e-3), (
        f"Llama γ-fold changed logits; max diff = "
        f"{(logits_before - logits_after).abs().max().item():.3e}"
    )

    for layer in m.model.layers:
        assert torch.allclose(
            layer.input_layernorm.weight,
            torch.ones_like(layer.input_layernorm.weight),
            atol=1e-6,
        )
        assert torch.allclose(
            layer.post_attention_layernorm.weight,
            torch.ones_like(layer.post_attention_layernorm.weight),
            atol=1e-6,
        )
