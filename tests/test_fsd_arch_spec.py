"""ArchitectureSpec interpreter tests (task #11).

Pins equivalence: expand_branches(spec) must reproduce the existing _detect_* output
for GPT-2 and Llama, and enumerate per-expert edges for MoE.
"""
import pytest

from fasd.arch import (
    GPT2_SPEC,
    LLAMA_SPEC,
    MIXTRAL_SPEC,
    QWEN3MOE_SPEC,
    ArchitectureSpec,
    expand_branches,
    register_arch,
    resolve_spec,
)
from fasd.autodetect import autodetect_branches


def _spec_to_tuples(branches):
    return [(b.name, b.module_path, b.kind, b.slice) for b in branches]


def _gpt2(n_layer=2, n_embd=32, n_head=4):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=n_embd, n_layer=n_layer,
                     n_head=n_head, n_inner=4 * n_embd)
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def _llama(n_layer=2, hidden=32, heads=4, kv=2):
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return None
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=n_layer, num_attention_heads=heads,
                      num_key_value_heads=kv, max_position_embeddings=32)
    return LlamaForCausalLM(cfg)


def _mixtral(n_layer=2, hidden=32, experts=4):
    try:
        from transformers import MixtralConfig, MixtralForCausalLM
    except ImportError:
        return None
    cfg = MixtralConfig(vocab_size=64, hidden_size=hidden, intermediate_size=2 * hidden,
                        num_hidden_layers=n_layer, num_attention_heads=4,
                        num_key_value_heads=2, num_local_experts=experts,
                        num_experts_per_tok=2, max_position_embeddings=32)
    return MixtralForCausalLM(cfg)


def test_gpt2_spec_matches_detector():
    m = _gpt2()
    if m is None:
        pytest.skip("transformers not installed")
    assert resolve_spec(m).name == "gpt2"
    spec_branches = _spec_to_tuples(expand_branches(m, GPT2_SPEC, mode="branch"))
    det_branches = _spec_to_tuples(autodetect_branches(m, mode="branch"))
    assert spec_branches == det_branches
    # residual mode too
    assert _spec_to_tuples(expand_branches(m, GPT2_SPEC, mode="residual")) == \
        _spec_to_tuples(autodetect_branches(m, mode="residual"))


def test_llama_spec_matches_detector():
    m = _llama()
    if m is None:
        pytest.skip("transformers not installed")
    assert resolve_spec(m).name == "llama"
    assert _spec_to_tuples(expand_branches(m, LLAMA_SPEC, mode="branch")) == \
        _spec_to_tuples(autodetect_branches(m, mode="branch"))


def test_mixtral_resolves_and_enumerates_experts():
    m = _mixtral(n_layer=2, experts=4)
    if m is None:
        pytest.skip("transformers not installed")
    assert resolve_spec(m).name == "mixtral"
    branches = expand_branches(m, MIXTRAL_SPEC, mode="branch")
    names = [b.name for b in branches]
    # 4 attention edges per layer + 4 experts * 3 edges per layer = 16 per layer.
    per_layer = [n for n in names if n.startswith("model.layers.0.")]
    assert sum("expert." in n for n in per_layer) == 4 * 3
    assert sum(n.endswith(".attn.q") for n in per_layer) == 1
    # Expert edges are addressable and distinct per expert.
    assert "model.layers.0.expert.3.ffn.down" in names


def test_custom_arch_registration():
    fake = ArchitectureSpec(name="fake", layers_path="x",
                            matches=lambda mm: type(mm).__name__ == "ZZTop")
    register_arch(fake)

    class ZZTop:
        pass
    assert resolve_spec(ZZTop()).name == "fake"


def test_qwen3moe_spec_uses_num_experts_attr():
    assert QWEN3MOE_SPEC.moe.num_experts_attr == "num_experts"
    assert MIXTRAL_SPEC.moe.num_experts_attr == "num_local_experts"
