"""Architecture registry: declarative specs for supported model families.

Adding a model family is a spec literal here (or via :func:`register_arch` from user
code) — no new ``_detect_*``/``_build_*``/``fold_*`` functions. A Llama-family variant
(Mistral, Qwen2.5) is just an added substring in ``matches``; a new MoE family is a
``replace(MIXTRAL_SPEC, ...)`` with three string changes.
"""
from __future__ import annotations

from dataclasses import replace

import torch.nn as nn

from .spec import ArchitectureSpec, EdgeTemplate, FoldTemplate, MoESpec

# -- GPT-2 (Conv1D, fused QKV, LayerNorm) -----------------------------------
GPT2_SPEC = ArchitectureSpec(
    name="gpt2",
    layers_path="transformer.h",
    embed_path="transformer.wte",
    final_norm_path="transformer.ln_f",
    lm_head_path="lm_head",
    attn_layout="fused_qkv",
    weight_layout="conv1d_gpt2",
    edges=(
        EdgeTemplate("attn.q", "attn.c_attn", slice_mult=0),
        EdgeTemplate("attn.k", "attn.c_attn", slice_mult=1),
        EdgeTemplate("attn.v", "attn.c_attn", slice_mult=2),
        EdgeTemplate("attn.o", "attn.c_proj"),
        EdgeTemplate("ffn.up", "mlp.c_fc"),
        EdgeTemplate("ffn.down", "mlp.c_proj"),
    ),
    folds=(
        FoldTemplate("ln_1", ("attn.c_attn",)),
        FoldTemplate("ln_2", ("mlp.c_fc",)),
        # ln_f intentionally excluded (tied to wte; cannot be γ-folded).
    ),
    hidden_attrs=("n_embd", "hidden_size"),
    matches=lambda m: "GPT2" in type(m).__name__,
)

# -- Llama / Mistral / Qwen2.5 (Linear, separate q/k/v, RMSNorm, GQA, dense FFN) --
LLAMA_SPEC = ArchitectureSpec(
    name="llama",
    layers_path="model.layers",
    embed_path="model.embed_tokens",
    final_norm_path="model.norm",
    lm_head_path="lm_head",
    attn_layout="separate_qkv",
    weight_layout="linear",
    edges=(
        EdgeTemplate("attn.q", "self_attn.q_proj"),
        EdgeTemplate("attn.k", "self_attn.k_proj"),
        EdgeTemplate("attn.v", "self_attn.v_proj"),
        EdgeTemplate("attn.o", "self_attn.o_proj"),
        EdgeTemplate("ffn.gate", "mlp.gate_proj"),
        EdgeTemplate("ffn.up", "mlp.up_proj"),
        EdgeTemplate("ffn.down", "mlp.down_proj"),
    ),
    folds=(
        FoldTemplate("input_layernorm",
                     ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        FoldTemplate("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
    ),
    hidden_attrs=("hidden_size",),
    # Mistral/Qwen2/Qwen3-dense share this layout exactly.
    matches=lambda m: any(k in type(m).__name__
                          for k in ("Llama", "Mistral", "Qwen2", "Qwen3ForCausal")),
)

# -- Mixtral / Qwen3-MoE (Llama attention + router + experts) ----------------
# Dense ffn edges are replaced by the router (full-rank) + per-expert edges.
MIXTRAL_SPEC = replace(
    LLAMA_SPEC,
    name="mixtral",
    edges=(
        EdgeTemplate("attn.q", "self_attn.q_proj"),
        EdgeTemplate("attn.k", "self_attn.k_proj"),
        EdgeTemplate("attn.v", "self_attn.v_proj"),
        EdgeTemplate("attn.o", "self_attn.o_proj"),
    ),
    folds=(
        FoldTemplate("input_layernorm",
                     ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        # post_attention_layernorm feeds the router + every expert (expanded by the
        # interpreter); the router itself stays full-rank.
        FoldTemplate("post_attention_layernorm", ("mlp.gate",)),
    ),
    moe=MoESpec(
        experts_rel="mlp.experts",
        router_rel="mlp.gate",
        num_experts_attr="num_local_experts",
        expert_edge_kinds=("ffn.gate", "ffn.up", "ffn.down"),
    ),
    matches=lambda m: any(k in type(m).__name__ for k in ("Mixtral", "Qwen3Moe")),
)

QWEN3MOE_SPEC = replace(
    MIXTRAL_SPEC,
    name="qwen3_moe",
    moe=MoESpec(
        experts_rel="mlp.experts",
        router_rel="mlp.gate",
        num_experts_attr="num_experts",
        expert_edge_kinds=("ffn.gate", "ffn.up", "ffn.down"),
    ),
    matches=lambda m: "Qwen3Moe" in type(m).__name__,
)


_REGISTRY: list[ArchitectureSpec] = [
    # MoE specs first: they match a superset name; resolve() returns the first hit,
    # and Mixtral/Qwen3Moe names don't collide with the dense Llama matcher anyway.
    QWEN3MOE_SPEC, MIXTRAL_SPEC, GPT2_SPEC, LLAMA_SPEC,
]


def register_arch(spec: ArchitectureSpec, *, front: bool = True) -> None:
    """Register a custom ArchitectureSpec (matched before built-ins by default)."""
    if front:
        _REGISTRY.insert(0, spec)
    else:
        _REGISTRY.append(spec)


def resolve_spec(model: nn.Module, name: str = "auto") -> ArchitectureSpec:
    """Return the ArchitectureSpec for ``model`` (by ``matches``) or by explicit name."""
    if name != "auto":
        for s in _REGISTRY:
            if s.name == name:
                return s
        raise ValueError(f"no registered ArchitectureSpec named {name!r}")
    for s in _REGISTRY:
        try:
            if s.matches(model):
                return s
        except Exception:
            continue
    raise ValueError(
        f"no ArchitectureSpec matches {type(model).__name__}; register one with "
        f"substill.arch.register_arch(...) or pass an explicit name"
    )


__all__ = ["GPT2_SPEC", "LLAMA_SPEC", "MIXTRAL_SPEC", "QWEN3MOE_SPEC",
           "register_arch", "resolve_spec"]
