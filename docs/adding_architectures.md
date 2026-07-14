# Adding a new model architecture

FSD/CPSD is architecture-agnostic via the declarative `ArchitectureSpec`
(`substill/arch/spec.py`). Adding a model family means writing **data, not code**: no new
`_detect_*`/`_build_*`/`fold_*` functions, and none of the old ~200-line-per-arch forks that
created the "3–5 days per architecture" problem.

## The 4-step checklist

### 1. Write an `ArchitectureSpec` (or `replace` an existing one)

```python
from dataclasses import replace
from substill.arch import LLAMA_SPEC, EdgeTemplate, FoldTemplate, register_arch

# Llama-family variant (Mistral, Qwen2.5) — usually just a matcher change:
register_arch(replace(LLAMA_SPEC, name="mistral",
                      matches=lambda m: "Mistral" in type(m).__name__))
```

For a genuinely new layout, declare the per-block template:

```python
from substill.arch import ArchitectureSpec, EdgeTemplate, FoldTemplate

MYARCH_SPEC = ArchitectureSpec(
    name="myarch",
    layers_path="model.layers",          # ModuleList of blocks
    embed_path="model.embed_tokens",
    final_norm_path="model.norm",
    lm_head_path="lm_head",
    attn_layout="separate_qkv",          # or "fused_qkv" (GPT-2 style)
    weight_layout="linear",              # or "conv1d_gpt2"
    edges=(                              # one per compressible linear
        EdgeTemplate("attn.q", "self_attn.q_proj"),
        EdgeTemplate("attn.k", "self_attn.k_proj"),
        EdgeTemplate("attn.v", "self_attn.v_proj"),
        EdgeTemplate("attn.o", "self_attn.o_proj"),
        EdgeTemplate("ffn.gate", "mlp.gate_proj"),
        EdgeTemplate("ffn.up", "mlp.up_proj"),
        EdgeTemplate("ffn.down", "mlp.down_proj"),
    ),
    folds=(                              # pre-norm -> consumer folds (γ-fold)
        FoldTemplate("input_layernorm",
                     ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        FoldTemplate("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
    ),
    hidden_attrs=("hidden_size",),
    matches=lambda m: "MyArch" in type(m).__name__,
)
register_arch(MYARCH_SPEC)
```

`kind` strings reuse the existing BranchKind vocabulary (`attn.q`, `ffn.up`, ...), so
profiling, `width_pruner` aggregation, losses, and the CPSD components work unchanged.

### 2. (MoE only) add a `MoESpec`

```python
from substill.arch import MoESpec
moe = MoESpec(
    experts_rel="mlp.experts",          # path to the experts container/tensor
    router_rel="mlp.gate",              # router (stays full-rank)
    num_experts_attr="num_experts",     # config field giving the expert count
    expert_edge_kinds=("ffn.gate", "ffn.up", "ffn.down"),
)
```
The interpreter enumerates one branch per `(expert, edge_kind)`, which is the surface for
per-expert rank allocation (DDR). One caveat: on recent `transformers` the experts are a
*fused* batched tensor rather than a `ModuleList`. Branch *enumeration* works; the absorbed
*build* on the fused tensor layout is the remaining builder work (see HANDOFF / task #11).

### 3. Verify branch detection

```python
from substill.arch import resolve_spec, expand_branches
spec = resolve_spec(model)                       # or pass name=
for b in expand_branches(model, spec)[:8]:
    print(b.name, b.module_path, b.kind, b.slice)
```
Confirm names/paths point at real modules.

### 4. Verify γ-fold preserves logits (atol ~1e-3)

Parametrize the `test_fsd_gamma_fold.py` logit-preservation pattern over your spec's
`folds`. If logits are preserved, the fold inventory is correct.

## What you do NOT write

No `_detect_myarch`, no `_build_myarch`, no `myarch_fold_edges`. Profiling, scoring,
rank allocation, the CPSD components (CPI/MT/DDR), losses, and training are all
spec-agnostic. A non-MoE Llama-family model is step 1 (one `matches` substring) only.

## Current coverage

| Family | Spec | Status |
|---|---|---|
| GPT-2 | `GPT2_SPEC` | branch-equivalence pinned to `_detect_gpt2`; CPSD-factored conversion wired |
| Llama 3.x / Mistral / Qwen2.5 / Qwen3-dense | `LLAMA_SPEC` | branch-equivalence pinned to `_detect_llama` |
| Mixtral / Qwen3-MoE | `MIXTRAL_SPEC` / `QWEN3MOE_SPEC` | branch + per-expert enumeration; fused-tensor absorbed-build remaining |
