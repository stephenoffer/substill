"""Declarative ArchitectureSpec — make adding a model family data, not code.

Today branch detection, γ-fold inventories, and builders each fork per architecture
(``_detect_gpt2``/``_detect_llama``, ``gpt2_fold_edges``/``llama_fold_edges``,
``_build_gpt2``/``_build_llama``). Adding Qwen3/Mistral/Mixtral means a new ~200-line
function each. ``ArchitectureSpec`` replaces those forks with a declarative description
that one generic interpreter consumes (see :mod:`fasd.arch.interpreter`).

A spec describes a per-block *template* (the compressible edges + their module paths +
attention/fold layout), parameterized by layer index ``i`` (and expert index ``e`` for
MoE). The interpreter materializes per-layer ``BranchSpec`` lists identical to today's
detectors — pinned by equivalence tests — so this layer is purely additive.

This module is the strangler-fig step 1 (add alongside, pin equivalence). Wiring the
detectors/builders to *delegate* here is incremental and gated on those tests.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch.nn as nn

WeightLayout = Literal["linear", "conv1d_gpt2"]
AttnLayout = Literal["separate_qkv", "fused_qkv"]


@dataclass(frozen=True)
class EdgeTemplate:
    """One compressible edge inside a block, addressed relative to the block root.

    ``kind`` reuses the existing BranchKind vocabulary ("attn.q", "ffn.up", ...) so
    profiling, width_pruner aggregation, and losses work unchanged. ``name_suffix``
    is the branch-name tail (defaults to ``kind``; "residual" uses a different tail).
    ``module_rel`` is the module path relative to the block (empty = the block itself,
    for the residual branch). ``slice_mult`` selects a fused-QKV column third
    (0/1/2 -> [mult*H, (mult+1)*H)); ``None`` means no slice.
    """

    kind: str
    module_rel: str = ""
    name_suffix: str | None = None
    slice_mult: int | None = None
    hook_point: Literal["output", "input"] = "output"

    def suffix(self) -> str:
        return self.name_suffix if self.name_suffix is not None else self.kind


@dataclass(frozen=True)
class FoldTemplate:
    """A (norm -> consumers) pre-norm fold, relative to the block root."""

    norm_rel: str
    consumer_rels: tuple[str, ...]


@dataclass(frozen=True)
class MoESpec:
    """How to enumerate experts inside one block (router stays full-rank).

    Works for both the classic ModuleList-of-experts layout and the newer fused
    (batched 3D weight) layout — enumeration only produces per-expert edge *names*;
    physical resolution (module vs tensor slice) is the builder/profiler's job.
    """

    experts_rel: str
    router_rel: str
    num_experts_attr: str
    expert_edge_kinds: tuple[str, ...] = ("ffn.gate", "ffn.up", "ffn.down")


@dataclass(frozen=True)
class ArchitectureSpec:
    """Declarative description of a model architecture for the FSD pipeline."""

    name: str
    layers_path: str
    embed_path: str = ""
    final_norm_path: str = ""
    lm_head_path: str = "lm_head"
    # per-block template
    residual: EdgeTemplate = field(
        default=EdgeTemplate(kind="block.residual", module_rel="", name_suffix="residual")
    )
    edges: tuple[EdgeTemplate, ...] = ()
    folds: tuple[FoldTemplate, ...] = ()
    attn_layout: AttnLayout = "separate_qkv"
    weight_layout: WeightLayout = "linear"
    moe: MoESpec | None = None
    # how to read the hidden size from a model (for fused-QKV slices)
    hidden_attrs: tuple[str, ...] = ("hidden_size", "n_embd")
    matches: Callable[[nn.Module], bool] = field(default=lambda m: False, compare=False)

    def hidden_size(self, model: nn.Module) -> int:
        cfg = getattr(model, "config", None)
        for a in self.hidden_attrs:
            v = getattr(cfg, a, None)
            if v is not None:
                return int(v)
        raise ValueError(f"{self.name}: could not read hidden size from config")


__all__ = ["EdgeTemplate", "FoldTemplate", "MoESpec", "ArchitectureSpec",
           "WeightLayout", "AttnLayout"]
