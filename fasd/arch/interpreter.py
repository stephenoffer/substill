"""Generic interpreter: materialize per-layer BranchSpecs from an ArchitectureSpec.

Replaces the per-architecture ``_detect_*`` walkers with one function driven by the
declarative spec. Output is pinned to be identical to today's detectors for GPT-2 and
Llama (see tests/test_fsd_arch_spec.py), so this is a drop-in alongside the existing
autodetect path.
"""
from __future__ import annotations

import torch.nn as nn

from ..autodetect import BranchSpec
from .spec import ArchitectureSpec


def _resolve(root: nn.Module, dotted: str):
    obj = root
    for p in dotted.split("."):
        if p == "":
            continue
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _num_layers(model: nn.Module, spec: ArchitectureSpec) -> int:
    return len(_resolve(model, spec.layers_path))


def _num_experts(model: nn.Module, spec: ArchitectureSpec) -> int:
    if spec.moe is None:
        return 0
    cfg = getattr(model, "config", None)
    v = getattr(cfg, spec.moe.num_experts_attr, None)
    if v is None:
        raise ValueError(
            f"{spec.name}: config has no {spec.moe.num_experts_attr!r} for expert count"
        )
    return int(v)


def expand_branches(
    model: nn.Module,
    spec: ArchitectureSpec,
    *,
    mode: str = "branch",
) -> list[BranchSpec]:
    """Materialize the per-layer (and per-expert) BranchSpec list for ``model``.

    ``mode="residual"`` yields one residual branch per layer (matching the detectors'
    residual mode); ``mode="branch"`` yields the full per-edge list plus, for MoE, one
    edge per (expert, expert_edge_kind).
    """
    n_layers = _num_layers(model, spec)
    hidden = spec.hidden_size(model) if spec.attn_layout == "fused_qkv" else None
    n_exp = _num_experts(model, spec)
    out: list[BranchSpec] = []

    for i in range(n_layers):
        block = f"{spec.layers_path}.{i}"
        if mode == "residual":
            r = spec.residual
            out.append(BranchSpec(f"{block}.{r.suffix()}", block, r.kind))
            continue
        for e in spec.edges:
            name = f"{block}.{e.suffix()}"
            mpath = f"{block}.{e.module_rel}" if e.module_rel else block
            sl = None
            if e.slice_mult is not None:
                sl = (e.slice_mult * hidden, (e.slice_mult + 1) * hidden)
            out.append(BranchSpec(name, mpath, e.kind, slice=sl, hook_point=e.hook_point))
        if spec.moe is not None:
            for x in range(n_exp):
                for kind in spec.moe.expert_edge_kinds:
                    # e.g. "model.layers.3.expert.5.ffn.up"
                    name = f"{block}.expert.{x}.{kind}"
                    mpath = f"{block}.{spec.moe.experts_rel}"
                    out.append(BranchSpec(name, mpath, kind))
    return out


__all__ = ["expand_branches"]
