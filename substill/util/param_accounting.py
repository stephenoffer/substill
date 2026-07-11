"""Parameter accounting for distillation runs.

The matched-compression search in [scripts/fasd_ablation.py](../../scripts/fasd_ablation.py)
counts via ``sum(p.numel())``, which silently double-counts shared parameters
(tied LM head, shared embeddings) and gives no per-edge breakdown. The rank
allocator needs per-edge cost to budget rank assignments, so we centralize
accounting here.

Usage::

    from substill.util.param_accounting import count_params, breakdown

    n = count_params(model)                       # int, tied weights counted once
    bd = breakdown(model)                         # ParamBreakdown with per-bucket totals
    print(bd.summary())                           # human-readable
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

_EDGE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (bucket name, substrings that must appear in the parameter's qualified name)
    ("embed.token", ("wte", "embed_tokens", "tok_embeddings")),
    ("embed.position", ("wpe", "position_embeddings")),
    ("attn.q", ("q_proj", "query")),
    ("attn.k", ("k_proj", "key")),
    ("attn.v", ("v_proj", "value")),
    ("attn.o", ("o_proj", "out_proj", "c_proj")),
    # GPT-2 fuses QKV into c_attn; bucket separately.
    ("attn.qkv_fused", ("c_attn",)),
    ("ffn.gate", ("gate_proj",)),
    ("ffn.up", ("up_proj", "c_fc")),
    ("ffn.down", ("down_proj",)),
    ("norm", ("layernorm", "layer_norm", "ln_", "ln_f", "norm.weight", "norm.bias", "rms")),
    ("lm_head", ("lm_head",)),
)


def _bucket_for(name: str) -> str:
    """Bucket a parameter qualified name. Order matters; first match wins."""
    lower = name.lower()
    for bucket, needles in _EDGE_PATTERNS:
        for needle in needles:
            if needle in lower:
                return bucket
    return "other"


@dataclass
class ParamBreakdown:
    """Per-bucket parameter count.

    ``total`` deduplicates by tensor identity; tied weights (e.g. ``lm_head.weight is wte.weight``)
    are counted once and bucketed under the *first* qualified name encountered in the iteration
    order of ``model.named_parameters()``.
    """

    total: int = 0
    trainable: int = 0
    by_bucket: dict[str, int] = field(default_factory=dict)
    tied_groups: list[list[str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"total={self.total:,}  trainable={self.trainable:,}"]
        for bucket in sorted(self.by_bucket, key=lambda k: -self.by_bucket[k]):
            n = self.by_bucket[bucket]
            pct = 100.0 * n / max(1, self.total)
            lines.append(f"  {bucket:20s} {n:>15,}  ({pct:5.1f}%)")
        if self.tied_groups:
            lines.append(f"tied groups: {len(self.tied_groups)}")
            for grp in self.tied_groups:
                lines.append(f"  {' = '.join(grp)}")
        return "\n".join(lines)


def count_params(model: nn.Module, *, only_trainable: bool = False) -> int:
    """Total parameters, deduplicating tied tensors by ``id()``.

    ``model.parameters()`` iterates ``_parameters`` plus submodule parameters; PyTorch's
    Module already deduplicates by tensor identity in this iterator (a single shared
    weight appears once). The naive ``sum(p.numel())`` is therefore correct for plain
    tying, but explicit accounting via ``named_parameters()`` is more transparent and
    survives unusual sharing patterns.
    """
    seen: set[int] = set()
    total = 0
    for _name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if only_trainable and not p.requires_grad:
            continue
        seen.add(id(p))
        total += int(p.numel())
    return total


def breakdown(model: nn.Module) -> ParamBreakdown:
    """Per-bucket, dedup-aware parameter breakdown."""
    seen: dict[int, str] = {}  # id(p) -> first qualified name
    by_bucket: dict[str, int] = {}
    tied: dict[int, list[str]] = {}
    total = 0
    trainable = 0

    # remove_duplicate=False lets us see every name a tied tensor is registered under.
    for name, p in model.named_parameters(remove_duplicate=False):
        pid = id(p)
        if pid in seen:
            tied.setdefault(pid, [seen[pid]]).append(name)
            continue
        seen[pid] = name
        n = int(p.numel())
        bucket = _bucket_for(name)
        by_bucket[bucket] = by_bucket.get(bucket, 0) + n
        total += n
        if p.requires_grad:
            trainable += n

    tied_groups = [grp for grp in tied.values() if len(grp) > 1]

    return ParamBreakdown(
        total=total,
        trainable=trainable,
        by_bucket=by_bucket,
        tied_groups=tied_groups,
    )


def count_per_edge(model: nn.Module) -> dict[str, int]:
    """Bucket-counts only — convenience for the rank allocator."""
    return dict(breakdown(model).by_bucket)


__all__ = ["ParamBreakdown", "count_params", "breakdown", "count_per_edge"]
