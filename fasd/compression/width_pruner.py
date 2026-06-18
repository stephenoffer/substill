"""Derive a compressed transformer config from a branch profile.

Encodes the Minitron findings:

- **Width-first**. Primary reduction comes from ``hidden_size`` (the
  residual branch rank) and ``intermediate_size`` (FFN up/gate ranks).
- **Retain attention heads**. ``num_attention_heads`` is kept equal
  to the teacher unless a branch rank would force it down.
- **GQA-friendly**. ``num_key_value_heads`` can drop independently of
  ``num_attention_heads`` — K/V branch ranks are usually lower.
- **Contiguous depth drops**. When the user requests depth reduction,
  layers are removed as a contiguous block, never scattered.

Also exposes :func:`plan_progressive_stages` — the teacher-assistant
chain helper. When teacher-to-student compression is aggressive
(``ratio > max_single_step``, default 3x), returns a sequence of
intermediate configs so the driver can run a progressive distillation
chain rather than one big jump.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

DepthPolicy = Literal["keep", "contiguous_tail", "contiguous_middle"]


@dataclass
class StudentConfig:
    """Minimal transformer config derived from a teacher profile."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int

    def as_dict(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
        }


def _round_up(x: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(x))
    return int(math.ceil(max(1, x) / multiple) * multiple)


def _branch_rank_by_kind(
    branches: Iterable, kind: str, reducer: str = "max",
    rank_map: dict[str, int] | None = None,
) -> int | None:
    """Aggregate behavioral rank across branches matching ``kind``.

    If ``rank_map`` is provided, the rank for branch ``b`` is taken as
    ``rank_map.get(b.name, b.behavioral_rank)``. This lets the exact rank
    allocator (:mod:`fasd.compression.rank_allocator`) override per-branch
    ranks while preserving the existing aggregation logic.
    """
    if rank_map is not None:
        values = [
            int(rank_map.get(b.name, b.behavioral_rank))
            for b in branches if b.kind == kind
        ]
    else:
        values = [int(b.behavioral_rank) for b in branches if b.kind == kind]
    if not values:
        return None
    if reducer == "max":
        return max(values)
    if reducer == "mean":
        return int(round(sum(values) / len(values)))
    if reducer == "min":
        return min(values)
    raise ValueError(f"unknown reducer: {reducer!r}")


def profile_to_student_config(
    profile,
    *,
    teacher_config,
    arch_multiplier: float = 1.0,
    head_multiple: int | None = None,
    min_hidden: int = 64,
    depth_policy: DepthPolicy = "keep",
    depth_keep: int | None = None,
    rank_map: dict[str, int] | None = None,
) -> StudentConfig:
    """Turn a :class:`TeacherProfile` into a compressed student config.

    Parameters
    ----------
    profile
        :class:`TeacherProfile` with behavioral ranks per branch.
    teacher_config
        The teacher's config (must expose ``hidden_size``,
        ``intermediate_size``, ``num_attention_heads``,
        ``num_key_value_heads``, ``num_hidden_layers``). Falls back to
        ``num_attention_heads`` for ``num_key_value_heads`` if the
        teacher doesn't expose a GQA attribute.
    arch_multiplier
        Scale factor on the retained ranks (default 1.0 — purely
        profile-driven). Ignored when ``rank_map`` is provided.
    head_multiple
        Round ``hidden_size`` up to this multiple; defaults to the
        teacher's ``num_attention_heads`` so head-dim stays integer.
    min_hidden
        Floor on ``hidden_size``.
    depth_policy
        ``"keep"`` preserves the teacher's layer count. Otherwise
        ``depth_keep`` is the target layer count; ``"contiguous_tail"``
        drops from the end, ``"contiguous_middle"`` drops from the
        middle.
    depth_keep
        Target ``num_hidden_layers`` when ``depth_policy != "keep"``.
    rank_map
        Optional dict mapping branch name to per-branch rank, as produced
        by the exact greedy q/cost knapsack allocator
        (:func:`fasd.compression.rank_allocator.allocate_ranks`). When
        provided, the ranks override each branch's stored ``behavioral_rank``
        and ``arch_multiplier`` is set to 1.0 (the rank-map already encodes
        the budget; further scaling would corrupt it).
    """
    branches = list(profile.branches if hasattr(profile, "branches") else profile)
    if arch_multiplier <= 0:
        raise ValueError(f"arch_multiplier must be > 0, got {arch_multiplier}")
    if rank_map is not None:
        # rank_map already encodes the budget; do not re-scale.
        arch_multiplier = 1.0

    t_hidden = int(getattr(teacher_config, "hidden_size", 0)) or int(
        getattr(teacher_config, "n_embd", 0)
    )
    t_interm = int(getattr(teacher_config, "intermediate_size", 0)) or int(
        getattr(teacher_config, "n_inner", 0) or 4 * t_hidden
    )
    t_heads = int(getattr(teacher_config, "num_attention_heads", 0)) or int(
        getattr(teacher_config, "n_head", 1)
    )
    t_kv_heads = int(
        getattr(teacher_config, "num_key_value_heads", t_heads) or t_heads
    )
    t_layers = int(getattr(teacher_config, "num_hidden_layers", 0)) or int(
        getattr(teacher_config, "n_layer", 1)
    )
    if t_hidden < 1:
        raise ValueError("teacher config missing hidden_size / n_embd")

    if head_multiple is None:
        head_multiple = max(1, t_heads)

    # Residual rank drives hidden_size if we have a residual branch in
    # the profile; otherwise use the max Q/K/V/O rank as a proxy.
    resid = _branch_rank_by_kind(branches, "block.residual", reducer="max", rank_map=rank_map)
    if resid is None:
        resid = max(
            _branch_rank_by_kind(branches, "attn.q", reducer="max", rank_map=rank_map) or 0,
            _branch_rank_by_kind(branches, "attn.o", reducer="max", rank_map=rank_map) or 0,
            _branch_rank_by_kind(branches, "ffn.down", reducer="max", rank_map=rank_map) or 0,
        )
    if resid <= 0:
        resid = t_hidden  # unable to determine → keep teacher size
    hidden_size = min(
        t_hidden, _round_up(int(round(resid * arch_multiplier)), head_multiple)
    )
    # Floor at min_hidden, but never exceed the teacher (a compression method must
    # not inflate). On tiny teachers (hidden < min_hidden) the cap wins, which also
    # keeps the absorbed bases V_in/V_out valid Stiefel points (n >= k).
    hidden_size = max(hidden_size, min(min_hidden, t_hidden))

    # Intermediate driven by FFN branches.
    ffn_up = _branch_rank_by_kind(branches, "ffn.up", reducer="max", rank_map=rank_map)
    ffn_gate = _branch_rank_by_kind(branches, "ffn.gate", reducer="max", rank_map=rank_map)
    ffn_max = max(ffn_up or 0, ffn_gate or 0)
    if ffn_max <= 0:
        ffn_max = t_interm
    intermediate_size = min(
        t_interm, _round_up(int(round(ffn_max * arch_multiplier)), head_multiple)
    )
    intermediate_size = max(intermediate_size, hidden_size)  # floor at hidden

    # Attention heads — preferentially keep teacher's count. Head dim
    # divides hidden_size; if not, reduce heads until it does.
    num_heads = t_heads
    while num_heads > 1 and hidden_size % num_heads != 0:
        num_heads -= 1

    # KV heads — drop independently based on K/V branch ranks.
    kv_max = max(
        _branch_rank_by_kind(branches, "attn.k", reducer="max", rank_map=rank_map) or 0,
        _branch_rank_by_kind(branches, "attn.v", reducer="max", rank_map=rank_map) or 0,
    )
    if kv_max <= 0:
        kv_heads = t_kv_heads
    else:
        # Head dim from Q branch: hidden_size / num_heads
        head_dim = max(1, hidden_size // max(1, num_heads))
        kv_heads = max(1, int(round(kv_max / max(1, head_dim))))
        kv_heads = min(kv_heads, num_heads)
        # KV heads must evenly divide query heads in GQA.
        while num_heads % kv_heads != 0 and kv_heads > 1:
            kv_heads -= 1

    # Depth.
    if depth_policy == "keep" or depth_keep is None:
        num_layers = t_layers
    else:
        num_layers = int(max(1, min(depth_keep, t_layers)))

    return StudentConfig(
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_heads),
        num_key_value_heads=int(kv_heads),
        num_hidden_layers=int(num_layers),
    )


def contiguous_layer_mapping(
    num_teacher_layers: int, num_student_layers: int, policy: DepthPolicy
) -> list[int]:
    """Return the indices of teacher layers that correspond to student layers.

    - ``"keep"`` → ``[0, 1, ..., num_student_layers-1]`` (with
      ``num_student_layers == num_teacher_layers``).
    - ``"contiguous_tail"`` → keep the first ``num_student_layers``.
    - ``"contiguous_middle"`` → drop a contiguous block from the
      middle of the teacher's layer stack.
    """
    if num_student_layers > num_teacher_layers:
        raise ValueError(
            f"student layers ({num_student_layers}) cannot exceed teacher ({num_teacher_layers})"
        )
    if policy == "keep":
        return list(range(num_teacher_layers))[:num_student_layers]
    if policy == "contiguous_tail":
        return list(range(num_student_layers))
    if policy == "contiguous_middle":
        # Keep equal halves at the start and end; drop the middle.
        half = num_student_layers // 2
        front = list(range(half))
        back = list(range(num_teacher_layers - (num_student_layers - half), num_teacher_layers))
        return front + back
    raise ValueError(f"unknown depth_policy: {policy!r}")


def plan_progressive_stages(
    teacher_config,
    target: StudentConfig,
    *,
    max_single_step: float = 3.0,
    n_stages: int | None = None,
) -> list[StudentConfig]:
    """Teacher-assistant chain planner.

    If the compression ratio from teacher to target exceeds
    ``max_single_step`` (measured by ``hidden_size``), returns one or
    more intermediate configs between teacher and target. Otherwise
    returns ``[target]``.
    """
    t_hidden = int(getattr(teacher_config, "hidden_size", 0)) or int(
        getattr(teacher_config, "n_embd", 0)
    )
    t_interm = int(getattr(teacher_config, "intermediate_size", 0)) or int(
        getattr(teacher_config, "n_inner", 0) or 4 * t_hidden
    )
    t_heads = int(getattr(teacher_config, "num_attention_heads", 0)) or int(
        getattr(teacher_config, "n_head", 1)
    )
    t_kv = int(getattr(teacher_config, "num_key_value_heads", t_heads) or t_heads)
    t_layers = int(getattr(teacher_config, "num_hidden_layers", 0)) or int(
        getattr(teacher_config, "n_layer", 1)
    )

    if target.hidden_size <= 0:
        return [target]
    ratio = max(t_hidden / max(1, target.hidden_size), 1.0)
    if n_stages is None:
        n_stages = max(1, int(math.ceil(math.log(ratio) / math.log(max_single_step))))
    if n_stages <= 1:
        return [target]

    stages: list[StudentConfig] = []
    for i in range(1, n_stages + 1):
        frac = i / n_stages
        hidden = int(round(t_hidden * (target.hidden_size / t_hidden) ** frac))
        interm = int(round(t_interm * (target.intermediate_size / t_interm) ** frac))
        heads = int(round(t_heads * (target.num_attention_heads / t_heads) ** frac))
        heads = max(1, heads)
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        kv = int(round(t_kv * (target.num_key_value_heads / t_kv) ** frac))
        kv = max(1, min(kv, heads))
        while heads % kv != 0 and kv > 1:
            kv -= 1
        layers = int(round(t_layers * (target.num_hidden_layers / t_layers) ** frac))
        stages.append(
            StudentConfig(
                hidden_size=hidden,
                intermediate_size=max(interm, hidden),
                num_attention_heads=heads,
                num_key_value_heads=kv,
                num_hidden_layers=max(1, layers),
            )
        )
    return stages


__all__ = [
    "StudentConfig",
    "DepthPolicy",
    "profile_to_student_config",
    "contiguous_layer_mapping",
    "plan_progressive_stages",
]
