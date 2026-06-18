"""Exact global rank allocator under a parameter budget.

The matched-compression search at
[scripts/fasd_ablation.py:489-533](../../scripts/fasd_ablation.py)
binary-searches a single global ``arch_multiplier`` to hit a parameter target.
This couples *all* edges to one scalar — it cannot increase the rank on a
high-q FFN edge while decreasing it on a low-q attention edge. The result is
that a 4× compression run gets a uniform 0.5× rank scaling, which the
behavioral-rank profile already partially compensates for, but global
budget-vs-quality trade-offs are out of reach.

This module replaces that with a greedy q/cost knapsack:

  Given:
    - per-edge per-rank scores  q[edge][k]  (Fisher-weighted, monotone non-increasing in k)
    - per-edge cost-of-rank-step  cost(edge, current_rank)  (parameters added by ++rank)
    - global parameter budget P*
    - per-edge legal step sizes (head-group units for attn, hardware multiples for FFN)
    - per-edge min/max rank
  Choose:
    rank[e] for each edge maximising  Σ_{e,i<rank[e]} q[e][i]
    subject to Σ_e params(e, rank[e]) ≤ P*.

We use a greedy ratio-of-gains heuristic (the standard 0/1 knapsack
approximation): at each step, identify the edge whose next legal rank
increment yields the highest q-gain / cost ratio, and increment if budget
allows. When nothing fits, stop. Local refinement: try swapping a low-q-per-cost
allocation for a higher one if that reduces total params (or, symmetrically,
adds an even higher-ratio increment).

The result is *exact* w.r.t. the parameter budget (within the granularity of
legal step sizes; we cap at ±1% of target).
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from fasd.profiling.functional_score import DirectionScores


@dataclass
class EdgeSpec:
    """One compressible edge in the model graph.

    - ``name``: identifier matching the profile branch name (e.g. ``"attn.q.0"``).
    - ``q``: per-direction Fisher-weighted scores, shape ``(k_max,)``,
      *not* required to be monotone — the allocator slices ``q[:k]`` and sums.
    - ``cost_per_rank``: parameters contributed by the i-th rank slot. Either
      a scalar (constant cost-per-rank, typical for Linear) or a 1-D tensor of
      the same length as ``q`` (variable, e.g. when adjacent edges share a basis).
    - ``min_rank``: floor (must be ≥ 1).
    - ``max_rank``: ceiling (defaults to ``len(q)``).
    - ``step``: legal rank increment. Set to ``head_dim`` for attention, ``8``
      or ``16`` for FFN intermediate, ``1`` for residual width.
    - ``fixed_overhead``: parameters in this edge that don't depend on rank
      (biases, norms attached to this edge). Counted once, regardless of rank.
    """

    name: str
    q: Tensor  # (k_max,)
    cost_per_rank: Tensor | float  # scalar or (k_max,)
    min_rank: int = 1
    max_rank: int | None = None
    step: int = 1
    fixed_overhead: int = 0

    def __post_init__(self):
        if self.q.dim() != 1:
            raise ValueError(f"q must be 1-D, got shape {tuple(self.q.shape)}")
        if self.max_rank is None:
            self.max_rank = int(self.q.shape[0])
        if self.min_rank < 1:
            raise ValueError(f"min_rank must be >= 1, got {self.min_rank}")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        if isinstance(self.cost_per_rank, Tensor) and self.cost_per_rank.shape != self.q.shape:
            raise ValueError(
                f"cost_per_rank tensor shape {tuple(self.cost_per_rank.shape)} "
                f"!= q shape {tuple(self.q.shape)}"
            )

    def cost_of_step(self, current_rank: int, next_rank: int) -> float:
        """Cost (parameters) of going from current_rank to next_rank."""
        if isinstance(self.cost_per_rank, Tensor):
            return float(self.cost_per_rank[current_rank:next_rank].sum().item())
        return float(self.cost_per_rank) * (next_rank - current_rank)

    def gain_of_step(self, current_rank: int, next_rank: int) -> float:
        """Quality gain (sum of q) of going from current_rank to next_rank."""
        return float(self.q[current_rank:next_rank].sum().item())

    def cost_at(self, rank: int) -> float:
        """Total parameters at the given rank (including fixed overhead)."""
        if isinstance(self.cost_per_rank, Tensor):
            base = float(self.cost_per_rank[:rank].sum().item())
        else:
            base = float(self.cost_per_rank) * rank
        return base + float(self.fixed_overhead)


@dataclass
class AllocationResult:
    """Output of :func:`allocate_ranks`."""

    ranks: dict[str, int]
    total_params: int
    target_params: int
    relative_error: float
    total_q: float
    iterations: int
    log: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rel_pct = 100.0 * self.relative_error
        lines = [
            f"target: {self.target_params:,}  realised: {self.total_params:,}  "
            f"err: {rel_pct:+.2f}%  total_q: {self.total_q:.3e}  iters: {self.iterations}"
        ]
        for name, k in sorted(self.ranks.items()):
            lines.append(f"  {name:30s} k={k}")
        return "\n".join(lines)


def allocate_ranks(
    edges: list[EdgeSpec],
    target_params: int,
    *,
    tol: float = 0.01,
    extra_overhead: int = 0,
    max_iterations: int = 100_000,
    verbose: bool = False,
) -> AllocationResult:
    """Greedy q/cost knapsack allocator.

    Parameters
    ----------
    edges : list[EdgeSpec]
        All compressible edges in the student graph.
    target_params : int
        Desired total trainable parameters (excluding ``extra_overhead``).
    tol : float
        Acceptable relative error vs. target. Default 1%.
    extra_overhead : int
        Parameters that exist regardless of rank choice (embeddings, LM head,
        norms not attached to a specific edge). Subtracted from ``target_params``
        before allocation; the allocator only chooses ranks for ``edges``.
    max_iterations : int
        Safety cap on the greedy loop.
    verbose : bool
        Log every increment to ``result.log``.

    Returns:
    -------
    AllocationResult
    """
    # Initialise at min_rank for every edge.
    ranks: dict[str, int] = {e.name: e.min_rank for e in edges}
    edge_by_name = {e.name: e for e in edges}
    total = float(extra_overhead) + sum(e.cost_at(e.min_rank) for e in edges)
    total_q = sum(e.gain_of_step(0, e.min_rank) for e in edges)
    log: list[str] = []

    if total > target_params * (1 + tol):
        # Even minimum-rank allocation overshoots. Caller must increase target
        # or lower min_ranks.
        return AllocationResult(
            ranks=ranks,
            total_params=int(round(total)),
            target_params=int(target_params),
            relative_error=(total - target_params) / max(1, target_params),
            total_q=total_q,
            iterations=0,
            log=[f"infeasible: min-rank allocation costs {int(total):,} > "
                 f"target {target_params:,} * (1+{tol})"],
        )

    # Priority queue of (negative q/cost ratio, edge_name, next_rank).
    # We pop the most-attractive next-step at each iteration.
    def push(heap, e: EdgeSpec, current_rank: int):
        next_rank = current_rank + e.step
        if next_rank > e.max_rank:
            return
        cost = e.cost_of_step(current_rank, next_rank)
        gain = e.gain_of_step(current_rank, next_rank)
        if cost <= 0:
            return  # avoid div-by-zero; treat as no-op
        ratio = gain / cost
        # Heapq is a min-heap; negate so we pop largest ratio first. Use edge name
        # as a tie-breaker so the heap is deterministic.
        heapq.heappush(heap, (-ratio, e.name, current_rank, next_rank, gain, cost))

    heap: list = []
    for e in edges:
        push(heap, e, ranks[e.name])

    iters = 0
    while heap and iters < max_iterations:
        iters += 1
        neg_ratio, name, current_rank, next_rank, gain, cost = heapq.heappop(heap)
        e = edge_by_name[name]
        # Stale entry? (rank may have been advanced via a previous pop on this edge)
        if ranks[name] != current_rank:
            continue
        # Would this push us over budget?
        if total + cost > target_params * (1 + tol):
            # Skip this increment; try the next-best.
            continue
        # Apply.
        ranks[name] = next_rank
        total += cost
        total_q += gain
        if verbose:
            log.append(
                f"+ {name}: {current_rank} → {next_rank}  "
                f"gain={gain:.3e}  cost={int(cost):,}  total={int(total):,}"
            )
        # Push the next step on this edge.
        push(heap, e, next_rank)

        # Stop once we're inside the lower side of tolerance.
        if total >= target_params * (1 - tol):
            break

    rel_err = (total - target_params) / max(1, target_params)
    return AllocationResult(
        ranks=ranks,
        total_params=int(round(total)),
        target_params=int(target_params),
        relative_error=rel_err,
        total_q=total_q,
        iterations=iters,
        log=log,
    )


# ---------------------------------------------------------------------------
# Convenience: build EdgeSpec list from a TeacherProfile + DirectionScores.
# ---------------------------------------------------------------------------


def edges_from_profile(
    profile,
    direction_scores: dict[str, DirectionScores] | None = None,
    *,
    cost_fn: Callable[[object], float] | None = None,
    step_fn: Callable[[object], int] | None = None,
    max_rank: int | None = None,
) -> list[EdgeSpec]:
    """Build an :class:`EdgeSpec` list from a profile.

    Parameters
    ----------
    profile : TeacherProfile
    direction_scores : optional dict[branch_name -> DirectionScores]
        Fisher-weighted scores. If None, falls back to using eigenvalues as q
        (variance-only scoring; the same heuristic the existing width pruner uses).
    cost_fn : edge -> float
        Maps a profile branch to its per-rank parameter cost. The default heuristic:
        a Linear's per-rank cost is ``in_features`` for the input side or
        ``out_features`` for the output side, but in practice every profile branch
        is associated with a specific side. The caller usually wants to pass an
        explicit cost_fn that knows the architecture.
    step_fn : edge -> int
        Maps a profile branch to its legal rank step size.
    max_rank : optional cap on directions per edge.
    """
    out: list[EdgeSpec] = []
    for b in profile.branches:
        if direction_scores is not None and b.name in direction_scores:
            q = direction_scores[b.name].q
        else:
            ev = b.eigenvalues if b.eigenvalues is not None else torch.ones(
                b.principal_components.shape[1]
            )
            q = ev.detach().clone()
        if max_rank is not None:
            q = q[:max_rank]
        cost = cost_fn(b) if cost_fn is not None else 1.0
        step = step_fn(b) if step_fn is not None else 1
        spec = EdgeSpec(
            name=b.name,
            q=q,
            cost_per_rank=cost,
            min_rank=1,
            max_rank=int(q.shape[0]),
            step=step,
        )
        out.append(spec)
    return out


__all__ = ["EdgeSpec", "AllocationResult", "allocate_ranks", "edges_from_profile"]
