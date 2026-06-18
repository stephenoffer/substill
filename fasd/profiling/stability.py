"""Stability diagnostics for fasd.

Re-exports :func:`bootstrap_principal_angles` and :class:`StabilityStats`
from asd, and adds :func:`stability_adjusted_rank`, which caps a
behavioral rank at the largest k whose subspace stays stable across
bootstrap splits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from asd.profiling.stability import (  # noqa: F401
    StabilityStats,
    bootstrap_principal_angles,
    stability_stats_to_json,
)


def _principal_angles_deg(V_a: Tensor, V_b: Tensor) -> Tensor:
    """Principal angles (degrees) between two column-orthonormal bases."""
    k = min(V_a.shape[1], V_b.shape[1])
    if k == 0:
        return torch.zeros(0)
    M = V_a[:, :k].T @ V_b[:, :k]
    s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles = torch.arccos(s) * (180.0 / math.pi)
    angles, _ = torch.sort(angles)
    return angles


def stability_adjusted_rank(
    bases: Sequence[Tensor],
    proposed_rank: int,
    *,
    angle_cap_deg: float = 15.0,
    min_rank: int = 1,
) -> tuple[int, float]:
    """Cap a rank at the largest stable k across bootstrap bases.

    Each entry of ``bases`` is a ``(C, K)`` orthonormal basis from one
    bootstrap split. For each candidate ``k = 1..proposed_rank``, we
    compute the median pairwise principal angle across the ``k``-
    truncated bases. The returned rank is the largest ``k`` for which
    this median is below ``angle_cap_deg``.

    Returns ``(k_stable, median_angle_at_k_stable)``. If no candidate
    meets the cap, returns ``(min_rank, median_angle_at_min_rank)``.
    """
    if proposed_rank < 1:
        raise ValueError(f"proposed_rank must be >= 1, got {proposed_rank}")
    if len(bases) < 2:
        return int(proposed_rank), 0.0

    k_max = min(proposed_rank, min(b.shape[1] for b in bases))
    if k_max < 1:
        return int(min_rank), float("inf")

    k_stable = min_rank
    angle_at_stable = float("inf")
    for k in range(1, k_max + 1):
        angles = []
        for a in range(len(bases)):
            for b in range(a + 1, len(bases)):
                ang = _principal_angles_deg(bases[a][:, :k], bases[b][:, :k])
                if ang.numel() > 0:
                    angles.append(ang)
        if not angles:
            continue
        flat = torch.cat(angles)
        # Use the MAX pairwise principal angle: the cap reflects the
        # worst retained direction, not the typical one.
        worst = float(flat.max().item())
        if worst <= angle_cap_deg:
            k_stable = k
            angle_at_stable = worst
        else:
            break
    if angle_at_stable == float("inf"):
        angles = []
        for a in range(len(bases)):
            for b in range(a + 1, len(bases)):
                ang = _principal_angles_deg(
                    bases[a][:, :min_rank], bases[b][:, :min_rank]
                )
                if ang.numel() > 0:
                    angles.append(ang)
        angle_at_stable = float(torch.cat(angles).max().item()) if angles else 0.0
    return int(k_stable), float(angle_at_stable)


__all__ = [
    "StabilityStats",
    "bootstrap_principal_angles",
    "stability_adjusted_rank",
    "stability_stats_to_json",
]
