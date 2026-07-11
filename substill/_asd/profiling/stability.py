"""Subspace-stability diagnostic via principal-angle bootstrap.

Measures whether the top-k subspace used for rank sizing is stable
under resampling of the calibration set. An unstable subspace
undermines any width claim derived from it.

Given a teacher, a calibration dataset, and a list of layers to
profile:

1. Build ``n_boot`` random half-size calibration subsets.
2. Run the capture/SVD pipeline on each.
3. For every pair of bootstrap runs, compute principal angles between
   their retained top-k subspaces via ``arccos(svd(V_a^T V_b).S)``.

Returns per-layer statistics: median angle, P90 angle, and the angle
at index ``k - 1`` (the most unstable retained direction). Cross-
reference with the per-stage profile to see whether the worst stage
is also the most unstable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .activation_capture import ActivationCaptureEngine
from .svd_analysis import SVDAnalyzer


@dataclass
class StabilityStats:
    """Per-layer stability summary across bootstrap runs."""

    name: str
    k: int
    median_angle_deg: float
    p90_angle_deg: float
    max_angle_deg: float
    n_pairs: int


def _principal_angles(V_a: Tensor, V_b: Tensor) -> Tensor:
    """Return principal angles in degrees, sorted ascending.

    ``V_a`` and ``V_b`` are column-orthonormal ``(C, k)``. If they
    have different ``k``, the comparison uses ``min(k_a, k_b)``: the
    smaller subspace is the "hardest-to-hit" one and trailing cosines
    from the larger subspace are discarded.
    """
    k = min(V_a.shape[1], V_b.shape[1])
    if k == 0:
        return torch.zeros(0)
    M = V_a[:, :k].T @ V_b[:, :k]
    s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles_deg = torch.arccos(s) * (180.0 / math.pi)
    angles_deg, _ = torch.sort(angles_deg)
    return angles_deg


def bootstrap_principal_angles(
    teacher: torch.nn.Module,
    calib_dataset,
    layer_names: Sequence[str],
    *,
    n_boot: int = 5,
    frac: float = 0.5,
    variance_threshold: float = 0.95,
    rank_definition: str = "variance",
    activation_source: str = "output",
    covariance_mode: str = "per_pixel",
    spatial_subsample: int = 1,
    batch_size: int = 128,
    num_workers: int = 2,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, StabilityStats]:
    """Bootstrap principal angles between top-k subspaces per layer.

    Runs the capture+SVD pipeline ``n_boot`` times on independent
    random subsets of ``calib_dataset`` (each of size
    ``frac * len(dataset)``), then computes pairwise principal angles
    between the retained subspaces.

    Returns one :class:`StabilityStats` per layer, with the
    median / P90 / max angle across all ``C(n_boot, 2)`` pair
    comparisons and the number of contributing pairs.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2 to form pairs, got {n_boot}")
    if not 0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    N = len(calib_dataset)
    subset_size = max(1, int(round(N * frac)))

    per_layer_subspaces: dict[str, list[Tensor]] = {name: [] for name in layer_names}

    rng = torch.Generator()
    rng.manual_seed(seed)

    for _ in range(n_boot):
        indices = torch.randperm(N, generator=rng)[:subset_size].tolist()
        subset = Subset(calib_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        engine = ActivationCaptureEngine(
            teacher,
            list(layer_names),
            covariance_mode=covariance_mode,
            spatial_subsample=spatial_subsample,
            source=activation_source,
        )
        accumulators = engine.run(loader, device=device)

        svd = SVDAnalyzer(
            variance_threshold=variance_threshold,
            definition=rank_definition,
        )

        for name in layer_names:
            acc = accumulators[name]
            cov = acc.finalize()
            profile = svd.analyze(name, cov, source=activation_source)
            per_layer_subspaces[name].append(
                profile.principal_components.detach().cpu().clone()
            )

    out: dict[str, StabilityStats] = {}
    for name in layer_names:
        subspaces = per_layer_subspaces[name]
        angles_deg_all: list[Tensor] = []
        for a in range(len(subspaces)):
            for b in range(a + 1, len(subspaces)):
                angles_deg_all.append(_principal_angles(subspaces[a], subspaces[b]))
        if not angles_deg_all:
            continue
        min_k = min(v.shape[0] for v in angles_deg_all)
        if min_k == 0:
            continue
        stacked = torch.stack([v[:min_k] for v in angles_deg_all])
        flat = stacked.flatten()
        out[name] = StabilityStats(
            name=name,
            k=int(min_k),
            median_angle_deg=float(flat.median().item()),
            p90_angle_deg=float(torch.quantile(flat, 0.9).item()),
            max_angle_deg=float(flat.max().item()),
            n_pairs=int(stacked.shape[0]),
        )

    return out


def stability_stats_to_json(stats: dict[str, StabilityStats]) -> dict:
    """Serialize stability stats for :func:`json.dump`."""
    return {
        name: {
            "k": s.k,
            "median_angle_deg": s.median_angle_deg,
            "p90_angle_deg": s.p90_angle_deg,
            "max_angle_deg": s.max_angle_deg,
            "n_pairs": s.n_pairs,
        }
        for name, s in stats.items()
    }
