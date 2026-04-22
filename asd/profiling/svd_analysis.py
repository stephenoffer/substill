"""SVD decomposition of activation covariance to find principal activation subspace."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .sparsity_analysis import SparsityStats


@dataclass
class LayerProfile:
    """Complete activation profile for one layer."""
    name: str
    eigenvalues: Tensor             # Eigenvalues of activation covariance (descending)
    principal_components: Tensor    # Top-k eigenvectors, shape (C, k)
    effective_rank: int             # k where cumsum(eigenvalues) >= threshold * total
    total_channels: int
    compression_ratio: float        # effective_rank / total_channels
    sparsity_stats: SparsityStats


class SVDAnalyzer:
    """Analyze activation covariance matrices via eigendecomposition.

    `definition` controls how effective rank is derived from the spectrum:

    - "variance" (default): smallest k with cumulative variance ≥
      `variance_threshold`. Classic; sensitive to tail.
    - "stable": ⌈Σ λ_i / λ_max⌉. Dimensionless, no threshold. Uses ceiling so
      heavy-tailed spectra don't collapse to 1.
    - "participation": ⌈(Σ λ_i)² / Σ λ_i²⌉. Measures spread of the spectrum;
      robust to long tails.
    - "entropy": ⌈exp(H(p))⌉ where p_i = λ_i / Σ λ. Smooth, between stable and
      variance rank.

    Eigenvalues smaller than `eps_relative * λ_max` are treated as zero before
    the rank statistic is computed. This guards against floating-point noise in
    the bottom of the spectrum inflating `stable` / `participation` / `entropy`
    ranks (and also keeps `clamp(min=0)` from silently masking genuine negative
    eigenvalues — we explicitly assert PSD-within-tolerance instead).
    """

    _DEFINITIONS = ("variance", "stable", "participation", "entropy")

    def __init__(
        self,
        variance_threshold: float = 0.95,
        definition: str = "variance",
        eps_relative: float = 1e-6,
    ):
        if definition not in self._DEFINITIONS:
            raise ValueError(
                f"definition must be one of {self._DEFINITIONS}, got {definition!r}"
            )
        if not 0 < variance_threshold < 1:
            raise ValueError(
                f"variance_threshold must be in (0, 1), got {variance_threshold}"
            )
        self.variance_threshold = variance_threshold
        self.definition = definition
        self.eps_relative = eps_relative

    def analyze(
        self,
        name: str,
        covariance: Tensor,
        sparsity_stats: SparsityStats,
    ) -> LayerProfile:
        """Eigendecompose the covariance and compute effective rank."""
        # Symmetrize defensively — torch.linalg.eigh assumes Hermitian, but
        # accumulated covariance can drift by ~1e-6 due to floating-point
        # non-associativity.
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Any sizeable negative eigenvalue means the accumulator is broken; fail
        # loudly rather than masking it with clamp(min=0) as before.
        lam_max = eigenvalues.max().clamp(min=0)
        if lam_max > 0:
            most_negative = eigenvalues.min()
            if most_negative < -1e-4 * lam_max:
                raise ValueError(
                    f"{name}: covariance is not PSD (λ_min/λ_max = "
                    f"{(most_negative / lam_max).item():.2e}). Check the accumulator."
                )
        eigenvalues = eigenvalues.clamp(min=0)

        effective_rank = self.compute_effective_rank(eigenvalues)
        total_channels = covariance.shape[0]

        return LayerProfile(
            name=name,
            eigenvalues=eigenvalues,
            principal_components=eigenvectors[:, :effective_rank],
            effective_rank=effective_rank,
            total_channels=total_channels,
            compression_ratio=effective_rank / total_channels,
            sparsity_stats=sparsity_stats,
        )

    def compute_effective_rank(self, eigenvalues: Tensor) -> int:
        ev = self._denoise(eigenvalues)
        if self.definition == "variance":
            return self._variance_rank(ev, self.variance_threshold)
        if self.definition == "stable":
            return self._stable_rank(ev)
        if self.definition == "participation":
            return self._participation_rank(ev)
        if self.definition == "entropy":
            return self._entropy_rank(ev)
        raise RuntimeError(f"Unreachable: {self.definition}")

    def _denoise(self, eigenvalues: Tensor) -> Tensor:
        """Zero out eigenvalues below eps_relative·λ_max.

        Accumulated covariance in float32 typically has a 1e-7 noise floor
        relative to the top eigenvalue. Without this, the entropy/participation
        ranks would include hundreds of noise components and never look low.
        """
        lam_max = eigenvalues.max()
        if lam_max <= 0:
            return eigenvalues
        return torch.where(
            eigenvalues >= self.eps_relative * lam_max,
            eigenvalues,
            torch.zeros_like(eigenvalues),
        )

    @staticmethod
    def _variance_rank(eigenvalues: Tensor, threshold: float) -> int:
        total = eigenvalues.sum()
        if total < 1e-10:
            return 1
        cumulative = torch.cumsum(eigenvalues, dim=0)
        ratio = cumulative / total
        mask = ratio >= threshold
        if not mask.any():
            return len(eigenvalues)
        k = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
        return max(1, k)

    @staticmethod
    def _stable_rank(eigenvalues: Tensor) -> int:
        lam_max = eigenvalues.max()
        if lam_max < 1e-10:
            return 1
        # Ceiling — "at least this many components fit into λ_max's budget".
        # `round()` previously collapsed heavy-tailed spectra to 1-2 even when
        # the tail carried meaningful signal.
        k = int(math.ceil((eigenvalues.sum() / lam_max).item()))
        return max(1, min(k, len(eigenvalues)))

    @staticmethod
    def _participation_rank(eigenvalues: Tensor) -> int:
        denom = (eigenvalues ** 2).sum()
        if denom < 1e-20:
            return 1
        k = int(math.ceil(((eigenvalues.sum() ** 2) / denom).item()))
        return max(1, min(k, len(eigenvalues)))

    @staticmethod
    def _entropy_rank(eigenvalues: Tensor) -> int:
        total = eigenvalues.sum()
        if total < 1e-10:
            return 1
        p = eigenvalues / total
        p = p[p > 0]
        entropy = -(p * p.log()).sum()
        k = int(math.ceil(entropy.exp().item()))
        return max(1, min(k, len(eigenvalues)))


def aggregate_stage_profile(
    stage_profiles: list[LayerProfile],
    mode: str = "last",
) -> LayerProfile:
    """Reduce a stage's per-block profiles to one stage-level profile.

    Modes:

    - "last": use the last block's profile. Corresponds to the stage output.
      Back-compat with the original pipeline.
    - "max_rank": pick the block with the highest effective rank (i.e., the
      stage's information bottleneck — the block that is hardest to compress).
    - "average": sum per-block covariances (reconstructed as V Λ Vᵀ) and
      re-eigendecompose. This produces components that capture variance
      observed anywhere in the stage. Effective rank is recomputed as the
      variance rank at threshold 0.95.

    All blocks in `stage_profiles` must share the same `total_channels`.
    """
    if not stage_profiles:
        raise ValueError("stage_profiles is empty")
    channels = stage_profiles[0].total_channels
    for p in stage_profiles:
        if p.total_channels != channels:
            raise ValueError(
                f"stage aggregation requires matching channel counts, got "
                f"{p.total_channels} vs {channels}"
            )

    if mode == "last":
        return stage_profiles[-1]

    if mode == "max_rank":
        return max(stage_profiles, key=lambda p: p.effective_rank)

    if mode == "average":
        # Sum block covariances in the full channel basis, re-eigendecompose.
        # Each block stored only the top-k eigenvectors/eigenvalues, so we
        # reconstruct the rank-k covariance approximation V Λ Vᵀ and sum those.
        C = channels
        cov_sum = torch.zeros(C, C, dtype=stage_profiles[0].eigenvalues.dtype)
        for p in stage_profiles:
            V = p.principal_components                          # (C, k)
            lam = p.eigenvalues[: V.shape[1]].clamp(min=0)      # (k,)
            cov_sum = cov_sum + (V * lam.unsqueeze(0)) @ V.T
        cov_sum = 0.5 * (cov_sum + cov_sum.T)
        eigvals, eigvecs = torch.linalg.eigh(cov_sum)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx].clamp(min=0)
        eigvecs = eigvecs[:, idx]

        # Re-derive effective rank at the same variance threshold used elsewhere.
        total = eigvals.sum()
        if total < 1e-10:
            k = max(p.effective_rank for p in stage_profiles)
        else:
            ratio = torch.cumsum(eigvals, dim=0) / total
            mask = ratio >= 0.95
            k = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1 if mask.any() else C
        k = max(1, min(k, C))

        last = stage_profiles[-1]
        return LayerProfile(
            name=f"{last.name}:stage_avg",
            eigenvalues=eigvals,
            principal_components=eigvecs[:, :k],
            effective_rank=k,
            total_channels=C,
            compression_ratio=k / C,
            sparsity_stats=last.sparsity_stats,
        )

    raise ValueError(f"Unknown stage aggregation mode: {mode!r}")


def group_profiles_by_stage(profiles: list[LayerProfile]) -> dict[int, list[LayerProfile]]:
    """Group profiles by stage (identified by channel count)."""
    stage_map: dict[int, list[LayerProfile]] = {}
    for p in profiles:
        stage_map.setdefault(p.total_channels, []).append(p)
    return stage_map


def profiles_to_stage_widths(
    profiles: list[LayerProfile],
    min_width: int = 16,
    width_multiple: int = 8,
    rank_reduction: str = "max",
) -> list[int]:
    """Convert layer profiles to per-stage student widths.

    Groups profiles by stage (based on total_channels) and reduces per-block
    ranks to one rank per stage. Rounds up to width_multiple.

    rank_reduction:
      - "max" (default): take the largest per-block rank in the stage.
        Conservative — never undersizes relative to any block.
      - "mean": average over blocks in the stage. Less conservative; uses
        the fact that later blocks are typically what the next stage sees.
      - "last": use the last block (the stage's output). Matches the
        "last"-aggregation subspace loss exactly.
    """
    if rank_reduction not in ("max", "mean", "last"):
        raise ValueError(f"Unknown rank_reduction: {rank_reduction!r}")

    stage_map = group_profiles_by_stage(profiles)

    widths = []
    for channels in sorted(stage_map.keys()):
        ranks = [p.effective_rank for p in stage_map[channels]]
        if rank_reduction == "max":
            rank = max(ranks)
        elif rank_reduction == "mean":
            rank = int(math.ceil(sum(ranks) / len(ranks)))
        else:  # "last"
            rank = ranks[-1]
        width = max(min_width, ((rank + width_multiple - 1) // width_multiple) * width_multiple)
        widths.append(width)

    return widths


def profiles_to_stage_blocks(
    profiles: list[LayerProfile],
    min_blocks: int = 1,
    max_blocks: int = 4,
    saturation_tol: float = 0.05,
) -> list[int]:
    """Derive per-stage block counts from rank evolution within a stage.

    Heuristic: if effective rank plateaus after k blocks (i.e., subsequent
    blocks contribute < `saturation_tol` relative increase in rank), the stage
    only needs ~k blocks in the student. This is a data-driven alternative to
    the hand-set `blocks_per_stage` hyperparameter.

    Falls back to `min_blocks` if a stage has only one profiled block, and
    clamps to `[min_blocks, max_blocks]`.
    """
    stage_map = group_profiles_by_stage(profiles)
    blocks = []
    for channels in sorted(stage_map.keys()):
        ranks = [p.effective_rank for p in stage_map[channels]]
        if len(ranks) <= 1:
            blocks.append(min_blocks)
            continue
        # Scan forward; stop at the first block where the rank stops growing.
        saturated_at = len(ranks)
        for i in range(1, len(ranks)):
            prev = max(ranks[i - 1], 1)
            rel = (ranks[i] - ranks[i - 1]) / prev
            if rel < saturation_tol:
                saturated_at = i
                break
        blocks.append(max(min_blocks, min(max_blocks, saturated_at)))
    return blocks


def save_profiles(profiles: list[LayerProfile], path: str) -> None:
    """Save profiles to disk."""
    data = {
        "profiles": [
            {
                "name": p.name,
                "eigenvalues": p.eigenvalues,
                "principal_components": p.principal_components,
                "effective_rank": p.effective_rank,
                "total_channels": p.total_channels,
                "compression_ratio": p.compression_ratio,
                "sparsity_stats": {
                    "sparsity_ratio": p.sparsity_stats.sparsity_ratio,
                    "activation_histogram": p.sparsity_stats.activation_histogram,
                    "bin_edges": p.sparsity_stats.bin_edges,
                    "entropy": p.sparsity_stats.entropy,
                    "mean_activation": p.sparsity_stats.mean_activation,
                    "std_activation": p.sparsity_stats.std_activation,
                },
            }
            for p in profiles
        ]
    }
    torch.save(data, path)


def load_profiles(path: str) -> list[LayerProfile]:
    """Load profiles from disk."""
    data = torch.load(path, weights_only=False)
    profiles = []
    for d in data["profiles"]:
        ss = d["sparsity_stats"]
        sparsity_stats = SparsityStats(
            sparsity_ratio=ss["sparsity_ratio"],
            activation_histogram=ss["activation_histogram"],
            bin_edges=ss["bin_edges"],
            entropy=ss["entropy"],
            mean_activation=ss["mean_activation"],
            std_activation=ss["std_activation"],
        )
        profiles.append(LayerProfile(
            name=d["name"],
            eigenvalues=d["eigenvalues"],
            principal_components=d["principal_components"],
            effective_rank=d["effective_rank"],
            total_channels=d["total_channels"],
            compression_ratio=d["compression_ratio"],
            sparsity_stats=sparsity_stats,
        ))
    return profiles
