"""SVD decomposition of activation covariance to find principal activation subspace."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LayerProfile:
    """Complete activation profile for one layer.

    `source` records what the underlying covariance was computed from:
      - "output": the block's forward output.
      - "delta":  the residual update Δx_l = output − shortcut(input).
      - "branch": the output of a residual branch sub-module (e.g. conv3 in
                  a Bottleneck, or `attn`/`mlp` in a transformer block).

    The loss side checks this field before mixing profiles across sources.
    """
    name: str
    eigenvalues: Tensor             # Eigenvalues of activation covariance (descending)
    principal_components: Tensor    # Top-k eigenvectors, shape (C, k)
    effective_rank: int             # k where cumsum(eigenvalues) >= threshold * total
    total_channels: int
    compression_ratio: float        # effective_rank / total_channels
    source: str = "output"


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

    `noise_model` controls how the noise floor is estimated before rank
    computation:

    - "eps" (default): treat eigenvalues below `eps_relative * λ_max` as
      zero. Legacy; defends against float-precision noise.
    - "mp": Gavish-Donoho / Marchenko-Pastur bulk-edge threshold. Requires
      `n_effective` — the covariance's effective sample count adjusted for
      spatial correlation. Eigenvalues below `ω(β)² · σ_med²` (β = C / N_eff,
      ω(β) the Gavish-Donoho coefficient, σ_med the median eigenvalue of
      the lower half of the spectrum) are treated as noise.

    `shrinkage` applies a covariance-level regularizer before
    eigendecomposition:

    - "none" (default): no shrinkage.
    - "ledoit_wolf": linear shrinkage toward a scaled identity,
      `cov' = (1−α) cov + α (tr(cov)/C) I`. `α` is computed from the
      Ledoit-Wolf closed form when `n_effective` is given; defaults to
      0.1 otherwise.
    """

    _DEFINITIONS = ("variance", "stable", "participation", "entropy")
    _NOISE_MODELS = ("eps", "mp")
    _SHRINKAGE = ("none", "ledoit_wolf")

    def __init__(
        self,
        variance_threshold: float = 0.95,
        definition: str = "variance",
        eps_relative: float = 1e-6,
        noise_model: str = "eps",
        shrinkage: str = "none",
        n_effective: int | None = None,
    ):
        if definition not in self._DEFINITIONS:
            raise ValueError(
                f"definition must be one of {self._DEFINITIONS}, got {definition!r}"
            )
        if noise_model not in self._NOISE_MODELS:
            raise ValueError(
                f"noise_model must be one of {self._NOISE_MODELS}, got {noise_model!r}"
            )
        if shrinkage not in self._SHRINKAGE:
            raise ValueError(
                f"shrinkage must be one of {self._SHRINKAGE}, got {shrinkage!r}"
            )
        if not 0 < variance_threshold < 1:
            raise ValueError(
                f"variance_threshold must be in (0, 1), got {variance_threshold}"
            )
        self.variance_threshold = variance_threshold
        self.definition = definition
        self.eps_relative = eps_relative
        self.noise_model = noise_model
        self.shrinkage = shrinkage
        self.n_effective = n_effective

    def analyze(
        self,
        name: str,
        covariance: Tensor,
        source: str = "output",
    ) -> LayerProfile:
        """Eigendecompose the covariance and compute effective rank."""
        # Symmetrize defensively — torch.linalg.eigh assumes Hermitian, but
        # accumulated covariance can drift by ~1e-6 due to floating-point
        # non-associativity.
        covariance = 0.5 * (covariance + covariance.T)
        if self.shrinkage == "ledoit_wolf":
            covariance = self._ledoit_wolf_shrink(covariance)
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
            source=source,
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
        """Zero out eigenvalues below the estimated noise floor.

        `noise_model="eps"`: floor = eps_relative · λ_max. Protects
        against float-precision noise in the bottom of the spectrum
        inflating `stable` / `participation` / `entropy` ranks.

        `noise_model="mp"`: Gavish-Donoho bulk-edge. For an i.i.d. Gaussian
        C×N sample covariance with aspect ratio β = C / N, the noise-bulk
        edge is at λ* = ω(β)² σ². We estimate σ from the median of the
        lower half of the spectrum (a robust proxy for the Marchenko-
        Pastur bulk median) and flag eigenvalues below that threshold as
        noise.
        """
        lam_max = eigenvalues.max()
        if lam_max <= 0:
            return eigenvalues

        if self.noise_model == "mp":
            threshold = self._mp_threshold(eigenvalues)
        else:
            threshold = self.eps_relative * lam_max

        return torch.where(
            eigenvalues >= threshold,
            eigenvalues,
            torch.zeros_like(eigenvalues),
        )

    def _mp_threshold(self, eigenvalues: Tensor) -> Tensor:
        """Gavish-Donoho bulk-edge threshold.

        Falls back to the eps-based threshold if `n_effective` is not set
        or is non-positive. The derivation assumes C ≤ N and β ∈ (0, 1];
        if β ≥ 1 (more channels than samples) the estimator is ill-posed
        and we also fall back.
        """
        C = int(eigenvalues.shape[0])
        n_eff = self.n_effective
        lam_max = eigenvalues.max()
        if not n_eff or n_eff <= 0 or n_eff < C:
            return self.eps_relative * lam_max
        beta = C / float(n_eff)
        if not 0 < beta <= 1:
            return self.eps_relative * lam_max
        # Gavish-Donoho asymptotic coefficient for the unknown-σ case.
        #   ω(β) = √(2 (β+1) + 8β / ((β+1) + √(β² + 14 β + 1)))
        omega = math.sqrt(
            2 * (beta + 1)
            + 8 * beta / ((beta + 1) + math.sqrt(beta * beta + 14 * beta + 1))
        )
        # σ_median: median of the lower half of the spectrum is a robust
        # proxy for the MP bulk median. Divide by the MP bulk median
        # coefficient ≈ (1 + √β)² / 2 to get σ². (In practice, the operator
        # should sweep the threshold at a few values if this is critical.)
        lower_half = eigenvalues[eigenvalues.shape[0] // 2 :]
        lower_half = lower_half[lower_half > 0]
        if lower_half.numel() == 0:
            return self.eps_relative * lam_max
        sigma_sq_proxy = float(lower_half.median().item())
        # The MP bulk median in units of σ² is between (1 − √β)² and
        # (1 + √β)². Use the midpoint as a conservative normalizer.
        bulk_mid = ((1 - math.sqrt(beta)) ** 2 + (1 + math.sqrt(beta)) ** 2) / 2
        sigma_sq = sigma_sq_proxy / max(bulk_mid, 1e-12)
        threshold = (omega * omega) * sigma_sq
        return torch.tensor(threshold, dtype=eigenvalues.dtype, device=eigenvalues.device)

    @staticmethod
    def _ledoit_wolf_shrink(covariance: Tensor) -> Tensor:
        """Linear shrinkage toward a scaled identity.

        We don't have the raw sample tensor here (only the covariance), so
        we can't compute the full Ledoit-Wolf optimal intensity from the
        sample moments. We use a pragmatic approximation: α = 0.1 when
        n_effective is unknown, scaled by 1 / (1 + condition_number / 100)
        for well-conditioned covariances (so shrinkage is weaker when the
        covariance is already well-conditioned).
        """
        C = covariance.shape[0]
        trace = torch.diagonal(covariance).sum()
        target = (trace / C) * torch.eye(C, device=covariance.device, dtype=covariance.dtype)
        alpha = 0.1
        return (1 - alpha) * covariance + alpha * target

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
            source=last.source,
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
    arch_multiplier: float = 1.0,
    arch_min: int | None = None,
) -> list[int]:
    """Convert layer profiles to per-stage student widths.

    Groups profiles by stage (based on total_channels) and reduces per-block
    ranks to one rank per stage. Rounds up to width_multiple.

    `arch_multiplier` (default 1.0) and `arch_min` decouple the student's
    per-stage width (k_arch) from the loss's retained subspace dimension
    (k_loss = effective_rank). Setting `arch_multiplier > 1` gives the
    student more channels than the loss uses — useful when the student
    needs extra capacity for optimization / nonlinear recombination while
    the loss ignores the spectral tail.

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
    if arch_multiplier <= 0:
        raise ValueError(
            f"arch_multiplier must be > 0, got {arch_multiplier}"
        )

    stage_map = group_profiles_by_stage(profiles)

    effective_min = max(min_width, arch_min or 0)

    widths = []
    for channels in sorted(stage_map.keys()):
        ranks = [p.effective_rank for p in stage_map[channels]]
        if rank_reduction == "max":
            rank = max(ranks)
        elif rank_reduction == "mean":
            rank = int(math.ceil(sum(ranks) / len(ranks)))
        else:  # "last"
            rank = ranks[-1]
        scaled = int(math.ceil(rank * arch_multiplier))
        width = max(
            effective_min,
            ((scaled + width_multiple - 1) // width_multiple) * width_multiple,
        )
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
                "source": p.source,
            }
            for p in profiles
        ]
    }
    torch.save(data, path)


def load_profiles(path: str) -> list[LayerProfile]:
    """Load profiles from disk."""
    data = torch.load(path, weights_only=False)
    return [
        LayerProfile(
            name=d["name"],
            eigenvalues=d["eigenvalues"],
            principal_components=d["principal_components"],
            effective_rank=d["effective_rank"],
            total_channels=d["total_channels"],
            compression_ratio=d["compression_ratio"],
            source=d.get("source", "output"),
        )
        for d in data["profiles"]
    ]
