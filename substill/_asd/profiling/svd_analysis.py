"""SVD decomposition of activation covariance to find the principal subspace."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LayerProfile:
    """Activation profile for one layer.

    ``source`` records what the covariance was computed from:

    - ``"output"``: the block's forward output.
    - ``"delta"``: the residual update
      ``dx_l = output - shortcut(input)``.
    - ``"branch"``: the output of a residual branch sub-module (for
      example ``conv3`` in a Bottleneck, or ``attn`` / ``mlp`` in a
      transformer block).

    The loss checks this field before mixing profiles across sources.
    """

    name: str
    eigenvalues: Tensor
    principal_components: Tensor
    effective_rank: int
    total_channels: int
    compression_ratio: float
    source: str = "output"


class SVDAnalyzer:
    """Analyze activation covariance matrices via eigendecomposition.

    ``definition`` controls how effective rank is derived from the
    spectrum:

    - ``"variance"`` (default): smallest ``k`` with cumulative
      variance at least ``variance_threshold``. Classic, sensitive
      to the tail.
    - ``"stable"``: ``ceil(sum(lam) / lam_max)``. Dimensionless, no
      threshold. Uses ceiling so heavy-tailed spectra do not collapse
      to 1.
    - ``"participation"``: ``ceil((sum(lam))^2 / sum(lam^2))``.
      Measures spread of the spectrum; robust to long tails.
    - ``"entropy"``: ``ceil(exp(H(p)))`` where ``p_i = lam_i / sum(lam)``.
      Smooth, between stable and variance rank.

    ``noise_model`` controls how the noise floor is estimated before
    rank computation:

    - ``"eps"`` (default): treat eigenvalues below
      ``eps_relative * lam_max`` as zero. Defends against
      float-precision noise.
    - ``"mp"``: Gavish-Donoho / Marchenko-Pastur bulk-edge threshold.
      Requires ``n_effective``, the covariance's effective sample
      count adjusted for spatial correlation. Eigenvalues below
      ``omega(beta)^2 * sigma_med^2`` are treated as noise, where
      ``beta = C / N_eff``, ``omega(beta)`` is the Gavish-Donoho
      coefficient, and ``sigma_med`` is the median eigenvalue of the
      lower half of the spectrum.

    ``shrinkage`` applies a covariance-level regularizer before
    eigendecomposition:

    - ``"none"`` (default): no shrinkage.
    - ``"ledoit_wolf"``: linear shrinkage toward a scaled identity,
      ``cov' = (1-alpha) * cov + alpha * (tr(cov) / C) * I``.
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
        # Symmetrize defensively: accumulated covariance can drift by
        # ~1e-6 due to floating-point non-associativity.
        covariance = 0.5 * (covariance + covariance.T)
        if self.shrinkage == "ledoit_wolf":
            covariance = self._ledoit_wolf_shrink(covariance)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # A sizeable negative eigenvalue means the accumulator is
        # broken. Fail loudly rather than masking it with clamp.
        lam_max = eigenvalues.max().clamp(min=0)
        if lam_max > 0:
            most_negative = eigenvalues.min()
            if most_negative < -1e-4 * lam_max:
                raise ValueError(
                    f"{name}: covariance is not PSD (lam_min/lam_max = "
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

        For ``noise_model="eps"`` the floor is
        ``eps_relative * lam_max``. For ``"mp"`` it is the
        Gavish-Donoho bulk-edge threshold.
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

        Falls back to the eps-based threshold when ``n_effective`` is
        not set, non-positive, or smaller than C. The derivation
        assumes ``C <= N`` and ``beta in (0, 1]``. If ``beta >= 1``
        the estimator is ill-posed, so the eps fallback also applies.
        """
        C = int(eigenvalues.shape[0])
        n_eff = self.n_effective
        lam_max = eigenvalues.max()
        if not n_eff or n_eff <= 0 or n_eff < C:
            return self.eps_relative * lam_max
        beta = C / float(n_eff)
        if not 0 < beta <= 1:
            return self.eps_relative * lam_max
        omega = math.sqrt(
            2 * (beta + 1)
            + 8 * beta / ((beta + 1) + math.sqrt(beta * beta + 14 * beta + 1))
        )
        lower_half = eigenvalues[eigenvalues.shape[0] // 2 :]
        lower_half = lower_half[lower_half > 0]
        if lower_half.numel() == 0:
            return self.eps_relative * lam_max
        sigma_sq_proxy = float(lower_half.median().item())
        bulk_mid = ((1 - math.sqrt(beta)) ** 2 + (1 + math.sqrt(beta)) ** 2) / 2
        sigma_sq = sigma_sq_proxy / max(bulk_mid, 1e-12)
        threshold = (omega * omega) * sigma_sq
        return torch.tensor(threshold, dtype=eigenvalues.dtype, device=eigenvalues.device)

    @staticmethod
    def _ledoit_wolf_shrink(covariance: Tensor) -> Tensor:
        """Linear shrinkage toward a scaled identity.

        With only the covariance (not the raw samples) the full
        Ledoit-Wolf optimal intensity cannot be computed from sample
        moments. A pragmatic ``alpha = 0.1`` is used.
        """
        C = covariance.shape[0]
        trace = torch.diagonal(covariance).sum()
        target = (trace / C) * torch.eye(
            C, device=covariance.device, dtype=covariance.dtype,
        )
        alpha = 0.1
        return (1 - alpha) * covariance + alpha * target

    @staticmethod
    def _variance_rank(eigenvalues: Tensor, threshold: float) -> int:
        total = eigenvalues.sum()
        if total < 1e-10:
            return 1
        cumulative = torch.cumsum(eigenvalues, dim=0)
        mask = cumulative / total >= threshold
        if not mask.any():
            return len(eigenvalues)
        k = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
        return max(1, k)

    @staticmethod
    def _stable_rank(eigenvalues: Tensor) -> int:
        lam_max = eigenvalues.max()
        if lam_max < 1e-10:
            return 1
        # Ceiling keeps heavy-tailed spectra from collapsing to 1-2
        # even when the tail carries meaningful signal.
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

    - ``"last"``: use the last block's profile (stage output).
      Matches the original pipeline.
    - ``"max_rank"``: pick the block with the highest effective rank,
      the stage's information bottleneck.
    - ``"average"``: sum per-block covariances (reconstructed as
      ``V L V^T``) and re-eigendecompose. Produces components that
      capture variance observed anywhere in the stage. Effective
      rank is recomputed as the variance rank at threshold 0.95.

    All blocks in ``stage_profiles`` must share the same
    ``total_channels``.
    """
    if not stage_profiles:
        raise ValueError("stage_profiles is empty")
    channels = stage_profiles[0].total_channels
    for p in stage_profiles:
        if p.total_channels != channels:
            raise ValueError(
                "stage aggregation requires matching channel counts, got "
                f"{p.total_channels} vs {channels}"
            )

    if mode == "last":
        return stage_profiles[-1]

    if mode == "max_rank":
        return max(stage_profiles, key=lambda p: p.effective_rank)

    if mode == "average":
        C = channels
        cov_sum = torch.zeros(C, C, dtype=stage_profiles[0].eigenvalues.dtype)
        for p in stage_profiles:
            V = p.principal_components
            lam = p.eigenvalues[: V.shape[1]].clamp(min=0)
            cov_sum = cov_sum + (V * lam.unsqueeze(0)) @ V.T
        cov_sum = 0.5 * (cov_sum + cov_sum.T)
        eigvals, eigvecs = torch.linalg.eigh(cov_sum)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx].clamp(min=0)
        eigvecs = eigvecs[:, idx]

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
    """Group profiles by stage, identified by channel count."""
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

    Groups profiles by stage (by ``total_channels``) and reduces
    per-block ranks to one rank per stage. Rounds up to
    ``width_multiple``.

    ``arch_multiplier`` (default 1.0) and ``arch_min`` decouple the
    student's per-stage width (``k_arch``) from the loss's retained
    subspace dimension (``k_loss = effective_rank``). Setting
    ``arch_multiplier > 1`` gives the student more channels than the
    loss uses, which helps when the student needs extra capacity for
    optimization or nonlinear recombination while the loss ignores
    the spectral tail.

    ``rank_reduction``:

    - ``"max"`` (default): take the largest per-block rank in the
      stage. Conservative; never undersizes relative to any block.
    - ``"mean"``: average over blocks in the stage.
    - ``"last"``: use the last block (the stage's output). Matches
      the ``"last"`` aggregation subspace loss exactly.
    """
    if rank_reduction not in ("max", "mean", "last"):
        raise ValueError(f"Unknown rank_reduction: {rank_reduction!r}")
    if arch_multiplier <= 0:
        raise ValueError(f"arch_multiplier must be > 0, got {arch_multiplier}")

    stage_map = group_profiles_by_stage(profiles)
    effective_min = max(min_width, arch_min or 0)

    widths = []
    for channels in sorted(stage_map.keys()):
        ranks = [p.effective_rank for p in stage_map[channels]]
        if rank_reduction == "max":
            rank = max(ranks)
        elif rank_reduction == "mean":
            rank = int(math.ceil(sum(ranks) / len(ranks)))
        else:
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

    Heuristic: if effective rank plateaus after ``k`` blocks (that
    is, subsequent blocks contribute less than ``saturation_tol``
    relative increase in rank), the stage only needs ``k`` blocks in
    the student. A data-driven alternative to a fixed
    ``blocks_per_stage`` hyperparameter.

    Falls back to ``min_blocks`` when a stage has only one profiled
    block, and clamps to ``[min_blocks, max_blocks]``.
    """
    stage_map = group_profiles_by_stage(profiles)
    blocks = []
    for channels in sorted(stage_map.keys()):
        ranks = [p.effective_rank for p in stage_map[channels]]
        if len(ranks) <= 1:
            blocks.append(min_blocks)
            continue
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
    """Save profiles to ``path`` via :func:`torch.save`."""
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
    """Load profiles saved with :func:`save_profiles`."""
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
