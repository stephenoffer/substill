"""Sparsity Pattern Loss — KL divergence between soft activation histograms."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..profiling.svd_analysis import LayerProfile, group_profiles_by_stage


class SparsityPatternLoss(nn.Module):
    """Match activation sparsity patterns between student and teacher.

    Uses differentiable soft histograms (Gaussian kernel binning) so gradients
    flow through the histogram computation. Hard histograms have zero gradient.

    Loss = (1/log N) · KL(teacher_hist ‖ student_soft_hist) + ratio_term

    The KL is divided by log(num_bins) so both terms are ~O(1) regardless of
    the bin count; otherwise a 64-bin KL could be 20× larger than a 4-bin KL
    at nothing-matches and silently dominate the sparsity-ratio term.

    ratio_term controls how the sparsity *ratio* (fraction of near-zero
    activations) is matched:
      - "bce" (default): binary cross-entropy — the natural distance between
        two Bernoulli-like probabilities in [0, 1].
      - "mse": legacy L2 distance between ratios.
    """

    def __init__(
        self,
        profiles: list[LayerProfile],
        num_bins: int = 64,
        sigma: float | None = None,
        sparsity_weight: float = 1.0,
        ratio_loss: str = "bce",
        adaptive_tau: bool = True,
        kernel_bw_factor: float = 1.5,
    ):
        super().__init__()
        if ratio_loss not in ("bce", "mse"):
            raise ValueError(f"ratio_loss must be 'bce' or 'mse', got {ratio_loss!r}")
        self.num_bins = num_bins
        # `sigma` is kept as an optional *absolute* override. When None (default)
        # we derive a per-stage Gaussian-kernel bandwidth from that stage's own
        # bin width — the previous constant-0.1 default silently over-smoothed
        # late-stage (small-scale) activations into uniform histograms.
        self.sigma_override = sigma
        self.kernel_bw_factor = kernel_bw_factor
        self.sparsity_weight = sparsity_weight
        self.ratio_loss = ratio_loss
        self.adaptive_tau = adaptive_tau
        self.num_stages = 4
        # log(N) normalizer for KL so it is comparable to the Bernoulli-BCE term.
        self._log_num_bins = math.log(max(num_bins, 2))

        stage_map = group_profiles_by_stage(profiles)

        for stage_idx, channels in enumerate(sorted(stage_map.keys())):
            stage_profiles = stage_map[channels]
            profile = stage_profiles[-1]

            self.register_buffer(
                f"teacher_hist_{stage_idx}",
                profile.sparsity_stats.activation_histogram.clone(),
            )
            self.register_buffer(
                f"teacher_bin_edges_{stage_idx}",
                profile.sparsity_stats.bin_edges.clone(),
            )
            self.register_buffer(
                f"teacher_sparsity_{stage_idx}",
                torch.tensor(profile.sparsity_stats.sparsity_ratio),
            )

    def _stage_sigma(self, bin_edges: Tensor) -> float:
        """Per-stage kernel bandwidth tied to that stage's bin width.

        A bandwidth that's too large (e.g., the previous fixed 0.1 applied to a
        late-stage activation range of [0, 0.3]) smooths the histogram into a
        uniform distribution; too small, gradients stop flowing between bins.
        Tying to `kernel_bw_factor · bin_width` gives a kernel that just bridges
        the gap between adjacent bins — enough for gradient flow without
        washing out the histogram shape.
        """
        bin_width = (bin_edges[-1] - bin_edges[0]) / (len(bin_edges) - 1)
        derived = float((bin_width * self.kernel_bw_factor).item())
        if self.sigma_override is not None:
            return max(self.sigma_override, derived)
        return max(derived, 1e-6)

    def soft_histogram(self, x: Tensor, bin_edges: Tensor) -> Tensor:
        """Differentiable soft histogram on the teacher's bin grid.

        Student values are clamped into the teacher range (keeping gradients).
        """
        x = torch.clamp(x, min=bin_edges[0], max=bin_edges[-1])

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (num_bins,)
        sigma = self._stage_sigma(bin_edges)

        x_2d = x.unsqueeze(1)             # (N, 1)
        centers_2d = centers.unsqueeze(0)  # (1, num_bins)

        weights = torch.exp(-0.5 * ((x_2d - centers_2d) / sigma) ** 2)
        hist = weights.sum(dim=0)
        hist = hist / hist.sum().clamp(min=1e-10)
        return hist

    def _student_sparsity(self, x: Tensor) -> Tensor:
        """Differentiable approximation to the fraction of near-zero entries.

        With adaptive_tau=True the sigmoid bandwidth scales with the activation
        std so the indicator has meaningful gradient support regardless of the
        layer's activation magnitude.
        """
        if self.adaptive_tau:
            with torch.no_grad():
                scale = x.detach().abs().std().clamp(min=1e-3).item()
            tau = max(scale * 0.1, 1e-3)
            eps = max(scale * 0.1, 1e-3)
        else:
            tau = 0.01
            eps = 0.01
        return torch.sigmoid(-(x.abs() - eps) / tau).mean()

    def forward(self, student_features: list[Tensor]) -> Tensor:
        assert len(student_features) == self.num_stages, \
            f"Expected {self.num_stages} feature maps, got {len(student_features)}"

        total_loss = torch.zeros((), device=student_features[0].device)

        for stage_idx in range(self.num_stages):
            s_feat = student_features[stage_idx]
            teacher_hist = getattr(self, f"teacher_hist_{stage_idx}")
            bin_edges = getattr(self, f"teacher_bin_edges_{stage_idx}")
            teacher_sparsity = getattr(self, f"teacher_sparsity_{stage_idx}")

            s_flat = s_feat.reshape(-1)

            # Sparsity-ratio matching term
            student_sparsity = self._student_sparsity(s_flat)

            if self.ratio_loss == "bce":
                # BCE(teacher || student) — natural Bernoulli distance
                p_t = teacher_sparsity.clamp(min=1e-6, max=1 - 1e-6)
                p_s = student_sparsity.clamp(min=1e-6, max=1 - 1e-6)
                ratio_term = -(p_t * torch.log(p_s) + (1 - p_t) * torch.log(1 - p_s))
            else:
                ratio_term = F.mse_loss(student_sparsity, teacher_sparsity)

            # Soft histogram KL on non-zero values
            with torch.no_grad():
                nonzero_mask = s_flat != 0
                num_nonzero = int(nonzero_mask.sum().item())

            if num_nonzero > 100:
                s_nonzero = s_flat[nonzero_mask]
                if len(s_nonzero) > 10000:
                    with torch.no_grad():
                        indices = torch.randperm(len(s_nonzero), device=s_nonzero.device)[:10000]
                    s_nonzero = s_nonzero[indices]

                s_hist = self.soft_histogram(s_nonzero, bin_edges)
                t_hist = teacher_hist.clamp(min=1e-10)
                s_hist = s_hist.clamp(min=1e-10)
                # Normalize KL by log(num_bins) so it sits in roughly [0, 1] and
                # can't dominate the Bernoulli ratio term just because we chose
                # a high bin count.
                kl_loss = (t_hist * (t_hist.log() - s_hist.log())).sum() / self._log_num_bins
            else:
                kl_loss = torch.zeros((), device=s_feat.device)

            total_loss = total_loss + kl_loss + self.sparsity_weight * ratio_term

        return total_loss / self.num_stages
