"""Sparsity Pattern Loss — KL divergence between soft activation histograms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..profiling.svd_analysis import LayerProfile


class SparsityPatternLoss(nn.Module):
    """Match activation sparsity patterns between student and teacher.

    Uses differentiable soft histograms (Gaussian kernel binning) so gradients
    flow through the histogram computation. Hard histograms have zero gradient.

    Loss = KL(teacher_hist || student_soft_hist) + lambda * MSE(sparsity_ratios)
    """

    def __init__(
        self,
        profiles: list[LayerProfile],
        num_bins: int = 64,
        sigma: float = 0.1,
        sparsity_weight: float = 1.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.sparsity_weight = sparsity_weight
        self.num_stages = 4

        # Store teacher histograms and sparsity ratios as buffers
        stage_map: dict[int, list[LayerProfile]] = {}
        for p in profiles:
            stage_map.setdefault(p.total_channels, []).append(p)

        for stage_idx, channels in enumerate(sorted(stage_map.keys())):
            stage_profiles = stage_map[channels]
            profile = stage_profiles[-1]

            # Teacher histogram (already normalized)
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

    def soft_histogram(self, x: Tensor, bin_edges: Tensor) -> Tensor:
        """Compute differentiable soft histogram using Gaussian kernel binning.

        Uses adaptive sigma based on bin width to handle activation range mismatch
        between student and teacher. If student activations are far outside the
        teacher's range, the bins are extended to cover the student's range.

        Args:
            x: Flattened activation values (N,)
            bin_edges: (num_bins + 1,) bin edge values from teacher

        Returns:
            (num_bins,) normalized soft histogram (sums to ~1)
        """
        # Adapt bin range to cover student activations if they fall outside teacher range
        with torch.no_grad():
            x_min = x.min()
            x_max = x.max()
            t_min = bin_edges[0]
            t_max = bin_edges[-1]
            # Expand range if student activations exceed teacher range
            new_min = torch.minimum(x_min, t_min)
            new_max = torch.maximum(x_max, t_max)

        # Use adaptive bin edges spanning the union of teacher and student ranges
        adapted_edges = torch.linspace(
            new_min.item(), new_max.item(), len(bin_edges), device=bin_edges.device
        )

        # Bin centers
        centers = (adapted_edges[:-1] + adapted_edges[1:]) / 2  # (num_bins,)

        # Adaptive sigma: scale with bin width so Gaussians always overlap
        bin_width = (adapted_edges[-1] - adapted_edges[0]) / (len(adapted_edges) - 1)
        sigma = max(self.sigma, (bin_width * 1.5).item())

        # Gaussian kernel: for each value, compute contribution to each bin
        x_2d = x.unsqueeze(1)          # (N, 1)
        centers_2d = centers.unsqueeze(0)  # (1, num_bins)

        weights = torch.exp(-0.5 * ((x_2d - centers_2d) / sigma) ** 2)  # (N, num_bins)

        hist = weights.sum(dim=0)  # (num_bins,)
        hist = hist / hist.sum().clamp(min=1e-10)

        return hist

    def forward(self, student_features: list[Tensor]) -> Tensor:
        """Compute sparsity pattern loss across all stages.

        Args:
            student_features: list of 4 tensors (B, C, H, W) — raw student activations

        Returns:
            Scalar loss tensor
        """
        assert len(student_features) == self.num_stages, \
            f"Expected {self.num_stages} feature maps, got {len(student_features)}"

        total_loss = torch.zeros(1, device=student_features[0].device)

        for stage_idx in range(self.num_stages):
            s_feat = student_features[stage_idx]
            teacher_hist = getattr(self, f"teacher_hist_{stage_idx}")
            bin_edges = getattr(self, f"teacher_bin_edges_{stage_idx}")
            teacher_sparsity = getattr(self, f"teacher_sparsity_{stage_idx}")

            s_flat = s_feat.reshape(-1)

            # Student sparsity ratio (differentiable approximation using sigmoid)
            # sigmoid(-(|x| - eps) / tau) approximates indicator(x ≈ 0)
            tau = 0.01
            student_sparsity = torch.sigmoid(-(s_flat.abs() - 0.01) / tau).mean()

            # Sparsity ratio MSE
            sparsity_loss = F.mse_loss(student_sparsity, teacher_sparsity)

            # Soft histogram KL divergence (on non-zero values)
            # Use detached mask to select values, but keep gradient through soft_histogram
            with torch.no_grad():
                nonzero_mask = s_flat != 0
                num_nonzero = nonzero_mask.sum().item()

            if num_nonzero > 100:
                s_nonzero = s_flat[nonzero_mask]
                # Subsample for efficiency (detach selection, keep values differentiable)
                if len(s_nonzero) > 10000:
                    with torch.no_grad():
                        indices = torch.randperm(len(s_nonzero), device=s_nonzero.device)[:10000]
                    s_nonzero = s_nonzero[indices]

                s_hist = self.soft_histogram(s_nonzero, bin_edges)

                # KL(teacher || student) — teacher is the target distribution
                t_hist = teacher_hist.clamp(min=1e-10)
                s_hist = s_hist.clamp(min=1e-10)
                kl_loss = (t_hist * (t_hist.log() - s_hist.log())).sum()
            else:
                kl_loss = torch.zeros(1, device=s_feat.device)

            total_loss = total_loss + kl_loss + self.sparsity_weight * sparsity_loss

        return total_loss / self.num_stages
