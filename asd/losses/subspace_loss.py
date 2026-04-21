"""Subspace Matching Loss — MSE between student projections and teacher SVD components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..profiling.svd_analysis import LayerProfile


class SubspaceMatchingLoss(nn.Module):
    """Match student's projected features to teacher's principal activation subspace.

    For each stage:
    1. Global average pool teacher features → (B, C_teacher)
    2. Project onto top-k SVD components → (B, k)
    3. Global average pool student projected features → (B, k)
    4. MSE between them, optionally weighted by singular value magnitude
    """

    def __init__(self, profiles: list[LayerProfile], sv_weighted: bool = True):
        super().__init__()
        self.sv_weighted = sv_weighted
        self.num_stages = 4

        # Group profiles by stage and store components + SVs
        stage_map: dict[int, list[LayerProfile]] = {}
        for p in profiles:
            stage_map.setdefault(p.total_channels, []).append(p)

        for stage_idx, channels in enumerate(sorted(stage_map.keys())):
            stage_profiles = stage_map[channels]
            # Use last block as representative
            profile = stage_profiles[-1]
            self.register_buffer(
                f"components_{stage_idx}",
                profile.principal_components.clone(),  # (C, k)
            )
            if sv_weighted:
                sv = profile.eigenvalues[:profile.effective_rank].clone()
                # Normalize so weights sum to k (keeping scale manageable)
                sv = sv / sv.mean()
                self.register_buffer(f"sv_weights_{stage_idx}", sv)

    def forward(
        self,
        student_projected: list[Tensor],
        teacher_features: list[Tensor],
    ) -> Tensor:
        """Compute subspace matching loss across all stages.

        Args:
            student_projected: list of 4 tensors (B, k_i, H_i, W_i) from projector bank
            teacher_features: list of 4 tensors (B, C_i, H_i, W_i) from teacher

        Returns:
            Scalar loss tensor
        """
        assert len(student_projected) == self.num_stages, \
            f"Expected {self.num_stages} projected features, got {len(student_projected)}"
        assert len(teacher_features) == self.num_stages, \
            f"Expected {self.num_stages} teacher features, got {len(teacher_features)}"

        total_loss = torch.zeros(1, device=teacher_features[0].device)

        for stage_idx in range(self.num_stages):
            components = getattr(self, f"components_{stage_idx}")  # (C, k)
            t_feat = teacher_features[stage_idx]  # (B, C, H, W)
            s_feat = student_projected[stage_idx]  # (B, k, H, W)

            # GAP teacher → (B, C) → project to subspace → (B, k)
            t_pooled = t_feat.mean(dim=(2, 3))  # (B, C)
            t_subspace = t_pooled @ components  # (B, k)

            # GAP student projected → (B, k)
            s_pooled = s_feat.mean(dim=(2, 3))  # (B, k)

            k = components.shape[1]
            assert s_pooled.shape[1] == k, \
                f"Stage {stage_idx}: student projected dim {s_pooled.shape[1]} != subspace dim {k}"

            if self.sv_weighted:
                sv_w = getattr(self, f"sv_weights_{stage_idx}")  # (k,)
                assert sv_w.shape[0] == k, \
                    f"Stage {stage_idx}: SV weights dim {sv_w.shape[0]} != subspace dim {k}"
                # Weighted MSE: weight each component by its singular value
                diff_sq = (s_pooled - t_subspace) ** 2  # (B, k)
                stage_loss = (diff_sq * sv_w.unsqueeze(0)).mean()
            else:
                stage_loss = F.mse_loss(s_pooled, t_subspace)

            total_loss = total_loss + stage_loss

        return total_loss / self.num_stages
