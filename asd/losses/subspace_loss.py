"""Subspace Matching Loss — MSE between student projections and teacher SVD components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..profiling.svd_analysis import (
    LayerProfile,
    aggregate_stage_profile,
    group_profiles_by_stage,
)


def _sv_weights(eigenvalues: Tensor, k: int, mode: str) -> Tensor | None:
    """Compute per-component weights for the subspace loss.

    Large eigenvalues span many orders of magnitude; using them raw (mode="linear")
    causes the loss to be dominated by the top 1-2 components and the student
    effectively only matches those. "sqrt" and "uniform" dampen this so that all
    k components matter in the loss.
    """
    sv = eigenvalues[:k].clone()
    if mode == "uniform" or mode == "none":
        return None
    if mode == "linear":
        return sv / sv.mean().clamp(min=1e-10)
    if mode == "sqrt":
        sv = sv.clamp(min=0).sqrt()
        return sv / sv.mean().clamp(min=1e-10)
    raise ValueError(f"Unknown sv_weighting mode: {mode!r}")


class SubspaceMatchingLoss(nn.Module):
    """Match student's projected features to teacher's principal activation subspace.

    Modes:

    - mode="spatial" (default): project teacher per-spatial-position into its
      top-k PC subspace and match the student's (B, k, H, W) feature map with
      MSE at every spatial position. Dense supervision.

    - mode="gap": pool teacher features to (B, C), project to (B, k), pool
      student to (B, k), and MSE in k dims. Legacy behavior — loses spatial
      structure.

    - mode="cosine_spatial": like spatial, but use 1 − cos(student, teacher)
      per spatial position. Scale-invariant (bounded in [0, 2]) — immune to
      feature-magnitude mismatch between student and teacher, which is the
      dominant failure mode at high compression. SV weighting is applied as a
      per-channel weighting of the normalized-MSE equivalent (the previous
      silent-ignore was a bug).

    `stage_aggregation` controls which per-block profile becomes the stage's
    principal components:
      - "last" (default): last block in the stage (the stage output).
      - "max_rank": the block with the highest effective rank.
      - "average": re-eigendecomposition of the summed per-block covariance
        approximations (Σ V_b Λ_b V_bᵀ). Uses every profiled block rather than
        discarding all but one.
    """

    def __init__(
        self,
        profiles: list[LayerProfile],
        sv_weighted: bool = True,
        mode: str = "spatial",
        sv_weighting: str = "sqrt",
        normalize_features: bool = False,
        stage_aggregation: str = "last",
    ):
        super().__init__()
        if mode not in ("spatial", "gap", "cosine_spatial"):
            raise ValueError(f"Unknown mode: {mode!r}")
        # sv_weighted kept for backward compat — maps False to "uniform" weights.
        if not sv_weighted:
            sv_weighting = "uniform"
        self.mode = mode
        self.sv_weighting = sv_weighting
        self.normalize_features = normalize_features  # L2-normalize channel axis
        self.num_stages = 4
        self.stage_aggregation = stage_aggregation

        stage_map = group_profiles_by_stage(profiles)

        for stage_idx, channels in enumerate(sorted(stage_map.keys())):
            profile = aggregate_stage_profile(stage_map[channels], mode=stage_aggregation)
            self.register_buffer(
                f"components_{stage_idx}",
                profile.principal_components.clone(),  # (C, k)
            )
            w = _sv_weights(profile.eigenvalues, profile.effective_rank, sv_weighting)
            if w is not None:
                self.register_buffer(f"sv_weights_{stage_idx}", w)

    def _stage_loss(self, components: Tensor, sv_w: Tensor | None, s_feat: Tensor, t_feat: Tensor) -> Tensor:
        """Compute per-stage loss in either mode."""
        k = components.shape[1]
        assert s_feat.shape[1] == k, \
            f"student projected dim {s_feat.shape[1]} != subspace dim {k}"

        if self.mode == "spatial":
            # Project teacher per-spatial-position into k-dim subspace.
            t_proj = torch.einsum("bchw,ck->bkhw", t_feat, components)
            if self.normalize_features:
                # L2-normalize along channel axis. Makes MSE scale-invariant to
                # feature magnitude mismatch between student (post-BN) and
                # teacher projections — the dominant failure mode at high
                # compression when ranks differ widely per stage.
                s_feat = F.normalize(s_feat, p=2, dim=1, eps=1e-6)
                t_proj = F.normalize(t_proj, p=2, dim=1, eps=1e-6)
            diff_sq = (s_feat - t_proj) ** 2  # (B, k, H, W)
            if sv_w is None:
                return diff_sq.mean()
            return (diff_sq * sv_w.view(1, -1, 1, 1)).mean()

        if self.mode == "cosine_spatial":
            # Scale-invariant per-position matching. With sv_w given we use the
            # per-channel normalized-MSE form (2·(1 − weighted_cos) up to a
            # constant), which honors the eigenvalue weights; without, we fall
            # back to plain 1 − cos. Previously sv_w was silently ignored.
            t_proj = torch.einsum("bchw,ck->bkhw", t_feat, components)  # (B, k, H, W)
            s_norm = F.normalize(s_feat, p=2, dim=1, eps=1e-6)
            t_norm = F.normalize(t_proj, p=2, dim=1, eps=1e-6)
            if sv_w is None:
                cos = (s_norm * t_norm).sum(dim=1)  # (B, H, W)
                return (1.0 - cos).mean()
            # Weighted squared error between unit-norm vectors =
            # Σ w_i (s_i − t_i)² averaged over (B, H, W, channel). Equivalent to
            # 2·(1 − Σ w_i s_i t_i) when Σ w_i = 1; we keep the raw form for
            # numerical stability with non-normalized weights.
            diff_sq = (s_norm - t_norm) ** 2  # (B, k, H, W)
            return (diff_sq * sv_w.view(1, -1, 1, 1)).mean()

        # "gap" mode
        t_pooled = t_feat.mean(dim=(2, 3))  # (B, C)
        t_subspace = t_pooled @ components  # (B, k)
        s_pooled = s_feat.mean(dim=(2, 3))  # (B, k)
        diff_sq = (s_pooled - t_subspace) ** 2
        if sv_w is None:
            return diff_sq.mean()
        return (diff_sq * sv_w.unsqueeze(0)).mean()

    def forward(
        self,
        student_projected: list[Tensor],
        teacher_features: list[Tensor],
    ) -> Tensor:
        assert len(student_projected) == self.num_stages, \
            f"Expected {self.num_stages} projected features, got {len(student_projected)}"
        assert len(teacher_features) == self.num_stages, \
            f"Expected {self.num_stages} teacher features, got {len(teacher_features)}"

        total_loss = torch.zeros((), device=teacher_features[0].device)
        for i in range(self.num_stages):
            components = getattr(self, f"components_{i}")
            sv_w = getattr(self, f"sv_weights_{i}", None)
            total_loss = total_loss + self._stage_loss(
                components, sv_w, student_projected[i], teacher_features[i],
            )
        return total_loss / self.num_stages
