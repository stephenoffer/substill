"""Relational Distillation — match inter-sample relationships.

Park et al. "Relational Knowledge Distillation" (CVPR 2019). Instead of matching
features absolutely, match the structure of pairwise relationships in a batch.
Two components:

  - Distance term: match pairwise Euclidean distances between batch samples,
    normalized by the mean batch distance (scale-invariant).
  - Angle term: match triplet angles (cos of the angle formed by three samples).

Matching teacher's relational structure is complementary to feature MSE: it
captures instance-discrimination and works even when the student's feature
space has different dimensionality than the teacher's.

We apply this on the last-stage features (most semantic) after global average
pool, as is standard in the RKD paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _pairwise_distances(x: Tensor) -> Tensor:
    """Pairwise Euclidean distance matrix. x: (B, D). Returns (B, B)."""
    # Use the numerically stable form: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq = (x * x).sum(dim=1, keepdim=True)            # (B, 1)
    sq_sum = sq + sq.T                                # (B, B)
    inner = x @ x.T                                   # (B, B)
    d2 = (sq_sum - 2 * inner).clamp(min=1e-12)
    d = d2.sqrt()
    # Zero out diagonal (self-distance) before computing mean; scale by mean of
    # off-diagonal distances so that the result is invariant to global scale.
    mask = 1 - torch.eye(d.shape[0], device=d.device)
    mean = (d * mask).sum() / mask.sum().clamp(min=1)
    return d / mean.clamp(min=1e-12)


class RelationalLoss(nn.Module):
    """Distance + angle relational distillation on a feature vector per sample.

    Input features: (B, D) — typically last-stage student features (already
    projected, after GAP) and teacher features (after GAP, and optionally
    projected).

    Weights are smooth-L1 penalties between student and teacher structural
    matrices (standard in RKD).
    """

    def __init__(self, distance_weight: float = 1.0, angle_weight: float = 2.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    def distance_loss(self, s: Tensor, t: Tensor) -> Tensor:
        ds = _pairwise_distances(s)
        dt = _pairwise_distances(t)
        return F.smooth_l1_loss(ds, dt)

    def angle_loss(self, s: Tensor, t: Tensor, chunk_size: int = 64) -> Tensor:
        """Triplet-angle loss with bounded intermediate memory.

        For each ordered triple (i, j, k), the angle at i is
        cos(x_j − x_i, x_k − x_i). The dense (B, B, D) intermediate blows up
        quickly for high D — we chunk along the anchor index i so peak memory
        is O(chunk_size · B · D) rather than O(B² · D).
        """
        B = s.shape[0]

        def _angle_block(x: Tensor, anchors: Tensor) -> Tensor:
            # anchors: (M, D); x: (B, D). Returns (M, B, B) — for each anchor a,
            # the (j, k) matrix of cos between (x_j - a) and (x_k - a).
            diff = x.unsqueeze(0) - anchors.unsqueeze(1)        # (M, B, D)
            norm = F.normalize(diff, p=2, dim=-1, eps=1e-12)    # (M, B, D)
            return torch.einsum("mjd,mkd->mjk", norm, norm)

        loss = torch.zeros((), device=s.device, dtype=s.dtype)
        total = 0
        for i in range(0, B, chunk_size):
            j = min(i + chunk_size, B)
            with torch.no_grad():
                t_chunk = _angle_block(t, t[i:j])
            s_chunk = _angle_block(s, s[i:j])
            # smooth_l1 with 'sum' reduction + manual normalization keeps the
            # chunked loss identical to the un-chunked version.
            loss = loss + F.smooth_l1_loss(s_chunk, t_chunk, reduction="sum")
            total += s_chunk.numel()
        return loss / max(total, 1)

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        """Both arguments are (B, D)."""
        loss = 0.0
        if self.distance_weight > 0:
            loss = loss + self.distance_weight * self.distance_loss(student, teacher)
        if self.angle_weight > 0:
            loss = loss + self.angle_weight * self.angle_loss(student, teacher)
        return loss
