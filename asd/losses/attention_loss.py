"""Attention Transfer (AT) — match spatial attention maps per stage.

Zagoruyko & Komodakis, "Paying More Attention to Attention" (ICLR 2017).

Aggregates channel dimension into a spatial attention map via Σ|feat|^p, then
matches L2-normalized vectors. This is orthogonal to subspace-channel matching:
subspace asks "which channels carry information"; attention asks "which spatial
regions matter".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def spatial_attention(feat: Tensor, p: float = 2.0) -> Tensor:
    """Channel-aggregated spatial attention map. feat: (B, C, H, W) → (B, H*W)."""
    # |x|^p summed over channels gives a (B, H, W) map; then flatten and L2-normalize.
    att = feat.pow(p).abs().sum(dim=1)            # (B, H, W)
    att = att.reshape(att.shape[0], -1)           # (B, H*W)
    att = F.normalize(att, p=2, dim=1, eps=1e-12)
    return att


class AttentionTransferLoss(nn.Module):
    """Sum per-stage MSE between L2-normalized spatial attention maps."""

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(self, student_features: list[Tensor], teacher_features: list[Tensor]) -> Tensor:
        assert len(student_features) == len(teacher_features), \
            f"stage count mismatch: {len(student_features)} vs {len(teacher_features)}"

        total = torch.zeros(1, device=teacher_features[0].device)
        for s, t in zip(student_features, teacher_features):
            # Student and teacher have different channel counts but same spatial dims.
            a_s = spatial_attention(s, self.p)
            a_t = spatial_attention(t, self.p)
            total = total + F.mse_loss(a_s, a_t)
        return total / len(student_features)
