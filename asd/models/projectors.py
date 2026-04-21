"""Learnable 1x1 conv projectors mapping student features to teacher's SVD subspace."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class SubspaceProjectorBank(nn.Module):
    """Bank of 1x1 conv projectors — one per stage.

    Each projector maps student feature maps (B, student_width, H, W)
    to teacher's SVD subspace dimension (B, teacher_rank, H, W).
    After global average pooling, the projected features are compared
    to the teacher's SVD-projected features.
    """

    def __init__(self, student_widths: list[int], teacher_ranks: list[int]):
        super().__init__()
        assert len(student_widths) == len(teacher_ranks) == 4

        self.projectors = nn.ModuleList()
        for s_width, t_rank in zip(student_widths, teacher_ranks):
            proj = nn.Sequential(
                nn.Conv2d(s_width, t_rank, kernel_size=1, bias=False),
                nn.BatchNorm2d(t_rank),
            )
            self.projectors.append(proj)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, student_features: list[Tensor]) -> list[Tensor]:
        """Project each stage's student features to teacher subspace.

        Args:
            student_features: list of 4 tensors (B, student_width_i, H_i, W_i)

        Returns:
            list of 4 tensors (B, teacher_rank_i, H_i, W_i)
        """
        assert len(student_features) == len(self.projectors), \
            f"Expected {len(self.projectors)} feature maps, got {len(student_features)}"
        return [proj(feat) for proj, feat in zip(self.projectors, student_features)]
