"""Combined ASD Loss — task + subspace matching + sparsity pattern."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..profiling.svd_analysis import LayerProfile
from .subspace_loss import SubspaceMatchingLoss
from .sparsity_loss import SparsityPatternLoss


class ASDLoss(nn.Module):
    """Combined Activation Subspace Distillation loss.

    total = alpha * task_loss + beta * subspace_loss + gamma * sparsity_loss

    Where gamma is subject to warmup scheduling (sparsity matching is
    meaningless when student activations are random at initialization).
    """

    def __init__(
        self,
        profiles: list[LayerProfile],
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        sv_weighted: bool = True,
        num_bins: int = 64,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # Will be modulated by scheduler

        self.subspace_loss = SubspaceMatchingLoss(profiles, sv_weighted=sv_weighted)
        self.sparsity_loss = SparsityPatternLoss(profiles, num_bins=num_bins)

    def forward(
        self,
        student_logits: Tensor,
        student_projected: list[Tensor],
        student_features: list[Tensor],
        teacher_features: list[Tensor],
        labels: Tensor,
        gamma_scale: float = 1.0,
    ) -> dict[str, Tensor]:
        """Compute combined ASD loss.

        Args:
            student_logits: (B, num_classes) student predictions
            student_projected: list of 4 projected feature tensors from projector bank
            student_features: list of 4 raw student feature tensors (for sparsity)
            teacher_features: list of 4 teacher feature tensors
            labels: (B,) ground truth class labels
            gamma_scale: warmup multiplier for sparsity loss (0→1 over warmup period)

        Returns:
            dict with 'total', 'task', 'subspace', 'sparsity' loss values
        """
        loss_task = F.cross_entropy(student_logits, labels)
        loss_subspace = self.subspace_loss(student_projected, teacher_features)
        loss_sparsity = self.sparsity_loss(student_features)

        effective_gamma = self.gamma * gamma_scale

        total = (
            self.alpha * loss_task
            + self.beta * loss_subspace
            + effective_gamma * loss_sparsity
        )

        return {
            "total": total,
            "task": loss_task.detach(),
            "subspace": loss_subspace.detach(),
            "sparsity": loss_sparsity.detach(),
        }
