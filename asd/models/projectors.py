"""Learnable 1x1 conv projectors mapping student features to teacher's SVD subspace."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SubspaceProjectorBank(nn.Module):
    """Bank of 1x1 conv projectors — one per stage.

    Each projector maps student feature maps (B, student_width, H, W)
    to teacher's SVD subspace dimension (B, teacher_rank, H, W).

    init_mode controls how the 1x1 conv is initialized:
      - "orthogonal" (default): torch.nn.init.orthogonal_ — preserves activation
        variance better than Kaiming for linear (no ReLU) layers and produces
        well-conditioned mixing of student channels into the subspace.
      - "linear_kaiming": nn.init.kaiming_normal_(nonlinearity="linear") —
        previous default.

    `use_bn` (default False): whether to append a BatchNorm after the 1x1 conv.
    BN post-projection rescales features independently of the teacher's
    subspace magnitudes, which made subspace-MSE fight with the BN γ/β
    parameters and motivated the `subspace_normalize_features` patch on the
    loss side. We now default to no BN; if a caller wants BN back (to stabilize
    early training), set `use_bn=True`.

    `freeze` (bool): if True, parameters are frozen from the start — useful for
    ablating the "projector can cheat" hypothesis. A passing result with frozen
    projectors means the student's features themselves (not the projector)
    carry the transferred knowledge.
    """

    def __init__(
        self,
        student_widths: list[int],
        teacher_ranks: list[int],
        init_mode: str = "orthogonal",
        freeze: bool = False,
        use_bn: bool = False,
    ):
        super().__init__()
        assert len(student_widths) == len(teacher_ranks) == 4
        if init_mode not in ("orthogonal", "linear_kaiming"):
            raise ValueError(f"Unknown init_mode: {init_mode!r}")
        self.init_mode = init_mode
        self.use_bn = use_bn

        self.projectors = nn.ModuleList()
        for s_width, t_rank in zip(student_widths, teacher_ranks):
            layers: list[nn.Module] = [
                nn.Conv2d(s_width, t_rank, kernel_size=1, bias=not use_bn),
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(t_rank))
            self.projectors.append(nn.Sequential(*layers))

        self._init_weights()
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_mode == "orthogonal":
                    # Orthogonal init on the (out, in) weight matrix. For 1x1
                    # convs, this gives a random orthogonal projection — the
                    # best unbiased default when the projector output is linear.
                    w = m.weight  # (out, in, 1, 1)
                    nn.init.orthogonal_(w.view(w.shape[0], -1))
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self) -> int:
        """Projector params — callers should report this alongside student params.

        Non-trivial at low student width / high teacher rank: a 48→320 1x1 conv
        is 15,360 params and the full 4-stage bank with BN adds up to ~100k.
        Hiding it in the "student" count understates the effective compression.
        """
        return sum(p.numel() for p in self.parameters())

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
