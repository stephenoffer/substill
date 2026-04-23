"""SlimNet: student with per-stage widths derived from activation rank analysis."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: two 3x3 convs plus a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet Bottleneck block: 1x1, 3x3, 1x1, plus a skip connection.

    Internal hidden width is ``out_channels // expansion``, which
    makes the block roughly ``expansion^2`` cheaper in parameters
    than a :class:`BasicBlock` at the same output width. This is what
    lets per-stage width equal the teacher's effective rank translate
    into fewer parameters than the teacher.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        min_hidden: int = 8,
    ):
        super().__init__()
        hidden = max(min_hidden, out_channels // expansion)

        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


_BLOCK_TYPES = {"basic": BasicBlock, "bottleneck": Bottleneck}
_STEM_TYPES = ("cifar", "imagenet")


class SlimNet(nn.Module):
    """Compact student with per-stage channel widths set from a profile.

    Four-stage ResNet structure:

    - Bottleneck blocks by default (matches ResNet-50).
    - Per-stage widths set to the teacher's effective activation rank.
    - CIFAR stem by default (3x3 stride-1, no maxpool); ImageNet stem
      available.
    """

    def __init__(
        self,
        stage_widths: list[int],
        blocks_per_stage: int = 2,
        num_classes: int = 10,
        block_type: str = "bottleneck",
        stem_type: str = "cifar",
    ):
        super().__init__()
        if len(stage_widths) != 4:
            raise ValueError(
                f"stage_widths must have exactly 4 entries, got {len(stage_widths)}"
            )
        if block_type not in _BLOCK_TYPES:
            raise ValueError(
                f"Unknown block_type {block_type!r}; "
                f"expected one of {list(_BLOCK_TYPES)}"
            )
        if stem_type not in _STEM_TYPES:
            raise ValueError(
                f"stem_type must be one of {_STEM_TYPES}, got {stem_type!r}"
            )

        self.stage_widths = stage_widths
        self.blocks_per_stage = blocks_per_stage
        self.block_type = block_type
        self.stem_type = stem_type
        self._block_cls = _BLOCK_TYPES[block_type]

        if stem_type == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, stage_widths[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stage_widths[0]),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, stage_widths[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stage_widths[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.stage1 = self._make_stage(stage_widths[0], stage_widths[0], blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(stage_widths[0], stage_widths[1], blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(stage_widths[1], stage_widths[2], blocks_per_stage, stride=2)
        self.stage4 = self._make_stage(stage_widths[2], stage_widths[3], blocks_per_stage, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_widths[3], num_classes)

        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [self._block_cls(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(self._block_cls(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Return logits and per-stage feature maps.

        Returns:
            logits: ``(B, num_classes)``.
            features: list of four tensors, one per stage, with
                shapes ``[(B, w1, 32, 32), (B, w2, 16, 16),
                (B, w3, 8, 8), (B, w4, 4, 4)]`` for a CIFAR stem.
        """
        features = []
        x = self.stem(x)
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x), features

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"SlimNet(stage_widths={self.stage_widths}, "
            f"block_type={self.block_type}, "
            f"params={params:,} ({params / 1e6:.2f}M))"
        )
