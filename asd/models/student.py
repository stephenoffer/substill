"""SlimNet — student architecture with per-stage widths derived from activation rank analysis."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (two 3x3 convs + skip connection)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
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
        out = self.relu(out)
        return out


class SlimNet(nn.Module):
    """Compact student network with per-stage channel widths from activation rank analysis.

    Architecture mirrors ResNet's 4-stage structure but uses:
    - BasicBlocks (not Bottleneck) for simplicity
    - Per-stage widths set to teacher's effective activation rank
    - CIFAR-10 stem (3x3 stride 1, no maxpool)
    """

    def __init__(
        self,
        stage_widths: list[int],
        blocks_per_stage: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()
        assert len(stage_widths) == 4, "Need exactly 4 stage widths"

        self.stage_widths = stage_widths

        # CIFAR-10 stem: 3x3 conv stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(3, stage_widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stage_widths[0]),
            nn.ReLU(inplace=True),
        )

        # Build 4 stages
        self.stage1 = self._make_stage(stage_widths[0], stage_widths[0], blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(stage_widths[0], stage_widths[1], blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(stage_widths[1], stage_widths[2], blocks_per_stage, stride=2)
        self.stage4 = self._make_stage(stage_widths[2], stage_widths[3], blocks_per_stage, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_widths[3], num_classes)

        self._init_weights()

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int,
    ) -> nn.Sequential:
        blocks = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass returning logits and per-stage feature maps.

        Returns:
            logits: (B, num_classes)
            features: list of 4 tensors, shapes:
                [(B, w1, 32, 32), (B, w2, 16, 16), (B, w3, 8, 8), (B, w4, 4, 4)]
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
        logits = self.fc(x)

        return logits, features

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"SlimNet(stage_widths={self.stage_widths}, "
            f"params={params:,} ({params / 1e6:.2f}M))"
        )
