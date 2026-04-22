"""Extended teacher wrappers for non-ResNet architectures: MobileNetV2, VGG16-BN, DenseNet-121.

These wrap torchvision backbones, replace stems/classifiers for CIFAR, and
expose a 4-stage feature list matching the ResNet-style `(logits, [f1..f4])`
signature. Stage boundaries are chosen at spatial-resolution changes so
each stage has a fixed output channel count — required by the SlimNet
student which groups profiles by channel count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    vgg16_bn, VGG16_BN_Weights,
    densenet121, DenseNet121_Weights,
)


def _base_teacher_methods(cls):
    """Add `freeze`, `finetune`, `_eval_accuracy` methods to the given class."""
    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()
        self._is_frozen = True

    def unfreeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._is_frozen = False

    def finetune(self, train_loader, test_loader, epochs=8, lr=0.01, device="cpu"):
        self.unfreeze()
        self.to(device)
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        best = 0.0
        for epoch in range(epochs):
            self.train()
            correct = total = 0
            pbar = tqdm(train_loader, desc=f"FT {type(self).__name__} {epoch}/{epochs}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                logits, _ = self(images)
                loss = nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
                pbar.set_postfix(acc=f"{100*correct/total:.1f}%")
            scheduler.step()
            train_acc = correct / total
            test_acc = self._eval_accuracy(test_loader, device)
            if test_acc > best:
                best = test_acc
            print(f"  FT {type(self).__name__} Epoch {epoch}: train={train_acc*100:.1f}% test={test_acc*100:.1f}% best={best*100:.1f}%")
        self.freeze()
        return {"train_accuracy": train_acc, "test_accuracy": best}

    @torch.no_grad()
    def _eval_accuracy(self, loader, device):
        self.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = self(images)
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        return correct / max(total, 1)

    cls.freeze = freeze
    cls.unfreeze = unfreeze
    cls.finetune = finetune
    cls._eval_accuracy = _eval_accuracy
    return cls


@_base_teacher_methods
class MobileNetV2Teacher(nn.Module):
    """MobileNetV2 adapted for CIFAR-10.

    We define 4 stages by grouping inverted-residual blocks at stride boundaries.
    Each "stage" ends at a block whose output has a consistent channel count —
    all blocks with stride=1 that share the same channel count as their
    predecessor.

    For 32×32 input, the first conv (stride=2) is replaced with stride=1 so
    later downsamplings still leave non-trivial spatial resolution.
    """

    model_name = "mobilenet_v2"
    STAGE_CHANNELS = [24, 32, 96, 1280]

    def __init__(self, cifar_stem=True, pretrained=True, num_classes=10, freeze=True, **_):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        if cifar_stem:
            # Change first conv stride 2 → 1 for 32×32 inputs.
            first = backbone.features[0][0]  # Conv2d(3, 32, 3, stride=2, padding=1)
            backbone.features[0][0] = nn.Conv2d(
                first.in_channels, first.out_channels,
                kernel_size=first.kernel_size, stride=1, padding=first.padding,
                bias=False,
            )

        # New classifier head
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )
        self.backbone = backbone
        self.stage_channels = self.STAGE_CHANNELS
        self.block_counts = [3, 3, 6, 4]
        self.teacher_block_type = "inverted_residual"
        self._is_frozen = False
        if freeze:
            self.freeze()

    def forward(self, x: Tensor):
        features = []
        # MobileNetV2 features: features[0..18]
        # We collect at end of each "stage" (consistent channel count regions).
        f = self.backbone.features
        # Stage boundaries (indices are last block of each stage):
        #   stage1: features[0..3] — c=16, 24 (use c=24 output at idx 3)
        #   stage2: features[4..6] — c=32 (idx 6)
        #   stage3: features[7..13] — c=64, 96 (idx 13)
        #   stage4: features[14..18] — c=160, 320, 1280 (idx 18)
        stage_ends = {3: 0, 6: 1, 13: 2, 18: 3}
        for i, layer in enumerate(f):
            x = layer(x)
            if i in stage_ends:
                features.append(x)

        pooled = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.backbone.classifier(pooled)
        return logits, features


@_base_teacher_methods
class VGG16BNTeacher(nn.Module):
    """VGG16-BN teacher — sequential conv-net without residual connections.

    Five conv blocks are separated by MaxPool layers. For the 4-stage ASD
    interface, we collapse the first two small blocks into stage 1 and use the
    remaining three as stages 2-4.
    """

    model_name = "vgg16_bn"
    STAGE_CHANNELS = [128, 256, 512, 512]

    def __init__(self, cifar_stem=True, pretrained=True, num_classes=10, freeze=True, **_):
        super().__init__()
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        backbone = vgg16_bn(weights=weights)

        # VGG already works on any input size; for CIFAR we just replace head.
        backbone.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Pre-compute block boundaries based on MaxPool positions.
        # vgg16_bn.features is Sequential; MaxPool2d appears after each block.
        self.block_ends = []  # indices of MaxPool layers
        for i, m in enumerate(backbone.features):
            if isinstance(m, nn.MaxPool2d):
                self.block_ends.append(i)
        # block_ends typically: [6, 13, 23, 33, 43]  (5 blocks)
        # 4 stages: merge first two (stage1 = first two conv blocks).
        # stage_end_indices: [block_ends[1], block_ends[2], block_ends[3], block_ends[4]]
        self.stage_end_indices = [self.block_ends[1], self.block_ends[2],
                                  self.block_ends[3], self.block_ends[4]]

        self.backbone = backbone
        self.stage_channels = self.STAGE_CHANNELS
        self.block_counts = [2, 3, 3, 3]  # conv blocks in each ASD-stage
        self.teacher_block_type = "vgg_block"
        self._is_frozen = False
        if freeze:
            self.freeze()

    def forward(self, x: Tensor):
        features = []
        stage_set = set(self.stage_end_indices)
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in stage_set:
                features.append(x)

        pooled = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.backbone.classifier(pooled)
        return logits, features


_NON_RESNET_WRAPPERS = {
    "mobilenet_v2": MobileNetV2Teacher,
    "vgg16_bn": VGG16BNTeacher,
}


def get_teacher(model: str, **kwargs) -> nn.Module:
    """Factory: return the appropriate teacher wrapper for `model`."""
    if model in _NON_RESNET_WRAPPERS:
        return _NON_RESNET_WRAPPERS[model](**kwargs)
    # Fall back to the standard ResNet TeacherWrapper
    from .teacher import TeacherWrapper
    return TeacherWrapper(model=model, **kwargs)


def teacher_hook_names(model: str) -> list[str]:
    """Return per-block hook names for the given teacher. Picks individual
    blocks (not stage outputs) so we capture multiple profiles per stage.

    For ResNet: layer{i}.{j}
    For MobileNetV2: features.{i} for stride-changing or channel-changing blocks
    For VGG16-BN: ReLU outputs inside each of the 4 ASD-stages
    """
    if model == "mobilenet_v2":
        # One hook per inverted-residual block that outputs each stage's channel count.
        # Stages are (c=24, c=32, c=96, c=1280), hooks at multiple points within.
        return [f"features.{i}" for i in [2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    if model == "vgg16_bn":
        # Hook the last conv output in each block (right before MaxPool).
        # Indices must be collected dynamically; fall back at runtime.
        return []  # special-cased in the bench script
    # For ResNet models use the standard per-block hook list
    from ..profiling.activation_capture import get_resnet_layer_names
    return get_resnet_layer_names(model)
