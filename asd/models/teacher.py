"""Teacher model wrapper — ResNet50 adapted for CIFAR-10 with SVD buffers.

IMPORTANT: When using pretrained=True with cifar_stem=True, the new conv1 and fc
layers are randomly initialized. The teacher MUST be fine-tuned on CIFAR-10 before
profiling — otherwise activation statistics are meaningless.
Use scripts/00_finetune_teacher.py or call TeacherWrapper.finetune().
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

from ..profiling.svd_analysis import LayerProfile


_BACKBONES = {
    "resnet50": {
        "ctor": resnet50,
        "weights": ResNet50_Weights.DEFAULT,
        "stage_channels": [256, 512, 1024, 2048],
        "block_counts": [3, 4, 6, 3],
        "block_type": "bottleneck",
        "fc_in": 2048,
    },
    "resnet18": {
        "ctor": resnet18,
        "weights": ResNet18_Weights.DEFAULT,
        "stage_channels": [64, 128, 256, 512],
        "block_counts": [2, 2, 2, 2],
        "block_type": "basic",
        "fc_in": 512,
    },
    "resnet34": {
        "ctor": resnet34,
        "weights": ResNet34_Weights.DEFAULT,
        "stage_channels": [64, 128, 256, 512],
        "block_counts": [3, 4, 6, 3],
        "block_type": "basic",
        "fc_in": 512,
    },
    "resnet101": {
        "ctor": resnet101,
        "weights": ResNet101_Weights.DEFAULT,
        "stage_channels": [256, 512, 1024, 2048],
        "block_counts": [3, 4, 23, 3],
        "block_type": "bottleneck",
        "fc_in": 2048,
    },
}


class TeacherWrapper(nn.Module):
    """ResNet teacher (resnet50 / resnet18) exposing per-stage features + SVD projection."""

    # Kept as a class-level default so legacy callers that reference
    # TeacherWrapper.STAGE_CHANNELS still work when the teacher is ResNet50.
    STAGE_CHANNELS = _BACKBONES["resnet50"]["stage_channels"]

    def __init__(
        self,
        profiles: list[LayerProfile] | None = None,
        cifar_stem: bool = True,
        pretrained: bool = True,
        num_classes: int = 10,
        freeze: bool = True,
        model: str = "resnet50",
    ):
        super().__init__()
        if model not in _BACKBONES:
            raise ValueError(f"Unknown backbone: {model!r}. Available: {list(_BACKBONES)}")

        spec = _BACKBONES[model]
        self.model_name = model
        self.stage_channels = list(spec["stage_channels"])
        self.block_counts = list(spec["block_counts"])
        self.teacher_block_type = spec["block_type"]

        weights = spec["weights"] if pretrained else None
        backbone = spec["ctor"](weights=weights)

        if cifar_stem:
            # Replace 7x7 stride-2 conv + maxpool with 3x3 stride-1 conv for 32x32 inputs.
            # NOTE: This layer is randomly initialized and requires fine-tuning.
            first_out = spec["stage_channels"][0] if model == "resnet18" else 64
            backbone.conv1 = nn.Conv2d(3, first_out, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()

        # Replace classifier head only when the number of classes differs from
        # the pretrained model's output — otherwise (e.g., ImageNet with
        # num_classes=1000) we preserve the trained fc so the teacher is
        # usable out of the box without fine-tuning.
        if num_classes != 1000:
            backbone.fc = nn.Linear(spec["fc_in"], num_classes)

        self.backbone = backbone
        self._is_frozen = False

        if freeze:
            self.freeze()

        # Store SVD components as non-trainable buffers
        self._stage_profiles: list[LayerProfile | None] = [None] * 4
        if profiles is not None:
            self._load_profiles(profiles)

    def freeze(self) -> None:
        """Freeze all parameters (call after fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._is_frozen = True
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._is_frozen = False

    def _load_profiles(self, profiles: list[LayerProfile]) -> None:
        """Group profiles by stage and store principal components as buffers."""
        stage_map: dict[int, list[LayerProfile]] = {}
        for p in profiles:
            stage_map.setdefault(p.total_channels, []).append(p)

        for stage_idx, channels in enumerate(sorted(stage_map.keys())):
            stage_profiles = stage_map[channels]
            profile = stage_profiles[-1]
            self._stage_profiles[stage_idx] = profile

            self.register_buffer(
                f"pca_components_{stage_idx}",
                profile.principal_components.clone(),
            )
            self.register_buffer(
                f"pca_eigenvalues_{stage_idx}",
                profile.eigenvalues[:profile.effective_rank].clone(),
            )

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass returning logits and per-stage feature maps.

        With cifar_stem=True and 32x32 input, feature spatial dims are:
            stage1: (B, 256, 32, 32)
            stage2: (B, 512, 16, 16)
            stage3: (B, 1024, 8, 8)
            stage4: (B, 2048, 4, 4)
        """
        features = []

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        features.append(x)

        x = self.backbone.layer2(x)
        features.append(x)

        x = self.backbone.layer3(x)
        features.append(x)

        x = self.backbone.layer4(x)
        features.append(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.backbone.fc(x)

        return logits, features

    def finetune(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 20,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> dict[str, float]:
        """Fine-tune the teacher on CIFAR-10 to adapt the new stem and classifier.

        Returns dict with final train/test accuracy.
        """
        self.unfreeze()
        self.to(device)
        self.train()

        optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        for epoch in range(epochs):
            # Train
            self.train()
            correct = 0
            total = 0
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"FT Epoch {epoch}/{epochs}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                logits, _ = self(images)
                loss = nn.functional.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")

            scheduler.step()
            train_acc = correct / total

            # Evaluate
            test_acc = self._eval_accuracy(test_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc

            print(
                f"  FT Epoch {epoch:2d} | "
                f"train_acc {train_acc*100:.1f}% | "
                f"test_acc {test_acc*100:.1f}% | "
                f"best {best_acc*100:.1f}%"
            )

        # Re-freeze after fine-tuning
        self.freeze()
        return {"train_accuracy": train_acc, "test_accuracy": best_acc}

    @torch.no_grad()
    def _eval_accuracy(self, loader: DataLoader, device: str) -> float:
        self.eval()
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = self(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return correct / total
