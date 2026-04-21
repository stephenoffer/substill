#!/usr/bin/env python3
"""Evaluate trained student model and compare with teacher."""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.data.cifar10 import get_cifar10_loaders
from asd.models.student import SlimNet
from asd.models.teacher import TeacherWrapper


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """Evaluate a model's accuracy and loss."""
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output
        loss = nn.functional.cross_entropy(logits, labels)

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        num_batches += 1

    return {
        "accuracy": correct / total,
        "loss": total_loss / num_batches,
        "correct": correct,
        "total": total,
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASD student vs teacher")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/training/checkpoint.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Load data
    loaders = get_cifar10_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation="none",
    )
    test_loader = loaders["test"]

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    stage_widths = ckpt["stage_widths"]

    # Build and load student
    student = SlimNet(stage_widths=stage_widths, blocks_per_stage=cfg.student.blocks_per_stage)
    student.load_state_dict(ckpt["student_state_dict"])

    # Build teacher
    teacher = TeacherWrapper(
        cifar_stem=cfg.teacher.cifar_stem,
        pretrained=cfg.teacher.pretrained,
    )

    # Evaluate both
    print("\nEvaluating teacher (ResNet50)...")
    teacher_metrics = evaluate_model(teacher, test_loader, device)

    print("Evaluating student (SlimNet)...")
    student_metrics = evaluate_model(student, test_loader, device)

    # Report
    teacher_params = count_params(teacher)
    student_params = count_params(student)

    print(f"\n{'=' * 60}")
    print("COMPARISON: Teacher vs Student")
    print(f"{'=' * 60}")
    print(f"{'':>20} {'Teacher':>15} {'Student':>15}")
    print(f"{'-' * 50}")
    print(f"{'Parameters':>20} {teacher_params:>15,} {student_params:>15,}")
    print(f"{'Accuracy':>20} {teacher_metrics['accuracy']*100:>14.2f}% {student_metrics['accuracy']*100:>14.2f}%")
    print(f"{'Loss':>20} {teacher_metrics['loss']:>15.4f} {student_metrics['loss']:>15.4f}")
    print(f"{'Compression':>20} {'1.0x':>15} {teacher_params/student_params:>14.1f}x")
    print(f"{'Accuracy Drop':>20} {'—':>15} {(teacher_metrics['accuracy'] - student_metrics['accuracy'])*100:>14.2f}%")
    print(f"{'Stage Widths':>20} {'[256,512,1024,2048]':>15} {str(stage_widths):>15}")


if __name__ == "__main__":
    main()
