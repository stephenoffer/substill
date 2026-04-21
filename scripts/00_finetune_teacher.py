#!/usr/bin/env python3
"""Phase 0: Fine-tune pretrained ResNet50 on CIFAR-10 after stem replacement.

This step is REQUIRED before profiling. The pretrained ResNet50 has its conv1
and fc layers replaced for CIFAR-10 (32x32 images, 10 classes). These new layers
are randomly initialized and must be adapted before activation profiles are meaningful.
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.data.cifar10 import get_cifar10_loaders
from asd.models.teacher import TeacherWrapper


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 teacher on CIFAR-10")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="outputs/teacher_finetuned.pt", help="Path to save fine-tuned weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Override fine-tuning epochs")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    ft_epochs = args.epochs if args.epochs is not None else cfg.teacher.finetune_epochs

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load data
    print("Loading CIFAR-10...")
    loaders = get_cifar10_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation=cfg.data.augmentation,
    )

    # Build teacher (unfrozen for fine-tuning)
    print("Building ResNet50 teacher with CIFAR-10 stem...")
    teacher = TeacherWrapper(
        profiles=None,
        cifar_stem=cfg.teacher.cifar_stem,
        pretrained=cfg.teacher.pretrained,
        freeze=False,
    )

    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Parameters: {total_params:,}")

    # Fine-tune
    print(f"\nFine-tuning for {ft_epochs} epochs...")
    metrics = teacher.finetune(
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        epochs=ft_epochs,
        lr=cfg.teacher.finetune_lr,
        device=device,
    )

    print(f"\nFine-tuning complete.")
    print(f"  Best test accuracy: {metrics['test_accuracy']*100:.2f}%")

    # Save fine-tuned weights
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(teacher.backbone.state_dict(), args.output)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
