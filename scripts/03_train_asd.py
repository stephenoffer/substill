#!/usr/bin/env python3
"""Phase 3: Train student via Activation Subspace Distillation."""

import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.data.cifar10 import get_cifar10_loaders
from asd.losses.combined_loss import ASDLoss
from asd.models.projectors import SubspaceProjectorBank
from asd.models.student import SlimNet
from asd.models.teacher import TeacherWrapper
from asd.profiling.svd_analysis import load_profiles, profiles_to_stage_widths
from asd.training.scheduler import LossWeightScheduler
from asd.training.trainer import ASDTrainer


def main():
    parser = argparse.ArgumentParser(description="ASD distillation training")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--profiles", default="outputs/profiles.pt")
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    num_epochs = args.epochs if args.epochs is not None else cfg.training.epochs

    # Resolve device
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

    # Load profiles
    print("Loading activation profiles...")
    profiles = load_profiles(args.profiles)

    # Compute student architecture
    stage_widths = profiles_to_stage_widths(
        profiles,
        min_width=cfg.student.min_width,
        width_multiple=cfg.student.width_multiple,
    )
    teacher_ranks = [
        max(p.effective_rank for p in profiles if p.total_channels == ch)
        for ch in sorted(set(p.total_channels for p in profiles))
    ]

    print(f"Student stage widths: {stage_widths}")
    print(f"Teacher effective ranks: {teacher_ranks}")

    # Build models
    print("Building models...")
    teacher = TeacherWrapper(
        profiles=profiles,
        cifar_stem=cfg.teacher.cifar_stem,
        pretrained=cfg.teacher.pretrained,
    )
    student = SlimNet(
        stage_widths=stage_widths,
        blocks_per_stage=cfg.student.blocks_per_stage,
    )
    projectors = SubspaceProjectorBank(
        student_widths=stage_widths,
        teacher_ranks=teacher_ranks,
    )

    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {student.count_parameters():,}")
    print(f"Projector params: {sum(p.numel() for p in projectors.parameters()):,}")

    # Loss function
    loss_fn = ASDLoss(
        profiles=profiles,
        alpha=cfg.training.loss_alpha,
        beta=cfg.training.loss_beta,
        gamma=cfg.training.loss_gamma,
        sv_weighted=True,
        num_bins=cfg.profiling.histogram_bins,
    )

    # Optimizer — train student + projectors jointly
    params = list(student.parameters()) + list(projectors.parameters())
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss weight scheduler
    loss_scheduler = LossWeightScheduler(warmup_epochs=cfg.training.gamma_warmup_epochs)

    # Data
    print("Loading CIFAR-10...")
    loaders = get_cifar10_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation=cfg.data.augmentation,
    )

    # Train
    trainer = ASDTrainer(
        teacher=teacher,
        student=student,
        projectors=projectors,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_scheduler=loss_scheduler,
        device=device,
    )

    history = trainer.train(
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        num_epochs=num_epochs,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save({
        "student_state_dict": student.state_dict(),
        "projector_state_dict": projectors.state_dict(),
        "stage_widths": stage_widths,
        "teacher_ranks": teacher_ranks,
    }, os.path.join(args.output_dir, "checkpoint.pt"))

    # Save history as JSON-serializable
    json_history = []
    for record in history:
        json_history.append({
            "epoch": record["epoch"],
            "train_total": record["train"]["total"],
            "train_task": record["train"]["task"],
            "train_subspace": record["train"]["subspace"],
            "train_sparsity": record["train"]["sparsity"],
            "eval_accuracy": record["eval"]["accuracy"],
            "eval_loss": record["eval"]["loss"],
            "lr": record["lr"],
            "gamma_scale": record["gamma_scale"],
            "elapsed": record["elapsed"],
        })

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(json_history, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
