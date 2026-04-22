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
from asd.profiling.svd_analysis import (
    aggregate_stage_profile,
    group_profiles_by_stage,
    load_profiles,
    profiles_to_stage_widths,
)
from asd.training.scheduler import BetaWarmupScheduler, LossWeightScheduler
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

    # Teacher ranks — must match the subspace loss's stage aggregation so the
    # projector's output dim equals the stored subspace dim.
    stage_aggregation = cfg.training.get("subspace_stage_aggregation", "last")
    stage_groups = group_profiles_by_stage(profiles)
    teacher_ranks = [
        aggregate_stage_profile(stage_groups[ch], mode=stage_aggregation).effective_rank
        for ch in sorted(stage_groups)
    ]

    blocks_per_stage = cfg.student.blocks_per_stage
    print(f"Student stage widths: {stage_widths}")
    print(f"Teacher effective ranks ({stage_aggregation}): {teacher_ranks}")
    print(f"Blocks per stage: {blocks_per_stage}")

    # Build models
    print("Building models...")
    teacher = TeacherWrapper(
        profiles=profiles,
        cifar_stem=cfg.teacher.cifar_stem,
        pretrained=False,
        freeze=True,
    )
    weights_path = cfg.teacher.weights_path
    if os.path.exists(weights_path):
        print(f"  Loading fine-tuned teacher weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        teacher.backbone.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Fine-tuned teacher weights not found at {weights_path}. "
            "Run scripts/00_finetune_teacher.py first."
        )
    student = SlimNet(
        stage_widths=stage_widths,
        blocks_per_stage=blocks_per_stage,
        block_type=cfg.student.get("block_type", "bottleneck"),
        stem_type=cfg.student.get("stem_type", "cifar"),
    )
    projectors = SubspaceProjectorBank(
        student_widths=stage_widths,
        teacher_ranks=teacher_ranks,
        init_mode=cfg.training.get("projector_init", "orthogonal"),
        use_bn=cfg.training.get("projector_use_bn", False),
        freeze=cfg.training.get("projector_freeze", False),
    )

    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {student.count_parameters():,}")
    print(f"Projector params: {projectors.count_parameters():,}")

    # Loss function
    loss_fn = ASDLoss(
        profiles=profiles,
        alpha=cfg.training.loss_alpha,
        beta=cfg.training.loss_beta,
        gamma=cfg.training.loss_gamma,
        delta=cfg.training.get("loss_delta", 1.0),
        sv_weighted=True,
        num_bins=cfg.profiling.histogram_bins,
        subspace_mode=cfg.training.get("subspace_mode", "spatial"),
        sv_weighting=cfg.training.get("sv_weighting", "sqrt"),
        subspace_normalize_features=cfg.training.get("subspace_normalize_features", False),
        subspace_stage_aggregation=stage_aggregation,
        sparsity_ratio_loss=cfg.training.get("sparsity_ratio_loss", "bce"),
        sparsity_adaptive_tau=cfg.training.get("sparsity_adaptive_tau", True),
        use_logit_kd=cfg.training.get("use_logit_kd", True),
        logit_temperature=cfg.training.get("logit_temperature", 4.0),
        combination=cfg.training.get("combination", "fixed"),
        auto_normalize=cfg.training.get("auto_normalize", False),
        auto_norm_momentum=cfg.training.get("auto_norm_momentum", 0.95),
    )

    # Optimizer — train student + projectors jointly
    params = list(student.parameters()) + list(projectors.parameters())
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )

    # LR scheduler — cosine runs over the POST-warmup epochs so it actually
    # reaches eta_min. The trainer steps the cosine scheduler only after the
    # linear warmup completes; constructing with T_max=num_epochs would
    # under-step by `lr_warmup_epochs`.
    lr_warmup_epochs = cfg.training.get("lr_warmup_epochs", 0)
    cosine_t_max = max(1, num_epochs - lr_warmup_epochs)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max)

    # Loss weight schedulers
    loss_scheduler = LossWeightScheduler(warmup_epochs=cfg.training.gamma_warmup_epochs)
    beta_scheduler = BetaWarmupScheduler(
        warmup_epochs=cfg.training.get("beta_warmup_epochs", 0),
    )

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
        beta_scheduler=beta_scheduler,
        lr_warmup_epochs=lr_warmup_epochs,
        base_lr=cfg.training.lr,
        keep_best=cfg.training.get("keep_best", True),
        restore_best_on_exit=cfg.training.get("restore_best_on_exit", True),
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
        "blocks_per_stage": blocks_per_stage,
        "block_type": cfg.student.get("block_type", "bottleneck"),
        "stem_type": cfg.student.get("stem_type", "cifar"),
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
            "train_logit": record["train"].get("logit", 0.0),
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
