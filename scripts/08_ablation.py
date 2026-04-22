#!/usr/bin/env python3
"""Ablation runner — ResNet50/CIFAR-10 at one operating point with one component disabled.

Supports variants:
  - full            : all improvements on (baseline-for-ablation = our method)
  - no_logit_kd     : disable Hinton KD
  - no_sparsity     : disable sparsity loss
  - gap_subspace    : use legacy GAP subspace loss
  - gap_cov         : use legacy GAP covariance in profiling
  - linear_sv       : use linear (not sqrt) SV weighting
  - uncertainty     : use uncertainty-weighted loss combination
  - classical_kd    : only task CE + Hinton KD (no ASD components)
  - task_only       : just task CE (pure supervised baseline, no teacher)
  - with_relation   : full + RKD relational loss (epsilon=0.5)
"""

import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.data.cifar10 import get_cifar_loaders
from asd.losses.combined_loss import ASDLoss
from asd.models.projectors import SubspaceProjectorBank
from asd.models.student import SlimNet
from asd.models.teacher import TeacherWrapper
from asd.profiling.activation_capture import ActivationCaptureEngine, get_resnet_layer_names
from asd.profiling.sparsity_analysis import SparsityAnalyzer
from asd.profiling.svd_analysis import SVDAnalyzer, profiles_to_stage_widths
from asd.training.scheduler import LossWeightScheduler
from asd.training.trainer import ASDTrainer


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    model.to(device)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        logits = out[0] if isinstance(out, tuple) else out
        correct += int(logits.argmax(1).eq(labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


VARIANT_CONFIGS = {
    "full": {},
    "no_logit_kd": {"use_logit_kd": False, "delta": 0.0},
    "no_sparsity": {"gamma": 0.0},
    "gap_subspace": {"subspace_mode": "gap"},
    "gap_cov": {"covariance_mode": "gap"},
    "linear_sv": {"sv_weighting": "linear"},
    "uncertainty": {"combination": "uncertainty"},
    "classical_kd": {"beta": 0.0, "gamma": 0.0, "use_logit_kd": True, "delta": 1.0},
    "task_only": {"beta": 0.0, "gamma": 0.0, "use_logit_kd": False, "delta": 0.0},
    "with_relation": {"use_relation": True, "epsilon": 0.5},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_CONFIGS))
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--teacher-weights", default="outputs/teacher_finetuned.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    overrides = VARIANT_CONFIGS[args.variant]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== ABLATION: {args.variant} (τ={args.threshold}) ===")
    print(f"Overrides: {overrides}")

    # Build overridden config values
    covariance_mode = overrides.get("covariance_mode", cfg.profiling.covariance_mode)
    subspace_mode = overrides.get("subspace_mode", cfg.training.subspace_mode)
    sv_weighting = overrides.get("sv_weighting", cfg.training.sv_weighting)
    use_logit_kd = overrides.get("use_logit_kd", cfg.training.use_logit_kd)
    use_relation = overrides.get("use_relation", False)
    combination = overrides.get("combination", cfg.training.combination)
    beta = overrides.get("beta", cfg.training.loss_beta)
    gamma = overrides.get("gamma", cfg.training.loss_gamma)
    delta = overrides.get("delta", cfg.training.loss_delta)
    epsilon = overrides.get("epsilon", 0.0)

    num_classes = 100 if args.dataset == "cifar100" else 10

    loaders = get_cifar_loaders(
        dataset=args.dataset, data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size, num_workers=cfg.data.num_workers,
        augmentation=cfg.data.augmentation,
    )
    calib_loaders = get_cifar_loaders(
        dataset=args.dataset, data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size, num_workers=cfg.data.num_workers,
        augmentation="none",
        calibration_samples=cfg.profiling.num_calibration_samples,
    )

    # Teacher
    teacher = TeacherWrapper(
        profiles=None, cifar_stem=cfg.teacher.cifar_stem, pretrained=False,
        num_classes=num_classes, freeze=True, model=args.model,
    )
    state = torch.load(args.teacher_weights, map_location="cpu", weights_only=True)
    teacher.backbone.load_state_dict(state)
    teacher.to(device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    teacher_acc = eval_accuracy(teacher, loaders["test"], device)
    print(f"Teacher: {teacher_params:,} params, acc={teacher_acc*100:.2f}%")

    # Profile
    print(f"Capturing covariances (mode={covariance_mode})...")
    layer_names = get_resnet_layer_names(args.model)
    engine = ActivationCaptureEngine(teacher.backbone, layer_names, covariance_mode=covariance_mode)
    accumulators = engine.run(calib_loaders["calibration"], device=device)
    svd = SVDAnalyzer(variance_threshold=args.threshold)
    sparsity_analyzer = SparsityAnalyzer(num_bins=cfg.profiling.histogram_bins)
    profiles = []
    for name in layer_names:
        acc = accumulators[name]
        cov = acc.finalize()
        ss = sparsity_analyzer.analyze(acc.sparsity_ratio, acc.get_activation_sample())
        profiles.append(svd.analyze(name, cov, ss))

    stage_widths = profiles_to_stage_widths(
        profiles, min_width=cfg.student.min_width, width_multiple=cfg.student.width_multiple,
    )
    stage_groups: dict[int, list] = {}
    for p in profiles:
        stage_groups.setdefault(p.total_channels, []).append(p)
    teacher_ranks = [stage_groups[ch][-1].effective_rank for ch in sorted(stage_groups)]
    print(f"Student widths: {stage_widths}, teacher_ranks: {teacher_ranks}")

    # Student + projectors
    student = SlimNet(
        stage_widths=stage_widths, blocks_per_stage=cfg.student.blocks_per_stage,
        num_classes=num_classes, block_type=cfg.student.block_type,
    )
    projectors = SubspaceProjectorBank(
        student_widths=stage_widths, teacher_ranks=teacher_ranks,
        init_mode=cfg.training.projector_init,
    )
    student_params = student.count_parameters()
    print(f"Student: {student_params:,} params ({teacher_params/student_params:.2f}x)")

    # Loss with variant overrides
    loss_fn = ASDLoss(
        profiles=profiles,
        alpha=cfg.training.loss_alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon,
        sv_weighted=True, num_bins=cfg.profiling.histogram_bins,
        subspace_mode=subspace_mode, sv_weighting=sv_weighting,
        sparsity_ratio_loss=cfg.training.sparsity_ratio_loss,
        sparsity_adaptive_tau=cfg.training.sparsity_adaptive_tau,
        use_logit_kd=use_logit_kd, logit_temperature=cfg.training.logit_temperature,
        use_relation=use_relation, combination=combination,
    )

    params = list(student.parameters()) + list(projectors.parameters())
    if combination == "uncertainty":
        params = params + list(loss_fn.parameters())
    optimizer = torch.optim.SGD(
        params, lr=cfg.training.lr, momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_sched = LossWeightScheduler(warmup_epochs=min(cfg.training.gamma_warmup_epochs, max(1, args.epochs // 3)))

    trainer = ASDTrainer(
        teacher=teacher, student=student, projectors=projectors, loss_fn=loss_fn,
        optimizer=optimizer, lr_scheduler=lr_sched, loss_scheduler=loss_sched,
        device=device,
    )

    t0 = time.time()
    history = trainer.train(loaders["train"], loaders["test"], num_epochs=args.epochs)
    elapsed = time.time() - t0

    best = max(history, key=lambda r: r["eval"]["accuracy"])
    result = {
        "variant": args.variant,
        "overrides": overrides,
        "model": args.model,
        "dataset": args.dataset,
        "threshold": args.threshold,
        "stage_widths": stage_widths,
        "teacher_ranks": teacher_ranks,
        "teacher_params": teacher_params,
        "teacher_acc": teacher_acc,
        "student_params": student_params,
        "student_acc_best": best["eval"]["accuracy"],
        "student_acc_final": history[-1]["eval"]["accuracy"],
        "compression": teacher_params / student_params,
        "acc_drop_pp": (teacher_acc - best["eval"]["accuracy"]) * 100,
        "elapsed_sec": elapsed,
        "epochs": args.epochs,
    }
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n→ {args.variant}: compression={result['compression']:.2f}x, "
          f"acc={result['student_acc_best']*100:.2f}% (drop {result['acc_drop_pp']:.2f} pp), "
          f"{result['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
