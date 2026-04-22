#!/usr/bin/env python3
"""Extended benchmark runner — same pipeline as 07_bench.py but supports
non-ResNet teachers (MobileNetV2, VGG16-BN, DenseNet-121) via
asd.models.teacher_ext.
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
from asd.models.teacher_ext import get_teacher
from asd.profiling.activation_capture import (
    ActivationCaptureEngine, get_resnet_layer_names,
)
from asd.profiling.sparsity_analysis import SparsityAnalyzer
from asd.profiling.svd_analysis import SVDAnalyzer, profiles_to_stage_widths
from asd.training.scheduler import LossWeightScheduler
from asd.training.trainer import ASDTrainer


def hook_names_for(teacher, model: str, device: str) -> list[str]:
    """Return a list of layer-name strings in the teacher's backbone that we
    should hook to build the activation profiles.

    For non-ResNet architectures, we hook one layer per ASD-stage (the last
    feature before the stage output). This gives four per-stage profiles
    matching what the student and losses expect.
    """
    if model in ("resnet18", "resnet34", "resnet50", "resnet101"):
        return get_resnet_layer_names(model)

    if model == "mobilenet_v2":
        # Hook the last block of each ASD-stage — see MobileNetV2Teacher.forward.
        # stage boundaries at indices 3, 6, 13, 18.
        return ["backbone.features.3", "backbone.features.6",
                "backbone.features.13", "backbone.features.18"]

    if model == "vgg16_bn":
        # Hook the conv block just before MaxPool at each of the 4 ASD-stages.
        # For VGG16-BN, ASD-stages end at features indices: [13, 23, 33, 43]
        return [f"backbone.features.{i}" for i in teacher.stage_end_indices]

    raise ValueError(f"Unknown model: {model}")


def resolve_submodule(root, name: str):
    obj = root
    for part in name.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_v2",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101",
                                 "mobilenet_v2", "vgg16_bn"])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.85, 0.95])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ft-epochs", type=int, default=6)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--teacher-weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    output_dir = args.output_dir or f"outputs/bench_{args.model}_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10}[args.dataset]
    print(f"=== {args.model.upper()} / {args.dataset.upper()} ===")

    # Data
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
    tw_path = args.teacher_weights or os.path.join(output_dir, "teacher.pt")
    teacher = get_teacher(
        model=args.model, cifar_stem=cfg.teacher.cifar_stem,
        pretrained=True, num_classes=num_classes, freeze=False,
    )
    if os.path.exists(tw_path):
        print(f"Loading teacher weights from {tw_path}")
        state = torch.load(tw_path, map_location="cpu", weights_only=True)
        teacher.backbone.load_state_dict(state)
        teacher.freeze()
        teacher.to(device)
        teacher_acc = eval_accuracy(teacher, loaders["test"], device)
    else:
        print(f"Fine-tuning teacher for {args.ft_epochs} epochs...")
        m = teacher.finetune(loaders["train"], loaders["test"],
                             epochs=args.ft_epochs, lr=cfg.teacher.finetune_lr, device=device)
        teacher_acc = m["test_accuracy"]
        torch.save(teacher.backbone.state_dict(), tw_path)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {teacher_params:,} params, acc={teacher_acc*100:.2f}%")

    # Profile — capture per-stage covariance
    cov_mode = cfg.profiling.get("covariance_mode", "per_pixel")
    hook_names = hook_names_for(teacher, args.model, device)
    print(f"Hooking {len(hook_names)} layers: {hook_names}")
    engine = ActivationCaptureEngine(teacher, hook_names, covariance_mode=cov_mode)
    # Work around: engine._get_module treats model=teacher, not teacher.backbone,
    # so hook names start with "backbone." for non-ResNet wrappers.
    accumulators = engine.run(calib_loaders["calibration"], device=device)
    captured = {}
    for name in hook_names:
        acc = accumulators[name]
        captured[name] = {
            "cov": acc.finalize(),
            "sparsity_ratio": acc.sparsity_ratio,
            "activation_sample": acc.get_activation_sample(),
        }

    # Sweep thresholds
    results = []
    for threshold in args.thresholds:
        print(f"\n--- threshold = {threshold} ---")
        svd = SVDAnalyzer(variance_threshold=threshold)
        sparsity_analyzer = SparsityAnalyzer(num_bins=cfg.profiling.histogram_bins)
        profiles = []
        for name in hook_names:
            c = captured[name]
            ss = sparsity_analyzer.analyze(c["sparsity_ratio"], c["activation_sample"])
            profiles.append(svd.analyze(name, c["cov"], ss))

        stage_widths = profiles_to_stage_widths(
            profiles, min_width=cfg.student.min_width,
            width_multiple=cfg.student.width_multiple,
        )
        stage_groups: dict[int, list] = {}
        for p in profiles:
            stage_groups.setdefault(p.total_channels, []).append(p)
        teacher_ranks = [stage_groups[ch][-1].effective_rank for ch in sorted(stage_groups)]
        print(f"Stage widths: {stage_widths}, teacher_ranks: {teacher_ranks}")

        # Student: always SlimNet (ResNet-style). This is fine because the
        # student only needs to reproduce the teacher's per-stage *output*
        # features. The teacher's block type doesn't dictate the student's.
        student = SlimNet(
            stage_widths=stage_widths, blocks_per_stage=cfg.student.blocks_per_stage,
            num_classes=num_classes, block_type=cfg.student.block_type,
        )
        projectors = SubspaceProjectorBank(
            student_widths=stage_widths, teacher_ranks=teacher_ranks,
            init_mode=cfg.training.projector_init,
        )

        loss_fn = ASDLoss(
            profiles=profiles,
            alpha=cfg.training.loss_alpha, beta=cfg.training.loss_beta,
            gamma=cfg.training.loss_gamma, delta=cfg.training.loss_delta,
            sv_weighted=True, num_bins=cfg.profiling.histogram_bins,
            subspace_mode=cfg.training.subspace_mode, sv_weighting=cfg.training.sv_weighting,
            sparsity_ratio_loss=cfg.training.sparsity_ratio_loss,
            sparsity_adaptive_tau=cfg.training.sparsity_adaptive_tau,
            use_logit_kd=cfg.training.use_logit_kd,
            logit_temperature=cfg.training.logit_temperature,
            combination=cfg.training.combination,
        )

        params = list(student.parameters()) + list(projectors.parameters())
        optimizer = torch.optim.SGD(
            params, lr=cfg.training.lr, momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        warmup = min(cfg.training.gamma_warmup_epochs, max(1, args.epochs // 3))
        loss_sched = LossWeightScheduler(warmup_epochs=warmup)

        trainer = ASDTrainer(
            teacher=teacher, student=student, projectors=projectors, loss_fn=loss_fn,
            optimizer=optimizer, lr_scheduler=lr_sched, loss_scheduler=loss_sched,
            device=device,
        )
        t0 = time.time()
        history = trainer.train(loaders["train"], loaders["test"], num_epochs=args.epochs)
        elapsed = time.time() - t0
        best = max(history, key=lambda r: r["eval"]["accuracy"])

        r = {
            "threshold": threshold, "model": args.model, "dataset": args.dataset,
            "stage_widths": stage_widths, "teacher_ranks": teacher_ranks,
            "student_params": student.count_parameters(),
            "student_acc_best": best["eval"]["accuracy"],
            "student_acc_final": history[-1]["eval"]["accuracy"],
            "teacher_params": teacher_params, "teacher_acc": teacher_acc,
            "compression": teacher_params / student.count_parameters(),
            "acc_drop_pp": (teacher_acc - best["eval"]["accuracy"]) * 100,
            "elapsed_sec": elapsed, "epochs": args.epochs,
        }
        print(f"τ={threshold}: compression={r['compression']:.2f}x, "
              f"acc={r['student_acc_best']*100:.2f}% (drop {r['acc_drop_pp']:.2f} pp), "
              f"{elapsed:.0f}s")
        results.append(r)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  τ={r['threshold']:.2f} {r['compression']:>7.2f}× "
              f"acc={r['student_acc_best']*100:>6.2f}%")


if __name__ == "__main__":
    main()
