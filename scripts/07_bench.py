#!/usr/bin/env python3
"""End-to-end ASD benchmark runner for any (model, dataset) combo.

For one (model, dataset) pair:
  1. Fine-tune teacher on the dataset  (skipped if --teacher-weights exists)
  2. Capture activation covariances once
  3. For each threshold: re-run SVD → build student → distill → evaluate
  4. Persist per-run results + an aggregated JSON over all thresholds

Used to build the research-paper result matrix across many models/datasets.
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


def finetune_teacher(teacher, loaders, epochs, lr, device):
    """Fine-tune teacher on the current dataset."""
    return teacher.finetune(
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        epochs=epochs, lr=lr, device=device,
    )


def capture_covariances(teacher, backbone_name, calib_loader, device, covariance_mode):
    layer_names = get_resnet_layer_names(backbone_name)
    engine = ActivationCaptureEngine(teacher.backbone, layer_names, covariance_mode=covariance_mode)
    accumulators = engine.run(calib_loader, device=device)
    captured = {}
    for name in layer_names:
        acc = accumulators[name]
        captured[name] = {
            "cov": acc.finalize(),
            "sparsity_ratio": acc.sparsity_ratio,
            "activation_sample": acc.get_activation_sample(),
        }
    return layer_names, captured


def run_threshold(
    threshold, layer_names, captured, teacher, teacher_info,
    train_loader, test_loader, cfg, epochs, device, output_dir, num_classes,
):
    svd = SVDAnalyzer(
        variance_threshold=threshold,
        definition=cfg.profiling.get("rank_definition", "variance"),
    )
    sparsity_analyzer = SparsityAnalyzer(num_bins=cfg.profiling.histogram_bins)
    profiles = []
    for name in layer_names:
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

    # Student uses Bottleneck blocks by default; teacher's natural block type
    # (basic vs bottleneck) is tracked separately in teacher_info.
    student = SlimNet(
        stage_widths=stage_widths,
        blocks_per_stage=cfg.student.blocks_per_stage,
        num_classes=num_classes,
        block_type=cfg.student.get("block_type", "bottleneck"),
    )
    projectors = SubspaceProjectorBank(
        student_widths=stage_widths,
        teacher_ranks=teacher_ranks,
        init_mode=cfg.training.get("projector_init", "orthogonal"),
    )

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
        sparsity_ratio_loss=cfg.training.get("sparsity_ratio_loss", "bce"),
        sparsity_adaptive_tau=cfg.training.get("sparsity_adaptive_tau", True),
        use_logit_kd=cfg.training.get("use_logit_kd", True),
        logit_temperature=cfg.training.get("logit_temperature", 4.0),
        combination=cfg.training.get("combination", "fixed"),
    )

    params = list(student.parameters()) + list(projectors.parameters())
    if cfg.training.get("combination", "fixed") == "uncertainty":
        params = params + list(loss_fn.parameters())
    optimizer = torch.optim.SGD(
        params, lr=cfg.training.lr, momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = min(cfg.training.gamma_warmup_epochs, max(1, epochs // 3))
    loss_sched = LossWeightScheduler(warmup_epochs=warmup)

    trainer = ASDTrainer(
        teacher=teacher, student=student, projectors=projectors, loss_fn=loss_fn,
        optimizer=optimizer, lr_scheduler=lr_sched, loss_scheduler=loss_sched,
        device=device,
    )

    t0 = time.time()
    history = trainer.train(train_loader, test_loader, num_epochs=epochs)
    elapsed = time.time() - t0

    best = max(history, key=lambda r: r["eval"]["accuracy"])

    ckpt_path = os.path.join(output_dir, f"ckpt_t{threshold:.2f}.pt")
    torch.save({
        "student_state_dict": student.state_dict(),
        "stage_widths": stage_widths,
        "teacher_ranks": teacher_ranks,
        "blocks_per_stage": cfg.student.blocks_per_stage,
        "block_type": cfg.student.get("block_type", "bottleneck"),
        "threshold": threshold,
        "num_classes": num_classes,
    }, ckpt_path)

    return {
        "threshold": threshold,
        "stage_widths": stage_widths,
        "teacher_ranks": teacher_ranks,
        "student_params": student.count_parameters(),
        "student_acc_best": best["eval"]["accuracy"],
        "student_acc_final": history[-1]["eval"]["accuracy"],
        "elapsed_sec": elapsed,
        "epochs": epochs,
        "checkpoint": ckpt_path,
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end ASD benchmark")
    parser.add_argument("--model", default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.85, 0.95])
    parser.add_argument("--epochs", type=int, default=25, help="Distillation epochs per threshold")
    parser.add_argument("--ft-epochs", type=int, default=8, help="Teacher fine-tune epochs")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--teacher-weights", default=None,
                        help="Path to pre-finetuned teacher weights (skip fine-tune)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    output_dir = args.output_dir or f"outputs/bench_{args.model}_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    num_classes = 100 if args.dataset == "cifar100" else 10
    print(f"=== {args.model.upper()} / {args.dataset.upper()} ===")
    print(f"Output dir: {output_dir}")
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
    teacher_weights = args.teacher_weights or os.path.join(output_dir, "teacher.pt")
    teacher = TeacherWrapper(
        profiles=None, cifar_stem=cfg.teacher.cifar_stem, pretrained=True,
        num_classes=num_classes, freeze=False, model=args.model,
    )

    if os.path.exists(teacher_weights):
        print(f"Loading teacher from {teacher_weights}")
        state = torch.load(teacher_weights, map_location="cpu", weights_only=True)
        teacher.backbone.load_state_dict(state)
        teacher.freeze()
        teacher.to(device)
        print("Evaluating teacher...")
        teacher_acc = eval_accuracy(teacher, loaders["test"], device)
    else:
        print(f"Fine-tuning teacher for {args.ft_epochs} epochs...")
        m = finetune_teacher(teacher, loaders, args.ft_epochs, cfg.teacher.finetune_lr, device)
        teacher_acc = m["test_accuracy"]
        torch.save(teacher.backbone.state_dict(), teacher_weights)
        print(f"Saved teacher to {teacher_weights}")

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher ({args.model}): {teacher_params:,} params, acc={teacher_acc*100:.2f}%")

    teacher_info = {
        "name": args.model,
        "stage_channels": teacher.stage_channels,
        "block_type": teacher.teacher_block_type,
    }

    # Profile
    cov_mode = cfg.profiling.get("covariance_mode", "per_pixel")
    print(f"\nCapturing covariances ({cov_mode})...")
    t0 = time.time()
    layer_names, captured = capture_covariances(
        teacher, args.model, calib_loaders["calibration"], device, cov_mode,
    )
    print(f"  Capture took {time.time()-t0:.1f}s")

    # Sweep thresholds
    results = []
    for threshold in args.thresholds:
        print(f"\n--- threshold = {threshold} ---")
        r = run_threshold(
            threshold=threshold, layer_names=layer_names, captured=captured,
            teacher=teacher, teacher_info=teacher_info,
            train_loader=loaders["train"], test_loader=loaders["test"],
            cfg=cfg, epochs=args.epochs, device=device, output_dir=output_dir,
            num_classes=num_classes,
        )
        r["teacher_acc"] = teacher_acc
        r["teacher_params"] = teacher_params
        r["compression"] = teacher_params / r["student_params"]
        r["acc_drop_pp"] = (teacher_acc - r["student_acc_best"]) * 100
        r["model"] = args.model
        r["dataset"] = args.dataset
        print(
            f"τ={threshold}: stage_widths={r['stage_widths']}, "
            f"student_params={r['student_params']:,} ({r['compression']:.2f}x), "
            f"acc={r['student_acc_best']*100:.2f}% "
            f"(drop {r['acc_drop_pp']:.2f} pp), {r['elapsed_sec']:.0f}s"
        )
        results.append(r)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Teacher ({args.model}/{args.dataset}): {teacher_acc*100:.2f}%")
    for r in results:
        print(f"  τ={r['threshold']:.2f}  {r['compression']:>7.2f}x  "
              f"{r['student_acc_best']*100:>6.2f}%  drop={r['acc_drop_pp']:>5.2f}pp")


if __name__ == "__main__":
    main()
