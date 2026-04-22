#!/usr/bin/env python3
"""Sweep variance_threshold to trace the compression vs accuracy tradeoff.

Runs profiling ONCE (covariance capture is the expensive step) and then re-runs
the SVD analyzer at each threshold to derive a different student. Each resulting
student is trained from scratch with ASD and evaluated on the CIFAR-10 test set.

Outputs:
  outputs/sweep/sweep_results.json
  outputs/sweep/compression_vs_accuracy.png
  outputs/sweep/checkpoint_t{threshold}.pt  (per-threshold student weights)
"""

import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.data.cifar10 import get_cifar10_loaders
from asd.losses.combined_loss import ASDLoss
from asd.models.projectors import SubspaceProjectorBank
from asd.models.student import SlimNet
from asd.models.teacher import TeacherWrapper
from asd.profiling.activation_capture import ActivationCaptureEngine, get_resnet50_layer_names
from asd.profiling.sparsity_analysis import SparsityAnalyzer
from asd.profiling.svd_analysis import SVDAnalyzer, profiles_to_stage_widths
from asd.training.scheduler import LossWeightScheduler
from asd.training.trainer import ASDTrainer


@torch.no_grad()
def eval_accuracy(model, loader, device: str) -> float:
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


def capture_covariances(teacher: TeacherWrapper, calib_loader, device: str, covariance_mode: str = "per_pixel"):
    """Capture per-layer covariance + sparsity sample once."""
    layer_names = get_resnet50_layer_names()
    engine = ActivationCaptureEngine(teacher.backbone, layer_names, covariance_mode=covariance_mode)
    accumulators = engine.run(calib_loader, device=device)

    sparsity_analyzer = SparsityAnalyzer(num_bins=64)  # num_bins overridden per-threshold later
    captured = {}
    for name in layer_names:
        acc = accumulators[name]
        cov = acc.finalize()
        sample = acc.get_activation_sample()
        captured[name] = {
            "cov": cov,
            "sparsity_ratio": acc.sparsity_ratio,
            "activation_sample": sample,
        }
    return layer_names, captured


def run_one_threshold(
    threshold: float,
    layer_names: list[str],
    captured: dict,
    teacher: TeacherWrapper,
    train_loader,
    test_loader,
    cfg,
    epochs: int,
    device: str,
    output_dir: str,
) -> dict:
    """Build + train + evaluate a student for a given variance_threshold."""
    # Re-run SVD + sparsity analysis at this threshold
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
        profiles,
        min_width=cfg.student.min_width,
        width_multiple=cfg.student.width_multiple,
    )
    stage_groups: dict[int, list] = {}
    for p in profiles:
        stage_groups.setdefault(p.total_channels, []).append(p)
    teacher_ranks = [stage_groups[ch][-1].effective_rank for ch in sorted(stage_groups)]

    block_type = cfg.student.get("block_type", "bottleneck")
    student = SlimNet(
        stage_widths=stage_widths,
        blocks_per_stage=cfg.student.blocks_per_stage,
        block_type=block_type,
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
    optimizer = torch.optim.SGD(
        params, lr=cfg.training.lr, momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Keep warmup proportional when epochs is small
    warmup = min(cfg.training.gamma_warmup_epochs, max(1, epochs // 3))
    loss_scheduler = LossWeightScheduler(warmup_epochs=warmup)

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

    t0 = time.time()
    history = trainer.train(train_loader, test_loader, num_epochs=epochs)
    elapsed = time.time() - t0

    best_epoch = max(history, key=lambda r: r["eval"]["accuracy"])
    final = history[-1]

    # Save the (final) student checkpoint
    ckpt_path = os.path.join(output_dir, f"checkpoint_t{threshold:.2f}.pt")
    torch.save({
        "student_state_dict": student.state_dict(),
        "projector_state_dict": projectors.state_dict(),
        "stage_widths": stage_widths,
        "teacher_ranks": teacher_ranks,
        "blocks_per_stage": cfg.student.blocks_per_stage,
        "block_type": block_type,
        "threshold": threshold,
    }, ckpt_path)

    return {
        "threshold": threshold,
        "stage_widths": stage_widths,
        "teacher_ranks": teacher_ranks,
        "student_params": student.count_parameters(),
        "student_acc_best": best_epoch["eval"]["accuracy"],
        "student_acc_final": final["eval"]["accuracy"],
        "elapsed_sec": elapsed,
        "epochs": epochs,
        "checkpoint": ckpt_path,
    }


def plot_tradeoff(results: list[dict], teacher_params: int, teacher_acc: float, save_path: str) -> None:
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda r: r["threshold"])
    thresholds = [r["threshold"] for r in results]
    compressions = [teacher_params / r["student_params"] for r in results]
    accs = [r["student_acc_best"] * 100 for r in results]
    drops = [(teacher_acc - r["student_acc_best"]) * 100 for r in results]
    params_m = [r["student_params"] / 1e6 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ASD: Compression vs Accuracy Tradeoff", fontsize=13)

    ax1 = axes[0]
    ax1.plot(compressions, accs, "o-", color="#2E7D32", linewidth=2, markersize=8, label="Student")
    ax1.axhline(y=teacher_acc * 100, color="red", linestyle="--", alpha=0.7,
                label=f"Teacher {teacher_acc*100:.2f}%")
    for c, a, t, pm in zip(compressions, accs, thresholds, params_m):
        ax1.annotate(
            f"τ={t}\n{pm:.2f}M",
            (c, a),
            xytext=(6, -4), textcoords="offset points", fontsize=9,
        )
    ax1.set_xlabel("Compression ratio  (teacher_params / student_params)")
    ax1.set_ylabel("CIFAR-10 test accuracy (%)")
    ax1.set_title("Accuracy vs Compression")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(compressions, drops, "o-", color="#C62828", linewidth=2, markersize=8)
    for c, d, t in zip(compressions, drops, thresholds):
        ax2.annotate(f"τ={t}", (c, d), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax2.set_xlabel("Compression ratio")
    ax2.set_ylabel("Accuracy drop vs teacher (pp)")
    ax2.set_title("Accuracy Drop vs Compression")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved tradeoff plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Sweep variance thresholds for ASD tradeoff curve")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.70, 0.85, 0.95, 0.99],
        help="Variance thresholds to sweep (passed to SVDAnalyzer)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Distillation epochs per threshold")
    parser.add_argument("--output-dir", default="outputs/sweep")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load fine-tuned teacher
    weights_path = cfg.teacher.weights_path
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Fine-tuned teacher weights not found at {weights_path}. "
            "Run scripts/00_finetune_teacher.py first."
        )
    print(f"Loading teacher from {weights_path}")
    teacher = TeacherWrapper(
        profiles=None, cifar_stem=cfg.teacher.cifar_stem, pretrained=False, freeze=True,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    teacher.backbone.load_state_dict(state)
    teacher.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())

    # Loaders
    calib_loaders = get_cifar10_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation="none",
        calibration_samples=cfg.profiling.num_calibration_samples,
    )
    train_loaders = get_cifar10_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation=cfg.data.augmentation,
    )

    print("Evaluating teacher on test set...")
    teacher_acc = eval_accuracy(teacher, train_loaders["test"], device)
    print(f"Teacher: {teacher_params:,} params, acc={teacher_acc*100:.2f}%")

    cov_mode = cfg.profiling.get("covariance_mode", "per_pixel")
    print(f"\nCapturing activation covariances once (mode={cov_mode}, shared across thresholds)...")
    t0 = time.time()
    layer_names, captured = capture_covariances(
        teacher, calib_loaders["calibration"], device, covariance_mode=cov_mode,
    )
    print(f"  Covariance capture took {time.time()-t0:.1f}s")

    results = []
    for threshold in args.thresholds:
        print(f"\n{'=' * 70}\n=== variance_threshold = {threshold} ===\n{'=' * 70}")

        result = run_one_threshold(
            threshold=threshold,
            layer_names=layer_names,
            captured=captured,
            teacher=teacher,
            train_loader=train_loaders["train"],
            test_loader=train_loaders["test"],
            cfg=cfg,
            epochs=args.epochs,
            device=device,
            output_dir=args.output_dir,
        )
        result["teacher_acc"] = teacher_acc
        result["teacher_params"] = teacher_params
        result["compression"] = teacher_params / result["student_params"]
        result["acc_drop_pp"] = (teacher_acc - result["student_acc_best"]) * 100

        print(
            f"\n→ τ={threshold:.2f}: stage_widths={result['stage_widths']}, "
            f"student_params={result['student_params']:,} "
            f"({result['compression']:.2f}x), "
            f"acc={result['student_acc_best']*100:.2f}% "
            f"(drop {result['acc_drop_pp']:.2f} pp), "
            f"{result['elapsed_sec']:.0f}s"
        )

        results.append(result)

        # Persist incrementally so partial sweeps are still useful
        with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    plot_tradeoff(
        results, teacher_params, teacher_acc,
        save_path=os.path.join(args.output_dir, "compression_vs_accuracy.png"),
    )

    # Text summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Teacher: {teacher_params:,} params, {teacher_acc*100:.2f}%")
    print(f"{'τ':>6} {'stage_widths':>26} {'params':>12} {'compression':>12} "
          f"{'acc':>8} {'drop(pp)':>10}")
    for r in sorted(results, key=lambda x: x["threshold"]):
        print(
            f"{r['threshold']:>6.2f} {str(r['stage_widths']):>26} "
            f"{r['student_params']:>12,} {r['compression']:>11.2f}x "
            f"{r['student_acc_best']*100:>7.2f}% {r['acc_drop_pp']:>10.2f}"
        )


if __name__ == "__main__":
    main()
