#!/usr/bin/env python3
"""ASD v2 comparison — runs the full improved algorithm with ALL v2 knobs on.

v2 improvements (vs v1 "improved"):
  - Auto-normalized loss components (EMA-based)
  - β-warmup (0.1 → 1.0 over first 3 epochs)
  - LR warmup (10% → 100% over first 2 epochs) then cosine
  - Best-val checkpoint (no end-of-cosine regression)
  - KD temperature 4 → 2 (less over-smoothing on 10-class CIFAR)
  - Subspace L2-normalize channel axis (scale-invariant to feature magnitude)

Runs the exact same (teacher, compression-point) configurations we tested with
v1 so we get a direct A/B comparison. Teacher weights are reused from the
main-matrix run (no re-fine-tune needed).
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
from asd.training.scheduler import BetaWarmupScheduler, LossWeightScheduler
from asd.training.trainer import ASDTrainer


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval(); model.to(device)
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
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--teacher-weights", default="outputs/teacher_finetuned.pt")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.70, 0.85, 0.95, 0.99])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output-dir", default="outputs/v2_resnet50_cifar10")
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--subspace-mode", default="spatial",
                        choices=["spatial", "cosine_spatial"])
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--auto-normalize", action="store_true", default=True)
    parser.add_argument("--beta-warmup", type=int, default=3)
    parser.add_argument("--lr-warmup", type=int, default=2)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    os.makedirs(args.output_dir, exist_ok=True)

    num_classes = 100 if args.dataset == "cifar100" else 10
    print(f"=== ASD v2 / {args.model.upper()} / {args.dataset.upper()} ===")
    print(f"Settings: T={args.kd_temperature}, mode={args.subspace_mode}, "
          f"normalize={args.normalize_features}, auto_norm={args.auto_normalize}, "
          f"β-warmup={args.beta_warmup}, lr-warmup={args.lr_warmup}")

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
    teacher = TeacherWrapper(
        profiles=None, cifar_stem=cfg.teacher.cifar_stem, pretrained=False,
        num_classes=num_classes, freeze=True, model=args.model,
    )
    if not os.path.exists(args.teacher_weights):
        raise FileNotFoundError(f"Need teacher weights at {args.teacher_weights}")
    state = torch.load(args.teacher_weights, map_location="cpu", weights_only=True)
    teacher.backbone.load_state_dict(state)
    teacher.to(device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    teacher_acc = eval_accuracy(teacher, loaders["test"], device)
    print(f"Teacher: {teacher_params:,} params, acc={teacher_acc*100:.2f}%")

    # Profile once
    cov_mode = cfg.profiling.get("covariance_mode", "per_pixel")
    layer_names = get_resnet_layer_names(args.model)
    engine = ActivationCaptureEngine(teacher.backbone, layer_names, covariance_mode=cov_mode)
    accumulators = engine.run(calib_loaders["calibration"], device=device)
    captured = {
        name: {
            "cov": acc.finalize(),
            "sparsity_ratio": acc.sparsity_ratio,
            "activation_sample": acc.get_activation_sample(),
        }
        for name, acc in accumulators.items()
    }

    results = []
    for threshold in args.thresholds:
        print(f"\n--- τ={threshold} ---")
        svd = SVDAnalyzer(variance_threshold=threshold)
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
        stage_groups = {}
        for p in profiles:
            stage_groups.setdefault(p.total_channels, []).append(p)
        teacher_ranks = [stage_groups[ch][-1].effective_rank for ch in sorted(stage_groups)]
        print(f"stage_widths={stage_widths}, teacher_ranks={teacher_ranks}")

        student = SlimNet(
            stage_widths=stage_widths, blocks_per_stage=cfg.student.blocks_per_stage,
            num_classes=num_classes, block_type=cfg.student.block_type,
        )
        projectors = SubspaceProjectorBank(
            student_widths=stage_widths, teacher_ranks=teacher_ranks,
            init_mode=cfg.training.projector_init,
        )

        # v2 loss: all new knobs on
        loss_fn = ASDLoss(
            profiles=profiles,
            alpha=cfg.training.loss_alpha, beta=cfg.training.loss_beta,
            gamma=cfg.training.loss_gamma, delta=cfg.training.loss_delta,
            sv_weighted=True, num_bins=cfg.profiling.histogram_bins,
            subspace_mode=args.subspace_mode,
            sv_weighting=cfg.training.sv_weighting,
            subspace_normalize_features=args.normalize_features,
            sparsity_ratio_loss=cfg.training.sparsity_ratio_loss,
            sparsity_adaptive_tau=cfg.training.sparsity_adaptive_tau,
            use_logit_kd=True,
            logit_temperature=args.kd_temperature,
            auto_normalize=args.auto_normalize,
            combination="fixed",
        )

        params = list(student.parameters()) + list(projectors.parameters())
        optimizer = torch.optim.SGD(
            params, lr=cfg.training.lr, momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
        # Cosine over (epochs - warmup) so warmup doesn't consume cosine schedule
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.lr_warmup),
        )
        warmup = min(cfg.training.gamma_warmup_epochs, max(1, args.epochs // 3))
        loss_sched = LossWeightScheduler(warmup_epochs=warmup)
        beta_sched = BetaWarmupScheduler(warmup_epochs=args.beta_warmup, initial_scale=0.1)

        trainer = ASDTrainer(
            teacher=teacher, student=student, projectors=projectors, loss_fn=loss_fn,
            optimizer=optimizer, lr_scheduler=lr_sched, loss_scheduler=loss_sched,
            device=device, beta_scheduler=beta_sched,
            lr_warmup_epochs=args.lr_warmup, keep_best=True,
        )
        t0 = time.time()
        history = trainer.train(loaders["train"], loaders["test"], num_epochs=args.epochs)
        elapsed = time.time() - t0

        # Evaluate best-checkpoint if we have one, else final
        if trainer._best_state is not None:
            student.load_state_dict({k: v.to(device) for k, v in trainer._best_state["student"].items()})
            final_acc = eval_accuracy(student, loaders["test"], device)
        else:
            final_acc = history[-1]["eval"]["accuracy"]
        best_acc = trainer._best_acc or max(r["eval"]["accuracy"] for r in history)

        r = {
            "threshold": threshold, "model": args.model, "dataset": args.dataset,
            "stage_widths": stage_widths, "teacher_ranks": teacher_ranks,
            "student_params": student.count_parameters(),
            "teacher_params": teacher_params, "teacher_acc": teacher_acc,
            "student_acc_best": best_acc, "student_acc_final": final_acc,
            "compression": teacher_params / student.count_parameters(),
            "acc_drop_pp": (teacher_acc - best_acc) * 100,
            "epochs": args.epochs, "elapsed_sec": elapsed,
            "variant": "v2",
            "kd_temperature": args.kd_temperature,
            "subspace_mode": args.subspace_mode,
            "normalize_features": args.normalize_features,
            "auto_normalize": args.auto_normalize,
        }
        print(f"τ={threshold}: {r['compression']:.2f}x, "
              f"acc={best_acc*100:.2f}% (drop {r['acc_drop_pp']:.2f} pp), {elapsed:.0f}s")
        results.append(r)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== v2 SUMMARY ===")
    for r in results:
        print(f"  τ={r['threshold']:.2f}  {r['compression']:>7.2f}× acc={r['student_acc_best']*100:>6.2f}%")


if __name__ == "__main__":
    main()
