#!/usr/bin/env python3
"""Dense threshold sweep + multi-seed runs for the Pareto curve.

More points and seeds make the compression/accuracy curve convincingly smooth.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--teacher-weights", default=None)
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Random seeds for student initialization; multiple → multi-seed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ft-epochs", type=int, default=6)
    parser.add_argument("--output-dir", default="outputs/dense_sweep")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    num_classes = 100 if args.dataset == "cifar100" else 10

    # Data loaders
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

    # Teacher: reuse existing weights if provided
    teacher = TeacherWrapper(
        profiles=None, cifar_stem=cfg.teacher.cifar_stem,
        pretrained=(args.teacher_weights is None), num_classes=num_classes,
        freeze=False, model=args.model,
    )
    tw_path = args.teacher_weights or os.path.join(args.output_dir, "teacher.pt")
    if os.path.exists(tw_path):
        print(f"Loading teacher weights from {tw_path}")
        state = torch.load(tw_path, map_location="cpu", weights_only=True)
        teacher.backbone.load_state_dict(state)
        teacher.freeze()
        teacher.to(device)
    else:
        print(f"Fine-tuning teacher for {args.ft_epochs} epochs...")
        teacher.to(device)
        m = teacher.finetune(loaders["train"], loaders["test"],
                             epochs=args.ft_epochs, lr=cfg.teacher.finetune_lr, device=device)
        torch.save(teacher.backbone.state_dict(), tw_path)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    teacher_acc = eval_accuracy(teacher, loaders["test"], device)
    print(f"Teacher ({args.model}): {teacher_params:,} params, acc={teacher_acc*100:.2f}%")

    # Capture covariances ONCE (shared across all thresholds)
    cov_mode = cfg.profiling.get("covariance_mode", "per_pixel")
    print(f"Capturing covariances (mode={cov_mode}) — shared across thresholds")
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
        for seed in args.seeds:
            print(f"\n=== τ={threshold}, seed={seed} ===")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

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
            stage_groups: dict[int, list] = {}
            for p in profiles:
                stage_groups.setdefault(p.total_channels, []).append(p)
            teacher_ranks = [stage_groups[ch][-1].effective_rank for ch in sorted(stage_groups)]

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
                subspace_mode=cfg.training.subspace_mode,
                sv_weighting=cfg.training.sv_weighting,
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
            loss_sched = LossWeightScheduler(
                warmup_epochs=min(cfg.training.gamma_warmup_epochs, max(1, args.epochs // 3)),
            )

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
                "threshold": threshold,
                "seed": seed,
                "model": args.model,
                "dataset": args.dataset,
                "stage_widths": stage_widths,
                "teacher_ranks": teacher_ranks,
                "student_params": student.count_parameters(),
                "teacher_params": teacher_params,
                "teacher_acc": teacher_acc,
                "student_acc_best": best["eval"]["accuracy"],
                "student_acc_final": history[-1]["eval"]["accuracy"],
                "compression": teacher_params / student.count_parameters(),
                "acc_drop_pp": (teacher_acc - best["eval"]["accuracy"]) * 100,
                "epochs": args.epochs,
                "elapsed_sec": elapsed,
            }
            print(f"τ={threshold}, seed={seed}: compression={r['compression']:.2f}x, "
                  f"acc={r['student_acc_best']*100:.2f}% (drop {r['acc_drop_pp']:.2f} pp), "
                  f"{elapsed:.0f}s")
            results.append(r)
            # Save incrementally
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Teacher: {teacher_params:,} params, acc={teacher_acc*100:.2f}%")
    for r in sorted(results, key=lambda x: (x["threshold"], x["seed"])):
        print(f"  τ={r['threshold']:.2f}  seed={r['seed']}  "
              f"{r['compression']:>7.2f}×  acc={r['student_acc_best']*100:>6.2f}%")


if __name__ == "__main__":
    main()
