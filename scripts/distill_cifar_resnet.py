#!/usr/bin/env python3
"""End-to-end CIFAR-10 distillation example using the `asd` library.

    python scripts/finetune_teacher.py --output outputs/teacher.pt
    python scripts/distill_cifar_resnet.py \\
        --teacher outputs/teacher.pt --epochs 20

Outputs a trained student checkpoint and prints final accuracy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asd
from asd.data.cifar10 import get_cifar10_loaders


def adapt_resnet_for_cifar(resnet, num_classes=10):
    import torch.nn as nn
    first_out = resnet.conv1.out_channels
    resnet.conv1 = nn.Conv2d(3, first_out, 3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True,
                    help="Path to fine-tuned teacher state_dict (see finetune_teacher.py)")
    ap.add_argument("--teacher-arch", default="resnet50",
                    choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    ap.add_argument("--source", default="delta", choices=["output", "delta"])
    ap.add_argument("--noise-model", default="mp", choices=["eps", "mp"])
    ap.add_argument("--objective", default="gram",
                    choices=["coord_mse", "gram", "cka"])
    ap.add_argument("--arch-multiplier", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0, help="task CE weight")
    ap.add_argument("--beta", type=float, default=0.5, help="subspace weight")
    ap.add_argument("--delta", type=float, default=1.0, help="KD weight")
    ap.add_argument("--kd-T", type=float, default=4.0, help="KD temperature")
    ap.add_argument("--calib-samples", type=int, default=5000)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output-dir", default="outputs/distill_cifar")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    loaders = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, augmentation="standard",
    )
    calib = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, augmentation="none",
        calibration_samples=args.calib_samples,
    )["calibration"]

    # Teacher
    from torchvision.models import resnet18, resnet34, resnet50, resnet101
    ctor = {"resnet18": resnet18, "resnet34": resnet34,
            "resnet50": resnet50, "resnet101": resnet101}[args.teacher_arch]
    teacher = adapt_resnet_for_cifar(ctor(weights=None)).to(device)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"teacher ({args.teacher_arch}): "
          f"{sum(p.numel() for p in teacher.parameters()):,} params")

    # Profile
    print(f"\nprofiling teacher (source={args.source}, noise_model={args.noise_model})...")
    t0 = time.time()
    profile = asd.profile(
        teacher, calib,
        source=args.source,
        noise_model=args.noise_model,
        n_effective=args.calib_samples,
    )
    print(f"  profiled in {time.time()-t0:.1f}s; effective ranks: {profile.effective_ranks()}")

    # Student
    student = asd.build_student(
        teacher, profile,
        arch_multiplier=args.arch_multiplier,
        num_classes=10, stem_type="cifar",
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"student: {s_params:,} params ({t_params/s_params:.2f}× compression)")

    # Loss (with projectors as parameters)
    loss_fn = asd.SubspaceLoss(profile, objective=args.objective).to(device)

    # Run one dummy forward to build projectors before we construct the optimizer.
    x0, _ = next(iter(loaders["train"]))
    x0 = x0.to(device)
    with asd.capture(teacher, profile) as t_cap0:
        teacher(x0)
    with asd.capture(student, profile) as s_cap0:
        student(x0)
    _ = loss_fn(s_cap0.values(), t_cap0.values())

    opt = torch.optim.SGD(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Training loop
    best = 0.0
    for epoch in range(args.epochs):
        student.train()
        running = {"task": 0.0, "sub": 0.0, "kd": 0.0, "total": 0.0, "n": 0}
        t0 = time.time()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            with asd.capture(teacher, profile) as t_cap:
                with torch.no_grad():
                    t_logits = teacher(x)
            with asd.capture(student, profile) as s_cap:
                s_logits = student(x)

            task = F.cross_entropy(s_logits, y)
            sub = loss_fn(s_cap.values(), t_cap.values())
            T = args.kd_T
            kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            total = args.alpha * task + args.beta * sub + args.delta * kd

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(loss_fn.parameters()), 5.0,
            )
            opt.step()
            for k in ("task", "sub", "kd", "total"):
                running[k] += float(locals()[k if k != "total" else "total"].detach().item())
            running["n"] += 1
        sched.step()

        student.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in loaders["test"]:
                x, y = x.to(device), y.to(device)
                correct += int(student(x).argmax(-1).eq(y).sum().item())
                n += int(y.size(0))
        acc = correct / max(n, 1)
        best = max(best, acc)
        avg = {k: v / max(running["n"], 1) for k, v in running.items() if k != "n"}
        print(f"epoch {epoch:2d} | task {avg['task']:.4f} sub {avg['sub']:.4f} "
              f"kd {avg['kd']:.4f} | acc {acc*100:5.2f}% (best {best*100:.2f}%) | "
              f"{time.time()-t0:.1f}s")

    print(f"\nbest test acc: {best*100:.2f}%")
    torch.save({"student": student.state_dict(),
                "stage_widths": student.stage_widths,
                "objective": args.objective},
               os.path.join(args.output_dir, "student.pt"))
    profile.save(os.path.join(args.output_dir, "teacher.profile"))


if __name__ == "__main__":
    main()
