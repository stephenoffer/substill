"""ResNet50 activation-subspace distillation — the vision analogue of the GPT-2 ladder.

Builds a channel-narrowed ResNet student of a ResNet50 teacher TWO ways at matched
compression — random-init vs absorbed-init (FASD) — distils both with class-logit KD, and
reports top-1. The absorbed-init vs random-init gap is the vision counterpart of the §1
"concrete win vs naive baseline" we report for GPT-2.

Smoke (no download, CPU, seconds):
    python scripts/resnet50_distill.py --smoke

Real (CIFAR-10, needs torchvision datasets):
    python scripts/resnet50_distill.py --dataset cifar10 --steps 2000 --width-ratio 0.5
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from fasd.vision import (
    build_resnet_student,
    channel_variance_scores,
    distill_classifier,
    top1_accuracy,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny random ResNet + synthetic data")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    p.add_argument("--width-ratio", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--head-warmup-steps", type=int, default=300,
                   help="train the teacher's new classification head (backbone frozen) so "
                        "the ImageNet teacher is a real classifier on the target dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="runs/resnet50_distill.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load(args):
    if args.smoke:
        from torchvision.models.resnet import Bottleneck, ResNet
        torch.manual_seed(0)
        teacher = ResNet(Bottleneck, [1, 1, 1, 1], num_classes=10).eval()
        train = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(8)]
        return teacher, train, train[:2], 10

    import torchvision
    import torchvision.transforms as T
    teacher = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
    # Adapt the 1000-class head to CIFAR-10 for a self-contained demo.
    teacher.fc = torch.nn.Linear(teacher.fc.in_features, 10)
    tf = T.Compose([T.Resize(224), T.ToTensor()])
    root = "./data"
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    va = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    train = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val = torch.utils.data.DataLoader(va, batch_size=args.batch_size)
    return teacher, train, val, 10


def warmup_head(teacher, train, steps, device):
    """Train only ``teacher.fc`` (backbone frozen) so the ImageNet teacher becomes a real
    classifier on the target dataset — otherwise the random head makes the distillation
    comparison degenerate."""
    if steps <= 0:
        return
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in teacher.fc.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(teacher.fc.parameters(), lr=1e-3)
    teacher.train()
    step = 0
    while step < steps:
        for x, y in train:
            if step >= steps:
                break
            logits = teacher(x.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0:
                print(f"[resnet] head-warmup step {step}/{steps} ce={loss.item():.4f}", flush=True)
            step += 1
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    teacher, train, val, _ = load(args)
    teacher.to(args.device)
    if not args.smoke:
        warmup_head(teacher, train, args.head_warmup_steps, args.device)

    scores = channel_variance_scores(teacher, train, n_batches=args.calib_batches,
                                     device=args.device)
    results = {}
    for variant in ("random_init", "absorbed_init"):
        student, info = build_resnet_student(
            copy.deepcopy(teacher), scores, width_ratio=args.width_ratio,
            absorbed_init=(variant == "absorbed_init"))
        n_params = sum(p.numel() for p in student.parameters())
        init_top1 = top1_accuracy(student, val, device=args.device, max_batches=16)
        out = distill_classifier(teacher, student, train, total_steps=args.steps,
                                 lr=args.lr, temperature=args.temperature,
                                 val_loader=val, device=args.device)
        results[variant] = {
            "params": int(n_params),
            "init_top1": init_top1,
            "final_top1": out.get("student_top1"),
        }
        print(f"[resnet] {variant}: params={n_params/1e6:.2f}M init_top1={init_top1:.4f} "
              f"final_top1={results[variant]['final_top1']}")

    summary = {
        "dataset": "smoke" if args.smoke else args.dataset,
        "width_ratio": args.width_ratio, "steps": args.steps,
        "teacher_top1": top1_accuracy(teacher, val, device=args.device, max_batches=16),
        "results": results,
        "absorbed_minus_random_top1": (
            (results["absorbed_init"]["final_top1"] or 0)
            - (results["random_init"]["final_top1"] or 0)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[resnet] absorbed - random (final top1) = "
          f"{summary['absorbed_minus_random_top1']:+.4f}; wrote {args.output}")


if __name__ == "__main__":
    main()
