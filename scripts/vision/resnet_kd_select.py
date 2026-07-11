"""ResNet50 / CIFAR-10: is KD-driven channel selection better than variance selection?

The vision counterpart of the LRD question. `substill/vision/resnet.py` keeps a bottleneck's
inner channels by **variance** — the same surrogate `docs/init_findings.md` found never
beats an arbitrary choice on transformers. `substill/vision/gated.py` chooses them **against
the KD loss, through the whole network** (soft channel gates under a budget, then hardened
to the identical width). ReLU forbids the residual-stream *rotation* LRD uses on
transformers, but selection commutes with ReLU exactly, so the hardened student is a
function-preserving compression — the vision analog.

Three arms at the **same** compressed width:

  random     narrow student, random init            (the naive floor)
  variance   narrow student, variance-selected      (substill.vision baseline)
  kd_select  narrow student, KD-gate-selected       (ours)

Uses the CIFAR-native ResNet50 teacher on shared storage (3x3 stem, 32x32 input) when
present, so it runs at CIFAR resolution rather than the 224-upsample the ImageNet path needs.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def cifar_resnet50(num_classes=10):
    """torchvision ResNet50 with a CIFAR stem (3x3 stride-1 conv, no maxpool)."""
    import torchvision
    m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def load(args):
    import torchvision
    import torchvision.transforms as T
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tf_test = T.Compose([T.ToTensor()])
    root = args.data_root
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf_train)
    va = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf_test)
    train = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                                        num_workers=4, drop_last=True)
    val = torch.utils.data.DataLoader(va, batch_size=256, num_workers=4)

    teacher = cifar_resnet50(10)
    ck = Path(args.teacher_ckpt)
    if ck.exists():
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        teacher.load_state_dict(sd)
        print(f"loaded CIFAR teacher {ck}", flush=True)
    else:
        raise FileNotFoundError(f"no teacher at {ck}; train one or pass --teacher-ckpt")
    return teacher.eval(), train, val


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/mnt/shared_storage/cifar10")
    p.add_argument("--teacher-ckpt",
                   default="/mnt/shared_storage/asd/teacher_resnet50_cifar10.pt")
    p.add_argument("--width-ratio", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--gate-steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate-lr", type=float, default=5e-2)
    p.add_argument("--budget-weight", type=float, default=30.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--arms", nargs="+", default=["random", "variance", "kd_select"])
    p.add_argument("--output", default="runs/resnet_kd_select.json")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_arm(arm, teacher, train, val, args, seed, device):
    from substill.vision.resnet import (
        build_resnet_student,
        channel_variance_scores,
        distill_classifier,
    )
    torch.manual_seed(seed)
    t0 = time.time()

    if arm == "kd_select":
        from substill.vision.gated import distill_gated_then_harden
        # phase-1 gate budget spends `gate_steps`; give it the rest as finetune so the total
        # KD budget matches the other arms' `steps`.
        out = distill_gated_then_harden(
            teacher, train, width_ratio=args.width_ratio, gate_steps=args.gate_steps,
            finetune_steps=max(1, args.steps - args.gate_steps), lr=args.lr,
            gate_lr=args.gate_lr, budget_weight=args.budget_weight,
            temperature=args.temperature, val_loader=val, device=device)
        return {"arm": arm, "seed": seed, "params": out["params"],
                    "final_top1": out["final_top1"], "secs": time.time() - t0}

    scores = channel_variance_scores(teacher, train, n_batches=args.calib_batches,
                                     device=device)
    student, info = build_resnet_student(copy.deepcopy(teacher), scores,
                                         width_ratio=args.width_ratio,
                                         absorbed_init=(arm != "random"))
    out = distill_classifier(teacher, student, train, total_steps=args.steps, lr=args.lr,
                             temperature=args.temperature, val_loader=val, device=device)
    return {"arm": arm, "seed": seed, "params": int(sum(p.numel() for p in student.parameters())),
                "final_top1": out["student_top1"], "secs": time.time() - t0}


def main():
    args = parse_args()
    dev = args.device
    teacher, train, val = load(args)
    teacher.to(dev)
    from substill.vision.resnet import top1_accuracy
    t_top1 = top1_accuracy(teacher, val, device=dev)
    print(f"teacher CIFAR-10 top1 = {t_top1:.4f}\n", flush=True)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm(arm, teacher, train, val, args, seed, dev)
            rows.append(r)
            print(f"{arm:<10} seed={seed}  params={r['params']/1e6:.2f}M  "
                  f"top1={r['final_top1']:.4f}  ({r['secs']:.0f}s)", flush=True)
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"teacher_top1": t_top1, "width_ratio": args.width_ratio, "args": vars(args),
         "rows": rows}, indent=2))

    import statistics as s
    print("\n" + "=" * 50)
    print(f"teacher top1 = {t_top1:.4f}   width_ratio = {args.width_ratio}")
    for arm in args.arms:
        v = [r["final_top1"] for r in rows if r["arm"] == arm]
        sd = s.stdev(v) if len(v) > 1 else 0.0
        print(f"  {arm:<10} top1 = {s.mean(v):.4f} +/- {sd:.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
