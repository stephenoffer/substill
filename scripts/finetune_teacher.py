#!/usr/bin/env python3
"""Adapt a torchvision ResNet to CIFAR-10 to serve as a distillation teacher.

Torchvision ResNets ship with an ImageNet-style 7x7 stride-2 stem and
a 1000-class classifier. For CIFAR-10 (32x32 inputs, 10 classes) we
swap the stem for a 3x3 stride-1 conv, drop the maxpool, and replace
the classifier head, then fine-tune. The resulting checkpoint feeds
into ``asd.profile``.

Usage::

    python scripts/finetune_teacher.py --epochs 8 --output outputs/teacher.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.data.cifar10 import get_cifar10_loaders


def adapt_resnet_for_cifar(resnet: nn.Module, num_classes: int = 10) -> nn.Module:
    """Replace the 7x7 stem with a 3x3 stride-1 conv, drop the maxpool,
    and replace the classifier head. Works on any torchvision ResNet."""
    first_out = resnet.conv1.out_channels
    resnet.conv1 = nn.Conv2d(3, first_out, 3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    return resnet


def train_epoch(model, loader, opt, device):
    model.train()
    correct = total = 0
    running_loss = 0.0
    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += float(loss.item())
        correct += int(logits.argmax(-1).eq(y).sum().item())
        total += int(y.size(0))
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100 * correct / max(total, 1):.1f}%")
    return {"train_acc": correct / max(total, 1),
            "train_loss": running_loss / max(len(loader), 1)}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int(model(x).argmax(-1).eq(y).sum().item())
        total += int(y.size(0))
    return {"test_acc": correct / max(total, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet50",
                    choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", default="outputs/teacher.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"device: {device}")
    print("loading CIFAR-10...")
    loaders = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation="standard",
    )

    from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnet34, ResNet34_Weights,
        resnet50, ResNet50_Weights,
        resnet101, ResNet101_Weights,
    )
    constructors = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
        "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
    }
    ctor, weights = constructors[args.model]
    model = adapt_resnet_for_cifar(ctor(weights=weights)).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    for epoch in range(args.epochs):
        tr = train_epoch(model, loaders["train"], opt, device)
        te = evaluate(model, loaders["test"], device)
        sched.step()
        print(f"epoch {epoch:2d} | train_acc {tr['train_acc']*100:5.2f}% | "
              f"test_acc {te['test_acc']*100:5.2f}%")
        if te["test_acc"] > best:
            best = te["test_acc"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nbest test acc: {best*100:.2f}%")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
