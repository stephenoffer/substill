"""KD-driven channel selection for ResNet bottlenecks — the vision analog of LRD.

`substill/compression/restricted.py` trains a *rotation* of a transformer's residual stream
against the KD loss, because RMSNorm makes that stream rotation-equivariant. A ReLU-CNN has
no such stream: a BN+ReLU sits between every pair of convs, and ReLU does not commute with a
dense basis rotation. So the residual-stream rotation LRD relies on is unavailable here.

What *is* available is the same **principle** one level down. `substill/vision/resnet.py` keeps a
bottleneck's inner channels by **variance** — a surrogate, exactly the kind
`docs/init_findings.md` §2/§9 found never beats an arbitrary choice on transformers. The
un-surrogated move is to choose the channels **against the KD loss, through the whole
network**: put a soft gate on every inner channel, train the gates (and the weights) against
KD under a channel budget, then harden to the same width the variance baseline uses. Selection
commutes with ReLU exactly, so hardening is function-preserving on the kept channels — the CNN
counterpart of `RestrictedLlama.fold()`.

This is to variance-selection what LRD is to PCA: the compression decision is made by the
distillation objective itself, not by a proxy computed before training.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from ..losses.generative_kd import forward_kl, skew_kl
from .resnet import (
    _bottlenecks,
    build_resnet_student,
)

__all__ = ["GatedBottleneck", "install_channel_gates", "distill_gated_then_harden"]


class GatedBottleneck(nn.Module):
    """Wrap a Bottleneck so its two inner activations are multiplied by soft channel gates.

    A gate is ``sigmoid(logit / tau)`` per inner channel, applied right after ``bn1``/``bn2``
    (before ``conv2``/``conv3`` read them). At ``logit → +∞`` the gate is 1 (channel kept); a
    small ``logit`` closes it. Because the gate multiplies the *post-ReLU-able* activation and
    selection commutes with ReLU, a hardened gate (0/1) is exactly equivalent to dropping the
    channel from the narrow student.
    """

    def __init__(self, block: nn.Module, tau: float = 1.0):
        super().__init__()
        self.block = block
        self.tau = tau
        w = int(block.conv1.out_channels)
        # New parameters must land on the block's device, or a CUDA model gets CPU gates and
        # the forward hits a device mismatch (only visible on GPU, not in a CPU smoke test).
        dev = block.conv1.weight.device
        # start fully open: sigmoid(3) ≈ 0.95, so the student begins ≈ the teacher block
        self.g1 = nn.Parameter(torch.full((w,), 3.0, device=dev))
        self.g2 = nn.Parameter(torch.full((w,), 3.0, device=dev))

    def gates(self) -> tuple[Tensor, Tensor]:
        return torch.sigmoid(self.g1 / self.tau), torch.sigmoid(self.g2 / self.tau)

    def forward(self, x: Tensor) -> Tensor:
        b = self.block
        s1, s2 = self.gates()
        identity = b.downsample(x) if b.downsample is not None else x
        out = b.relu(b.bn1(b.conv1(x))) * s1[None, :, None, None]
        out = b.relu(b.bn2(b.conv2(out))) * s2[None, :, None, None]
        out = b.bn3(b.conv3(out))
        return b.relu(out + identity)


def install_channel_gates(model: nn.Module, tau: float = 1.0) -> dict[str, GatedBottleneck]:
    """Replace every Bottleneck in ``model`` with a `GatedBottleneck`, in place.

    Returns ``{name: GatedBottleneck}`` so the caller can read the learned gate values.
    """
    gated: dict[str, GatedBottleneck] = {}
    for name, blk in list(_bottlenecks(model)):
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        g = GatedBottleneck(blk, tau=tau)
        setattr(parent, leaf, g)
        gated[name] = g
    return gated


def _budget_loss(gated: dict[str, GatedBottleneck], keep_frac: float) -> Tensor:
    """Push the mean gate openness toward ``keep_frac`` (a soft L1 channel budget)."""
    tot = torch.zeros((), device=next(iter(gated.values())).g1.device)
    n = 0
    for g in gated.values():
        s1, s2 = g.gates()
        tot = tot + s1.sum() + s2.sum()
        n += s1.numel() + s2.numel()
    return (tot / max(1, n) - keep_frac).clamp_min(0.0)


@torch.no_grad()
def _harden_scores(gated: dict[str, GatedBottleneck]) -> dict[str, dict[str, Tensor]]:
    """Turn the learned gate logits into the score dict `build_resnet_student` consumes.

    Higher gate ⇒ more important, so the top-``width_ratio`` channels by gate value are the
    ones the hardened student keeps. This is the KD-chosen selection replacing the
    variance-chosen one.
    """
    return {name: {"conv1": g.g1.detach().clone(), "conv2": g.g2.detach().clone()}
            for name, g in gated.items()}


def distill_gated_then_harden(
    teacher: nn.Module,
    train_loader,
    *,
    width_ratio: float = 0.5,
    gate_steps: int = 500,
    finetune_steps: int = 1000,
    lr: float = 1e-3,
    gate_lr: float = 5e-2,
    budget_weight: float = 30.0,
    temperature: float = 4.0,
    generative_kd: str = "forward_kl",
    tau: float = 1.0,
    val_loader=None,
    device=None,
) -> dict:
    """Phase 1: learn channel gates against KD under a budget. Phase 2: harden + finetune.

    Returns the hardened student's top-1 and the per-block channel choice, so it can be
    compared head-to-head with the variance-selection student at the **same width**.
    """
    if device is None:
        device = next(teacher.parameters()).device
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- phase 1: gated full-width student, gates + weights trained against KD ---
    gstudent = copy.deepcopy(teacher).to(device)
    gated = install_channel_gates(gstudent, tau=tau)
    gstudent.train()
    gate_params = [p for g in gated.values() for p in (g.g1, g.g2)]
    gate_ids = {id(p) for p in gate_params}
    weight_params = [p for p in gstudent.parameters() if id(p) not in gate_ids]
    opt = torch.optim.AdamW(
        [{"params": weight_params, "lr": lr},
         {"params": gate_params, "lr": gate_lr}], weight_decay=0.0)
    kd_fn = skew_kl if generative_kd == "skew_kl" else forward_kl

    step = 0
    while step < gate_steps:
        for batch in train_loader:
            if step >= gate_steps:
                break
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            with torch.no_grad():
                tl = teacher(x)
            kd = kd_fn(gstudent(x), tl, temperature=temperature)
            loss = kd + budget_weight * _budget_loss(gated, width_ratio)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

    scores = _harden_scores(gated)

    # --- phase 2: harden to the target width, absorb, finetune ---
    from .resnet import distill_classifier
    student, info = build_resnet_student(copy.deepcopy(teacher), scores,
                                         width_ratio=width_ratio, absorbed_init=True)
    out = distill_classifier(teacher, student, train_loader, total_steps=finetune_steps,
                             lr=lr, temperature=temperature, generative_kd=generative_kd,
                             val_loader=val_loader, device=device)
    return {
        "params": int(sum(p.numel() for p in student.parameters())),
        "final_top1": out.get("student_top1"),
        "gate_choice": {n: {"conv1": s["conv1"], "conv2": s["conv2"]}
                        for n, s in scores.items()},
        "info": info,
    }
