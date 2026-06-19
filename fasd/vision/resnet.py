"""Activation-subspace compression + distillation for ResNet-family CNNs.

This is the non-LLM arm of FASD: the same activation-subspace idea (keep the directions
a trained network actually uses; absorb the teacher's weights into them; distil) applied
to convolutional vision models instead of transformer decoders. The core machinery is
already architecture-agnostic — covariance/channel statistics, the conv2d absorbed-init
(``V_out^T W V_in`` lifted over the kernel, see
:func:`fasd.compression.absorbed_init.absorbed_weight`), and the KD losses (which operate
on ``(B, num_classes)`` logits unchanged). This module supplies the CNN-specific glue.

**Compression target.** We narrow each ``Bottleneck`` block's *inner* channels
(``conv1``-out / ``conv2`` / ``conv3``-in) while keeping the block's input/output
(residual-stream) width fixed. This is the convolutional analogue of compressing a
transformer FFN's intermediate dimension: it needs no change to the downsample shortcut
or the residual add, so blocks compress independently with no cross-block coupling.

**Why channel *selection* (not PCA rotation).** A BN+ReLU sits between the convs, and
ReLU is element-wise — it does not commute with an arbitrary basis rotation. So we select
the top-variance inner channels (one-hot bases) exactly as the transformer path does for
the FFN intermediate dim. At full width the student reproduces the teacher bit-for-bit.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..compression.absorbed_init import absorbed_linear_init
from ..losses.generative_kd import forward_kl, skew_kl


def _is_bottleneck(m: nn.Module) -> bool:
    """Duck-type a torchvision ``Bottleneck`` (avoid importing torchvision here)."""
    return all(hasattr(m, a) for a in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3"))


def _bottlenecks(model: nn.Module):
    """Yield ``(name, block)`` for every Bottleneck in the model."""
    for name, m in model.named_modules():
        if _is_bottleneck(m):
            yield name, m


def _onehot(idx: Tensor, n: int) -> Tensor:
    """``(n, k)`` one-hot channel-selection basis from sorted indices ``idx``."""
    E = torch.zeros(n, idx.numel())
    E[idx, torch.arange(idx.numel())] = 1.0
    return E


@torch.no_grad()
def channel_variance_scores(
    model: nn.Module,
    loader: Iterable,
    *,
    n_batches: int = 8,
    device=None,
) -> dict[str, dict[str, Tensor]]:
    """Per-channel activation energy for each Bottleneck's ``conv1``/``conv2`` outputs.

    Returns ``{block_name: {"conv1": scores(width,), "conv2": scores(width,)}}`` where a
    higher score means the channel carries more activation variance (the importance signal
    the builder selects on). Cheap: a single forward over ``n_batches`` calibration batches.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval().to(device)
    blocks = list(_bottlenecks(model))
    acc: dict[str, dict[str, list]] = {
        name: {"conv1": [0.0, 0], "conv2": [0.0, 0]} for name, _ in blocks
    }
    caps: dict[str, dict[str, Tensor]] = {name: {} for name, _ in blocks}
    hooks = []
    for name, blk in blocks:
        hooks.append(blk.conv1.register_forward_hook(
            lambda m, i, o, n=name: caps[n].__setitem__("conv1", o.detach())))
        hooks.append(blk.conv2.register_forward_hook(
            lambda m, i, o, n=name: caps[n].__setitem__("conv2", o.detach())))
    try:
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(x.to(device))
            for name, _ in blocks:
                for key in ("conv1", "conv2"):
                    a = caps[name].get(key)
                    if a is None:
                        continue
                    # sum of squares per channel over (B, H, W)
                    s = (a.float() ** 2).sum(dim=(0, 2, 3))
                    acc[name][key][0] = acc[name][key][0] + s
                    acc[name][key][1] += a.shape[0] * a.shape[2] * a.shape[3]
    finally:
        for h in hooks:
            h.remove()
    return {
        name: {k: (acc[name][k][0] / max(1, acc[name][k][1])) for k in ("conv1", "conv2")}
        for name, _ in blocks
    }


@torch.no_grad()
def _copy_bn_slice(src: nn.Module, dst: nn.Module, idx: Tensor) -> None:
    """Slice a BatchNorm2d's per-channel buffers/params onto ``idx`` channels."""
    for attr in ("weight", "bias", "running_mean", "running_var"):
        s = getattr(src, attr, None)
        d = getattr(dst, attr, None)
        if s is not None and d is not None:
            d.data.copy_(s.detach()[idx].to(d.dtype))
    if getattr(src, "num_batches_tracked", None) is not None:
        dst.num_batches_tracked.data.copy_(src.num_batches_tracked.detach())


@torch.no_grad()
def build_resnet_student(
    teacher: nn.Module,
    scores: dict[str, dict[str, Tensor]],
    *,
    width_ratio: float = 0.5,
    absorbed_init: bool = True,
) -> tuple[nn.Module, dict]:
    """Build a channel-narrowed ResNet student of ``teacher`` and absorb its weights.

    Each Bottleneck's inner width ``w`` is reduced to ``round(w * width_ratio)`` by keeping
    the top-variance channels (per ``scores``); ``conv1/conv2/conv3`` and ``bn1/bn2`` are
    rebuilt at the narrowed width and initialized with the conv2d absorbed projection. The
    block input/output channels, downsample, and residual add are untouched. At
    ``width_ratio == 1.0`` the student equals the teacher exactly.

    Returns ``(student, info)`` where ``info`` records per-block original/new widths.
    """
    student = copy.deepcopy(teacher)
    info: dict[str, dict] = {}
    s_blocks = dict(_bottlenecks(student))
    for name, t_blk in _bottlenecks(teacher):
        s_blk = s_blocks[name]
        width = int(t_blk.conv1.out_channels)
        s_width = max(1, int(round(width * width_ratio)))
        info[name] = {"width": width, "s_width": s_width}
        if s_width == width:
            continue  # no compression for this block
        if int(getattr(t_blk.conv2, "groups", 1)) != 1:
            raise NotImplementedError(f"{name}: grouped conv2 (ResNeXt) not supported yet")

        sc = scores.get(name, {})
        idx1 = torch.argsort(sc.get("conv1", torch.ones(width)), descending=True)[:s_width]
        idx2 = torch.argsort(sc.get("conv2", torch.ones(width)), descending=True)[:s_width]
        idx1, idx2 = torch.sort(idx1).values, torch.sort(idx2).values

        c1, c2, c3 = t_blk.conv1, t_blk.conv2, t_blk.conv3
        new_c1 = nn.Conv2d(c1.in_channels, s_width, c1.kernel_size, stride=c1.stride,
                           padding=c1.padding, dilation=c1.dilation, bias=c1.bias is not None)
        new_bn1 = nn.BatchNorm2d(s_width)
        new_c2 = nn.Conv2d(s_width, s_width, c2.kernel_size, stride=c2.stride,
                           padding=c2.padding, dilation=c2.dilation, bias=c2.bias is not None)
        new_bn2 = nn.BatchNorm2d(s_width)
        new_c3 = nn.Conv2d(s_width, c3.out_channels, c3.kernel_size, stride=c3.stride,
                           padding=c3.padding, dilation=c3.dilation, bias=c3.bias is not None)

        if absorbed_init:
            E1, E2 = _onehot(idx1, width), _onehot(idx2, width)
            absorbed_linear_init(c1, new_c1, V_in=None, V_out=E1)
            _copy_bn_slice(t_blk.bn1, new_bn1, idx1)
            absorbed_linear_init(c2, new_c2, V_in=E1, V_out=E2)
            _copy_bn_slice(t_blk.bn2, new_bn2, idx2)
            absorbed_linear_init(c3, new_c3, V_in=E2, V_out=None)

        s_blk.conv1, s_blk.bn1 = new_c1, new_bn1
        s_blk.conv2, s_blk.bn2 = new_c2, new_bn2
        s_blk.conv3 = new_c3
    # The student is fully trainable even if ``teacher`` was frozen (we deepcopy it).
    student.requires_grad_(True)
    return student, info


def _classifier_logits(out) -> Tensor:
    return out.logits if hasattr(out, "logits") else out


@torch.no_grad()
def top1_accuracy(model: nn.Module, loader: Iterable, *, device=None,
                  max_batches: int | None = None) -> float:
    if device is None:
        device = next(model.parameters()).device
    model.eval().to(device)
    correct = total = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        pred = _classifier_logits(model(x)).argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    return correct / max(1, total)


def distill_classifier(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: Iterable,
    *,
    total_steps: int = 200,
    lr: float = 1e-3,
    temperature: float = 4.0,
    generative_kd: str = "forward_kl",
    ce_weight: float = 0.1,
    val_loader: Iterable | None = None,
    device=None,
    log_every: int = 50,
) -> dict:
    """Distil a classifier student from a (frozen) teacher on class logits.

    The KD objective is :func:`forward_kl` / :func:`skew_kl` over ``(B, num_classes)``
    logits — the same losses the LLM path uses, no sequence dimension. An optional
    cross-entropy term on the labels (``ce_weight``) stabilizes early training. No
    on-policy/rollout stages (those are language-generation specific).
    """
    if device is None:
        device = next(student.parameters()).device
    teacher.to(device).eval()
    student.to(device).train()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)
    history: list[dict] = []
    step = 0
    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            y = batch[1].to(device) if isinstance(batch, (tuple, list)) and len(batch) > 1 else None
            with torch.no_grad():
                t_logits = _classifier_logits(teacher(x))
            s_logits = _classifier_logits(student(x))
            if generative_kd == "skew_kl":
                kd = skew_kl(s_logits, t_logits, temperature=temperature)
            else:
                kd = forward_kl(s_logits, t_logits, temperature=temperature)
            ce = (F.cross_entropy(s_logits, y) if (y is not None and ce_weight > 0)
                  else torch.zeros((), device=device))
            loss = kd + ce_weight * ce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            history.append({"step": step, "kd": float(kd.item()), "ce": float(ce.item())})
            if log_every and step % log_every == 0:
                print(f"[fasd.vision] step={step}/{total_steps} kd={float(kd.item()):.4f}",
                      flush=True)
            step += 1
    out = {"history": history}
    if val_loader is not None:
        out["student_top1"] = top1_accuracy(student, val_loader, device=device)
        out["teacher_top1"] = top1_accuracy(teacher, val_loader, device=device)
    return out


__all__ = [
    "channel_variance_scores",
    "build_resnet_student",
    "distill_classifier",
    "top1_accuracy",
]
