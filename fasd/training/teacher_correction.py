"""Short teacher adaptation pass before profiling.

When the distillation corpus differs from the teacher's pretraining
distribution, recalibrating the teacher by a few hundred LM-loss steps
on the distillation corpus improves downstream profile quality. This
is the Minitron "teacher correction" step.

Not called by default. The driver enables it when
``teacher_correction_steps > 0``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _logits_and_labels(batch, output) -> tuple[Tensor, Tensor]:
    """Pull ``(logits, labels)`` from a Huggingface-style batch + output."""
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, Tensor):
        logits = output
    elif isinstance(output, (tuple, list)):
        logits = output[0]
    else:
        raise TypeError(f"unsupported output type: {type(output)}")

    # Derive labels from batch.
    if isinstance(batch, dict):
        if "labels" in batch:
            labels = batch["labels"]
        elif "input_ids" in batch:
            labels = batch["input_ids"]
        else:
            raise KeyError("batch must contain 'labels' or 'input_ids'")
    elif isinstance(batch, Tensor):
        labels = batch
    elif isinstance(batch, (tuple, list)) and len(batch) > 0 and isinstance(batch[0], Tensor):
        labels = batch[0]
    else:
        raise TypeError(f"unsupported batch type: {type(batch)}")
    return logits, labels


def _shift_lm_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Next-token cross-entropy, HuggingFace style."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def correct_teacher(
    teacher: nn.Module,
    loader,
    *,
    steps: int = 500,
    lr: float = 5e-6,
    grad_accum: int = 1,
    device: str | torch.device = "cpu",
    clip_grad: float | None = 1.0,
) -> dict[str, float]:
    """Run a short LM-loss fine-tune of ``teacher`` on ``loader``.

    The teacher's parameters are updated in place. Returns a small
    dict with ``{"initial_loss", "final_loss", "steps"}``.
    """
    teacher.train()
    teacher.to(device)
    opt = torch.optim.AdamW([p for p in teacher.parameters() if p.requires_grad], lr=lr)
    opt.zero_grad()

    initial_loss = float("nan")
    last_loss = float("nan")
    seen = 0
    micro = 0
    for batch in loader:
        if seen >= steps:
            break
        if isinstance(batch, dict):
            batch = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
            out = teacher(**batch)
        elif isinstance(batch, Tensor):
            batch = batch.to(device)
            out = teacher(batch)
        else:
            batch = tuple(b.to(device) if isinstance(b, Tensor) else b for b in batch)
            out = teacher(*batch)

        logits, labels = _logits_and_labels(batch, out)
        loss = _shift_lm_loss(logits, labels)
        if not torch.isfinite(loss):
            continue
        if seen == 0 and micro == 0:
            initial_loss = float(loss.detach().item())
        (loss / max(1, grad_accum)).backward()
        micro += 1
        if micro % max(1, grad_accum) == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), clip_grad)
            opt.step()
            opt.zero_grad()
            seen += 1
        last_loss = float(loss.detach().item())
    teacher.eval()
    return {"initial_loss": initial_loss, "final_loss": last_loss, "steps": seen}


__all__ = ["correct_teacher"]
