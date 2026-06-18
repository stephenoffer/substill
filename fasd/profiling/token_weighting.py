"""Token-level weighting for calibration and behavioral-rank loss.

Produces a ``(B, T)`` weight tensor over tokens that rescales
calibration loss contributions. The choice matters for rank selection:
weighting toward high-entropy or high-disagreement tokens is the
"where distillation is hard" signal, and weighting completion-only
tokens excludes the prompt part that the student doesn't need to
match.

All functions return non-negative weights; the caller is expected to
normalize (e.g. divide by the mean) before use as a KL reweighter.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

Method = Literal["uniform", "entropy", "disagreement", "completion"]


def uniform_weights(shape: tuple[int, ...], *, device=None, dtype=torch.float32) -> Tensor:
    """All-ones weight grid."""
    return torch.ones(shape, device=device, dtype=dtype)


def entropy_weights(teacher_logits: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Per-token teacher entropy in nats.

    High-entropy tokens are ambiguous for the teacher; weighting toward
    them emphasizes tokens where the student has more to learn.
    """
    if teacher_logits.dim() < 2:
        raise ValueError(
            f"teacher_logits must be at least (T, V), got {teacher_logits.shape}"
        )
    logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    p = logp.exp()
    # entropy = -sum p log p  (nats)
    return -(p * logp).sum(dim=-1)


def disagreement_weights(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Per-token teacher→student forward KL as a weight.

    High-disagreement tokens are where the student currently diverges
    from the teacher; weighting them emphasizes "where we're wrong."
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"teacher/student logits must share shape, got "
            f"{teacher_logits.shape} vs {student_logits.shape}"
        )
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_p = t_logp.exp()
    return (t_p * (t_logp - s_logp)).sum(dim=-1)


def completion_mask(
    shape: tuple[int, int],
    prompt_lens: Tensor,
    *,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """1.0 on completion tokens, 0.0 on prompt tokens.

    ``prompt_lens[b]`` is the number of prompt tokens in batch row b —
    positions ``[0, prompt_lens[b])`` are masked out.
    """
    B, T = shape
    if prompt_lens.shape != (B,):
        raise ValueError(
            f"prompt_lens must have shape ({B},), got {tuple(prompt_lens.shape)}"
        )
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    lens = prompt_lens.to(device=device).unsqueeze(1)
    return (idx >= lens).to(dtype)


def compute_weights(
    method: Method,
    *,
    teacher_logits: Tensor | None = None,
    student_logits: Tensor | None = None,
    prompt_lens: Tensor | None = None,
    shape: tuple[int, ...] | None = None,
    device=None,
    dtype=torch.float32,
    normalize: bool = True,
) -> Tensor:
    """Dispatcher. Returns a ``(B, T)`` weight grid.

    If ``normalize=True`` the returned weights are scaled so their mean
    (over the non-zero mask, if one is applied) is 1.0, which keeps
    downstream loss magnitudes stable across methods.
    """
    if method == "uniform":
        if shape is None and teacher_logits is not None:
            shape = teacher_logits.shape[:-1]
        if shape is None:
            raise ValueError("uniform weights require shape= or teacher_logits=")
        w = uniform_weights(shape, device=device, dtype=dtype)
    elif method == "entropy":
        if teacher_logits is None:
            raise ValueError("entropy weights require teacher_logits=")
        w = entropy_weights(teacher_logits).to(dtype=dtype)
    elif method == "disagreement":
        if teacher_logits is None or student_logits is None:
            raise ValueError(
                "disagreement weights require both teacher_logits= and student_logits="
            )
        w = disagreement_weights(teacher_logits, student_logits).to(dtype=dtype)
    elif method == "completion":
        if shape is None and teacher_logits is not None:
            shape = teacher_logits.shape[:-1]
        if shape is None or prompt_lens is None:
            raise ValueError(
                "completion weights require shape= (or teacher_logits=) and prompt_lens="
            )
        w = completion_mask(
            (shape[0], shape[1]), prompt_lens, device=device, dtype=dtype
        )
    else:
        raise ValueError(f"unknown method: {method!r}")

    if normalize:
        nz = w[w > 0]
        if nz.numel() > 0:
            mean = nz.mean()
            if mean > 0:
                w = w / mean
    return w


__all__ = [
    "Method",
    "uniform_weights",
    "entropy_weights",
    "disagreement_weights",
    "completion_mask",
    "compute_weights",
]
