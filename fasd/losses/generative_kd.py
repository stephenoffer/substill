"""Generative-KD losses.

- :func:`forward_kl` — the classical teacher-distribution-anchored KL
  used in Hinton-style distillation.
- :func:`reverse_kl` — student-distribution-anchored KL (mode-seeking),
  argued by MiniLLM to be better for generative LMs.
- :func:`skew_kl` — DistiLLM's interpolated target
  ``alpha * p_t + (1 - alpha) * p_s``.
- :func:`contrastive_response_loss` — DistiLLM-2-style margin loss that
  encourages higher student log-probability on teacher-generated
  responses than on student-generated responses.

All return token-averaged scalar tensors. A boolean or float mask
``(B, T)`` skips / downweights specific tokens (e.g. padding, prompt).
Use ``temperature`` to soften distributions like classic KD.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _masked_mean(per_tok: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return per_tok.mean()
    if mask.shape != per_tok.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match token shape {per_tok.shape}"
        )
    m = mask.to(per_tok.dtype)
    denom = m.sum().clamp_min(1e-8)
    return (per_tok * m).sum() / denom


def forward_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    mask: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """KL(teacher || student) averaged over tokens, in nats.

    At ``temperature`` ``T``, logits are divided by ``T`` before the
    softmax; the result is scaled by ``T^2`` (standard distillation
    compensation).
    """
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_p = t_logp.exp()
    per_tok = (t_p * (t_logp - s_logp)).sum(dim=-1)
    return _masked_mean(per_tok, mask) * (temperature ** 2)


def reverse_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    mask: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """KL(student || teacher) averaged over tokens, in nats."""
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    s_p = s_logp.exp()
    per_tok = (s_p * (s_logp - t_logp)).sum(dim=-1)
    return _masked_mean(per_tok, mask) * (temperature ** 2)


def skew_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    alpha: float = 0.1,
    direction: str = "student_to_mix",
    mask: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """DistiLLM's skew KL.

    The skewed target is ``alpha * p_t + (1 - alpha) * p_s`` by
    default. Two directions are supported:

    ``"student_to_mix"`` (default)
        ``KL(p_s || alpha * p_t + (1 - alpha) * p_s)``. Mode-seeking,
        bounded.
    ``"teacher_to_mix"``
        ``KL(p_t || alpha * p_t + (1 - alpha) * p_s)``. Mass-covering.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_p = t_logp.exp()
    s_p = s_logp.exp()
    mix = alpha * t_p + (1.0 - alpha) * s_p
    log_mix = mix.clamp_min(1e-12).log()

    if direction == "student_to_mix":
        per_tok = (s_p * (s_logp - log_mix)).sum(dim=-1)
    elif direction == "teacher_to_mix":
        per_tok = (t_p * (t_logp - log_mix)).sum(dim=-1)
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    return _masked_mean(per_tok, mask) * (temperature ** 2)


def _sequence_logprob(
    logits: Tensor,
    tokens: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Sum of log-prob of ``tokens`` under ``logits``, per sequence.

    ``logits`` is ``(B, T, V)`` produced by a forward pass; ``tokens``
    is ``(B, T)``, already shifted so index ``t`` predicts position
    ``t`` in ``tokens``. ``mask`` zeros out padding or prompt tokens.
    """
    if logits.shape[:2] != tokens.shape:
        raise ValueError(
            f"logits {logits.shape[:2]} vs tokens {tokens.shape} mismatch"
        )
    logp = F.log_softmax(logits, dim=-1)
    gather = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        if mask.shape != gather.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} vs {gather.shape}")
        gather = gather * mask.to(gather.dtype)
    return gather.sum(dim=-1)


def contrastive_response_loss(
    student_logits_on_teacher: Tensor,
    student_logits_on_student: Tensor,
    teacher_tokens: Tensor,
    student_tokens: Tensor,
    *,
    margin: float = 0.5,
    teacher_mask: Tensor | None = None,
    student_mask: Tensor | None = None,
) -> Tensor:
    """DistiLLM-2-style response contrastive loss.

    Encourages ``log p_student(teacher_tokens) > log p_student(student_tokens) + margin``:

        L = mean( max(0, margin - (logp_teacher - logp_student)) ).

    Both student forward passes must be computed ahead of time (with
    gradients enabled) — this function only combines them.
    """
    if margin < 0.0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    logp_t = _sequence_logprob(
        student_logits_on_teacher, teacher_tokens, mask=teacher_mask
    )
    logp_s = _sequence_logprob(
        student_logits_on_student, student_tokens, mask=student_mask
    )
    hinge = (margin - (logp_t - logp_s)).clamp_min(0.0)
    return hinge.mean()


__all__ = [
    "forward_kl",
    "reverse_kl",
    "skew_kl",
    "contrastive_response_loss",
]
