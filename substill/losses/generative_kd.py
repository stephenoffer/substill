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


def adaptive_skew_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    tau: float = 1.0,
    alpha_min: float = 0.1,
    alpha_max: float = 0.9,
    direction: str = "student_to_mix",
    mask: Tensor | None = None,
    temperature: float = 1.0,
    return_alpha: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Skew-KL with per-token alpha derived from the teacher/student entropy gap.

    Per-token alpha::

        alpha_t = clamp(sigmoid(tau * (H(p_t) - H(p_s))), alpha_min, alpha_max)

    High teacher entropy (uncertain teacher) → alpha → 1: lean teacher.
    Low teacher entropy with confident student disagreement → alpha → 0: lean student.

    The skewed target is ``alpha * p_t + (1 - alpha) * p_s`` (per token).
    The KL direction follows :func:`skew_kl`.

    Parameters
    ----------
    tau : float
        Sharpness of the entropy-gap → alpha mapping. Default 1.0 nat^-1.
    alpha_min, alpha_max : float
        Hard floor and ceiling on the per-token alpha (prevents collapse).
    return_alpha : bool
        If True, also return the per-token alpha tensor for diagnostics.
    """
    if not 0.0 < alpha_min < alpha_max < 1.0:
        raise ValueError(
            f"need 0 < alpha_min < alpha_max < 1, got {alpha_min}, {alpha_max}"
        )
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_p = t_logp.exp()
    s_p = s_logp.exp()

    # Per-token entropies in nats.
    H_t = -(t_p * t_logp).sum(dim=-1)
    H_s = -(s_p * s_logp).sum(dim=-1)
    raw_alpha = torch.sigmoid(tau * (H_t - H_s))
    alpha_t = raw_alpha.clamp(alpha_min, alpha_max).unsqueeze(-1)  # (..., 1)

    mix = alpha_t * t_p + (1.0 - alpha_t) * s_p
    log_mix = mix.clamp_min(1e-12).log()

    if direction == "student_to_mix":
        per_tok = (s_p * (s_logp - log_mix)).sum(dim=-1)
    elif direction == "teacher_to_mix":
        per_tok = (t_p * (t_logp - log_mix)).sum(dim=-1)
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    out = _masked_mean(per_tok, mask) * (temperature ** 2)
    if return_alpha:
        return out, alpha_t.squeeze(-1)
    return out


def unified_token_weights(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 1.0,
    normalise: bool = True,
) -> Tensor:
    """Per-token weights for unified KD/task loss reweighting.

    Score per token::

        w_t = H(p_t) * 0.5 * sum_i |p_t[i] - p_s[i]|

    The first factor (teacher entropy) emphasises tokens where the teacher
    has signal beyond the argmax. The second factor (TV distance) emphasises
    tokens where teacher and student disagree. Their product is the weight.

    Returns:
    -------
    Tensor with shape ``logits.shape[:-1]`` (per-token weights).

    If ``normalise=True``, weights are normalised so they average to 1 across
    the batch (preserving the overall scale of any subsequent loss).
    """
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_p = t_logp.exp()
    s_p = s_logp.exp()
    H_t = -(t_p * t_logp).sum(dim=-1)
    tv = 0.5 * (t_p - s_p).abs().sum(dim=-1)
    w = H_t * tv
    if normalise:
        denom = w.mean().clamp_min(1e-8)
        w = w / denom
    return w


class PlateauDetector:
    """Detect when an EMA-smoothed loss has stopped decreasing.

    Used to trigger schedule transitions (on-policy ramp, sequence-level loss
    activation, QAT). The motivation: fixed step-count phase boundaries are
    fragile to changes in batch size, learning rate, or model size; plateau
    detection adapts.

    Algorithm:
        ema_t = decay * ema_{t-1} + (1 - decay) * loss_t
        slope_t = (ema_{t} - ema_{t - window}) / window

        Trigger plateau if |slope_t| < tolerance for ``patience`` consecutive
        observations after a minimum step count.

    Parameters
    ----------
    decay : float
        EMA decay (default 0.99).
    window : int
        Number of EMA samples to compute the slope over.
    tolerance : float
        Slope threshold below which we consider it a plateau.
    patience : int
        Consecutive plateau observations needed before ``triggered()`` returns True.
    min_step : int
        Minimum step count before plateau detection can fire (warm-up).
    """

    def __init__(
        self,
        *,
        decay: float = 0.99,
        window: int = 50,
        tolerance: float = 1e-4,
        patience: int = 10,
        min_step: int = 100,
    ):
        self.decay = decay
        self.window = window
        self.tolerance = tolerance
        self.patience = patience
        self.min_step = min_step

        self._ema: float | None = None
        self._history: list[float] = []
        self._step = 0
        self._consecutive = 0
        self._triggered = False

    def update(self, loss: float) -> bool:
        """Record a loss observation; return True iff plateau is triggered now.

        True is returned only on the transition from not-triggered to triggered.
        """
        self._step += 1
        if self._ema is None:
            self._ema = float(loss)
        else:
            self._ema = self.decay * self._ema + (1.0 - self.decay) * float(loss)
        self._history.append(self._ema)

        if self._step < self.min_step or len(self._history) < self.window + 1:
            return False
        slope = (self._history[-1] - self._history[-1 - self.window]) / self.window
        if abs(slope) < self.tolerance:
            self._consecutive += 1
        else:
            self._consecutive = 0

        new_trigger = self._consecutive >= self.patience and not self._triggered
        if new_trigger:
            self._triggered = True
        return new_trigger

    def triggered(self) -> bool:
        return self._triggered

    def state(self) -> dict:
        """Diagnostic snapshot of the detector state."""
        slope = None
        if len(self._history) >= self.window + 1:
            slope = (self._history[-1] - self._history[-1 - self.window]) / self.window
        return {
            "step": self._step,
            "ema": self._ema,
            "slope": slope,
            "consecutive": self._consecutive,
            "triggered": self._triggered,
        }


__all__ = [
    "forward_kl",
    "reverse_kl",
    "skew_kl",
    "adaptive_skew_kl",
    "contrastive_response_loss",
    "unified_token_weights",
    "PlateauDetector",
]
