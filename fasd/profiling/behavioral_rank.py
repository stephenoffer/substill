"""Behavioral rank selection — F-ASD's headline novelty.

For a branch with orthonormal basis ``V in R^{C x C}`` (columns sorted
by descending eigenvalue), we choose the smallest rank ``k`` such that
patching the branch activation ``x`` with ``P_k x = V[:, :k] V[:, :k]^T x``
keeps the teacher's next-token log-probability nearly unchanged on a
calibration set.

Concretely the score for a candidate ``k`` is::

    R(k) = sum_t w_t * KL(p_T(.|x_t) || p_T^(k)(.|x_t)) / sum_t w_t

where ``w_t`` are optional per-token weights
(:mod:`fasd.profiling.token_weighting`). The returned rank is the
smallest ``k`` with ``R(k) <= tol``.

Supports two searches:

``"bisect"`` (default):
    Monotone bisection on ``[1, max_rank]``. Assumes ``R`` is roughly
    non-increasing in ``k`` (the usual case when eigenvalues are sorted
    descending). The returned :attr:`kl_curve` has one entry per
    tested ``k``.

``"linear"``:
    Walk a geometric ladder of ``k`` values and report the crossing
    point. Preferred when the user wants the full ``kl_curve`` for
    diagnostics.
"""

from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..autodetect import BranchSpec
from .activation_capture import _get_module


Search = Literal["bisect", "linear"]


def _default_ladder(max_rank: int) -> list[int]:
    """Geometric ladder of candidate ranks, always ending in max_rank."""
    if max_rank <= 0:
        return []
    out: list[int] = []
    k = 1
    while k < max_rank:
        out.append(k)
        k = max(k + 1, k * 2)
    out.append(int(max_rank))
    return out


@torch.no_grad()
def _teacher_logits(teacher: nn.Module, batch) -> Tensor:
    out = teacher(**batch) if isinstance(batch, dict) else teacher(batch)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return out[0]
    raise TypeError(f"teacher output has no .logits and is not a tensor: {type(out)}")


def _weighted_kl(
    ref_logits: Tensor,
    pat_logits: Tensor,
    weights: Tensor | None,
) -> float:
    """Token-weighted KL(ref || pat) averaged over tokens."""
    if ref_logits.shape != pat_logits.shape:
        raise ValueError(
            f"logits shape mismatch: {ref_logits.shape} vs {pat_logits.shape}"
        )
    logp_ref = F.log_softmax(ref_logits, dim=-1)
    logp_pat = F.log_softmax(pat_logits, dim=-1)
    p_ref = logp_ref.exp()
    per_tok = (p_ref * (logp_ref - logp_pat)).sum(dim=-1)
    if weights is None:
        return float(per_tok.mean().item())
    w = weights.to(per_tok.device, dtype=per_tok.dtype)
    if w.shape != per_tok.shape:
        raise ValueError(
            f"token weights shape {w.shape} does not match per-token shape {per_tok.shape}"
        )
    denom = w.sum().clamp_min(1e-8)
    return float((per_tok * w).sum().item() / denom.item())


def _patch_hook(spec: BranchSpec, V_k: Tensor):
    """Build a forward hook that projects the branch activation through V_k V_k^T."""

    def hook(mod, inputs, output):
        V = V_k.to(device=output[0].device if isinstance(output, (tuple, list)) else output.device,
                  dtype=output[0].dtype if isinstance(output, (tuple, list)) else output.dtype)

        if isinstance(output, Tensor):
            out = output
            is_tuple = False
            tup_rest = None
        elif isinstance(output, (tuple, list)):
            out = output[0]
            is_tuple = True
            tup_rest = tuple(output[1:])
        else:
            return output

        # Restrict to the sliced channels, project, write back.
        if spec.slice is not None:
            a, b = spec.slice
            head = out[..., :a]
            body = out[..., a:b]
            tail = out[..., b:]
            proj = body @ V @ V.T
            new_out = torch.cat([head, proj, tail], dim=-1)
        else:
            new_out = out @ V @ V.T

        if is_tuple:
            return (new_out,) + tup_rest
        return new_out

    return hook


def _run_teacher(teacher: nn.Module, batch, device) -> Tensor:
    if isinstance(batch, dict):
        b = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
        return _teacher_logits(teacher, b)
    if isinstance(batch, Tensor):
        return _teacher_logits(teacher, batch.to(device))
    if isinstance(batch, (tuple, list)):
        moved = tuple(b.to(device) if isinstance(b, Tensor) else b for b in batch)
        out = teacher(*moved)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, Tensor):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        raise TypeError(f"teacher output has no .logits: {type(out)}")
    return _teacher_logits(teacher, batch)


@torch.no_grad()
def _score_rank(
    teacher: nn.Module,
    spec: BranchSpec,
    basis: Tensor,
    k: int,
    calib_batches: list,
    device,
    ref_logits_per_batch: list[Tensor],
    weights_per_batch: list[Tensor | None] | None,
) -> float:
    """Average weighted KL across calib_batches under patching with V[:, :k]."""
    V_k = basis[:, :k].contiguous()
    module = _get_module(teacher, spec.module_path)
    hook = module.register_forward_hook(_patch_hook(spec, V_k))
    try:
        vals: list[float] = []
        for idx, batch in enumerate(calib_batches):
            pat_logits = _run_teacher(teacher, batch, device)
            w = weights_per_batch[idx] if weights_per_batch is not None else None
            vals.append(_weighted_kl(ref_logits_per_batch[idx], pat_logits, w))
        return sum(vals) / max(1, len(vals))
    finally:
        hook.remove()


@torch.no_grad()
def choose_behavioral_rank(
    teacher: nn.Module,
    spec: BranchSpec,
    basis: Tensor,
    calib_batches: Iterable,
    *,
    tol: float = 0.02,
    search: Search = "bisect",
    token_weights: Iterable[Tensor | None] | None = None,
    max_rank: int | None = None,
    min_rank: int = 1,
    device: str | torch.device | None = None,
) -> tuple[int, list[tuple[int, float]]]:
    """Choose the smallest behavioral rank meeting the KL tolerance.

    Parameters
    ----------
    teacher
        Frozen teacher model. Will be switched to ``eval()`` mode.
    spec
        :class:`BranchSpec` identifying the branch (module + slice + kind).
    basis
        ``(C_branch, C_branch)`` orthonormal eigenvector matrix with
        columns sorted by descending eigenvalue. (Rank is measured
        against the sliced channel dimension, not the full module
        output.)
    calib_batches
        Iterable of batches (dict / tensor / tuple) ready to pass to
        ``teacher(...)``.
    tol
        Maximum acceptable token-weighted KL between the unpatched and
        patched teacher.
    search
        ``"bisect"`` (default) or ``"linear"`` (geometric ladder).
    token_weights
        One weight tensor per calibration batch, matching the logits
        ``(B, T)`` shape. Use :mod:`fasd.profiling.token_weighting`.
    max_rank
        Upper bound on ``k``. Defaults to ``basis.shape[1]``.
    min_rank
        Lower bound on ``k`` (must be >= 1).
    device
        Device on which to run the teacher. Defaults to the teacher's
        current device.

    Returns
    -------
    (k, kl_curve)
        ``k`` is the chosen rank. ``kl_curve`` is a list of
        ``(rank_tested, kl)`` pairs, in the order they were evaluated,
        for diagnostics.
    """
    if min_rank < 1:
        raise ValueError(f"min_rank must be >= 1, got {min_rank}")
    if basis.dim() != 2:
        raise ValueError(f"basis must be (C, C) or (C, K), got {tuple(basis.shape)}")
    C = basis.shape[0]
    K_full = basis.shape[1]
    if max_rank is None:
        max_rank = K_full
    max_rank = int(min(max_rank, K_full))
    if max_rank < min_rank:
        raise ValueError(
            f"max_rank ({max_rank}) must be >= min_rank ({min_rank})"
        )

    teacher.eval()
    if device is None:
        device = next(teacher.parameters()).device
    teacher.to(device)

    calib_list = list(calib_batches)
    weights_list: list[Tensor | None] | None
    if token_weights is not None:
        weights_list = list(token_weights)
        if len(weights_list) != len(calib_list):
            raise ValueError(
                "token_weights must have the same length as calib_batches"
            )
    else:
        weights_list = None

    # Cache reference logits once.
    ref_logits: list[Tensor] = []
    for batch in calib_list:
        ref_logits.append(_run_teacher(teacher, batch, device).detach())

    def eval_k(k: int) -> float:
        return _score_rank(
            teacher, spec, basis, k, calib_list, device, ref_logits, weights_list
        )

    kl_curve: list[tuple[int, float]] = []

    if search == "linear":
        ladder = _default_ladder(max_rank)
        ladder = [k for k in ladder if k >= min_rank]
        if not ladder or ladder[-1] != max_rank:
            ladder.append(max_rank)
        chosen = max_rank
        for k in ladder:
            score = eval_k(k)
            kl_curve.append((k, score))
            if score <= tol:
                chosen = k
                break
        return chosen, kl_curve

    # Bisect on [lo, hi].
    lo, hi = min_rank, max_rank
    score_hi = eval_k(hi)
    kl_curve.append((hi, score_hi))
    if score_hi > tol:
        # Full rank itself doesn't meet tolerance — the teacher *needs* more
        # than `max_rank` directions to preserve its output. Return max_rank
        # as the best feasible answer and flag it via the KL curve so the
        # caller can decide whether to raise `max_rank`.
        return max_rank, kl_curve
    score_lo = eval_k(lo)
    kl_curve.append((lo, score_lo))
    if score_lo <= tol:
        return lo, kl_curve
    chosen = hi
    # Early-stop: if the KL between adjacent evaluations is below
    # `tol/10`, further bisection is just noise — treat as "converged".
    plateau_tol = tol / 10.0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        score_mid = eval_k(mid)
        kl_curve.append((mid, score_mid))
        if score_mid <= tol:
            chosen = mid
            hi = mid
        else:
            lo = mid
        # Plateau detection: if the last two evals are within plateau_tol
        # we're in the noise floor.
        if (
            len(kl_curve) >= 2
            and abs(kl_curve[-1][1] - kl_curve[-2][1]) < plateau_tol
            and kl_curve[-1][1] <= tol * 1.5
        ):
            break
    return chosen, kl_curve


__all__ = ["choose_behavioral_rank", "Search"]
