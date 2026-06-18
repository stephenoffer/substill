"""Absorbed weight initialization: ``W_s ~ V_out^T W_T V_in``.

Given a teacher linear with weight ``W_T (d_out, d_in)`` and orthonormal
bases ``V_in (d_in, k_in)`` and ``V_out (d_out, k_out)`` (from the
branch profiles of the linear's input and output branches), we
initialize the compressed student linear with weight::

    W_s = V_out^T @ W_T @ V_in          in R^{k_out x k_in}

This is the second novelty claim: the same teacher profile that
supplies the subspace-loss targets also supplies a principled initial
student. When ``k_in == d_in`` and ``k_out == d_out``, ``W_s`` equals
``W_T`` up to orthogonal rotations, so the full-rank path is a sanity
check.

Biases use only the output projection::

    b_s = V_out^T @ b_T                 in R^{k_out}

Special cases:

- HuggingFace GPT-2 ``Conv1D`` stores weight as ``(d_in, d_out)`` and
  expects ``x @ W + b`` — flipped w.r.t. ``nn.Linear``. This module
  handles both; use :func:`absorbed_linear_init` for either.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

WeightLayout = Literal["linear", "conv1d_gpt2"]


def _infer_layout(module: nn.Module) -> WeightLayout:
    """Detect whether a module is a standard Linear or HF GPT-2 Conv1D.

    HF Conv1D has a ``nf`` attribute and weight shape ``(d_in, d_out)``.
    """
    if hasattr(module, "nf") and not isinstance(module, nn.Linear):
        return "conv1d_gpt2"
    return "linear"


def absorbed_weight(
    W_teacher: Tensor,
    V_in: Tensor | None,
    V_out: Tensor | None,
    *,
    layout: WeightLayout = "linear",
) -> Tensor:
    """Compute ``V_out^T @ W_T @ V_in`` (or the layout-adjusted form).

    Parameters
    ----------
    W_teacher
        Teacher weight. For ``layout="linear"`` shape is ``(d_out, d_in)``;
        for ``layout="conv1d_gpt2"`` shape is ``(d_in, d_out)``.
    V_in
        ``(d_in, k_in)`` orthonormal basis on the input. Pass ``None``
        to keep the input dim unchanged.
    V_out
        ``(d_out, k_out)`` orthonormal basis on the output. Pass
        ``None`` to keep the output dim unchanged.
    """
    if layout == "linear":
        W = W_teacher
        if V_in is not None:
            W = W @ V_in  # (d_out, k_in)
        if V_out is not None:
            W = V_out.T @ W  # (k_out, k_in)
        return W
    if layout == "conv1d_gpt2":
        # W: (d_in, d_out). x @ W = (d_out); compressed equivalent:
        # x @ V_in @ (V_in^T W V_out) -> (k_out)
        W = W_teacher
        if V_in is not None:
            W = V_in.T @ W  # (k_in, d_out)
        if V_out is not None:
            W = W @ V_out  # (k_in, k_out)
        return W
    raise ValueError(f"unknown layout: {layout!r}")


def absorbed_bias(
    b_teacher: Tensor | None,
    V_out: Tensor | None,
) -> Tensor | None:
    """Project a teacher bias into the absorbed output basis ``V_out``."""
    if b_teacher is None:
        return None
    if V_out is None:
        return b_teacher.clone()
    return V_out.T @ b_teacher


@torch.no_grad()
def absorbed_linear_init(
    teacher_module: nn.Module,
    student_module: nn.Module,
    V_in: Tensor | None,
    V_out: Tensor | None,
) -> None:
    """Fill ``student_module``'s weights with the absorbed projection.

    Both modules must be the same kind (``nn.Linear`` or HF ``Conv1D``)
    and must have weights compatible with the passed bases:

    - ``V_in`` shape ``(d_in, k_in)`` where ``d_in`` is the teacher's
      input dim and ``k_in`` is the student's input dim.
    - ``V_out`` shape ``(d_out, k_out)`` where ``d_out`` is the teacher's
      output dim and ``k_out`` is the student's output dim.
    """
    layout_t = _infer_layout(teacher_module)
    layout_s = _infer_layout(student_module)
    if layout_t != layout_s:
        raise ValueError(
            f"teacher and student module layouts differ: {layout_t!r} vs {layout_s!r}"
        )
    W_t = teacher_module.weight.detach()
    b_t = getattr(teacher_module, "bias", None)
    b_t = b_t.detach() if b_t is not None else None

    # Move bases to teacher-weight device/dtype for matmul compatibility.
    if V_in is not None:
        V_in = V_in.to(device=W_t.device, dtype=W_t.dtype)
    if V_out is not None:
        V_out = V_out.to(device=W_t.device, dtype=W_t.dtype)

    W_s = absorbed_weight(W_t, V_in, V_out, layout=layout_t)
    b_s = absorbed_bias(b_t, V_out)

    # Copy onto student in-place (cross-device safe).
    if W_s.shape != student_module.weight.shape:
        raise ValueError(
            f"absorbed weight shape {W_s.shape} does not match student weight "
            f"{student_module.weight.shape}"
        )
    student_module.weight.data.copy_(
        W_s.to(student_module.weight.dtype).to(student_module.weight.device)
    )

    if b_s is not None and getattr(student_module, "bias", None) is not None:
        if b_s.shape != student_module.bias.shape:
            raise ValueError(
                f"absorbed bias shape {b_s.shape} does not match student bias "
                f"{student_module.bias.shape}"
            )
        student_module.bias.data.copy_(
            b_s.to(student_module.bias.dtype).to(student_module.bias.device)
        )


__all__ = [
    "absorbed_weight",
    "absorbed_bias",
    "absorbed_linear_init",
    "WeightLayout",
]
