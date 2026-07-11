"""Block-diagonal per-head sparse correction.

The absorbed-init weight ``W_S = V_out^T W_T V_in`` lives entirely in a low-rank
subspace defined by the bases. Outlier weights — channels that contribute
heavily to the output but along directions orthogonal to the retained subspace
— are *unrecoverable* by any V choice. Real LLM weights are heavy-tailed
(Dettmers/SmoothQuant etc.); a small but important set of channels falls in
this regime.

We add a structured-sparse residual to the absorbed weight::

    W_S = V_out^T W_T V_in + BlockDiag(S_1, …, S_H)

where each ``S_h ∈ R^{d_h × d_h}`` is dense within one attention head's
sub-space, and zero across heads. Cost: ``H · d_h² = d · d_h`` parameters per
linear, where ``d`` is the student hidden size and ``d_h = d/H`` the head
dimension. For a 2048-dim, 16-head Llama-3-1B-shape student, that's
``2048 * 128 = 262K`` extra params per attention/FFN linear — fully orthogonal
to the V structure, easy to ablate, cheap.

The block-diagonal structure (vs. unstructured sparsity) is principled: heavy-
tailed outliers in transformer weights cluster *within* attention heads
(Dettmers et al., Bondarenko et al.), not across heads. A per-head dense block
captures the heaviest residual directions per head exactly, while staying
budget-friendly.

Initialization: zeros. The student starts with W_S = absorbed weight only;
the sparse correction grows during distillation training as needed. To bias
training toward using the correction, callers can pre-fill S_h with the
top-d_h singular components of the per-head residual ``W_T - V_out V_out^T W_T V_in V_in^T``
restricted to head h.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BlockDiagonalCorrection(nn.Module):
    """Per-head block-diagonal residual ``S = BlockDiag(S_1, …, S_H)``.

    Stored as a single dense tensor of shape ``(num_heads, d_head, d_head)``
    rather than a true block-diagonal matrix; this is faster on GPU and the
    forward path applies it head-wise.

    ``forward(x)`` computes::

        x : (..., num_heads * d_head)         input
        x_h : (..., num_heads, d_head)         reshape
        y_h : (..., num_heads, d_head) = einsum('...hi,hij->...hj', x_h, S)
        y : (..., num_heads * d_head)          flatten back

    For per-head FFN intermediate dimensions where ``num_heads`` doesn't
    naturally apply, set ``num_heads=1`` and ``d_head=intermediate_size``;
    the correction degenerates to a single dense matrix (full-rank residual).

    Parameters
    ----------
    num_heads : int
    d_head : int
    init : str
        ``"zero"`` (default) starts the correction at zero; the absorbed weight
        is the entire student weight at init. ``"random"`` initialises with
        small Gaussian noise (rarely useful; mostly for ablation).
    """

    def __init__(
        self,
        num_heads: int,
        d_head: int,
        *,
        init: str = "zero",
    ):
        super().__init__()
        if num_heads < 1 or d_head < 1:
            raise ValueError(
                f"num_heads and d_head must be >= 1, got {num_heads}, {d_head}"
            )
        self.num_heads = num_heads
        self.d_head = d_head
        self.weight = nn.Parameter(torch.empty(num_heads, d_head, d_head))
        self._init(init)

    def _init(self, init: str) -> None:
        with torch.no_grad():
            if init == "zero":
                self.weight.zero_()
            elif init == "random":
                nn.init.normal_(self.weight, std=0.02)
            else:
                raise ValueError(f"unknown init {init!r}")

    def forward(self, x: Tensor) -> Tensor:
        d_in = self.num_heads * self.d_head
        if x.shape[-1] != d_in:
            raise ValueError(
                f"BlockDiagonalCorrection: expected last dim {d_in}, got {x.shape[-1]}"
            )
        leading = x.shape[:-1]
        x_h = x.view(*leading, self.num_heads, self.d_head)
        # Per-head matmul: y_h[h, i] = sum_j x_h[h, j] * S[h, j, i]
        y_h = torch.einsum("...hi,hij->...hj", x_h, self.weight)
        return y_h.reshape(*leading, d_in)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, d_head={self.d_head}"

    @torch.no_grad()
    def warm_init_from_residual(
        self,
        teacher_weight: Tensor,
        absorbed_weight: Tensor,
        *,
        layout: str = "linear",
    ) -> None:
        """Optional: initialise blocks from the per-head residual of the absorbed init.

        ``teacher_weight``: original teacher linear weight (output × input layout).
        ``absorbed_weight``: ``V_out^T W_T V_in`` already projected to student space.

        The residual ``R = W_T - V_out V_out^T W_T V_in V_in^T`` lives in the
        complement of the retained subspace. Restricting R to one head's
        in/out coordinates and projecting that into the student head's coords
        gives a non-trivial warm start for ``S_h``. We approximate by reshaping
        the *student-space* residual ``W_T_proj - V_out^T W_T V_in`` per head;
        callers supply ``W_T_proj`` (same shape as absorbed_weight).
        """
        if teacher_weight.shape != absorbed_weight.shape:
            # The intent is to call this with absorbed_weight already in student
            # space and a same-shape "ideal" teacher reconstruction. On a shape
            # mismatch the residual is reset to zero.
            self.weight.zero_()
            return
        residual = teacher_weight - absorbed_weight
        if layout == "linear":
            # residual shape (out, in). Reshape into (num_heads, d_head, num_heads, d_head)
            # and take the diagonal-block elements.
            out_dim, in_dim = residual.shape
            if out_dim != self.num_heads * self.d_head or in_dim != self.num_heads * self.d_head:
                self.weight.zero_()
                return
            r = residual.view(self.num_heads, self.d_head, self.num_heads, self.d_head)
            # Diagonal blocks: r[h, :, h, :].
            for h in range(self.num_heads):
                self.weight[h].copy_(r[h, :, h, :])
        else:
            self.weight.zero_()

    def num_extra_params(self) -> int:
        return int(self.weight.numel())


class CorrectedLinear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` that adds a block-diagonal correction.

    ``y = x @ W^T + b + BlockDiag-correction(x)``

    This is what the student's attention/FFN linears become after the
    block-diagonal correction is applied. The base linear's weight is the
    absorbed-init projection; the
    correction is trained in addition.

    Construct with the same args as ``nn.Linear``, plus the head structure.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_heads: int,
        d_head: int,
        bias: bool = True,
        correction_init: str = "zero",
    ):
        super().__init__()
        if num_heads * d_head != in_features:
            raise ValueError(
                f"num_heads * d_head = {num_heads * d_head} must equal in_features = {in_features}"
            )
        if num_heads * d_head != out_features:
            raise ValueError(
                f"num_heads * d_head = {num_heads * d_head} must equal "
                f"out_features = {out_features} "
                f"(use a separate correction module for non-square linears)"
            )
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.correction = BlockDiagonalCorrection(num_heads, d_head, init=correction_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.correction(x)


__all__ = ["BlockDiagonalCorrection", "CorrectedLinear"]
