"""FactoredLinear: trainable U_in, U_out, B, S factors (HANDOFF TODO #4 module).

**Status note (2026-05-02)**: This module is *implemented and tested in
isolation* (15 tests in test_fsd_factored_linear.py pass) but is **not yet
integrated into the absorbed-init builder pipeline**. A naive drop-in
replacement of the student's `nn.Linear` modules failed because the forward
chain ``(x @ U_in) @ B^T @ U_out^T`` expects teacher-dim input ``d_in``,
while the absorbed-init student's input has already been compressed to
``k_in``. Two architecturally-different paths to actually use FactoredLinear
in training:

  (a) **Uncompressed student** — replace the student's compressed linears
      with FactoredLinear(d_in=t_d_in, d_out=t_d_out, k_in=s_k_in, k_out=s_k_out).
      The student now has the same input/output dimensions as the teacher
      but with rank-bottlenecked weights. Loses the parameter-count savings
      of compression unless followed by a dimensionality projection layer.

  (b) **Feature-loss-only** — keep the student's compressed linears as plain
      nn.Linear, store U_in/U_out as side parameters on each module. They're
      trained by an auxiliary feature-distillation loss
      ``L_feat = ||U_out^T t_hidden - s_hidden||²`` where t_hidden is the
      teacher's full-dim activation and s_hidden is the student's compressed
      activation. The trainable U_in, U_out parameterize the teacher↔student
      subspace correspondence; they don't participate in the linear forward.

Both options require non-trivial restructuring of the trainer. This is
documented in HANDOFF.md as deferred to user implementation.

**The module itself remains useful**:
  - As a building block for future architectures.
  - As a convenient entry point for experimenting with Stiefel-trainable
    bases on small toy problems.
  - The from_teacher constructor produces a mathematically-correct
    factorization that's verified by the test suite.

Pillar 2 (trainable Stiefel bases) requires that V_in and V_out — the input/
output bases of the absorbed-init weight ``W_S = V_out^T W_T V_in`` — are
*standalone* `nn.Parameter` matrices, so the StiefelAdam optimizer can find
them and keep them on the manifold during training.

The current absorbed-init code in [fasd/builders.py](../builders.py) writes
``V_out^T W_T V_in`` into a plain `nn.Linear.weight`, which collapses the three
factors into one. After the absorb step, V_in and V_out are *unrecoverable*
from the linear's weight — they're not stored anywhere. The RR-Norm Q matrix
(in fasd/util/rr_norm.py) IS already a trainable Stiefel parameter — that's
what the headline experiment uses for Pillar 2. Wiring V_in/V_out as well
needs option (a) or (b) above.

This module provides::

    class FactoredLinear(nn.Module):
        weight_effective = U_out @ B @ U_in.T + S

where:
    U_in  ∈ R^{d_in × k_in}   trainable on Stiefel manifold
    U_out ∈ R^{d_out × k_out} trainable on Stiefel manifold
    B     ∈ R^{k_out × k_in}  trainable on Euclidean (the small core)
    S     ∈ optional sparse-block correction (Pillar 3)

The forward pass computes ``y = x @ weight_effective.T + b``. To avoid
materialising the full weight on every forward (which would defeat the
compression benefit when k_in ≪ d_in), we evaluate via the small-matrix
chain::

    y = ((x @ U_in) @ B.T) @ U_out.T + b

This is O(B · T · d_in · k_in + B · T · k_in · k_out + B · T · k_out · d_out).

Markers ``is_stiefel_q = True`` on U_in and U_out tell `stiefel_param_groups`
to register them in the Stiefel param-group; B and the bias go into the
Euclidean group.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .sparse_block import BlockDiagonalCorrection


class FactoredLinear(nn.Module):
    """Linear with U_in, U_out, B, optional sparse-block S all separately trainable.

    Drop-in replacement for ``nn.Linear(d_in, d_out)``. After init:

        - ``U_in.shape  = (d_in, k_in)``     — input basis, columns orthonormal
        - ``U_out.shape = (d_out, k_out)``   — output basis, columns orthonormal
        - ``B.shape     = (k_out, k_in)``    — small core
        - ``bias.shape  = (d_out,)``         — optional

    Construct directly with ``FactoredLinear(d_in, d_out, k_in, k_out)`` — random
    init places U_in, U_out on Stiefel via QR on Gaussian noise, B as the
    Kaiming-initialised core. To absorb a teacher linear, call
    :meth:`absorb_teacher`.

    Parameters
    ----------
    d_in, d_out : int
        Input and output dimensions.
    k_in, k_out : int
        Latent ranks. Must satisfy ``k_in <= d_in`` and ``k_out <= d_out``.
    bias : bool
    use_sparse_block : bool
        If True, add a `BlockDiagonalCorrection` (Pillar 3) between U_in and U_out.
        Requires ``d_in == d_out`` and ``num_heads`` set.
    num_heads : int | None
        Required if ``use_sparse_block``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k_in: int,
        k_out: int,
        *,
        bias: bool = True,
        use_sparse_block: bool = False,
        num_heads: int | None = None,
    ):
        super().__init__()
        if k_in > d_in or k_out > d_out:
            raise ValueError(
                f"k_in <= d_in and k_out <= d_out required; got "
                f"k_in={k_in}, d_in={d_in}, k_out={k_out}, d_out={d_out}"
            )
        self.d_in = d_in
        self.d_out = d_out
        self.k_in = k_in
        self.k_out = k_out

        # Initialise U_in, U_out on Stiefel via QR.
        with torch.no_grad():
            U_in_raw = torch.randn(d_in, k_in)
            U_in_q, _ = torch.linalg.qr(U_in_raw)
            U_out_raw = torch.randn(d_out, k_out)
            U_out_q, _ = torch.linalg.qr(U_out_raw)

        self.U_in = nn.Parameter(U_in_q)
        self.U_out = nn.Parameter(U_out_q)
        # Marker for stiefel_param_groups.
        self.U_in.is_stiefel = True  # type: ignore[attr-defined]
        self.U_out.is_stiefel = True  # type: ignore[attr-defined]

        # B: Kaiming-initialised core.
        self.B = nn.Parameter(torch.empty(k_out, k_in))
        nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias", None)

        self.correction: BlockDiagonalCorrection | None = None
        if use_sparse_block:
            if d_in != d_out:
                raise ValueError(
                    "use_sparse_block requires d_in == d_out (head structure is square)"
                )
            if num_heads is None or d_in % num_heads != 0:
                raise ValueError(
                    "use_sparse_block requires num_heads dividing "
                    f"d_in={d_in}; got num_heads={num_heads}"
                )
            self.correction = BlockDiagonalCorrection(num_heads, d_in // num_heads, init="zero")

    def forward(self, x: Tensor) -> Tensor:
        # y = ((x U_in) B^T) U_out^T + b ; optionally + correction(x)
        # x: (..., d_in)
        z = x @ self.U_in  # (..., k_in)
        z = z @ self.B.T  # (..., k_out)
        y = z @ self.U_out.T  # (..., d_out)
        if self.correction is not None:
            y = y + self.correction(x)
        if self.bias is not None:
            y = y + self.bias
        return y

    def stiefel_parameters(self) -> list[nn.Parameter]:
        """Return U_in and U_out — the parameters StiefelAdam should manage."""
        return [self.U_in, self.U_out]

    def effective_weight(self) -> Tensor:
        """Materialise the (d_out, d_in) weight as a single tensor.

        Useful for diagnostics, or for converting back to a plain `nn.Linear`
        after training (e.g. for inference deployment). The sparse-block
        correction is included; ``effective_weight() @ x.T + b`` reproduces
        the forward pass exactly (modulo float ordering).
        """
        W = self.U_out @ self.B @ self.U_in.T  # (d_out, d_in)
        if self.correction is not None:
            # The correction block-diag operates per-head. Construct full S matrix.
            H, d_h = self.correction.num_heads, self.correction.d_head
            S = torch.zeros(self.d_out, self.d_in, dtype=W.dtype, device=W.device)
            for h in range(H):
                S[h * d_h:(h + 1) * d_h, h * d_h:(h + 1) * d_h] = self.correction.weight[h]
            W = W + S
        return W

    @classmethod
    def from_teacher(
        cls,
        teacher_linear: nn.Linear,
        V_in: Tensor,
        V_out: Tensor,
        *,
        use_sparse_block: bool = False,
        num_heads: int | None = None,
    ) -> FactoredLinear:
        """Construct a FactoredLinear from absorbed-init bases.

        We set ``U_in = V_in``, ``U_out = V_out``, and the core
        ``B = V_out^T @ W_T @ V_in`` (shape ``(k_out, k_in)``). Then::

            effective_weight() = U_out @ B @ U_in^T
                               = V_out (V_out^T W_T V_in) V_in^T
                               = (V_out V_out^T) W_T (V_in V_in^T)

        i.e. the projection of ``W_T`` onto the subspaces spanned by ``V_out``
        (output) and ``V_in`` (input) — exactly what absorbed init computes. At
        full rank ``V V^T = I`` so this equals ``W_T``; at reduced rank it is the
        activation-subspace projection. Storing the three factors explicitly (vs
        the collapsed ``W_S``) is what lets the Stiefel optimizer train the bases.
        """
        d_out, d_in = teacher_linear.weight.shape
        k_out = V_out.shape[1]
        k_in = V_in.shape[1]
        if V_in.shape[0] != d_in or V_out.shape[0] != d_out:
            raise ValueError(
                f"V_in/V_out shapes don't match teacher linear: "
                f"V_in {V_in.shape} (expected first dim={d_in}), "
                f"V_out {V_out.shape} (expected first dim={d_out})"
            )
        m = cls(
            d_in=d_in, d_out=d_out, k_in=k_in, k_out=k_out,
            bias=teacher_linear.bias is not None,
            use_sparse_block=use_sparse_block, num_heads=num_heads,
        )
        with torch.no_grad():
            m.U_in.data.copy_(V_in)
            m.U_out.data.copy_(V_out)
            # B = V_out^T W_T V_in.
            W_T = teacher_linear.weight.detach().to(V_in.dtype)
            m.B.data.copy_(V_out.T @ W_T @ V_in)
            if m.bias is not None and teacher_linear.bias is not None:
                m.bias.data.copy_(V_out.T @ teacher_linear.bias.detach().to(V_out.dtype))
        return m


class TeacherFactoredLinear(nn.Module):
    """Compressed linear with a FROZEN teacher weight and TRAINABLE Stiefel bases.

    This is the CPSD manifold-training module (Phase 2-MT). Unlike
    :class:`FactoredLinear` — whose ``U_in`` consumes *teacher-dim* input and so
    cannot drop into a compressed student — this module operates on the
    *student-dim* residual stream ``x_S ∈ R^{k_in}`` while keeping the teacher
    weight ``W_T`` frozen, exposing the bases ``V_in, V_out`` as trainable Stiefel
    parameters. The collapsed compressed weight is ``W_S = V_out^T W_T V_in``.

    Forward routes through teacher dim WITHOUT materialising ``W_S``::

        y_S = ((x_S @ V_in^T) @ W_T^T) @ V_out  (+ V_out^T b_T) (+ S(x_S))

    which equals ``x_S @ W_S^T + b_S`` but lets gradients reach ``V_in, V_out``.
    ``StiefelAdam`` keeps them on the manifold (via the ``stiefel_parameters()``
    method, which ``stiefel_param_groups`` discovers).

    - **Inference:** call :meth:`fold` to collapse to a plain ``nn.Linear`` with
      weight ``W_S`` — zero overhead vs a normal compressed linear.
    - **Training cost:** routing through teacher dim costs ~6× the FLOPs of a
      collapsed linear (measured, ``runs/derisk/optim_derisk.py``); the deferral
      reason. Acceptable for circuit-critical edges; use plain linears elsewhere.

    Parameters
    ----------
    W_teacher : Tensor
        Frozen teacher weight ``(d_out, d_in)`` (``nn.Linear`` layout).
    V_in : Tensor
        ``(d_in, k_in)`` orthonormal input basis (trained on Stiefel).
    V_out : Tensor
        ``(d_out, k_out)`` orthonormal output basis (trained on Stiefel).
    b_teacher : Tensor | None
        Optional frozen teacher bias ``(d_out,)``; projected to ``V_out^T b_T``.
    use_sparse_block, num_heads
        Optional :class:`BlockDiagonalCorrection` on the student space (requires
        ``k_in == k_out`` and ``num_heads`` dividing it).
    """

    def __init__(
        self,
        W_teacher: Tensor,
        V_in: Tensor,
        V_out: Tensor,
        b_teacher: Tensor | None = None,
        *,
        layout: str = "linear",
        free_core: bool = False,
        use_sparse_block: bool = False,
        num_heads: int | None = None,
    ):
        super().__init__()
        if layout not in ("linear", "conv1d_gpt2"):
            raise ValueError(f"unknown layout {layout!r}")
        self.layout = layout
        # nn.Linear stores (d_out, d_in); HF GPT-2 Conv1D stores (d_in, d_out).
        if layout == "linear":
            d_out, d_in = W_teacher.shape
        else:
            d_in, d_out = W_teacher.shape
        if V_in.shape[0] != d_in or V_out.shape[0] != d_out:
            raise ValueError(
                f"basis/teacher mismatch ({layout}): W_T {tuple(W_teacher.shape)}, "
                f"V_in {tuple(V_in.shape)} (expected first dim {d_in}), "
                f"V_out {tuple(V_out.shape)} (expected first dim {d_out})"
            )
        self.d_in, self.d_out = d_in, d_out
        self.k_in, self.k_out = V_in.shape[1], V_out.shape[1]

        # Frozen teacher weight/bias as buffers (move with the module, never trained).
        self.register_buffer("W_T", W_teacher.detach().clone())
        if b_teacher is not None:
            self.register_buffer("b_T", b_teacher.detach().clone())
        else:
            self.b_T = None

        self.V_in = nn.Parameter(V_in.detach().clone())
        self.V_out = nn.Parameter(V_out.detach().clone())
        self.V_in.is_stiefel = True  # type: ignore[attr-defined]
        self.V_out.is_stiefel = True  # type: ignore[attr-defined]

        # Optional free Euclidean core correction (zero-init): adds full fitting
        # capacity in the compressed space on top of the basis-rotated frozen teacher,
        # while preserving exact-at-init (B_free=0) and inference compression (folds
        # into W_S). Without it the module can only ROTATE bases, which is too few
        # DOF to fit a KD target — basis rotation refines the subspace, B_free fits.
        # Stored in effective_weight orientation: linear (k_out,k_in), conv1d (k_in,k_out).
        self.B_free: nn.Parameter | None = None
        if free_core:
            shape = (self.k_out, self.k_in) if layout == "linear" else (self.k_in, self.k_out)
            self.B_free = nn.Parameter(torch.zeros(shape))

        self.correction: BlockDiagonalCorrection | None = None
        if use_sparse_block:
            if self.k_in != self.k_out:
                raise ValueError("use_sparse_block requires k_in == k_out (square)")
            if num_heads is None or self.k_in % num_heads != 0:
                raise ValueError(
                    f"use_sparse_block requires num_heads dividing k_in={self.k_in}; "
                    f"got num_heads={num_heads}"
                )
            self.correction = BlockDiagonalCorrection(
                num_heads, self.k_in // num_heads, init="zero"
            )

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., k_in). Route through teacher dim; no W_S materialisation.
        z = x @ self.V_in.T  # (..., d_in)
        # Linear: y = z W_T^T; Conv1D (x @ W): y = z W_T.
        z = z @ self.W_T.T if self.layout == "linear" else z @ self.W_T  # (..., d_out)
        y = z @ self.V_out  # (..., k_out)
        if self.B_free is not None:
            y = y + (x @ self.B_free.T if self.layout == "linear" else x @ self.B_free)
        if self.b_T is not None:
            y = y + (self.b_T @ self.V_out)
        if self.correction is not None:
            y = y + self.correction(x)
        return y

    def stiefel_parameters(self) -> list[nn.Parameter]:
        """V_in and V_out — the parameters StiefelAdam should manage."""
        return [self.V_in, self.V_out]

    def effective_weight(self) -> Tensor:
        """Collapsed compressed weight matching the layout.

        Linear: ``V_out^T W_T V_in`` (k_out, k_in). Conv1D: ``V_in^T W_T V_out``
        (k_in, k_out) — i.e. the same orientation the source module stores.
        """
        if self.layout == "linear":
            W = self.V_out.T @ self.W_T @ self.V_in
        else:
            W = self.V_in.T @ self.W_T @ self.V_out
        if self.B_free is not None:
            W = W + self.B_free
        return W

    def effective_bias(self) -> Tensor | None:
        if self.b_T is None:
            return None
        return self.b_T @ self.V_out

    @torch.no_grad()
    def fold(self) -> nn.Linear:
        """Collapse to a plain ``nn.Linear(k_in, k_out)`` for zero-overhead inference.

        ``nn.Linear`` stores weight as ``(k_out, k_in)`` and computes ``x @ W^T``;
        for the conv1d layout the effective weight is ``(k_in, k_out)`` so it is
        transposed. The sparse-block correction (square case) is folded in.
        """
        lin = nn.Linear(self.k_in, self.k_out, bias=self.b_T is not None)
        # linear: (k_out, k_in); conv1d: (k_in, k_out)
        W = self.effective_weight()
        W_lin = W if self.layout == "linear" else W.T    # -> (k_out, k_in)
        if self.correction is not None:
            H, d_h = self.correction.num_heads, self.correction.d_head
            S = torch.zeros_like(W_lin)
            for h in range(H):
                S[h * d_h:(h + 1) * d_h, h * d_h:(h + 1) * d_h] = self.correction.weight[h]
            W_lin = W_lin + S
        lin.weight.data.copy_(W_lin.to(lin.weight.dtype))
        b = self.effective_bias()
        if b is not None:
            lin.bias.data.copy_(b.to(lin.bias.dtype))
        return lin


def stiefel_parameters_of(module: nn.Module) -> list[nn.Parameter]:
    """Walk ``module`` and collect every parameter tagged ``is_stiefel = True``.

    Used by :func:`fasd.training.stiefel_optim.stiefel_param_groups` to identify
    parameters from `RRNorm` (its `q` matrix) and `FactoredLinear` (its `U_in`,
    `U_out` matrices) without each module needing to expose a special method.
    """
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for p in module.parameters():
        if id(p) in seen:
            continue
        if getattr(p, "is_stiefel", False):
            out.append(p)
            seen.add(id(p))
    return out


__all__ = ["FactoredLinear", "TeacherFactoredLinear", "stiefel_parameters_of"]
