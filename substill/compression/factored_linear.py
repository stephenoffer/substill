"""Stiefel-trainable factored linears for CPSD manifold training (MT).

This module provides two classes:

- :class:`TeacherFactoredLinear` — the **production CPSD-MT module**. It replaces an
  absorbed-init student linear with a frozen teacher weight ``W_T`` plus trainable
  Stiefel bases ``V_in/V_out`` (collapsed weight ``W_S = V_out^T W_T V_in``), optionally
  a zero-initialized Euclidean ``B_free`` core for extra fitting capacity. It operates on
  the *student-dim* residual stream (unlike :class:`FactoredLinear`, whose ``U_in`` consumes
  teacher-dim input), so it drops directly into a compressed student. It is wired into the
  pipeline by ``FSDPipeline(use_cpsd_factored=True)`` via ``convert_{gpt2,llama}_to_factored``
  and folds to a plain ``nn.Linear`` for zero-overhead inference
  (:meth:`TeacherFactoredLinear.fold`).
  :class:`GatedFactoredLinear` wraps it with a :class:`DifferentiableRankGate` for DDR.

- :class:`FactoredLinear` — a general ``U_out @ B @ U_in^T`` factorization (teacher-dim I/O,
  rank-bottlenecked core) used for research/toy experiments and as the ``from_teacher``
  reference factorization. It is not the module the pipeline swaps in.

Both keep ``V_in``/``V_out`` (resp. ``U_in``/``U_out``) as standalone Stiefel ``nn.Parameter``
matrices tagged ``is_stiefel = True`` so
:func:`substill.training.stiefel_optim.stiefel_param_groups`
places them in the Stiefel group (the Euclidean core ``B``/``B_free`` and bias go in the AdamW
group). Forwards evaluate via the small-matrix chain (no full-weight materialization), e.g.
``y = ((x @ U_in) @ B.T) @ U_out.T + b`` — O(B·T·(d_in·k_in + k_in·k_out + k_out·d_out)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .diff_rank import DifferentiableRankGate
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
        If True, add a `BlockDiagonalCorrection` between U_in and U_out.
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
      collapsed linear (measured); the deferral
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
        lin = nn.Linear(self.k_in, self.k_out, bias=self.b_T is not None).to(self.V_in.device)
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


class GatedFactoredLinear(nn.Module):
    """A :class:`TeacherFactoredLinear` gated on its compressed input latent.

    Combines CPSD manifold-training (MT) and distillation-driven differentiable-rank
    (DDR) into one trainable edge via a :class:`DifferentiableRankGate`.

    Forward applies the soft gate to ``x`` (the ``k_in`` latent) before routing it
    through the frozen teacher weight::

        y = tfl(gate(x))            # gate scales each of the k_in latent columns

    The gate's per-column keep-probabilities are trained against the KD loss under a
    global parameter budget (a :class:`RankBudgetController` holds the gates + costs).
    Because the gate multiplies the *input* latent, it folds exactly into the collapsed
    weight by scaling its columns — so inference is a single plain ``nn.Linear`` with
    zero gate overhead (see :meth:`fold`). The Stiefel bases ``V_in/V_out`` of the inner
    ``tfl`` remain manifold-trained (discovered via the inner module's
    ``stiefel_parameters()``); the gate ``alpha`` trains in the Euclidean group.

    This promotes the proven ``GatedCPSDLinear`` pattern from
    ``tests/test_fsd_cpsd_integration.py`` to a reusable, pipeline-wired module.

    Parameters
    ----------
    tfl : TeacherFactoredLinear
        The factored edge to gate.
    init_open : bool
        Start with all gates ≈ open (the absorbed-init full-rank edge); DDR then
        prunes columns down to the budget.
    temperature : float
        Initial sigmoid temperature; annealed toward ``0`` to sharpen to {0, 1}.
    monotone : bool
        Enforce a contiguous top-prefix rank (bases are ordered by importance).
    """

    def __init__(
        self,
        tfl: TeacherFactoredLinear,
        *,
        init_open: bool = True,
        temperature: float = 1.0,
        monotone: bool = False,
    ):
        super().__init__()
        self.tfl = tfl
        self.gate = DifferentiableRankGate(
            tfl.k_in, init_open=init_open, temperature=temperature, monotone=monotone
        )

    @property
    def k_in(self) -> int:
        return self.tfl.k_in

    @property
    def k_out(self) -> int:
        return self.tfl.k_out

    def forward(self, x: Tensor) -> Tensor:
        return self.tfl(self.gate(x))

    def cost(self) -> Tensor:
        """Per-column parameter cost (``d_in + d_out`` for each of the ``k_in`` columns)."""
        return torch.full(
            (self.tfl.k_in,), float(self.tfl.d_in + self.tfl.d_out)
        )

    def stiefel_parameters(self) -> list[nn.Parameter]:
        """Delegate to the inner factored linear (the gate is Euclidean, not Stiefel)."""
        return self.tfl.stiefel_parameters()

    def gate_parameters(self) -> list[nn.Parameter]:
        return list(self.gate.parameters())

    def expected_rank(self) -> Tensor:
        return self.gate.expected_rank()

    @torch.no_grad()
    def fold(self, *, harden: bool = True, threshold: float = 0.5) -> nn.Linear:
        """Collapse to a plain ``nn.Linear(k_in, k_out)`` with the gate folded in.

        Gating the input latent scales the columns of the collapsed weight, so we fold
        the inner ``tfl`` and multiply column ``j`` by gate value ``g_j``. With
        ``harden=True`` the gate is binarized at ``threshold`` (columns below it become
        exact zeros — the realized rank reduction); otherwise the soft gate is used.
        Bias is unaffected (the gate only scales ``x``).
        """
        lin = self.tfl.fold()  # nn.Linear(k_in, k_out), weight (k_out, k_in)
        g = self.gate.gate()
        if harden:
            g = (g >= threshold).to(g.dtype)
        g = g.to(device=lin.weight.device, dtype=lin.weight.dtype)
        lin.weight.data.mul_(g[None, :])  # scale input columns
        return lin


def stiefel_parameters_of(module: nn.Module) -> list[nn.Parameter]:
    """Walk ``module`` and collect every parameter tagged ``is_stiefel = True``.

    Used by :func:`substill.training.stiefel_optim.stiefel_param_groups` to identify
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


__all__ = ["FactoredLinear", "TeacherFactoredLinear", "GatedFactoredLinear",
           "stiefel_parameters_of"]
