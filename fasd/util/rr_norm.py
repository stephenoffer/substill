"""Rotation-equivariant normalization (RR-Norm) for the student.

After γ-fold (see [fasd/profiling/gamma_fold.py](../profiling/gamma_fold.py)),
the teacher's normalization layers are parameter-free: γ → 1, β → 0. The
student should mirror this so that absorbed-init plus a basis change V_r is
*exact* at the norm boundary.

We implement RR-Norm as **isotropic RMSNorm** (no γ, no β, no centering)
plus two optional knobs:

1. **Learnable scalar ``c``** — a single scaling factor per layer that
   absorbs the RMS energy ratio between teacher (full d_T) and student
   (compressed d_S < d_T). Initialised from a calibration pass; rarely needs
   to drift far from this.

2. **Learnable orthogonal correction ``Q ∈ O(d_S)``** — a Stiefel-trainable
   matrix initialised to the identity. After absorption, the student's
   activations ``x_S = V_r^T x_T`` are approximately the right inputs for
   the absorbed weights, but residual γ-fold artefacts (from approximate
   commutativity of centering and projection) leave a small basis
   misalignment that ``Q`` is free to correct. ``Q`` lives "inside" the
   basis frame: ``y = (Q · norm(x))`` then feeds the absorbed linear.

   ``Q`` is just an ``nn.Parameter`` of shape ``(d_S, d_S)`` initialised to
   ``I``. The Stiefel optimizer ([fasd/training/stiefel_optim.py](../training/stiefel_optim.py))
   keeps ``Q^T Q = I`` during training. Outside of FSD's optimizer, it
   behaves like an ordinary parameter — no harm.

A drop-in replacement for ``LayerNorm`` / ``RMSNorm`` in the student.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RRNorm(nn.Module):
    """Rotation-equivariant normalization with optional scale and Stiefel correction.

    ``forward(x)`` computes::

        y = c * norm0(x) @ Q     if Q is enabled
        y = c * norm0(x)         otherwise

    where ``norm0`` is parameter-free isotropic RMS::

        norm0(x) = x * rsqrt(mean(x², dim=-1, keepdim=True) + eps)

    Optionally with mean-subtraction (``center=True``) for LayerNorm-trained teachers::

        norm0(x) = (x - x.mean(-1, keepdim=True)) * rsqrt(var(x) + eps)

    Parameters
    ----------
    d : int
        Hidden dimension of the residual stream in student coordinates (d_S).
    eps : float
        Numerical stabiliser, default 1e-6.
    use_scale : bool
        Whether to add the learnable per-layer scalar ``c``. Default True.
    use_q : bool
        Whether to add the learnable Stiefel correction ``Q``. Default True.
    center : bool
        Whether to subtract the mean before scaling (LayerNorm-style). Default
        False (RMSNorm-style). Set to True if the teacher used LayerNorm and
        the student's input is *not* mean-zero.
    init_scale : float
        Initial value for ``c`` (calibration sets this to RMS_T/RMS_S; default 1.0).
    """

    is_stiefel_q: bool = True  # marker so the Stiefel optimizer can find Q params

    def __init__(
        self,
        d: int,
        eps: float = 1e-6,
        *,
        use_scale: bool = True,
        use_q: bool = True,
        center: bool = False,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.eps = eps
        self.center = center
        self.use_scale = use_scale
        self.use_q = use_q

        if use_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_parameter("scale", None)

        if use_q:
            # Q ∈ O(d_S). Initialised to I; trained by the Stiefel optimizer.
            self.q = nn.Parameter(torch.eye(d))
        else:
            self.register_parameter("q", None)

    def forward(self, x: Tensor) -> Tensor:
        if self.center:
            x = x - x.mean(dim=-1, keepdim=True)
        n2 = x.pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(n2 + self.eps)
        if self.scale is not None:
            y = y * self.scale
        if self.q is not None:
            # y has shape (..., d). Right-multiply by Q.
            y = y @ self.q
        return y

    def extra_repr(self) -> str:
        return (
            f"d={self.d}, eps={self.eps}, center={self.center}, "
            f"use_scale={self.use_scale}, use_q={self.use_q}"
        )

    @torch.no_grad()
    def calibrate_scale(self, rms_t: float, rms_s: float) -> None:
        """Set ``c`` so the student's post-norm RMS matches the teacher's.

        Used after profiling: pass the calibration RMS in teacher (full d_T)
        and student (V_r-projected d_S) coordinates. The folded teacher
        weights then reproduce the teacher's outputs in expectation.
        """
        if self.scale is None:
            return
        ratio = float(rms_t) / max(float(rms_s), 1e-8)
        self.scale.data.fill_(ratio)

    def stiefel_parameters(self) -> list[nn.Parameter]:
        """Yield parameters that should be optimized on the Stiefel manifold."""
        if self.q is not None:
            return [self.q]
        return []


def replace_layernorm_with_rrnorm(
    module: nn.Module,
    *,
    d_model: int,
    use_q: bool = True,
    use_scale: bool = True,
    center_for_layernorm: bool = False,
    eps: float = 1e-6,
    skip_unfoldable: bool = True,
    skip_paths: list[str] | None = None,
) -> int:
    """Replace every ``nn.LayerNorm`` / RMSNorm-like submodule with an ``RRNorm``.

    Walks ``module``'s tree and swaps any matching norm layer of dim ``d_model``
    in place. Returns the number of replacements.

    ``center_for_layernorm``: if True, set ``RRNorm.center=True`` for
    replaced ``nn.LayerNorm`` layers (preserving centering); otherwise drop
    centering (treats LayerNorm-trained teachers as RMS-trainable students).

    ``skip_unfoldable``: if True (default), refuse to replace any norm whose
    γ deviates from 1.0 or β deviates from 0.0 by more than a small tolerance.
    Such norms haven't had their γ/β folded into adjacent linears, and naive
    replacement would silently drop those parameters — typically catastrophic
    (the v9 5-14 OOM init disaster). The user must run γ-fold *first* (see
    :mod:`fasd.profiling.gamma_fold`); norms that cannot be folded (e.g. GPT-2's
    ``ln_f`` — tied to ``lm_head``) are left as plain ``nn.LayerNorm``.

    ``skip_paths``: optional list of dotted module paths to skip explicitly.
    """
    skip_set = set(skip_paths or [])
    n_replaced = 0
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_path = f"{parent_name}.{child_name}" if parent_name else child_name
            if full_path in skip_set:
                continue

            ref_param = next(child.parameters(), None)
            device = ref_param.device if ref_param is not None else torch.device("cpu")
            dtype = ref_param.dtype if ref_param is not None else torch.float32

            if isinstance(child, nn.LayerNorm):
                if int(child.normalized_shape[0]) != d_model:
                    continue
                if skip_unfoldable and not _looks_unit_affine(child):
                    # γ/β not folded — silently dropping would corrupt the model.
                    continue
                new = RRNorm(
                    d_model,
                    eps=child.eps if hasattr(child, "eps") else eps,
                    use_scale=use_scale,
                    use_q=use_q,
                    center=center_for_layernorm,
                ).to(device=device, dtype=dtype)
                setattr(parent, child_name, new)
                n_replaced += 1
            elif _looks_like_rmsnorm(child) and _rmsnorm_dim(child) == d_model:
                if skip_unfoldable and not _looks_unit_affine(child):
                    continue
                new = RRNorm(
                    d_model,
                    eps=getattr(child, "eps", None) or getattr(child, "variance_epsilon", eps),
                    use_scale=use_scale,
                    use_q=use_q,
                    center=False,
                ).to(device=device, dtype=dtype)
                setattr(parent, child_name, new)
                n_replaced += 1
    return n_replaced


def _looks_unit_affine(mod: nn.Module, tol: float = 1e-4) -> bool:
    """Detect whether γ ≈ 1.0 and β ≈ 0.0 (i.e. the norm has been γ-folded)."""
    w = getattr(mod, "weight", None)
    if w is None:
        return True
    if (w - 1.0).abs().max().item() > tol:
        return False
    b = getattr(mod, "bias", None)
    return not (b is not None and b.abs().max().item() > tol)


def _looks_like_rmsnorm(mod: nn.Module) -> bool:
    """Detect HF / Llama-style RMSNorm without importing it."""
    if isinstance(mod, nn.LayerNorm):
        return False
    name = type(mod).__name__
    if "RMSNorm" in name:
        return True
    # Has γ (weight), no β.
    return (
        hasattr(mod, "weight")
        and (hasattr(mod, "variance_epsilon") or hasattr(mod, "eps"))
        and mod.weight is not None
        and mod.weight.dim() == 1
        and getattr(mod, "bias", None) is None
    )


def _rmsnorm_dim(mod: nn.Module) -> int | None:
    if hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 1:
        return int(mod.weight.shape[0])
    return None


__all__ = ["RRNorm", "replace_layernorm_with_rrnorm"]
