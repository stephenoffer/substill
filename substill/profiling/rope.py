"""RoPE-aware circuit basis (CPSD Phase 2-CPI, QK half).

Why this module exists
----------------------
The shared-subspace QK construction in :mod:`substill.profiling.gqa_basis` preserves the
attention score only if the shared basis ``V`` **commutes with RoPE**. The original
docstring claimed it does "because RoPE acts within the head dim"; this is **false**
and was empirically refuted: an arbitrary PCA basis inflates the post-RoPE
score error ~7× vs the no-RoPE case.

RoPE applies a position-dependent rotation ``R(θ, pos)`` that is block-diagonal over
2D coordinate planes. A linear map commutes with all ``R(θ, Δ)`` iff it is itself
block-diagonal over those planes (cross-plane mixing breaks commutation, since 2×2
rotations in different planes don't commute through a mixing map). So a RoPE-aware
basis must respect the plane structure. The cleanest commuting compressor is
**plane-aligned selection**: keep whole 2D planes (ranked by energy), drop the rest.
This commutes exactly with RoPE; the cost is whole-plane (rank-2) granularity.

Convention
----------
HF Llama/Qwen/Mistral use the ``rotate_half`` convention: plane ``p`` (``0..d/2-1``)
couples dims ``p`` and ``p + d/2`` with angle ``pos · inv_freq[p]``. Plane-aligned
selection therefore keeps column **pairs** ``{p, p + d/2}``.

The OV/value circuit carries no RoPE, so it uses the free cross-plane shared basis in
:mod:`substill.compression.cpi` — only the QK projections need this module.
"""
from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """HF ``rotate_half``: (x1, x2) -> (-x2, x1) splitting the last dim in half."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def rope_cos_sin(positions: Tensor, d_h: int, base: float = 10000.0):
    """cos/sin tables (HF layout: freqs concatenated, not interleaved)."""
    half = d_h // 2
    inv_freq = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    ang = positions[:, None].float() * inv_freq[None, :]      # (T, half)
    emb = torch.cat([ang, ang], dim=-1)                        # (T, d_h)
    return emb.cos(), emb.sin()


def apply_rope(x: Tensor, positions: Tensor, base: float = 10000.0) -> Tensor:
    """Apply RoPE to ``x`` (..., T, d_h) at the given integer ``positions`` (T,)."""
    cos, sin = rope_cos_sin(positions, x.shape[-1], base)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


def plane_energy(cov: Tensor) -> Tensor:
    """Per-plane energy for the rotate_half convention: plane p uses dims p, p+d/2.

    ``cov`` is a ``(d_h, d_h)`` covariance; returns a length-``d_h/2`` vector.
    """
    d_h = cov.shape[0]
    half = d_h // 2
    diag = torch.diagonal(cov)
    return diag[:half] + diag[half:]


def rope_aware_basis(cov: Tensor, keep_planes: int) -> Tensor:
    """Plane-aligned RoPE-commuting basis: select the top-energy 2D planes.

    Returns a ``(d_h, 2*keep_planes)`` matrix whose columns are the identity columns
    of the retained planes (dims ``p`` and ``p + d/2`` for each kept plane ``p``).
    Being a selection of standard-basis columns over whole planes, it is block-diagonal
    over RoPE's planes and therefore commutes with ``R(θ, Δ)`` exactly.

    Columns are ordered ``[p1, p2, ..., p1+d/2, p2+d/2, ...]`` to match rotate_half.
    """
    d_h = cov.shape[0]
    half = d_h // 2
    if not (0 < keep_planes <= half):
        raise ValueError(f"keep_planes must be in (0, {half}], got {keep_planes}")
    energy = plane_energy(cov)
    kept = torch.sort(torch.argsort(energy, descending=True)[:keep_planes]).values
    cols = torch.cat([kept, kept + half])         # rotate_half pairing
    V = torch.zeros(d_h, 2 * keep_planes, dtype=cov.dtype)
    for j, c in enumerate(cols.tolist()):
        V[c, j] = 1.0
    return V


@torch.no_grad()
def qk_score_residual(
    q: Tensor,
    k: Tensor,
    V: Tensor,
    *,
    positions: Tensor | None = None,
    base: float = 10000.0,
) -> float:
    """Relative Frobenius error of QK scores under basis ``V``, optionally POST-RoPE.

    q, k : (T, d_h). V : (d_h, r) shared basis (projector ``P = V V^T``). If
    ``positions`` is given, RoPE is applied to both teacher and student q/k before
    scoring (the test the old ``attention_score_residual`` omitted). The student
    projects pre-RoPE (as absorbed init does) then applies RoPE.
    """
    P = V @ V.T
    if positions is None:
        q_t, k_t, q_s, k_s = q, k, q @ P, k @ P
    else:
        q_t = apply_rope(q, positions, base)
        k_t = apply_rope(k, positions, base)
        q_s = apply_rope(q @ P, positions, base)
        k_s = apply_rope(k @ P, positions, base)
    S_t = q_t @ k_t.T
    S_s = q_s @ k_s.T
    return float((S_t - S_s).norm() / S_t.norm().clamp_min(1e-9))


__all__ = [
    "rotate_half",
    "rope_cos_sin",
    "apply_rope",
    "plane_energy",
    "rope_aware_basis",
    "qk_score_residual",
]
