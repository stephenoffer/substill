"""Circuit-Preserving Initialization (CPSD Phase 2-CPI).

Independent per-matrix absorption uses *different* output bases for the
projections that compose into an attention circuit, which breaks the circuit:

  - QK score circuit: ``q_S^T k_S = q^T V_q V_k^T k`` ≠ ``q^T k`` unless ``V_q = V_k``.
  - OV value circuit: ``W_O V_o V_v^T W_V`` ≠ ``W_O W_V`` unless ``V_o = V_v``.

CPI uses a **shared subspace** so the ``V V^T`` factor cancels: per KV-group, one
orthonormal basis ``V`` is used for both sides of the circuit. This module provides
the construction helpers that turn per-group bases into the block-diagonal output
basis the builder feeds to :func:`absorbed_linear_init`, plus diagnostics.

Distinction from KQ-SVD (2512.05916): we do NOT compute the best rank-r SVD of the
operator ``W_Q^T W_K``; we preserve the circuit by *sharing the activation subspace*
(see papers/novel_mechanism.md §1.2). KQ-SVD owns the operator-SVD-with-bound claim
and is QK-only / KV-cache-only; the OV circuit + weight-side construction here is the
unclaimed delta.

RoPE caveat: the QK shared basis only commutes with RoPE when ``V`` respects RoPE's
2D rotation planes — see :mod:`fasd.profiling.gqa_basis` (RoPE-aware path). The OV/value
circuit carries no RoPE, so the shared-subspace construction applies directly there.
"""
from __future__ import annotations

import torch
from torch import Tensor


def block_diagonal_basis(
    group_bases: Tensor,
    head_groups: list[int],
    keep: int | list[int],
) -> Tensor:
    """Build a block-diagonal output basis from per-group shared bases.

    Each attention head occupies a ``d_h``-sized block of the projection's output
    space; head ``h`` uses the (sliced) basis of its KV-group. Stacking these as a
    block-diagonal matrix gives the ``(H*d_h, sum_keep)`` output basis for absorbing
    a fused attention projection while preserving the per-head circuit.

    Parameters
    ----------
    group_bases : Tensor
        ``(G, d_h, d_h)`` per-group orthonormal bases (columns sorted by descending
        eigenvalue), as produced by :func:`fasd.profiling.gqa_basis.collect_gqa_bases`.
    head_groups : list[int]
        Length ``H``; ``head_groups[h]`` is the KV-group index of head ``h``.
    keep : int | list[int]
        Retained columns per head. Scalar applies to all heads; a list gives a
        per-head rank.

    Returns:
    -------
    Tensor
        ``(H*d_h, sum(keep))`` block-diagonal basis. Columns within each block are
        orthonormal; blocks are orthogonal by construction (disjoint row support).
    """
    G, d_h, _ = group_bases.shape
    H = len(head_groups)
    keeps = [keep] * H if isinstance(keep, int) else list(keep)
    if len(keeps) != H:
        raise ValueError(f"keep list length {len(keeps)} != num heads {H}")
    blocks = []
    for h in range(H):
        g = head_groups[h]
        if not (0 <= g < G):
            raise ValueError(f"head {h} group {g} out of range [0,{G})")
        k = keeps[h]
        if not (0 < k <= d_h):
            raise ValueError(f"keep {k} out of range (0,{d_h}] for head {h}")
        blocks.append(group_bases[g][:, :k])  # (d_h, k)
    return torch.block_diag(*blocks)  # (H*d_h, sum_k)


@torch.no_grad()
def ov_circuit_residual(
    W_O: Tensor,
    W_V: Tensor,
    V_v: Tensor,
    V_o: Tensor | None = None,
) -> float:
    """Relative Frobenius error of the OV operator under (V_v, V_o) absorption.

    The teacher OV operator is ``M = W_O @ W_V``. Under absorption it becomes
    ``M' = W_O V_o V_o^T  ·  V_v V_v^T W_V``. With the **shared** basis
    (``V_o = V_v``) this is ``W_O P W_V`` with ``P = V V^T`` (one projector); with
    independent bases the two projectors do not cancel. Returns ``||M - M'|| / ||M||``.

    Shapes: ``W_O (d, d_h)``, ``W_V (d_h, d)``, ``V_v (d_h, k)``, ``V_o (d, ?)`` —
    for the value circuit ``V_o`` lives in the same ``d_h`` value space as ``V_v``
    (the head/value space), so pass ``V_o`` with first dim ``d_h``; default
    ``V_o = V_v`` (shared).
    """
    if V_o is None:
        V_o = V_v
    M = W_O @ W_V  # (d, d)
    P_v = V_v @ V_v.T  # (d_h, d_h)
    P_o = V_o @ V_o.T  # (d_h, d_h)
    M_approx = (W_O @ P_o) @ (P_v @ W_V)
    return float((M - M_approx).norm() / M.norm().clamp_min(1e-9))


__all__ = ["block_diagonal_basis", "ov_circuit_residual"]
