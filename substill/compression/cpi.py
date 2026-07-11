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
2D rotation planes — see :mod:`substill.profiling.gqa_basis` (RoPE-aware path). The OV/value
circuit carries no RoPE, so the shared-subspace construction applies directly there.
"""
from __future__ import annotations

import torch
import torch.nn as nn
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
        eigenvalue), as produced by :func:`substill.profiling.gqa_basis.collect_gqa_bases`.
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


def cpi_rank_map(teacher, profile, *, head_dim_ratio: float = 0.5,
                 ffn_ratio: float = 0.5) -> dict[str, int]:
    """Build a rank-map pinning a CPI-compatible student.

    Keeps the teacher's attention-head count H and KV-group count G, compressing
    only the per-head dim and the FFN. CPI's shared-per-group basis requires
    ``H_s == H_t`` and ``G_s == G_t`` so the per-group block-diagonal mapping is
    well-defined. ``head_dim_ratio`` sets ``s_d_h = round(d_h * ratio)`` (even).
    """
    cfg = teacher.config
    H = int(cfg.num_attention_heads)
    G = int(getattr(cfg, "num_key_value_heads", H))
    t_hidden = int(cfg.hidden_size)
    t_interm = int(getattr(cfg, "intermediate_size", 4 * t_hidden))
    d_h = t_hidden // H
    s_d_h = max(2, int(round(d_h * head_dim_ratio)))
    s_d_h -= s_d_h % 2  # even for RoPE plane alignment
    s_d_h = max(2, s_d_h)
    s_hidden, s_kv = H * s_d_h, G * s_d_h
    s_interm = max(8, int(round(t_interm * ffn_ratio)))
    rm: dict[str, int] = {}
    # NB ffn.down OUTPUTS to the residual dim (its branch rank drives the residual-size
    # fallback in width_pruner), so it must map to s_hidden, NOT s_interm. Only
    # ffn.up/ffn.gate output to the intermediate dim.
    for b in profile.branches:
        if b.kind in ("block.residual", "attn.q", "attn.o", "ffn.down"):
            rm[b.name] = s_hidden
        elif b.kind in ("attn.k", "attn.v"):
            rm[b.name] = s_kv
        elif b.kind in ("ffn.up", "ffn.gate"):
            rm[b.name] = s_interm
    return rm


@torch.no_grad()
def _accumulate_group_covariances(teacher, calib_batches, *, device=None):
    """Accumulate the joint per-KV-group covariance of q/k/v activations.

    Captures each layer's q/k/v post-projection and pre-RoPE. Returns a mapping
    ``layer_idx -> tensor(G, d_h, d_h)``.
    """
    from ..profiling.gqa_basis import gqa_config_from_model, joint_group_covariance

    if device is None:
        device = next(teacher.parameters()).device
    teacher.eval().to(device)
    cfg = gqa_config_from_model(teacher)
    layers = teacher.model.layers
    acc = [torch.zeros(cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim,
                       dtype=torch.float64, device=device) for _ in layers]

    caps: dict[int, dict] = {}
    hooks = []
    for i, layer in enumerate(layers):
        a = layer.self_attn
        caps[i] = {}
        hooks.append(a.q_proj.register_forward_hook(
            lambda m, inp, o, i=i: caps[i].__setitem__("q", o.detach())))
        hooks.append(a.k_proj.register_forward_hook(
            lambda m, inp, o, i=i: caps[i].__setitem__("k", o.detach())))
        hooks.append(a.v_proj.register_forward_hook(
            lambda m, inp, o, i=i: caps[i].__setitem__("v", o.detach())))
    try:
        for batch in calib_batches:
            if isinstance(batch, dict):
                batch = {k: (v.to(device) if isinstance(v, Tensor) else v)
                         for k, v in batch.items()}
                teacher(**batch)
            else:
                teacher(batch.to(device))
            for i in range(len(layers)):
                c = caps[i]
                if {"q", "k", "v"} <= c.keys():
                    acc[i] += joint_group_covariance(
                        c["q"].float(), c["k"].float(), c["v"].float(), cfg
                    ).double()
    finally:
        for h in hooks:
            h.remove()
    return {i: acc[i].float() for i in range(len(layers))}


@torch.no_grad()
def apply_ov_align_init(
    student: nn.Module,
    teacher: nn.Module,
    profile,
    calib_batches,
    *,
    device=None,
) -> int:
    """Align ``o_proj``'s input basis to ``v_proj``'s output basis at no energy cost.

    Uses V's own per-group PCA, with no compromise or shared basis.
    The disjoint baseline gives ``v_proj`` output basis ``V_v`` but ``o_proj`` input
    basis ``V_r`` (residual) — a pure mismatch that breaks ``W_O W_V`` on GQA models.
    This re-inits v_proj with its own per-group cross-plane PCA (no RoPE on V) and
    o_proj with the *same* per-head value basis as input, so the OV circuit is exactly
    preserved while V keeps its energy-optimal basis. Q/K are left as the baseline.
    Returns #layers re-initialized.
    """
    from ..builders import _residual_basis
    from ..profiling.gqa_basis import gqa_config_from_model
    from .absorbed_init import absorbed_linear_init

    t_cfg, s_cfg = teacher.config, student.config
    t_hidden, s_hidden = int(t_cfg.hidden_size), int(s_cfg.hidden_size)
    H = int(s_cfg.num_attention_heads)
    G = int(getattr(s_cfg, "num_key_value_heads", H))
    H_t = int(t_cfg.num_attention_heads)
    G_t = int(getattr(t_cfg, "num_key_value_heads", H_t))
    if H_t != H or G_t != G:
        raise ValueError(
            "apply_ov_align_init requires H_s==H_t and G_s==G_t (use cpi_rank_map); "
            f"got H_s={H} H_t={H_t} G_s={G} G_t={G_t}"
        )
    s_d_h = s_hidden // H
    gqa = gqa_config_from_model(teacher)
    d_h = gqa.head_dim
    heads_per_group = H // G

    # Per-group V-only covariance (V carries no RoPE -> cross-plane PCA is optimal).
    if device is None:
        device = next(teacher.parameters()).device
    teacher.eval().to(device)
    layers = teacher.model.layers
    acc = [torch.zeros(G, d_h, d_h, dtype=torch.float64, device=device) for _ in layers]
    caps: dict[int, Tensor] = {}
    hooks = [layers[i].self_attn.v_proj.register_forward_hook(
        lambda m, inp, o, i=i: caps.__setitem__(i, o.detach())) for i in range(len(layers))]
    try:
        for batch in calib_batches:
            if isinstance(batch, dict):
                teacher(**{k: (v.to(device) if isinstance(v, Tensor) else v)
                           for k, v in batch.items()})
            else:
                teacher(batch.to(device))
            for i in range(len(layers)):
                v = caps.get(i)
                if v is None:
                    continue
                vh = v.float().reshape(-1, G, d_h)            # (N, G, d_h)
                for g in range(G):
                    vg = vh[:, g, :]
                    acc[i][g] += (vg.T @ vg).double()
    finally:
        for h in hooks:
            h.remove()

    V_r = _residual_basis(profile, t_hidden, s_hidden)
    for i, (tb, sb) in enumerate(zip(layers, student.model.layers, strict=True)):
        cov = acc[i].float().to(V_r.dtype)
        v_basis = [torch.linalg.eigh(0.5 * (cov[g] + cov[g].T))[1][:, -s_d_h:]
                   for g in range(G)]                          # (d_h, s_d_h) per group
        Vv = torch.block_diag(*v_basis)                        # (t_kv, s_kv)
        Vo = torch.block_diag(*[v_basis[h // heads_per_group] for h in range(H)])
        absorbed_linear_init(tb.self_attn.v_proj, sb.self_attn.v_proj, V_in=V_r, V_out=Vv)
        absorbed_linear_init(tb.self_attn.o_proj, sb.self_attn.o_proj, V_in=Vo, V_out=V_r)
    return len(layers)


@torch.no_grad()
def apply_cpi_attention_init(
    student: nn.Module,
    teacher: nn.Module,
    profile,
    calib_batches,
    *,
    rope_aware: bool = True,
    device=None,
) -> int:
    """Re-initialize a Llama student's attention projections with a shared basis.

    Uses one shared-per-KV-group circuit-preserving basis, avoiding the GQA
    disjoint-basis mismatch that a per-projection basis would introduce.
    Per group g: one basis is shared by all query heads in g and the group's K (so
    ``q_h·k_g`` is preserved) and one by V and the matching O-input columns (so
    ``W_O W_V`` is preserved). The Q/K basis is **plane-aligned** when ``rope_aware``
    (commutes with RoPE); the V/O basis is cross-plane PCA (V carries no RoPE — the
    clean win). Overwrites the disjoint absorbed init in place. Returns #layers done.
    """
    from ..builders import _residual_basis
    from ..profiling.rope import rope_aware_basis
    from .absorbed_init import absorbed_linear_init

    t_cfg, s_cfg = teacher.config, student.config
    t_hidden, s_hidden = int(t_cfg.hidden_size), int(s_cfg.hidden_size)
    H = int(s_cfg.num_attention_heads)
    G = int(getattr(s_cfg, "num_key_value_heads", H))
    s_d_h = s_hidden // H
    if s_d_h % 2 != 0:
        raise ValueError(f"student head_dim {s_d_h} must be even for RoPE plane alignment")
    H_t = int(t_cfg.num_attention_heads)
    G_t = int(getattr(t_cfg, "num_key_value_heads", H_t))
    if H_t != H or G_t != G:
        raise ValueError(
            f"apply_cpi_attention_init requires the student to keep the teacher's head "
            f"structure (H_s==H_t, G_s==G_t); got H_s={H}, H_t={H_t}, G_s={G}, G_t={G_t}. "
            f"Build the student with substill.compression.cpi.cpi_rank_map(...) as rank_map."
        )
    covs = _accumulate_group_covariances(teacher, calib_batches, device=device)
    V_r = _residual_basis(profile, t_hidden, s_hidden)
    heads_per_group = H // G

    t_layers, s_layers = teacher.model.layers, student.model.layers
    for i, (tb, sb) in enumerate(zip(t_layers, s_layers, strict=True)):
        cov = covs[i].to(V_r.dtype)  # (G, d_h, d_h)
        qk_basis, v_basis = [], []
        for g in range(G):
            cg = 0.5 * (cov[g] + cov[g].T)
            if rope_aware:
                qk_basis.append(rope_aware_basis(cg, s_d_h // 2))           # (d_h, s_d_h)
            else:
                qk_basis.append(torch.linalg.eigh(cg)[1][:, -s_d_h:])
            v_basis.append(torch.linalg.eigh(cg)[1][:, -s_d_h:])           # cross-plane
        # Block-diagonal output bases (disjoint row support => orthonormal columns).
        # Block-diagonal output bases; shapes: Vq/Vo (t_hidden, s_hidden), Vk/Vv (t_kv, s_kv).
        Vq = torch.block_diag(*[qk_basis[h // heads_per_group] for h in range(H)])
        Vk = torch.block_diag(*[qk_basis[g] for g in range(G)])
        Vv = torch.block_diag(*[v_basis[g] for g in range(G)])
        Vo = torch.block_diag(*[v_basis[h // heads_per_group] for h in range(H)])
        absorbed_linear_init(tb.self_attn.q_proj, sb.self_attn.q_proj, V_in=V_r, V_out=Vq)
        absorbed_linear_init(tb.self_attn.k_proj, sb.self_attn.k_proj, V_in=V_r, V_out=Vk)
        absorbed_linear_init(tb.self_attn.v_proj, sb.self_attn.v_proj, V_in=V_r, V_out=Vv)
        absorbed_linear_init(tb.self_attn.o_proj, sb.self_attn.o_proj, V_in=Vo, V_out=V_r)
    return len(s_layers)


__all__ = ["block_diagonal_basis", "ov_circuit_residual", "apply_cpi_attention_init",
           "apply_ov_align_init", "cpi_rank_map"]
