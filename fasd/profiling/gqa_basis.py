"""GQA shared-per-KV-group basis (Sprint 2 fix for the disjoint-basis bug).

The bug
-------

The current absorbed-init code at [fasd/builders.py:477-485](../builders.py#L477-L485)
runs streaming PCA *per branch* — one basis ``V_q`` for q_proj output, one
``V_k`` for k_proj, one ``V_v`` for v_proj. For grouped-query attention (GQA,
``num_kv_heads < num_attention_heads``) this breaks attention preservation.

Per-head, the attention score is ``q_h · k_g`` where ``q_h`` is a query head
in KV-group g and ``k_g`` is that group's key (each d_h-dimensional). Under a
basis change ``q_h ← V_q[h]^T q_h``, ``k_g ← V_k[g]^T k_g``, the new inner
product is ``(V_q[h]^T q_h) · (V_k[g]^T k_g) = q_h · V_q[h] V_k[g]^T · k_g``,
which equals the original ``q_h · k_g`` only when ``V_q[h] V_k[g]^T = I_{d_h}``
on the relevant directions — i.e. when ``V_q[h] = V_k[g]``.

The fix
-------

Derive a *single shared basis per KV-group*. For each layer ``ℓ`` and group
``g`` containing query heads ``{h_1, …, h_{H/G}}``, run joint PCA on per-head
activations from that group::

    X_{ℓ,g} = stack(q_{h_1, ℓ}, q_{h_2, ℓ}, …, q_{h_{H/G}, ℓ}, k_{g, ℓ}, v_{g, ℓ})
            ∈ R^{(H/G + 2) · B · T × d_h}

The eigenvectors of ``cov(X_{ℓ,g})`` form a single orthonormal basis
``V_{ℓ,g} ∈ R^{d_h × d_h}``. Slice to top-``d'_h`` columns; use this slice
as the output basis for *every* q head in group g, plus the K and V edges
for group g. This makes attention preservation a per-group invariant.

The basis is captured *after* projection but *before* RoPE.

**RoPE caveat (corrected 2026-06-17).** An earlier version of this docstring
claimed the shared basis "commutes with RoPE because it acts within d_h". That
is **false** and was empirically refuted (``runs/derisk/rope_circuit_basis.py``:
an arbitrary cross-plane PCA basis inflates the *post-RoPE* QK score error ~7×).
RoPE is a position-dependent rotation that is block-diagonal over 2D coordinate
planes; a basis commutes with it only if it is itself block-diagonal over those
planes. Therefore:

  - **QK projections (RoPE):** use the RoPE-aware *plane-aligned* basis in
    :mod:`fasd.profiling.rope` (``rope_aware_basis``), not the free cross-plane
    PCA basis below. Validate with ``rope.qk_score_residual(..., positions=...)``.
  - **V projection / OV circuit (no RoPE):** the free cross-plane shared basis
    below is correct and is the easy win.

The functions below remain valid for the V/OV circuit and for the no-RoPE case;
``collect_gqa_bases`` still returns the joint [q;k;v] per-group bases, but the QK
slice must be replaced by / restricted to a RoPE-aware basis before use on a RoPE
model. ``attention_score_residual`` below only measures the no-RoPE residual.

Usage
-----

::

    from fasd.profiling.gqa_basis import collect_gqa_bases

    bases = collect_gqa_bases(
        teacher,                      # HF Llama / Qwen / Mistral
        calib_loader,
        device="cuda",
    )
    # bases[layer_idx][group_idx] → Tensor of shape (d_h, d_h), eigenvectors

The builder consumes ``bases`` to construct absorbed-init weights for the
attention sublayer. Per-layer per-group:
    - q_proj output basis for head h ∈ group g: bases[ℓ][g][:, :s_d_h]
    - k_proj output basis for group g:           bases[ℓ][g][:, :s_d_h]
    - v_proj output basis for group g:           bases[ℓ][g][:, :s_d_h]

The builder integration is a follow-up (see HANDOFF.md TODO #5); this
module supplies the math + tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GQAConfig:
    """Layout summary of a GQA attention block."""

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )


def gqa_config_from_model(model: nn.Module) -> GQAConfig:
    """Build a :class:`GQAConfig` from a HF model's attention config."""
    cfg = model.config
    H = int(cfg.num_attention_heads)
    G = int(getattr(cfg, "num_key_value_heads", H))
    h_size = int(cfg.hidden_size)
    return GQAConfig(num_attention_heads=H, num_key_value_heads=G, head_dim=h_size // H)


def _split_heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """Reshape (..., num_heads * head_dim) → (..., num_heads, head_dim)."""
    leading = x.shape[:-1]
    return x.view(*leading, num_heads, head_dim)


@torch.no_grad()
def joint_group_covariance(
    q_act: Tensor,
    k_act: Tensor,
    v_act: Tensor,
    gqa_cfg: GQAConfig,
) -> Tensor:
    """Compute the joint covariance per KV-group from a single forward pass.

    Inputs (post-projection, before RoPE):
        q_act : (..., H * d_h)
        k_act : (..., G * d_h)
        v_act : (..., G * d_h)

    Output:
        cov : (G, d_h, d_h) — per-group joint covariance over (H/G + 2) sources
              (queries in the group + key + value).
    """
    H, G, d_h = gqa_cfg.num_attention_heads, gqa_cfg.num_key_value_heads, gqa_cfg.head_dim
    H_per_G = gqa_cfg.heads_per_group

    q_h = _split_heads(q_act, H, d_h)  # (..., H, d_h)
    k_h = _split_heads(k_act, G, d_h)  # (..., G, d_h)
    v_h = _split_heads(v_act, G, d_h)  # (..., G, d_h)

    # Reshape Q heads into per-group: (..., G, H_per_G, d_h)
    leading = q_h.shape[:-2]
    q_grouped = q_h.view(*leading, G, H_per_G, d_h)

    # Per group, stack queries / key / value into (token-count, d_h) rows.
    cov = torch.zeros(G, d_h, d_h, dtype=q_act.dtype, device=q_act.device)
    for g in range(G):
        # queries in this group: (..., H_per_G, d_h)
        qg = q_grouped[..., g, :, :].reshape(-1, d_h)
        kg = k_h[..., g, :].reshape(-1, d_h)
        vg = v_h[..., g, :].reshape(-1, d_h)
        x = torch.cat([qg, kg, vg], dim=0)  # ((H_per_G + 2) · n_tokens, d_h)
        cov[g] = x.T @ x  # (d_h, d_h)
    return cov


@torch.no_grad()
def shared_bases_from_covariance(cov: Tensor) -> Tensor:
    """Eigen-decompose per-group joint covariance to get shared per-group bases.

    Input:
        cov : (G, d_h, d_h)

    Output:
        V : (G, d_h, d_h) — per-group orthonormal bases, columns sorted by
            descending eigenvalue.
    """
    G, d_h, d_h2 = cov.shape
    if d_h != d_h2:
        raise ValueError(f"cov[g] must be square; got shape {cov.shape}")
    V = torch.zeros_like(cov)
    for g in range(G):
        cov_g = 0.5 * (cov[g] + cov[g].T)
        eigvals, eigvecs = torch.linalg.eigh(cov_g)
        # Sort descending.
        order = torch.argsort(eigvals, descending=True)
        V[g] = eigvecs[:, order]
    return V


class _AttnHook:
    """Forward hook capturing q_act, k_act, v_act post-projection per layer."""

    def __init__(self):
        self.q: Tensor | None = None
        self.k: Tensor | None = None
        self.v: Tensor | None = None
        self._hooks: list = []

    def install(self, layer_self_attn: nn.Module):
        def fwd_q(_mod, _inp, out):
            self.q = out.detach() if isinstance(out, Tensor) else out[0].detach()
        def fwd_k(_mod, _inp, out):
            self.k = out.detach() if isinstance(out, Tensor) else out[0].detach()
        def fwd_v(_mod, _inp, out):
            self.v = out.detach() if isinstance(out, Tensor) else out[0].detach()

        self._hooks.append(layer_self_attn.q_proj.register_forward_hook(fwd_q))
        self._hooks.append(layer_self_attn.k_proj.register_forward_hook(fwd_k))
        self._hooks.append(layer_self_attn.v_proj.register_forward_hook(fwd_v))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


@torch.no_grad()
def collect_gqa_bases(
    teacher: nn.Module,
    calib_batches: Iterable,
    *,
    device: str | torch.device | None = None,
) -> dict[int, Tensor]:
    """Collect per-layer, per-group shared bases for a GQA Llama-like teacher.

    Returns:
    -------
    dict[layer_idx -> Tensor]
        Each value has shape ``(num_key_value_heads, head_dim, head_dim)``: per-group
        orthonormal basis matrices, columns sorted by descending eigenvalue.
    """
    if device is None:
        device = next(teacher.parameters()).device
    teacher.eval()
    teacher.to(device)

    cfg = gqa_config_from_model(teacher)
    layers = teacher.model.layers
    n_layers = len(layers)

    # Per-layer accumulator for the (G, d_h, d_h) joint covariance.
    accumulators: list[Tensor] = [
        torch.zeros(cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim,
                    dtype=torch.float64, device=device)
        for _ in range(n_layers)
    ]

    hooks = [_AttnHook() for _ in range(n_layers)]
    for i, h in enumerate(hooks):
        h.install(layers[i].self_attn)

    try:
        for batch in calib_batches:
            if isinstance(batch, dict):
                batch = {
                    k: (v.to(device) if isinstance(v, Tensor) else v)
                    for k, v in batch.items()
                }
            elif isinstance(batch, Tensor):
                batch = batch.to(device)

            # Reset captures.
            for h in hooks:
                h.q = h.k = h.v = None

            if isinstance(batch, dict):
                teacher(**batch)
            else:
                teacher(batch)

            for i, h in enumerate(hooks):
                if h.q is None or h.k is None or h.v is None:
                    continue
                cov = joint_group_covariance(h.q.float(), h.k.float(), h.v.float(), cfg)
                accumulators[i] = accumulators[i] + cov.to(dtype=torch.float64)
    finally:
        for h in hooks:
            h.remove()

    # Eigen-decompose each.
    bases = {}
    for i in range(n_layers):
        bases[i] = shared_bases_from_covariance(accumulators[i].float()).cpu()
    return bases


@torch.no_grad()
def attention_score_residual(
    q: Tensor, k: Tensor, V: Tensor
) -> float:
    """Diagnostic: how well does basis V preserve attention scores?

    Computes per-head attention scores under
      (a) the original Q, K
      (b) Q' = V V^T Q, K' = V V^T K (full-rank reprojection — should be exact)
      (c) Q' = V[:, :s_d_h] V[:, :s_d_h]^T Q (truncated to compressed dim)

    Returns the relative Frobenius error of (c) vs (a). Used by tests.
    """
    # q, k: (B, H, T, d_h); V: (d_h, d_h)
    s_d_h = V.shape[1]
    P = V[:, :s_d_h] @ V[:, :s_d_h].T
    q_proj = q @ P
    k_proj = k @ P
    score_orig = q @ k.transpose(-2, -1)
    score_proj = q_proj @ k_proj.transpose(-2, -1)
    rel = (score_orig - score_proj).norm() / score_orig.norm().clamp_min(1e-8)
    return float(rel.item())


__all__ = [
    "GQAConfig",
    "gqa_config_from_model",
    "joint_group_covariance",
    "shared_bases_from_covariance",
    "collect_gqa_bases",
    "attention_score_residual",
]
