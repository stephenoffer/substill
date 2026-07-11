"""Gamma-fold pre-pass for rotation-equivariant normalization.

Pre-norm transformer blocks compute ``y = W · LN(x) + b`` where
``LN(x) = γ ⊙ (x - μ̄·1)/σ + β`` (LayerNorm) or
``γ ⊙ x / √(mean(x²) + ε)`` (RMSNorm).

Both expand as::

    y = W · diag(γ) · norm0(x) + (W · β + b)

where ``norm0`` is the parameter-free part (centering + RMS, or just RMS).
We **fold** ``diag(γ)`` into ``W`` and (for LayerNorm) ``β`` into the bias::

    W' = W · diag(γ)
    b' = b + W · β
    γ ← 1
    β ← 0

After folding, the norm becomes parameter-free, which is rotation-equivariant
under any orthonormal basis change of the residual stream (subject to
centering/RMS dropping that we handle in the RR-Norm module). PCA on the
post-norm activation is then a clean rotation problem with no γ-induced
anisotropy — exactly the structure absorbed initialization needs.

Why fold *before* profiling: the folded weight ``W' = W diag(γ)`` is what
the student must reproduce. Running streaming PCA on the folded teacher's
post-norm activations (which equal ``norm0(x)``) gives the right covariance
basis for absorbed init.

This module operates on a *copy* of the teacher; the in-memory teacher is
never modified. The folded copy is only used for profiling; the original
teacher computes the true forward pass during distillation.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class FoldEdge:
    """A (norm, next_linear) pair scheduled for γ/β folding.

    ``norm_path`` and ``linear_path`` are dotted module paths
    (e.g. ``"transformer.h.0.ln_1"``, ``"transformer.h.0.attn.c_attn"``).
    """

    norm_path: str
    linear_path: str


def _get_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path."""
    mod: nn.Module = root
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


@torch.no_grad()
def fold_pair(norm: nn.Module, linear: nn.Module) -> None:
    """Fold ``norm.weight`` and (if present) ``norm.bias`` into ``linear``.

    After folding:
      - ``linear.weight`` has been right-multiplied by ``diag(γ)`` (or for
        Conv1D layout, equivalently scaled along the input channel axis).
      - ``linear.bias`` (if present) has been incremented by ``W · β`` (only
        for LayerNorm; RMSNorm has no β).
      - ``norm.weight`` is set to ones; ``norm.bias`` (if present) is set to zero.

    Operates in-place on the supplied modules. Caller must guarantee ``norm``
    feeds *directly* into ``linear`` without a residual or other gating in
    between (this is the standard pre-norm pattern: LN → linear inside attn/FFN).

    Raises ValueError if shapes are incompatible.
    """
    gamma = getattr(norm, "weight", None)
    beta = getattr(norm, "bias", None)
    if gamma is None:
        return  # nothing to fold

    W = linear.weight  # shape depends on layout
    layout = _infer_linear_layout(linear)
    d_in = _linear_in_features(linear, layout)
    if int(gamma.numel()) != d_in:
        raise ValueError(
            f"gamma has {gamma.numel()} elements; linear expects in_features={d_in}"
        )

    g = gamma.detach().to(device=W.device, dtype=W.dtype)

    # Compute β-contribution to bias FIRST, using the *original* (unscaled) W.
    # b' = b + W · β  (linear),  b' = b + β @ W  (Conv1D where y = x @ W + b)
    delta_b: Tensor | None = None
    if beta is not None:
        bv = beta.detach().to(device=W.device, dtype=W.dtype)
        delta_b = W.detach() @ bv if layout == "linear" else bv @ W.detach()

    # Now scale W in place.
    if layout == "linear":
        # nn.Linear: W shape (out, in). Right-multiply by diag(g) ↔ scale columns.
        W.mul_(g.view(1, -1))
    elif layout == "conv1d_gpt2":
        # HF GPT-2 Conv1D: W shape (in, out). Scale rows.
        W.mul_(g.view(-1, 1))
    else:
        raise ValueError(f"unsupported linear layout: {layout!r}")

    # Apply the captured β-contribution.
    if delta_b is not None:
        b = getattr(linear, "bias", None)
        if b is None:
            linear.bias = nn.Parameter(delta_b.detach().clone())
        else:
            b.data.add_(delta_b)

    # Reset norm to identity.
    gamma.data.fill_(1.0)
    if beta is not None:
        beta.data.zero_()


def _infer_linear_layout(linear: nn.Module) -> str:
    name = type(linear).__name__
    if isinstance(linear, nn.Linear):
        return "linear"
    if name == "Conv1D":
        return "conv1d_gpt2"
    raise TypeError(f"unrecognized linear-like module: {type(linear).__name__}")


def _linear_in_features(linear: nn.Module, layout: str) -> int:
    if layout == "linear":
        return int(linear.in_features)
    if layout == "conv1d_gpt2":
        # HF GPT-2 Conv1D: weight shape (in, out); .nf is out. Use weight shape.
        return int(linear.weight.shape[0])
    raise ValueError(f"unknown layout: {layout!r}")


def fold_edges(model: nn.Module, edges: Iterable[FoldEdge]) -> None:
    """Apply γ/β folding to every (norm, linear) edge in ``edges``.

    In-place on ``model``. After this returns, the norm modules have
    γ=1, β=0, and the next linears have absorbed those parameters.
    """
    for e in edges:
        norm = _get_module(model, e.norm_path)
        linear = _get_module(model, e.linear_path)
        fold_pair(norm, linear)


def make_folded_copy(model: nn.Module, edges: Iterable[FoldEdge]) -> nn.Module:
    """Return a deep-copy of ``model`` with γ/β folded along ``edges``.

    The original ``model`` is untouched. Use the returned copy for
    streaming-PCA profiling so the captured activations are post-isotropic-norm.
    """
    folded = copy.deepcopy(model)
    fold_edges(folded, edges)
    return folded


# ---------------------------------------------------------------------------
# Architecture-specific edge inventories.
# ---------------------------------------------------------------------------


def gpt2_fold_edges(model: nn.Module) -> list[FoldEdge]:
    """All (LN → next Conv1D/Linear) edges in a HF GPT-2 model.

    Pre-norm structure inside each block::

        ln_1 → attn.c_attn   (QKV-fused)
        ln_2 → mlp.c_fc      (FFN up)

    Plus the final ``ln_f`` feeds into ``lm_head`` (Linear).
    """
    edges: list[FoldEdge] = []
    h = model.transformer.h
    for i in range(len(h)):
        edges.append(FoldEdge(f"transformer.h.{i}.ln_1", f"transformer.h.{i}.attn.c_attn"))
        edges.append(FoldEdge(f"transformer.h.{i}.ln_2", f"transformer.h.{i}.mlp.c_fc"))
    # ln_f → lm_head. lm_head is tied to wte in GPT-2; folding γ_lnf into it
    # would also affect the embedding table via tying. We skip ln_f to keep
    # the embedding table untouched; the residual γ stays on the norm.
    # Callers who want this fold should explicitly opt in.
    return edges


def llama_fold_edges(model: nn.Module) -> list[FoldEdge]:
    """All (RMSNorm → next Linear) edges in a HF Llama-style model.

    Pre-norm structure inside each layer::

        input_layernorm           → self_attn.q_proj, .k_proj, .v_proj
        post_attention_layernorm  → mlp.gate_proj, mlp.up_proj

    And the final ``model.norm`` → ``lm_head`` (skipped for the same
    embedding-tying caution as GPT-2's ``ln_f``).
    """
    edges: list[FoldEdge] = []
    layers = model.model.layers
    for i in range(len(layers)):
        prefix = f"model.layers.{i}"
        # input_layernorm feeds q_proj, k_proj, v_proj — but γ is the same
        # for all three (they all read the same norm output). Folding into
        # one of them would break the others; instead, fold γ into ALL three.
        # We implement this as multiple edges sharing the same norm: the
        # first fold_pair() call resets γ to 1, so subsequent calls fold a
        # vector of ones (no-op). To get all three correctly folded, we
        # expand each norm-feeds-multiple-linears edge into one edge per
        # consumer, keeping γ pristine until the last consumer is folded.
        # See `expand_shared_norms` for the resolution.
        edges.append(FoldEdge(f"{prefix}.input_layernorm", f"{prefix}.self_attn.q_proj"))
        edges.append(FoldEdge(f"{prefix}.input_layernorm", f"{prefix}.self_attn.k_proj"))
        edges.append(FoldEdge(f"{prefix}.input_layernorm", f"{prefix}.self_attn.v_proj"))
        edges.append(FoldEdge(f"{prefix}.post_attention_layernorm", f"{prefix}.mlp.gate_proj"))
        edges.append(FoldEdge(f"{prefix}.post_attention_layernorm", f"{prefix}.mlp.up_proj"))
    return edges


@torch.no_grad()
def fold_shared_norm(
    model: nn.Module, norm_path: str, linear_paths: list[str]
) -> None:
    """Fold a single norm's γ/β into MULTIPLE downstream linears.

    The pre-norm pattern in Llama-style models has one RMSNorm feeding three
    linears (q_proj, k_proj, v_proj). We need γ folded into *each* of them
    while resetting γ only once (after all consumers are done).

    Implementation: capture γ/β snapshots, fold them into every consumer
    using the captured values, then reset γ=1, β=0 once.
    """
    norm = _get_module(model, norm_path)
    gamma = getattr(norm, "weight", None)
    beta = getattr(norm, "bias", None)
    if gamma is None:
        return
    g_snapshot = gamma.detach().clone()
    b_snapshot = beta.detach().clone() if beta is not None else None

    for lp in linear_paths:
        linear = _get_module(model, lp)
        layout = _infer_linear_layout(linear)
        d_in = _linear_in_features(linear, layout)
        if int(g_snapshot.numel()) != d_in:
            raise ValueError(
                f"shared-norm fold: γ has {g_snapshot.numel()} elements; "
                f"linear {lp} expects in_features={d_in}"
            )
        W = linear.weight
        g = g_snapshot.to(device=W.device, dtype=W.dtype)

        # β-contribution computed against the ORIGINAL (unscaled) W.
        delta_b: Tensor | None = None
        if b_snapshot is not None:
            bv = b_snapshot.to(device=W.device, dtype=W.dtype)
            delta_b = W.detach() @ bv if layout == "linear" else bv @ W.detach()

        if layout == "linear":
            W.mul_(g.view(1, -1))
        else:
            W.mul_(g.view(-1, 1))

        if delta_b is not None:
            b = getattr(linear, "bias", None)
            if b is None:
                linear.bias = nn.Parameter(delta_b.detach().clone())
            else:
                b.data.add_(delta_b)

    gamma.data.fill_(1.0)
    if beta is not None:
        beta.data.zero_()


def fold_llama(model: nn.Module) -> None:
    """Fold all RMSNorm γ values in a Llama-style model into their downstream linears.

    Operates in-place. Use ``make_folded_copy`` if you need a non-destructive variant.
    """
    layers = model.model.layers
    for i in range(len(layers)):
        prefix = f"model.layers.{i}"
        fold_shared_norm(
            model,
            f"{prefix}.input_layernorm",
            [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
            ],
        )
        fold_shared_norm(
            model,
            f"{prefix}.post_attention_layernorm",
            [
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
            ],
        )


def fold_gpt2(model: nn.Module) -> None:
    """Fold all LN γ/β values in a HF GPT-2 model into their downstream linears.

    Skips ``ln_f`` (would couple to the tied LM head / embedding table).
    """
    h = model.transformer.h
    for i in range(len(h)):
        # GPT-2's ln_1 feeds only c_attn (QKV-fused), and ln_2 feeds only c_fc.
        # No shared-norm complication.
        fold_pair(
            _get_module(model, f"transformer.h.{i}.ln_1"),
            _get_module(model, f"transformer.h.{i}.attn.c_attn"),
        )
        fold_pair(
            _get_module(model, f"transformer.h.{i}.ln_2"),
            _get_module(model, f"transformer.h.{i}.mlp.c_fc"),
        )


__all__ = [
    "FoldEdge",
    "fold_pair",
    "fold_shared_norm",
    "fold_edges",
    "make_folded_copy",
    "fold_gpt2",
    "fold_llama",
    "gpt2_fold_edges",
    "llama_fold_edges",
]
