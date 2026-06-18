"""Student constructors with absorbed-init.

``build_student(teacher, profile, template=..., absorbed_init=True)``
derives a transformer config from the teacher's branch profile (via
:func:`fasd.compression.width_pruner.profile_to_student_config`),
instantiates a fresh student, and — when ``absorbed_init=True`` —
fills its linear weights using
:func:`fasd.compression.absorbed_init.absorbed_linear_init`.

Supported templates:

- ``"gpt2"`` — HuggingFace ``GPT2LMHeadModel``.
- ``"llama"`` — HuggingFace ``LlamaForCausalLM``.
- ``"auto"`` — dispatch on the teacher's class name.

The student's hidden dimensions are set to the behavioral ranks found
during profiling (rounded for head divisibility). If the profile does
not cover a given linear — for example the embedding table or
layer-norm weights — the corresponding tensors are either copied
directly (when shapes happen to match) or initialized freshly from
the student's config.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .compression.absorbed_init import absorbed_linear_init
from .compression.width_pruner import StudentConfig, profile_to_student_config

Template = Literal["gpt2", "llama", "auto"]


# -- helpers -----------------------------------------------------------


def _residual_basis(profile, t_hidden: int, s_hidden: int) -> Tensor:
    """Return a ``(t_hidden, s_hidden)`` basis for the residual stream.

    v10 change: when ``s_hidden == t_hidden`` (no residual compression) we
    return the identity. v9 used a PCA rotation here even when no
    compression was happening, and that rotation broke initial PPL by
    5–14 orders of magnitude at d=768 because LayerNorm does not commute
    with arbitrary orthogonal rotations of its input — gamma/beta cannot
    be projected through V_r without losing the per-channel structure
    that GPT-2 relies on. Skipping the rotation when k=d is information-
    preserving and lets absorbed init reproduce the teacher exactly.

    For true residual compression (``s_hidden < t_hidden``) we still need
    a basis. We use channel-selection by per-channel residual variance
    (``diag(cov)``, reconstructed from PCA components and eigenvalues)
    instead of PCA rotation. Channel-selection commutes with LayerNorm's
    centering/scaling — gamma_s and beta_s can be sliced from gamma_t /
    beta_t, preserving per-channel structure.

    The legacy ``attn.o``/``ffn.down`` averaged-then-QR fallback is gone:
    it had no theoretical justification and made things worse in v9.
    """
    if s_hidden == t_hidden:
        return torch.eye(t_hidden)

    # Find block.residual branches and aggregate diag(cov).
    cov_diag_sum = None
    pc_first = None
    for b in profile.branches:
        if getattr(b, "kind", None) != "block.residual":
            continue
        if b.principal_components.shape[0] != t_hidden:
            continue
        V = b.principal_components.float()
        eigvals = getattr(b, "eigenvalues", None)
        # No eigenvalues stored — degrade to a uniformly-weighted diag estimate
        # (||V[i,:]||^2). This is a degenerate signal for fully-orthogonal V, so
        # we also keep the PC matrix as a tie-breaker fallback below.
        eigvals = torch.ones(V.shape[1], dtype=V.dtype) if eigvals is None else eigvals.float()
        # cov = V diag(eigvals) V^T  ⇒  cov.diag()[i] = sum_j V[i,j]^2 eigvals[j]
        cov_diag = (V * V * eigvals.unsqueeze(0)).sum(dim=1)
        cov_diag_sum = cov_diag if cov_diag_sum is None else cov_diag_sum + cov_diag
        if pc_first is None:
            pc_first = V
    if cov_diag_sum is not None:
        # If diag is degenerate (all equal — happens when eigvals are
        # uniform and V is orthogonal), fall back to slicing the first
        # s_hidden columns of the residual PC matrix. This preserves
        # the v9 semantics for tests that don't supply eigenvalues.
        if (cov_diag_sum.max() - cov_diag_sum.min()).item() < 1e-6:
            return pc_first[:, :s_hidden].contiguous()
        top = torch.argsort(cov_diag_sum, descending=True)[:s_hidden]
        E = torch.zeros(t_hidden, s_hidden)
        for j, i in enumerate(top.tolist()):
            E[i, j] = 1.0
        return E

    # Fallback to identity-truncated when no residual capture exists.
    return torch.eye(t_hidden, s_hidden)


def _channel_select_basis(profile, branch_name: str, k: int) -> Tensor:
    """Return a (channels, k) channel-selection basis for ``branch_name``.

    Each column is e_i for one of the top-``k`` channels by variance
    (diag of branch's covariance, reconstructed from PCA). Used for the
    FFN intermediate dimension where the activation function (GELU/SiLU)
    is element-wise and does *not* commute with PCA rotation: PCA-based
    V_up means ``act(z @ V_up) ≠ act(z) @ V_up``, which empirically
    inflated initial student PPL by 4 orders of magnitude on rung 4 of
    v9. Channel-selection slices instead of rotating, so
    ``act(teacher_intermediate)[kept_channels] == student_intermediate``
    exactly.
    """
    for b in profile.branches:
        if b.name == branch_name:
            n = b.principal_components.shape[0]
            if k >= n:
                # No compression — return identity.
                return torch.eye(n)
            V = b.principal_components.float()
            eigvals = getattr(b, "eigenvalues", None)
            eigvals = torch.ones(V.shape[1], dtype=V.dtype) if eigvals is None else eigvals.float()
            cov_diag = (V * V * eigvals.unsqueeze(0)).sum(dim=1)
            if (cov_diag.max() - cov_diag.min()).item() < 1e-6:
                # Degenerate scoring — fall back to first-k columns of PC
                # matrix (v9 semantics).
                return V[:, :k].contiguous()
            top = torch.argsort(cov_diag, descending=True)[:k]
            E = torch.zeros(n, k)
            for j, i in enumerate(top.tolist()):
                E[i, j] = 1.0
            return E
    raise KeyError(f"branch {branch_name!r} not in profile")


def _pad_cols_orthogonal(V: Tensor, cols_needed: int) -> Tensor:
    """Extend ``V`` to ``cols_needed`` columns with random orthonormal directions.

    Previously callers zero-padded when ``cols_needed > V.shape[1]`` (e.g. when
    a branch's behavioral_rank is smaller than the student's hidden_size). Zero
    columns make the absorbed weight rank-deficient in exactly those directions
    where the student is expected to do something — in v7-apr24 this caused
    absorbed-init students (rungs 4–7) to start at PPL 10^10 to 10^15.
    Random-orthogonal padding preserves column orthonormality so the absorbed
    weight ``V_out^T W V_in`` stays well-conditioned.
    """
    rows, current = V.shape
    if current >= cols_needed:
        return V[:, :cols_needed].contiguous()
    extra = cols_needed - current
    # Sample in the ambient space, project out V's column span, QR-orthonormalize.
    R = torch.randn(rows, extra, dtype=V.dtype, device=V.device)
    if current > 0:
        R = R - V @ (V.T @ R)
    Q, _ = torch.linalg.qr(R)
    if Q.shape[1] < extra:
        # Rank-deficient R; retry once with fresh noise projected against V and Q.
        R2 = torch.randn(rows, extra - Q.shape[1], dtype=V.dtype, device=V.device)
        if current > 0:
            R2 = R2 - V @ (V.T @ R2)
        R2 = R2 - Q @ (Q.T @ R2)
        Q2, _ = torch.linalg.qr(R2)
        Q = torch.cat([Q, Q2], dim=1)
    return torch.cat([V, Q[:, :extra]], dim=1).contiguous()


# -- GPT-2 -------------------------------------------------------------


def _build_gpt2(teacher, profile, student_cfg: StudentConfig, absorbed_init: bool):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fasd.build_student template='gpt2' requires the 'transformers' package"
        ) from e

    t_cfg = teacher.config
    s_cfg_kwargs = {
        "vocab_size": int(t_cfg.vocab_size),
        "n_positions": int(getattr(t_cfg, "n_positions", 1024)),
        "n_embd": int(student_cfg.hidden_size),
        "n_layer": int(student_cfg.num_hidden_layers),
        "n_head": int(student_cfg.num_attention_heads),
        "n_inner": int(student_cfg.intermediate_size),
        "activation_function": getattr(t_cfg, "activation_function", "gelu_new"),
        "resid_pdrop": float(getattr(t_cfg, "resid_pdrop", 0.1)),
        "embd_pdrop": float(getattr(t_cfg, "embd_pdrop", 0.1)),
        "attn_pdrop": float(getattr(t_cfg, "attn_pdrop", 0.1)),
        "layer_norm_epsilon": float(getattr(t_cfg, "layer_norm_epsilon", 1e-5)),
    }
    s_config = GPT2Config(**s_cfg_kwargs)
    student = GPT2LMHeadModel(s_config)

    if not absorbed_init:
        return student

    return _gpt2_absorb(teacher, student, profile)


def _gpt2_pad_rows(V: Tensor, rows: int) -> Tensor:
    if V.shape[0] == rows:
        return V
    if V.shape[0] > rows:
        return V[:rows].contiguous()
    pad = torch.zeros(rows - V.shape[0], V.shape[1], dtype=V.dtype)
    return torch.cat([V, pad], dim=0)


def gpt2_absorb_targets(teacher, student, profile):
    """Yield ``(name, t_module, s_module, V_in, V_out)`` for every absorbed weight.

    Used both by initial absorption (:func:`_gpt2_absorb`) and by periodic
    re-absorption (:mod:`fasd.training.reabsorb`). The bases are derived
    purely from ``profile`` and the *fixed* teacher/student hidden dims —
    re-running this with a refreshed profile gives a new set of bases on
    the same module structure, which is exactly what PRA needs.
    """
    t_h = teacher.transformer.h
    s_h = student.transformer.h
    if len(t_h) < len(s_h):
        raise ValueError(
            f"teacher has {len(t_h)} blocks but student needs {len(s_h)} — "
            f"use depth_policy='keep' or a teacher with >= student depth"
        )
    t_hidden = int(teacher.config.n_embd)
    s_hidden = int(student.config.n_embd)
    t_interm = int(teacher.config.n_inner or 4 * t_hidden)
    s_interm = int(student.config.n_inner or 4 * s_hidden)

    V_r = _residual_basis(profile, t_hidden, s_hidden)
    V_r_full = _gpt2_pad_rows(V_r, t_hidden)
    V_qkv = torch.block_diag(V_r_full, V_r_full, V_r_full)

    for i, s_block in enumerate(s_h):
        t_block = t_h[i]
        prefix = f"transformer.h.{i}"
        V_up = _gpt2_pad_rows(
            _channel_select_basis(profile, f"{prefix}.ffn.up", s_interm), t_interm
        )
        yield (f"{prefix}.attn.c_attn", t_block.attn.c_attn, s_block.attn.c_attn, V_r_full, V_qkv)
        yield (
            f"{prefix}.attn.c_proj", t_block.attn.c_proj, s_block.attn.c_proj,
            V_r_full, V_r_full,
        )
        yield (f"{prefix}.mlp.c_fc", t_block.mlp.c_fc, s_block.mlp.c_fc, V_r_full, V_up)
        yield (f"{prefix}.mlp.c_proj", t_block.mlp.c_proj, s_block.mlp.c_proj, V_up, V_r_full)


def gpt2_residual_basis(teacher, student, profile) -> Tensor:
    """Return the (t_hidden, s_hidden) residual basis used by the GPT-2 absorb."""
    return _residual_basis(profile, int(teacher.config.n_embd), int(student.config.n_embd))


def _gpt2_absorb(teacher, student, profile):
    """Fill student weights using V_out^T W_T V_in per GPT-2 linear."""
    t_hidden = int(teacher.config.n_embd)
    s_hidden = int(student.config.n_embd)
    V_r = _residual_basis(profile, t_hidden, s_hidden)

    # Embeddings + position table.
    with torch.no_grad():
        W_emb_t = teacher.transformer.wte.weight.detach()
        Vr_e = V_r.to(device=W_emb_t.device, dtype=W_emb_t.dtype)
        student.transformer.wte.weight.data.copy_(
            (W_emb_t @ Vr_e).to(
                student.transformer.wte.weight.dtype
            ).to(student.transformer.wte.weight.device)
        )
        W_pos_t = teacher.transformer.wpe.weight.detach()
        student.transformer.wpe.weight.data.copy_(
            (W_pos_t @ Vr_e).to(
                student.transformer.wpe.weight.dtype
            ).to(student.transformer.wpe.weight.device)
        )

    # Per-block absorption via the shared iterator.
    for _name, t_mod, s_mod, V_in, V_out in gpt2_absorb_targets(teacher, student, profile):
        absorbed_linear_init(t_mod, s_mod, V_in=V_in, V_out=V_out)

    # LayerNorms (diagonal projection — not part of the absorbed-target set).
    s_h = student.transformer.h
    t_h = teacher.transformer.h
    for i, s_block in enumerate(s_h):
        _copy_layernorm(t_h[i].ln_1, s_block.ln_1, V_r)
        _copy_layernorm(t_h[i].ln_2, s_block.ln_2, V_r)
    _copy_layernorm(teacher.transformer.ln_f, student.transformer.ln_f, V_r)

    # lm_head is tied to wte in GPT-2; HF's GPT2LMHeadModel ties automatically.
    return student


def _col_basis(profile, branch_name: str, cols_needed: int) -> Tensor:
    """Return the top ``cols_needed`` eigenvectors of branch ``branch_name``.

    ``principal_components`` is stored as the full ``(C, C)`` descending
    eigenvector matrix (api.py:222-225), so for any ``cols_needed <= C`` we
    slice directly from the teacher's PCA — no random padding needed. The
    ``_pad_cols_orthogonal`` fallback only triggers if a future caller asks
    for more columns than the branch's channel count, which a sensible
    width-pruner config should never produce; it warns loudly when it does.
    """
    for b in profile.branches:
        if b.name == branch_name:
            V = b.principal_components.float()
            if cols_needed <= V.shape[1]:
                return V[:, :cols_needed].contiguous()
            import warnings

            warnings.warn(
                f"_col_basis({branch_name!r}, cols_needed={cols_needed}) > "
                f"channels={V.shape[1]}; falling back to random-orthogonal padding. "
                "This indicates a width-pruner / profile mismatch.",
                stacklevel=2,
            )
            return _pad_cols_orthogonal(V.contiguous(), cols_needed)
    raise KeyError(f"branch {branch_name!r} not in profile")


@torch.no_grad()
def _copy_layernorm(src: nn.Module, dst: nn.Module, V_r: Tensor) -> None:
    """Project a teacher LayerNorm's diagonal scale into the student."""
    w = getattr(src, "weight", None)
    b = getattr(src, "bias", None)
    dw = getattr(dst, "weight", None)
    db = getattr(dst, "bias", None)
    if w is not None and dw is not None and dw.shape[0] == V_r.shape[1]:
        Vr = V_r.to(device=w.device, dtype=w.dtype)
        proj = (Vr.pow(2) * w.detach().unsqueeze(1)).sum(dim=0)
        dw.data.copy_(proj.to(dw.dtype).to(dw.device))
    if b is not None and db is not None and db.shape[0] == V_r.shape[1]:
        Vr = V_r.to(device=b.device, dtype=b.dtype)
        proj_b = Vr.T @ b.detach()
        db.data.copy_(proj_b.to(db.dtype).to(db.device))


# -- Llama -------------------------------------------------------------


def _build_llama(teacher, profile, student_cfg: StudentConfig, absorbed_init: bool):
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fasd.build_student template='llama' requires the 'transformers' package"
        ) from e

    t_cfg = teacher.config
    s_cfg = LlamaConfig(
        vocab_size=int(t_cfg.vocab_size),
        max_position_embeddings=int(getattr(t_cfg, "max_position_embeddings", 2048)),
        hidden_size=int(student_cfg.hidden_size),
        intermediate_size=int(student_cfg.intermediate_size),
        num_hidden_layers=int(student_cfg.num_hidden_layers),
        num_attention_heads=int(student_cfg.num_attention_heads),
        num_key_value_heads=int(student_cfg.num_key_value_heads),
        rms_norm_eps=float(getattr(t_cfg, "rms_norm_eps", 1e-6)),
        rope_theta=float(getattr(t_cfg, "rope_theta", 10000.0)),
        hidden_act=getattr(t_cfg, "hidden_act", "silu"),
    )
    student = LlamaForCausalLM(s_cfg)

    if not absorbed_init:
        return student

    t_layers = teacher.model.layers
    s_layers = student.model.layers
    t_hidden = int(t_cfg.hidden_size)
    s_hidden = int(s_cfg.hidden_size)
    s_interm = int(s_cfg.intermediate_size)

    V_r = _residual_basis(profile, t_hidden, s_hidden)

    with torch.no_grad():
        emb_t = teacher.model.embed_tokens.weight.detach()
        Vr = V_r.to(emb_t.device, emb_t.dtype)
        student.model.embed_tokens.weight.data.copy_(
            (emb_t @ Vr).to(student.model.embed_tokens.weight.dtype).to(
                student.model.embed_tokens.weight.device
            )
        )

    # For GQA, k_proj / v_proj output dim is `num_kv_heads * head_dim`, not hidden_size.
    head_dim = s_hidden // int(s_cfg.num_attention_heads)
    s_kv_out = int(s_cfg.num_key_value_heads) * head_dim
    t_hidden_int = int(t_cfg.hidden_size)
    t_interm_int = int(getattr(t_cfg, "intermediate_size", 4 * t_hidden_int))

    for i, s_block in enumerate(s_layers):
        t_block = t_layers[i]
        prefix = f"model.layers.{i}"

        _col_basis(profile, f"{prefix}.attn.q", s_hidden)
        V_k = _col_basis(profile, f"{prefix}.attn.k", s_kv_out)
        V_v = _col_basis(profile, f"{prefix}.attn.v", s_kv_out)
        # v10: channel-select for FFN intermediate. SiLU is element-wise so
        # commutes with channel slicing; PCA rotation breaks initial behavior.
        V_gate = _channel_select_basis(profile, f"{prefix}.ffn.gate", s_interm)
        V_up = _channel_select_basis(profile, f"{prefix}.ffn.up", s_interm)

        # Pad bases out to the full teacher dim if needed (eigh returns C x C,
        # but _col_basis may truncate).
        def _pad_rows(V, rows):
            if V.shape[0] == rows:
                return V
            if V.shape[0] > rows:
                return V[:rows]
            pad = torch.zeros(rows - V.shape[0], V.shape[1], dtype=V.dtype)
            return torch.cat([V, pad], dim=0)

        kv_rows = int(
            getattr(t_cfg, "num_key_value_heads", t_cfg.num_attention_heads)
        ) * (t_hidden_int // int(t_cfg.num_attention_heads))
        V_r_full = _pad_rows(V_r, t_hidden_int)
        V_k_full = _pad_rows(V_k, kv_rows)
        V_v_full = _pad_rows(V_v, kv_rows)
        V_up_full = _pad_rows(V_up, t_interm_int)

        # Mirror the GPT-2 fix: for non-GQA, q/k/v share V_r_full so Q·K^T is
        # preserved across the projection. GQA (num_kv_heads < num_attention_heads)
        # has a smaller k/v output dim; until we derive a kv-shared sub-basis,
        # GQA still uses the legacy per-branch V_k/V_v (latent disjoint-basis bug).
        is_gqa = s_kv_out != s_hidden
        attn, s_attn = t_block.self_attn, s_block.self_attn
        absorbed_linear_init(attn.q_proj, s_attn.q_proj, V_in=V_r_full, V_out=V_r_full)
        if is_gqa:
            absorbed_linear_init(attn.k_proj, s_attn.k_proj, V_in=V_r_full, V_out=V_k_full)
            absorbed_linear_init(attn.v_proj, s_attn.v_proj, V_in=V_r_full, V_out=V_v_full)
        else:
            absorbed_linear_init(attn.k_proj, s_attn.k_proj, V_in=V_r_full, V_out=V_r_full)
            absorbed_linear_init(attn.v_proj, s_attn.v_proj, V_in=V_r_full, V_out=V_r_full)
        absorbed_linear_init(attn.o_proj, s_attn.o_proj, V_in=V_r_full, V_out=V_r_full)
        mlp, s_mlp = t_block.mlp, s_block.mlp
        absorbed_linear_init(mlp.gate_proj, s_mlp.gate_proj, V_in=V_r_full, V_out=V_gate)
        absorbed_linear_init(mlp.up_proj, s_mlp.up_proj, V_in=V_r_full, V_out=V_up_full)
        absorbed_linear_init(mlp.down_proj, s_mlp.down_proj, V_in=V_up_full, V_out=V_r_full)

        # RMSNorm scale is a diagonal, project it.
        _copy_rmsnorm(t_block.input_layernorm, s_block.input_layernorm, V_r)
        _copy_rmsnorm(t_block.post_attention_layernorm, s_block.post_attention_layernorm, V_r)

    _copy_rmsnorm(teacher.model.norm, student.model.norm, V_r)

    # lm_head
    with torch.no_grad():
        W_lm = teacher.lm_head.weight.detach()
        Vr_lm = V_r.to(device=W_lm.device, dtype=W_lm.dtype)
        student.lm_head.weight.data.copy_(
            (W_lm @ Vr_lm).to(
                student.lm_head.weight.dtype
            ).to(student.lm_head.weight.device)
        )

    return student


def llama_absorb_targets(teacher, student, profile):
    """Yield ``(name, t_module, s_module, V_in, V_out)`` for every absorbed weight.

    Recomputes the same bases ``_build_llama`` used for a Llama-family student.
    Mirror of :func:`gpt2_absorb_targets` for the Llama path — lets the CPSD
    manifold-training conversion recover each linear's ``(V_in, V_out)`` to wrap it
    as a Stiefel-trainable :class:`TeacherFactoredLinear`. Layout is ``"linear"``.
    """
    t_cfg, s_cfg = teacher.config, student.config
    t_hidden = int(t_cfg.hidden_size)
    s_hidden = int(s_cfg.hidden_size)
    s_interm = int(s_cfg.intermediate_size)
    head_dim = s_hidden // int(s_cfg.num_attention_heads)
    s_kv_out = int(s_cfg.num_key_value_heads) * head_dim
    t_interm = int(getattr(t_cfg, "intermediate_size", 4 * t_hidden))
    t_kv = int(getattr(t_cfg, "num_key_value_heads", t_cfg.num_attention_heads)) \
        * (t_hidden // int(t_cfg.num_attention_heads))

    V_r = _residual_basis(profile, t_hidden, s_hidden)

    def _pad_rows(V, rows):
        if V.shape[0] == rows:
            return V
        if V.shape[0] > rows:
            return V[:rows]
        return torch.cat([V, torch.zeros(rows - V.shape[0], V.shape[1], dtype=V.dtype)], dim=0)

    V_r_full = _pad_rows(V_r, t_hidden)
    is_gqa = s_kv_out != s_hidden
    t_layers, s_layers = teacher.model.layers, student.model.layers
    for i, s_block in enumerate(s_layers):
        t_block = t_layers[i]
        prefix = f"model.layers.{i}"
        V_k = _pad_rows(_col_basis(profile, f"{prefix}.attn.k", s_kv_out), t_kv)
        V_v = _pad_rows(_col_basis(profile, f"{prefix}.attn.v", s_kv_out), t_kv)
        V_gate = _channel_select_basis(profile, f"{prefix}.ffn.gate", s_interm)
        V_up = _pad_rows(_channel_select_basis(profile, f"{prefix}.ffn.up", s_interm), t_interm)
        kv_out = (V_k, V_v) if is_gqa else (V_r_full, V_r_full)
        attn, s_attn = t_block.self_attn, s_block.self_attn
        mlp, s_mlp = t_block.mlp, s_block.mlp
        yield (f"{prefix}.self_attn.q_proj", attn.q_proj, s_attn.q_proj, V_r_full, V_r_full)
        yield (f"{prefix}.self_attn.k_proj", attn.k_proj, s_attn.k_proj, V_r_full, kv_out[0])
        yield (f"{prefix}.self_attn.v_proj", attn.v_proj, s_attn.v_proj, V_r_full, kv_out[1])
        yield (f"{prefix}.self_attn.o_proj", attn.o_proj, s_attn.o_proj, V_r_full, V_r_full)
        yield (f"{prefix}.mlp.gate_proj", mlp.gate_proj, s_mlp.gate_proj, V_r_full, V_gate)
        yield (f"{prefix}.mlp.up_proj", mlp.up_proj, s_mlp.up_proj, V_r_full, V_up)
        yield (f"{prefix}.mlp.down_proj", mlp.down_proj, s_mlp.down_proj, V_up, V_r_full)


@torch.no_grad()
def _copy_rmsnorm(src: nn.Module, dst: nn.Module, V_r: Tensor) -> None:
    w = getattr(src, "weight", None)
    dw = getattr(dst, "weight", None)
    if w is not None and dw is not None and dw.shape[0] == V_r.shape[1]:
        Vr = V_r.to(device=w.device, dtype=w.dtype)
        proj = (Vr.pow(2) * w.detach().unsqueeze(1)).sum(dim=0)
        dw.data.copy_(proj.to(dw.dtype).to(dw.device))


# -- dispatcher --------------------------------------------------------


def build_student(
    teacher: nn.Module,
    profile,
    *,
    arch_multiplier: float = 1.0,
    absorbed_init: bool = True,
    template: Template = "auto",
    depth_policy: str = "keep",
    depth_keep: int | None = None,
    rank_map: dict[str, int] | None = None,
):
    """Construct a compressed student from a teacher profile.

    ``rank_map`` (optional): per-branch rank dict produced by
    :func:`fasd.compression.rank_allocator.allocate_ranks`. When provided,
    overrides each branch's stored ``behavioral_rank`` and disables
    ``arch_multiplier`` scaling — the rank-map already encodes the budget.
    """
    if template == "auto":
        cls_name = type(teacher).__name__
        if "GPT2" in cls_name:
            template = "gpt2"
        elif "Llama" in cls_name or "Mistral" in cls_name or "Qwen" in cls_name:
            template = "llama"
        else:
            raise ValueError(
                f"build_student could not auto-detect template for {cls_name}; "
                "pass template='gpt2' or 'llama' explicitly"
            )
    cfg = profile_to_student_config(
        profile,
        teacher_config=teacher.config,
        arch_multiplier=arch_multiplier,
        depth_policy=depth_policy,
        depth_keep=depth_keep,
        rank_map=rank_map,
    )
    if template == "gpt2":
        return _build_gpt2(teacher, profile, cfg, absorbed_init)
    if template == "llama":
        return _build_llama(teacher, profile, cfg, absorbed_init)
    raise ValueError(f"unknown template: {template!r}")


__all__ = [
    "build_student",
    "Template",
    "StudentConfig",
    "gpt2_absorb_targets",
    "gpt2_residual_basis",
]
