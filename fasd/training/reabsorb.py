"""Periodic Re-Absorption (PRA) — re-derive teacher PCA bases mid-training
and re-project the student onto them while preserving the learned residual.

Why this exists
---------------
Static absorbed init places the student inside the teacher's PCA-derived
subspace, computed once from a calibration set. As training progresses
(and especially when on-policy rollouts kick in), the activation
distribution drifts off that calibration subspace, and the carefully
aligned bases go stale. v10-apr27 saw rung 5 (on-policy) regress from
71 → 77 PPL and rung 7 (full) from 71 → 105 PPL — adding mechanisms on
top of static absorption made things worse, not better.

PRA addresses this by periodically:
  1. Refreshing teacher PCA on a fresh batch (calibration or on-policy).
  2. Re-projecting student weights onto the new bases — but preserving
     the learned residual ``Δ = W_s - V_out_old^T W_T V_in_old`` rather
     than wiping it. The residual is rotated from the old basis to the
     new one with the closed-form transform derived below.
  3. Rotating Adam first moment ``m`` the same way; resetting second
     moment ``v`` (rotation does not preserve element-wise squares
     cleanly, so we accept a fresh ``v`` rather than corrupt it).

Math
----
For a Linear weight ``W`` of shape ``(k_out, k_in)`` we have
``y = W x + b``. If we trained ``W = V_out_old^T W_T V_in_old + Δ``
in the old basis, the same Δ expressed in a new basis is::

    R_in  = V_in_new^T  @ V_in_old      # (k_in,  k_in)  — new basis sees
    R_out = V_out_new^T @ V_out_old     # (k_out, k_out)   old basis input
    Δ_new = R_out @ Δ @ R_in.T

Derivation: rotate input ``x_new → x_old = R_in.T x_new`` (because
``V_in_old^T V_in_new = R_in.T`` projects the new compressed input
into the old subspace), apply Δ, then rotate output back.

For HF GPT-2 ``Conv1D`` storage ``(k_in, k_out)`` the same logic with
swapped roles gives ``Δ_new = R_in @ Δ @ R_out.T``.

For the embedding tables ``wte``, ``wpe`` (shape ``(vocab, k_r)``)
only the residual basis is on the output side, so::

    Δ_emb_new = Δ_emb @ R_r.T

LayerNorms are diagonal and are NOT rotated — basis change of a
diagonal scale is ill-defined for non-orthonormal channel-select
bases, and the LN parameters are tiny, so we keep the current trained
values.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from ..api import BranchProfile, TeacherProfile
from ..autodetect import BranchSpec
from ..builders import gpt2_absorb_targets, gpt2_residual_basis
from ..compression.absorbed_init import absorbed_weight, absorbed_bias, _infer_layout
from ..profiling.activation_capture import BranchCaptureEngine


@torch.no_grad()
def recompute_principal_components(
    teacher: nn.Module,
    profile: TeacherProfile,
    calib_batches: Iterable,
    *,
    device: str | torch.device | None = None,
) -> TeacherProfile:
    """Run PCA on a fresh calibration batch, return a new profile with
    refreshed ``principal_components`` and ``eigenvalues`` but the SAME
    ``behavioral_rank`` per branch as the input profile.

    Architecture-affecting fields (rank, channels, slice, kind, name) are
    preserved so re-absorption can re-use the same student layout.
    """
    if device is None:
        device = next(teacher.parameters()).device
    teacher.eval()
    teacher.to(device)

    branches = [
        BranchSpec(
            name=b.name,
            module_path=b.module_path,
            kind=b.kind,  # type: ignore[arg-type]
            slice=b.slice,
        )
        for b in profile.branches
    ]
    engine = BranchCaptureEngine(teacher, branches, accumulator_device=str(device))
    accumulators = engine.run(list(calib_batches), device=device)

    new_branches: list[BranchProfile] = []
    for old in profile.branches:
        acc = accumulators.get(old.name)
        if acc is None:
            # No fresh data for this branch — keep old PCA.
            new_branches.append(old)
            continue
        cov = acc.finalize().to(device).float()
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0.0)
        eigvecs = eigvecs[:, order].contiguous()
        new_branches.append(
            replace(
                old,
                principal_components=eigvecs.detach().cpu(),
                eigenvalues=eigvals.detach().cpu(),
            )
        )

    return TeacherProfile(branches=new_branches, meta={**profile.meta, "refreshed": True})


def _rotation_in(V_in_new: Tensor, V_in_old: Tensor) -> Tensor:
    """R_in = V_in_new^T @ V_in_old, shape (k_in, k_in)."""
    return (V_in_new.T @ V_in_old).contiguous()


def _rotation_out(V_out_new: Tensor, V_out_old: Tensor) -> Tensor:
    """R_out = V_out_new^T @ V_out_old, shape (k_out, k_out)."""
    return (V_out_new.T @ V_out_old).contiguous()


def _rotate_2d(W: Tensor, R_out: Tensor, R_in: Tensor, layout: str) -> Tensor:
    """Apply Δ_new = R_out @ Δ @ R_in.T (linear) or R_in @ Δ @ R_out.T (conv1d).

    Works in the weight's native dtype/device.
    """
    R_out = R_out.to(device=W.device, dtype=W.dtype)
    R_in = R_in.to(device=W.device, dtype=W.dtype)
    if layout == "linear":
        return R_out @ W @ R_in.T
    if layout == "conv1d_gpt2":
        return R_in @ W @ R_out.T
    raise ValueError(f"unknown layout {layout!r}")


def _rotate_1d(b: Tensor, R_out: Tensor) -> Tensor:
    R_out = R_out.to(device=b.device, dtype=b.dtype)
    return R_out @ b


def _reset_optim_state_for(optimizer, params: list[torch.nn.Parameter]) -> None:
    """Zero the Adam state for the given params (m and v both reset)."""
    if optimizer is None:
        return
    for p in params:
        if p in optimizer.state:
            optimizer.state[p] = {}


def _rotate_optim_state_for(
    optimizer,
    p: torch.nn.Parameter,
    R_out: Tensor,
    R_in: Tensor | None,
    layout: str,
    rotate_v: bool = False,
) -> None:
    """Reset Adam state for a re-projected parameter.

    The simple-and-correct option after a basis rotation: zero BOTH ``m`` and
    ``v``. Reasons we don't rotate ``m`` alone:

    - Adam's update is ``lr * m_hat / (sqrt(v_hat) + eps)``. After PRA we
      were rotating ``m`` (preserving magnitude) but zeroing ``v``. With
      ``v=0`` the denominator collapses to ``eps`` and the next step is
      ``m / eps`` — an effective learning-rate multiplier of ~10^8. Smoke
      v11-smoke2 step=50 task_loss=915 confirmed this catastrophic blowup.
    - ``v`` cannot be cleanly rotated (it is element-wise squared, so a
      rotation breaks non-negativity).
    - Resetting both means the next ~10 steps run as if Adam was just
      initialized for these params, which is mild — the student weight
      already has the rotated Δ, so the function is preserved; we just
      lose 5-20 steps of momentum buildup.
    """
    if optimizer is None:
        return
    state = optimizer.state.get(p)
    if not state:
        return
    m = state.get("exp_avg")
    if m is not None:
        state["exp_avg"] = torch.zeros_like(m)
    v = state.get("exp_avg_sq")
    if v is not None:
        state["exp_avg_sq"] = torch.zeros_like(v)
    # AdamW uses 'step' as a scalar tensor. Reset to 0 so bias correction
    # treats this param as fresh (otherwise the (1 - beta^t) correction
    # would over-amplify the first post-PRA gradient).
    if "step" in state:
        try:
            state["step"] = torch.zeros_like(state["step"]) if torch.is_tensor(state["step"]) else 0
        except Exception:
            state["step"] = 0


@torch.no_grad()
def reabsorb_gpt2(
    teacher: nn.Module,
    student: nn.Module,
    profile_old: TeacherProfile,
    calib_batches: Iterable,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device | None = None,
) -> TeacherProfile:
    """Periodic re-absorption for a GPT-2 student.

    Refreshes the teacher PCA on ``calib_batches``, then re-projects every
    absorbed weight in ``student`` from the old basis to the new one while
    preserving the learned residual ``Δ``. Optimizer state for each
    re-projected parameter is rotated (m) and reset (v).

    Returns the new profile, which the caller should also push into the
    loss function via ``loss_fn.refresh_from_profile(new_profile)``.
    """
    if device is None:
        device = next(student.parameters()).device

    profile_new = recompute_principal_components(
        teacher, profile_old, calib_batches, device=device
    )

    # Old / new V tuples per absorbed module.
    old_targets = list(gpt2_absorb_targets(teacher, student, profile_old))
    new_targets = list(gpt2_absorb_targets(teacher, student, profile_new))
    if len(old_targets) != len(new_targets):
        raise RuntimeError("re-absorb target count changed between profiles")

    # Per-weight rotation.
    for (name_o, t_mod, s_mod, V_in_old, V_out_old), (
        name_n, _t2, _s2, V_in_new, V_out_new,
    ) in zip(old_targets, new_targets):
        assert name_o == name_n, f"target order mismatch: {name_o} vs {name_n}"
        layout = _infer_layout(s_mod)

        # Old / new init weights and biases.
        W_T = t_mod.weight.detach()
        W_init_old = absorbed_weight(W_T, V_in_old.to(W_T), V_out_old.to(W_T), layout=layout)
        W_init_new = absorbed_weight(W_T, V_in_new.to(W_T), V_out_new.to(W_T), layout=layout)
        W_s = s_mod.weight.data
        Delta = (W_s.to(W_init_old) - W_init_old)

        R_in = _rotation_in(V_in_new, V_in_old)
        R_out = _rotation_out(V_out_new, V_out_old)
        Delta_new = _rotate_2d(Delta, R_out, R_in, layout)
        W_new = (W_init_new + Delta_new).to(W_s.dtype).to(W_s.device)
        if W_new.shape != W_s.shape:
            raise RuntimeError(
                f"reabsorb {name_o}: shape mismatch {W_new.shape} vs {W_s.shape}"
            )
        W_s.copy_(W_new)

        # Bias.
        b_t = getattr(t_mod, "bias", None)
        b_s = getattr(s_mod, "bias", None)
        if b_s is not None and b_t is not None and layout == "linear":
            b_t_d = b_t.detach()
            b_init_old = absorbed_bias(b_t_d, V_out_old.to(b_t_d))
            b_init_new = absorbed_bias(b_t_d, V_out_new.to(b_t_d))
            db = (b_s.data.to(b_init_old) - b_init_old)
            db_new = _rotate_1d(db, R_out)
            b_s.data.copy_((b_init_new + db_new).to(b_s.dtype).to(b_s.device))
        elif b_s is not None and b_t is not None and layout == "conv1d_gpt2":
            # Conv1D bias is in output space; treat the same way.
            b_t_d = b_t.detach()
            b_init_old = absorbed_bias(b_t_d, V_out_old.to(b_t_d))
            b_init_new = absorbed_bias(b_t_d, V_out_new.to(b_t_d))
            db = (b_s.data.to(b_init_old) - b_init_old)
            db_new = _rotate_1d(db, R_out)
            b_s.data.copy_((b_init_new + db_new).to(b_s.dtype).to(b_s.device))

        # Optimizer state: rotate m, reset v for both weight and bias.
        _rotate_optim_state_for(optimizer, s_mod.weight, R_out, R_in, layout)
        if b_s is not None:
            _rotate_optim_state_for(optimizer, b_s, R_out, None, layout)

    # Embedding tables: wte, wpe — output-side residual rotation only.
    V_r_old = gpt2_residual_basis(teacher, student, profile_old)
    V_r_new = gpt2_residual_basis(teacher, student, profile_new)
    R_r = _rotation_out(V_r_new, V_r_old)  # (s_h, s_h)
    W_emb_t = teacher.transformer.wte.weight.detach()
    W_pos_t = teacher.transformer.wpe.weight.detach()
    Vr_old_e = V_r_old.to(W_emb_t)
    Vr_new_e = V_r_new.to(W_emb_t)
    R_r_e = R_r.to(W_emb_t)

    wte = student.transformer.wte.weight
    wpe = student.transformer.wpe.weight
    wte_init_old = W_emb_t @ Vr_old_e
    wte_init_new = W_emb_t @ Vr_new_e
    d_wte = wte.data.to(wte_init_old) - wte_init_old
    wte.data.copy_(
        (wte_init_new + d_wte @ R_r_e.T).to(wte.dtype).to(wte.device)
    )

    wpe_init_old = W_pos_t @ Vr_old_e
    wpe_init_new = W_pos_t @ Vr_new_e
    d_wpe = wpe.data.to(wpe_init_old) - wpe_init_old
    wpe.data.copy_(
        (wpe_init_new + d_wpe @ R_r_e.T).to(wpe.dtype).to(wpe.device)
    )

    # Optimizer state for embeddings: m has shape (vocab, s_h). Right-multiply
    # by R_r.T to rotate output dim only.
    if optimizer is not None:
        for p in (wte, wpe):
            state = optimizer.state.get(p)
            if not state:
                continue
            m = state.get("exp_avg")
            if m is not None and m.dim() == 2:
                state["exp_avg"] = (m @ R_r.to(m).T).contiguous()
            v = state.get("exp_avg_sq")
            if v is not None:
                state["exp_avg_sq"] = torch.zeros_like(v)

    return profile_new


__all__ = ["reabsorb_gpt2", "recompute_principal_components"]
