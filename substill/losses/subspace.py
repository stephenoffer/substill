"""F-ASD branchwise subspace loss.

Given a :class:`TeacherProfile` (a list of :class:`BranchProfile`s),
this module:

1. Stores the teacher's per-branch principal components
   ``V[:, :k_behavioral]`` as non-trainable buffers.
2. Builds per-branch learnable projectors mapping the student's branch
   activation (last dim ``C_s``) to the teacher's retained rank ``k``.
   The default projector is **semi-orthogonal** — parameterized via
   :func:`torch.nn.utils.parametrizations.orthogonal` — so the loss is
   not free to compensate for a deficient student with an expressive
   projector.
3. Applies one of three comparison objectives on the
   ``(N, k)``-reshaped coefficients:

   - ``"gram"`` — Frobenius distance between Gram matrices ``Z Z^T``
     using the trace identity. Basis-invariant.
   - ``"cka"`` — centered kernel alignment. Basis- and scale-invariant.
   - ``"procrustes"`` — whitened orthogonal Procrustes plus covariance
     and norm calibration terms. The "middle-ground" objective from
     the design brief: rotation-invariant inside the subspace but
     tight enough to drive alignment.

4. Supports a step-indexed :class:`Schedule` that linearly anneals
   between the three objectives across training.

5. After warm-up, :meth:`fold_projectors_into_` converts each
   semi-orthogonal projector to an identity map — the student is then
   supervised directly, without a learnable bridge that could hide
   deficiencies. Requires the student's branch activation channel
   count to equal the teacher's retained rank (which
   :func:`substill.build_student` ensures by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal

from .procrustes import (
    covariance_calibration,
    norm_calibration,
    procrustes_distance,
)

Objective = Literal["gram", "cka", "procrustes"]


@dataclass
class ScheduleStage:
    """One phase of a training schedule."""

    start_frac: float
    end_frac: float
    weights: dict[str, float] = field(default_factory=dict)

    def contains(self, frac: float) -> bool:
        return self.start_frac <= frac < self.end_frac


@dataclass
class Schedule:
    """Step-fraction-indexed loss objective schedule.

    Each stage specifies weights over objectives ``{"gram", "cka",
    "procrustes"}``. Between stages, weights can optionally be linearly
    interpolated; by default they switch discretely.

    ``feature_weight`` is an overall multiplier on the subspace-feature
    loss as a whole (the per-objective weights sum to it inside a
    stage). Useful for a linear fade-out at the end of training.
    """

    stages: list[ScheduleStage]
    feature_weight_schedule: list[tuple[float, float]] | None = None
    interpolate: bool = True
    blend_width: float = 0.05  # fraction of step-budget to linearly blend across stage boundaries

    def objective_weights(self, frac: float) -> dict[str, float]:
        if not self.stages:
            return {"procrustes": 1.0}

        active = None
        for stage in self.stages:
            if stage.contains(frac):
                active = stage
                break
        if active is None:
            active = self.stages[-1]

        if not self.interpolate or self.blend_width <= 0:
            return dict(active.weights)

        # Linear blend across the last `blend_width` fraction of this stage
        # into the next stage's weights, so objectives don't switch abruptly.
        stage_idx = self.stages.index(active)
        if stage_idx == len(self.stages) - 1:
            return dict(active.weights)
        end = active.end_frac
        blend_start = end - self.blend_width
        if frac < blend_start:
            return dict(active.weights)
        t = (frac - blend_start) / max(1e-8, self.blend_width)
        t = max(0.0, min(1.0, t))
        next_stage = self.stages[stage_idx + 1]
        keys = set(active.weights) | set(next_stage.weights)
        out = {}
        for k in keys:
            a = float(active.weights.get(k, 0.0))
            b = float(next_stage.weights.get(k, 0.0))
            w = (1 - t) * a + t * b
            # Drop zero-weight entries so callers that iterate keys get a
            # minimal set; also keeps behavior consistent with the
            # non-blending path.
            if w > 0.0:
                out[k] = w
        return out

    def feature_weight(self, frac: float) -> float:
        if not self.feature_weight_schedule:
            return 1.0
        pts = self.feature_weight_schedule
        if frac <= pts[0][0]:
            return pts[0][1]
        if frac >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if x0 <= frac <= x1:
                if x1 == x0:
                    return y1
                return y0 + (y1 - y0) * (frac - x0) / (x1 - x0)
        return pts[-1][1]


def default_schedule() -> Schedule:
    """Default transformer schedule: gram → cka → procrustes → fade."""
    return Schedule(
        stages=[
            ScheduleStage(0.0, 0.10, {"gram": 1.0}),
            ScheduleStage(0.10, 0.40, {"cka": 1.0}),
            ScheduleStage(0.40, 1.0, {"procrustes": 1.0}),
        ],
        feature_weight_schedule=[(0.0, 1.0), (0.80, 1.0), (1.0, 0.0)],
    )


# -- loss primitives ---------------------------------------------------


def _l2_normalize(Z: Tensor, eps: float = 1e-6) -> Tensor:
    """L2 normalize per-sample over the last dim."""
    return F.normalize(Z, dim=-1, eps=eps)


def _to_nk(Z: Tensor) -> Tensor:
    """Flatten any leading batch/time dims to give ``(N, k)``."""
    if Z.dim() <= 1:
        raise ValueError(f"expected 2+D tensor, got {Z.shape}")
    if Z.dim() == 2:
        return Z
    return Z.reshape(-1, Z.shape[-1])


def gram_distance(Z_s: Tensor, Z_t: Tensor) -> Tensor:
    """``(1/N^2) * ||Z_s Z_s^T - Z_t Z_t^T||_F^2`` via the trace identity.

    ``||K_s - K_t||^2 = tr(K_s K_s) - 2 tr(K_s K_t) + tr(K_t K_t)``,
    and ``tr(K_s K_t) = ||Z_s^T Z_t||^2``. Only ``k x k`` inner
    products are materialized.
    """
    if Z_s.shape != Z_t.shape:
        raise ValueError(f"shape mismatch: {Z_s.shape} vs {Z_t.shape}")
    N = Z_s.shape[0]
    if N == 0:
        return Z_s.new_zeros(())
    ss = (Z_s.T @ Z_s).pow(2).sum()
    tt = (Z_t.T @ Z_t).pow(2).sum()
    st = (Z_s.T @ Z_t).pow(2).sum()
    return (ss + tt - 2.0 * st) / float(N * N)


def cka_distance(Z_s: Tensor, Z_t: Tensor, eps: float = 1e-8) -> Tensor:
    """``1 - linear CKA(Z_s, Z_t)``. Scale- and orthogonal-invariant."""
    if Z_s.shape != Z_t.shape:
        raise ValueError(f"shape mismatch: {Z_s.shape} vs {Z_t.shape}")
    N = Z_s.shape[0]
    if N < 2:
        return Z_s.new_zeros(())
    Zs = Z_s - Z_s.mean(dim=0, keepdim=True)
    Zt = Z_t - Z_t.mean(dim=0, keepdim=True)
    num = (Zs.T @ Zt).pow(2).sum()
    den_s = (Zs.T @ Zs).pow(2).sum()
    den_t = (Zt.T @ Zt).pow(2).sum()
    den = (den_s * den_t).clamp_min(eps).sqrt()
    sim = num / den
    return (1.0 - sim).clamp_min(0.0)


# -- the loss module ---------------------------------------------------


class F_ASDLoss(nn.Module):
    """Branchwise subspace distillation loss.

    Parameters
    ----------
    profile
        :class:`TeacherProfile` (or any iterable of :class:`BranchProfile`).
    student_widths
        Optional per-branch student output widths. If omitted, the
        projector is built lazily on first forward from the incoming
        hidden shape.
    objective
        Default objective if ``schedule`` is not supplied.
    schedule
        Optional :class:`Schedule` that overrides ``objective`` per step.
    normalize_features
        L2-normalize the projected student and teacher vectors before
        computing the loss (default True).
    projector
        ``"semiortho"`` (default) uses a semi-orthogonal parameterization
        so the projector cannot scale or skew features arbitrarily.
        ``"linear"`` uses an unconstrained ``nn.Linear``.
    calibration_lambdas
        ``(lambda_cov, lambda_norm)`` weights for the covariance- and
        norm-calibration terms added to the Procrustes loss. Default
        ``(0.01, 0.001)`` matches the design brief.
    instability_weights
        Optional dict ``branch_name -> [0, 1]`` multiplier on the
        per-branch loss contribution. Used by the driver's
        ``instability_downweight`` path.
    """

    def __init__(
        self,
        profile,
        *,
        student_widths: dict[str, int] | None = None,
        objective: Objective = "procrustes",
        schedule: Schedule | None = None,
        normalize_features: bool = True,
        projector: Literal["semiortho", "linear"] = "semiortho",
        calibration_lambdas: tuple[float, float] = (0.01, 0.001),
        instability_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        # Accept TeacherProfile or a raw list of BranchProfile.
        branches = list(profile.branches) if hasattr(profile, "branches") else list(profile)
        if not branches:
            raise ValueError("F_ASDLoss requires at least one BranchProfile")

        self.branch_names: list[str] = [b.name for b in branches]
        self.branch_ks: dict[str, int] = {
            b.name: int(b.behavioral_rank) for b in branches
        }
        self.branch_dims: dict[str, int] = {
            b.name: int(b.principal_components.shape[0]) for b in branches
        }
        self.objective: Objective = objective
        self.schedule = schedule
        self.normalize_features = normalize_features
        self.projector_kind = projector
        self.lambda_cov, self.lambda_norm = calibration_lambdas
        self.instability_weights = dict(instability_weights or {})

        # Store teacher bases V[:, :k] per branch as buffers.
        for i, b in enumerate(branches):
            V = b.principal_components[:, : int(b.behavioral_rank)].contiguous().float()
            self.register_buffer(f"V_{i}", V)
        self._branch_index: dict[str, int] = {
            name: i for i, name in enumerate(self.branch_names)
        }

        # Projectors are lazily built; optionally pre-allocated from student_widths.
        self.projectors = nn.ModuleDict()
        if student_widths is not None:
            for name, width in student_widths.items():
                k = self.branch_ks.get(name)
                if k is None:
                    continue
                self.projectors[self._safe_key(name)] = self._make_projector(width, k)
        self._folded: set[str] = set()

    # -- projector plumbing -------------------------------------------

    @staticmethod
    def _safe_key(name: str) -> str:
        return name.replace(".", "_")

    def _make_projector(self, in_dim: int, out_dim: int) -> nn.Module:
        layer = nn.Linear(in_dim, out_dim, bias=False)
        if self.projector_kind == "semiortho" and in_dim >= out_dim:
            layer = orthogonal(layer)
        return layer

    def _get_v(self, name: str) -> Tensor:
        i = self._branch_index[name]
        return getattr(self, f"V_{i}")

    def _project_teacher(self, name: str, hidden: Tensor) -> Tensor:
        V = self._get_v(name).to(device=hidden.device, dtype=hidden.dtype)
        return _to_nk(hidden @ V)

    def _project_student(self, name: str, hidden: Tensor) -> Tensor:
        key = self._safe_key(name)
        k = self.branch_ks[name]
        in_dim = hidden.shape[-1]
        if name in self._folded:
            if in_dim != k:
                raise ValueError(
                    f"branch {name!r} is folded but student hidden dim {in_dim} "
                    f"!= retained rank {k}. Re-fold after shape changes."
                )
            return _to_nk(hidden)
        if key not in self.projectors:
            self.projectors[key] = self._make_projector(in_dim, k).to(
                device=hidden.device, dtype=hidden.dtype
            )
        proj = self.projectors[key].to(device=hidden.device, dtype=hidden.dtype)
        return _to_nk(proj(hidden))

    # -- public forward -----------------------------------------------

    def forward(
        self,
        student_hiddens: dict[str, Tensor],
        teacher_hiddens: dict[str, Tensor],
        *,
        step_frac: float | None = None,
    ) -> Tensor:
        weights = self._resolve_weights(step_frac)
        if not weights:
            # nothing to compute
            return teacher_hiddens[self.branch_names[0]].new_zeros(())

        total = None
        count = 0
        for name in self.branch_names:
            if name not in student_hiddens or name not in teacher_hiddens:
                continue
            s_hid = student_hiddens[name]
            t_hid = teacher_hiddens[name]

            Z_s = self._project_student(name, s_hid)
            Z_t = self._project_teacher(name, t_hid).detach()
            if self.normalize_features:
                Z_s = _l2_normalize(Z_s)
                Z_t = _l2_normalize(Z_t)

            per_branch = self._compute_loss(Z_s, Z_t, weights)
            w = self.instability_weights.get(name, 1.0)
            per_branch = per_branch * float(w)

            total = per_branch if total is None else total + per_branch
            count += 1

        if total is None:
            # no matching branch names — return zero with the right device/dtype
            ref = next(iter(teacher_hiddens.values()))
            return ref.new_zeros(())
        loss = total / float(count)
        if self.schedule is not None and step_frac is not None:
            loss = loss * self.schedule.feature_weight(step_frac)
        return loss

    def _compute_loss(
        self, Z_s: Tensor, Z_t: Tensor, weights: dict[str, float]
    ) -> Tensor:
        parts = []
        if weights.get("gram", 0.0) > 0:
            parts.append(weights["gram"] * gram_distance(Z_s, Z_t))
        if weights.get("cka", 0.0) > 0:
            parts.append(weights["cka"] * cka_distance(Z_s, Z_t))
        if weights.get("procrustes", 0.0) > 0:
            p = procrustes_distance(Z_s, Z_t)
            p = p + self.lambda_cov * covariance_calibration(Z_s, Z_t)
            p = p + self.lambda_norm * norm_calibration(Z_s, Z_t)
            parts.append(weights["procrustes"] * p)
        if not parts:
            return Z_s.new_zeros(())
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out

    def _resolve_weights(self, step_frac: float | None) -> dict[str, float]:
        if self.schedule is not None and step_frac is not None:
            return self.schedule.objective_weights(step_frac)
        return {self.objective: 1.0}

    # -- runtime mutation: profile refresh ----------------------------

    def refresh_from_profile(self, profile) -> None:
        """Hot-swap per-branch bases (and ranks, where safe) from a refresh.

        For folded branches the student's hidden dim is locked to the old
        retained rank. The refresh may only update the *basis* (first
        ``old.shape[1]`` columns of the new PCA projected onto the same row
        space); rank changes are ignored to preserve the folded forward pass.

        For non-folded branches the projector is rebuilt on next forward when
        the retained rank changes.

        The on-policy refresh can produce a new PCA with a much larger
        ``behavioral_rank`` (e.g. 2664) for ``ffn.up`` while the student's
        intermediate dim is 768; a blind swap would break the next forward pass.
        """
        branches = list(profile.branches if hasattr(profile, "branches") else profile)
        for b in branches:
            if b.name not in self._branch_index:
                continue
            i = self._branch_index[b.name]
            old = getattr(self, f"V_{i}")
            new_rank = int(b.behavioral_rank)
            new_V = b.principal_components[:, :new_rank].contiguous().float()

            if b.name in self._folded:
                if (
                    new_V.shape[0] == old.shape[0]
                    and new_V.shape[1] >= old.shape[1]
                ):
                    old.copy_(
                        new_V[:, : old.shape[1]].to(device=old.device, dtype=old.dtype)
                    )
                # Rank is locked for folded branches — don't touch branch_ks.
                continue

            if new_V.shape != old.shape:
                delattr(self, f"V_{i}")
                self.register_buffer(f"V_{i}", new_V)
                key = self._safe_key(b.name)
                if key in self.projectors:
                    del self.projectors[key]
            else:
                old.copy_(new_V.to(device=old.device, dtype=old.dtype))
            self.branch_ks[b.name] = new_rank

    # -- runtime mutation: fold projectors ----------------------------

    def fold_projectors_into_(self, student: nn.Module) -> None:
        """Replace learned projectors with identity after warm-up.

        Semantics: the learned semi-orthogonal projector maps student
        hidden dim ``C_s`` to teacher retained rank ``k``. When
        ``C_s == k`` — which :func:`substill.build_student` ensures when
        ``absorbed_init=True`` — the projector is purely a rotation
        inside the student's branch space, and can be absorbed into
        the student by composing with the adjacent linear's weight
        without changing the loss landscape.

        For branches where ``C_s != k`` we cannot fold without
        rewriting the student architecture; we leave those projectors
        alone and log the skipped branch names in the returned list.

        Returns the list of skipped (unfolded) branch names.
        """
        skipped: list[str] = []
        for name in list(self.branch_names):
            if name in self._folded:
                continue
            key = self._safe_key(name)
            if key not in self.projectors:
                # Nothing learned yet → treat as folded if shapes match.
                self._folded.add(name)
                continue
            proj = self.projectors[key]
            # Extract the weight (handles orthogonal parametrization too).
            if isinstance(proj, nn.Linear):
                W = proj.weight.detach()
            else:
                # parametrized module exposes .weight through forward
                W = getattr(proj, "weight", None)
                if W is None:
                    skipped.append(name)
                    continue
                W = W.detach()
            k = self.branch_ks[name]
            if W.shape != (k, k):
                skipped.append(name)
                continue
            # Compose into the teacher basis buffer V_i:
            #   z_s = proj(s_hid); if we post-compose V <- V @ W^T on teacher,
            #   we can drop proj without changing loss.
            i = self._branch_index[name]
            V = getattr(self, f"V_{i}")
            new_V = (V @ W.T.to(V.device, V.dtype)).contiguous()
            delattr(self, f"V_{i}")
            self.register_buffer(f"V_{i}", new_V)
            del self.projectors[key]
            self._folded.add(name)
        return skipped


__all__ = [
    "F_ASDLoss",
    "Schedule",
    "ScheduleStage",
    "default_schedule",
    "gram_distance",
    "cka_distance",
]
