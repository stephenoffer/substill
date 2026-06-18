"""Top-level F-ASD API: profile, capture, TeacherProfile, BranchProfile.

Profile produces a pickle-safe snapshot, capture exposes per-branch hidden
states, and :class:`F_ASDLoss` plugs into the user's training loop.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .autodetect import BranchSpec, autodetect_branches
from .profiling.activation_capture import (
    BranchCaptureEngine,
    BranchHiddenCapture,
)
from .profiling.behavioral_rank import choose_behavioral_rank
from .profiling.stability import bootstrap_principal_angles  # noqa: F401
from .profiling.streaming_pca import auto_backend
from .profiling.token_weighting import Method as WeightMethod
from .profiling.token_weighting import compute_weights

# -- BranchProfile / TeacherProfile ------------------------------------


@dataclass
class BranchProfile:
    """Profile for a single branch."""

    name: str
    kind: str
    module_path: str
    principal_components: Tensor  # (C, C) descending eigenvector matrix
    eigenvalues: Tensor  # (C,)
    behavioral_rank: int
    variance_rank: int
    channels: int
    kl_curve: list[tuple[int, float]] = field(default_factory=list)
    slice: tuple[int, int] | None = None
    calibration_meta: dict = field(default_factory=dict)


@dataclass
class TeacherProfile:
    """Pickle-safe branchwise teacher profile."""

    branches: list[BranchProfile]
    meta: dict = field(default_factory=dict)

    def names(self) -> list[str]:
        return [b.name for b in self.branches]

    def get(self, name: str) -> BranchProfile | None:
        for b in self.branches:
            if b.name == name:
                return b
        return None

    # save/load round-trip ------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> TeacherProfile:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected TeacherProfile at {path}, got {type(obj)}")
        return obj


@dataclass
class DistillResult:
    """Return value from :func:`fasd.distill`."""

    student: nn.Module
    profile: TeacherProfile
    history: list[dict]
    best_metric: float | None = None
    final_metric: float | None = None
    teacher_metric: float | None = None
    val_kl_forward: float | None = None
    val_kl_reverse: float | None = None


# -- profile -----------------------------------------------------------


def _variance_rank(eigvals: Tensor, threshold: float = 0.95) -> int:
    eig = eigvals.clamp_min(0)
    total = eig.sum().clamp_min(1e-8)
    cum = torch.cumsum(eig, dim=0) / total
    k = int((cum >= threshold).to(torch.long).argmax().item()) + 1
    return max(1, min(k, eigvals.numel()))


def _collect_calib_batches(
    dataloader, n_batches: int, device
) -> list:
    batches = []
    for i, b in enumerate(dataloader):
        if i >= n_batches:
            break
        batches.append(b)
    return batches


def _teacher_forward_logits(model, batch, device):
    if isinstance(batch, dict):
        b = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
        out = model(**b)
    elif isinstance(batch, Tensor):
        out = model(batch.to(device))
    else:
        b = tuple(x.to(device) if isinstance(x, Tensor) else x for x in batch)
        out = model(*b)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"teacher output type unrecognized: {type(out)}")


def profile(
    model: nn.Module,
    dataloader,
    *,
    branches: Sequence[BranchSpec] | None = None,
    mode: Literal["branch", "residual"] = "branch",
    rank_tol: float = 0.02,
    token_weighting: WeightMethod = "entropy",
    variance_threshold: float = 0.95,
    max_rank: int | None = None,
    min_rank: int = 1,
    search: Literal["bisect", "linear"] = "bisect",
    pca_backend: Literal["auto", "exact", "randomized", "oja"] = "auto",
    n_calib_batches: int | None = None,
    behavioral_calib_batches: int | None = None,
    device: str | torch.device | None = None,
) -> TeacherProfile:
    """Run the F-ASD profile over a calibration dataloader.

    The pipeline:

    1. Resolve branches (autodetect or user-supplied).
    2. Run ``model`` over the loader, accumulate per-branch PCA.
    3. Gather the top principal components ``V`` and eigenvalues per
       branch.
    4. For each branch, run :func:`choose_behavioral_rank` on a small
       subset of calibration batches to pick the smallest ``k`` whose
       projection preserves teacher logits.
    5. Pack into a :class:`TeacherProfile`.

    Parameters
    ----------
    rank_tol
        Max acceptable KL between unpatched and patched teacher for a
        branch's chosen ``k``.
    token_weighting
        One of ``"uniform"``, ``"entropy"``, ``"disagreement"``,
        ``"completion"``. Applied only to the behavioral-rank scoring
        step; PCA accumulation is always uniform across tokens.
    pca_backend
        ``"auto"`` (default), or force ``"exact"`` / ``"randomized"``
        / ``"oja"``. See :mod:`fasd.profiling.streaming_pca`.
    n_calib_batches
        Max batches to consume from ``dataloader`` for PCA. ``None``
        means use all.
    behavioral_calib_batches
        Max batches used for the rank-picking forward passes. Smaller
        than ``n_calib_batches`` is fine; default takes up to 8.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    model.eval()
    model.to(device)

    if branches is None:
        branches = autodetect_branches(model, mode=mode)
    branches = list(branches)

    # PCA accumulation pass.
    engine = BranchCaptureEngine(model, branches, accumulator_device=str(device))
    acc_iter = dataloader if n_calib_batches is None else _limit_iter(dataloader, n_calib_batches)
    accumulators = engine.run(acc_iter, device=device)

    # For backend selection.
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        n_params = 0

    branch_profiles: list[BranchProfile] = []
    for spec in branches:
        acc = accumulators.get(spec.name)
        if acc is None:
            continue
        C = acc.num_channels
        backend = pca_backend
        if backend == "auto":
            backend = auto_backend(C, n_params)
        # Exact path re-uses the accumulator directly; other backends
        # would need a second pass, which we keep out of v0.1 profile
        # (StreamingPCA is independently exposed for users who want
        # to feed it manually). Here we decode via eigendecomposition
        # of the accumulated covariance.
        cov = acc.finalize().to(device).float()
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0.0)
        eigvecs = eigvecs[:, order].contiguous()
        var_rank = _variance_rank(eigvals, variance_threshold)

        # Behavioral-rank scoring on a small calibration sub-batch set.
        max_b = behavioral_calib_batches or min(8, n_calib_batches or 8)
        behavioral_batches = _collect_calib_batches(dataloader, max_b, device)

        # Optional per-batch token weights.
        weight_list: list[Tensor | None] | None = None
        if token_weighting != "uniform" and token_weighting is not None:
            weight_list = []
            with torch.no_grad():
                for b in behavioral_batches:
                    try:
                        t_logits = _teacher_forward_logits(model, b, device)
                    except Exception:
                        weight_list.append(None)
                        continue
                    if token_weighting == "entropy":
                        w = compute_weights(
                            "entropy",
                            teacher_logits=t_logits,
                            device=t_logits.device,
                        )
                    else:
                        # disagreement / completion require extra inputs we
                        # don't carry in profile(); fall back to uniform.
                        w = None
                    weight_list.append(w)

        k_ceiling = min(max_rank or C, C)
        chosen_k, kl_curve = choose_behavioral_rank(
            model,
            spec,
            eigvecs,
            behavioral_batches,
            tol=rank_tol,
            search=search,
            token_weights=weight_list,
            max_rank=k_ceiling,
            min_rank=min_rank,
            device=device,
        )

        branch_profiles.append(
            BranchProfile(
                name=spec.name,
                kind=spec.kind,
                module_path=spec.module_path,
                principal_components=eigvecs.detach().cpu(),
                eigenvalues=eigvals.detach().cpu(),
                behavioral_rank=int(chosen_k),
                variance_rank=int(var_rank),
                channels=int(C),
                kl_curve=list(kl_curve),
                slice=spec.slice,
                calibration_meta={
                    "n_samples": int(acc.n),
                    "pca_backend": backend,
                    "rank_tol": float(rank_tol),
                    "token_weighting": token_weighting or "uniform",
                },
            )
        )

    return TeacherProfile(
        branches=branch_profiles,
        meta={
            "n_branches": len(branch_profiles),
            "rank_tol": float(rank_tol),
            "mode": mode,
            "device": str(device),
        },
    )


def _limit_iter(loader, n: int):
    for i, b in enumerate(loader):
        if i >= n:
            break
        yield b


# -- capture -----------------------------------------------------------


def capture(
    model: nn.Module,
    profile: TeacherProfile,
    *,
    detach: bool = False,
) -> BranchHiddenCapture:
    """Context manager that hooks ``model`` at the profile's branches."""
    specs = [
        BranchSpec(
            name=b.name,
            module_path=b.module_path,
            kind=b.kind,  # type: ignore[arg-type]
            slice=b.slice,
        )
        for b in profile.branches
    ]
    return BranchHiddenCapture(model, specs, detach=detach)


__all__ = [
    "BranchProfile",
    "TeacherProfile",
    "DistillResult",
    "profile",
    "capture",
]
