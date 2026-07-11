"""Function-aware (Fisher-weighted) directional scoring.

Existing behavioral-rank selection (:mod:`substill.profiling.behavioral_rank`)
picks the smallest k such that patching the branch's activation onto
``V[:, :k] V[:, :k]^T`` keeps the teacher's output KL within tolerance.
That is *function-aware in spirit* — it asks "does the teacher still produce
the right logits if we project this branch onto the top-k directions?" —
but it is coarse: it scores *the rank choice*, not *individual directions*.

This module supplies the missing per-direction score::

    q_{e,i} = λ_{e,i} · E_x[ (u_{e,i}^T g_e(x))² ]

where ``λ_{e,i}`` is the eigenvalue of the i-th principal component on edge
``e`` (variance along that direction), and ``g_e(x) = ∂L_T(x)/∂h_e`` is the
teacher-loss gradient at the branch's hidden state. The expectation is over
a held-out calibration set with the teacher in eval mode.

Why this score:
  - λ alone (variance) penalises the loss by the *energy* in each direction
    but ignores task relevance. A direction with high variance that the
    teacher's output doesn't depend on can be safely dropped.
  - The gradient norm alone reflects task sensitivity but not how much the
    direction is *used* by typical inputs.
  - The product λ · E[(u^T g)²] is a Fisher-style score: it weights the
    direction's task sensitivity by the empirical density along it. This is
    formally equivalent to the diagonal of the Fisher information matrix
    in the eigenbasis when the loss is the model's negative log-likelihood
    (Theis & Kummerer, 2017; Hassibi & Stork, 1992).

The allocator (:mod:`substill.compression.rank_allocator`) consumes ``q`` to
choose ranks under a global parameter budget.

Usage::

    from substill.profiling.functional_score import score_directions

    scores = score_directions(
        teacher,
        profile,           # already-collected TeacherProfile with PCs + eigenvalues
        calib_loader,
        device="cuda",
    )
    # scores: dict[branch_name] -> Tensor of shape (k_max,)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activation_capture import _get_module


@dataclass
class DirectionScores:
    """Per-direction Fisher-weighted scores for one branch.

    ``q[i]`` corresponds to ``profile.principal_components[:, i]``. The
    eigenvalues and squared-gradient inner-products are reported separately
    for diagnostic plotting.
    """

    branch_name: str
    eigenvalues: Tensor  # (k_max,) λ_i
    grad_inner_sq: Tensor  # (k_max,) E_x[(u_i^T g_e(x))²]
    q: Tensor  # (k_max,) λ_i · E[(u_i^T g_e)²]

    def topk(self, k: int) -> Tensor:
        """Indices (into the original eigenvector ordering) of the top-k by q."""
        return torch.argsort(self.q, descending=True)[:k]


class _GradHook:
    """Forward hook returning the activation plus a tensor-grad capture.

    The grad capture lets us read ``∂L/∂a`` from the same forward pass.
    Implementation: save the activation tensor (after sliced extraction); on
    backward, the gradient flowing into that tensor is what we need.
    """

    def __init__(self, branch_name: str, slice_: tuple[int, int] | None):
        self.branch_name = branch_name
        self.slice_ = slice_
        self.captured_input: Tensor | None = None
        self.captured_grad: Tensor | None = None
        self._hook = None
        self._grad_hook = None

    def install(self, module: nn.Module):
        def fwd(mod, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if self.slice_ is not None:
                a, b = self.slice_
                sliced = out[..., a:b]
            else:
                sliced = out
            # Record activation for use in offline statistics if needed.
            self.captured_input = sliced.detach()
            # Register a grad hook on the *output* tensor to grab the upstream grad
            # for the sliced region.
            if sliced.requires_grad:
                def grad_cb(g):
                    self.captured_grad = g.detach()
                sliced.register_hook(grad_cb)

        self._hook = module.register_forward_hook(fwd)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()


@torch.enable_grad()
def score_directions(
    teacher: nn.Module,
    profile,
    calib_batches: Iterable,
    *,
    device: str | torch.device | None = None,
    loss_fn: str = "ce",
    max_rank: int | None = None,
) -> dict[str, DirectionScores]:
    """Compute per-direction Fisher-weighted scores for every branch in ``profile``.

    For each branch ``e`` with eigenvectors ``U_e ∈ R^{C_e × C_e}`` and eigenvalues
    ``λ_e``, we run the teacher with grad enabled, capture the gradient of the
    next-token CE loss w.r.t. the branch's sliced activation, project the gradient
    onto each eigenvector, square, and average over tokens / batches.

    Parameters
    ----------
    teacher
        Frozen teacher; we set ``requires_grad_(True)`` on the captured branch
        activations during the forward pass so the gradient hook fires. Teacher
        parameters do NOT have grads accumulated (we zero them out at the end).
    profile
        :class:`TeacherProfile` with principal components and eigenvalues.
    calib_batches
        Iterable of token batches.
    loss_fn
        ``"ce"`` for next-token cross-entropy (default; standard for language
        modelling). Other choices not yet implemented.
    max_rank
        Cap on directions scored per branch. Defaults to all eigenvectors.

    Returns:
    -------
    dict[branch_name -> DirectionScores]
    """
    if device is None:
        device = next(teacher.parameters()).device
    teacher.eval()  # disable dropout, but allow grads
    teacher.to(device)
    # We do NOT freeze teacher params: the autograd graph needs leaf parameters
    # with requires_grad=True for backward to compute the activation grads.
    # We zero out any accumulated grads on teacher params after each batch
    # so the teacher's state is unchanged on exit.
    saved_requires = {name: p.requires_grad for name, p in teacher.named_parameters()}
    for p in teacher.parameters():
        p.requires_grad_(True)

    # Install hooks on all branches.
    hooks: dict[str, _GradHook] = {}
    for b in profile.branches:
        h = _GradHook(b.name, b.slice)
        h.install(_get_module(teacher, b.module_path))
        hooks[b.name] = h

    # Pre-allocate accumulators for sum of squared grad-inner-products,
    # per branch, per direction.
    accumulators: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    for b in profile.branches:
        k = int(b.principal_components.shape[1])
        if max_rank is not None:
            k = min(k, max_rank)
        accumulators[b.name] = torch.zeros(k, dtype=torch.float64)
        counts[b.name] = 0

    try:
        for batch in calib_batches:
            # Move batch to device.
            if isinstance(batch, dict):
                batch = {
                    k: (v.to(device) if isinstance(v, Tensor) else v)
                    for k, v in batch.items()
                }
            elif isinstance(batch, Tensor):
                batch = batch.to(device)
            elif isinstance(batch, (tuple, list)):
                batch = tuple(b.to(device) if isinstance(b, Tensor) else b for b in batch)

            # We need the activations to require grad. Easiest: run forward,
            # capture activations via the hook (which records detached refs);
            # but for grad to flow we need them attached. So we re-do the
            # forward with grad enabled — which we already are.
            #
            # The hook captures the sliced output of each branch module. For
            # grad to flow back into the slice, the backward graph must include
            # those tensors. Forward inside teacher already builds the graph;
            # calling backward on the loss propagates into the slice.

            # Reset captured grads for this batch.
            for h in hooks.values():
                h.captured_grad = None
                h.captured_input = None

            # Forward.
            if isinstance(batch, dict):
                out = teacher(**batch)
            elif isinstance(batch, Tensor):
                out = teacher(batch)
            else:
                out = teacher(*batch)
            logits = out.logits if hasattr(out, "logits") else out

            # Compute next-token CE.
            if loss_fn != "ce":
                raise NotImplementedError(f"loss_fn={loss_fn!r} not supported yet")
            labels = (
                batch["input_ids"]
                if isinstance(batch, dict) and "input_ids" in batch
                else None
            )
            if labels is None:
                # Use shifted logits as a fallback (teacher-forced).
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = (
                    batch["labels"]
                    if isinstance(batch, dict) and "labels" in batch
                    else logits.argmax(-1)[..., 1:].contiguous()
                )
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

            # Backward.
            ce.backward()

            # Project per-branch grad onto its eigenvectors.
            for b in profile.branches:
                h = hooks[b.name]
                if h.captured_grad is None:
                    continue
                g = h.captured_grad.to(dtype=torch.float64)  # (..., C_branch)
                U = b.principal_components.to(device=g.device, dtype=g.dtype)  # (C, C)
                k = accumulators[b.name].shape[0]
                Uk = U[:, :k]  # (C, k)
                # g shape (..., C). Project: (..., k) = g @ Uk.
                proj = g @ Uk
                # Sum over all leading dims of squared projection.
                sq = proj.pow(2).reshape(-1, k).sum(dim=0)  # (k,)
                accumulators[b.name] = accumulators[b.name] + sq.cpu()
                counts[b.name] += int(proj.reshape(-1, k).shape[0])

            # Zero grads accumulated on teacher params (none, since frozen) and clear the graph.
            for p in teacher.parameters():
                if p.grad is not None:
                    p.grad = None
    finally:
        for h in hooks.values():
            h.remove()
        # Restore requires_grad.
        for name, p in teacher.named_parameters():
            p.requires_grad_(saved_requires.get(name, p.requires_grad))

    # Build DirectionScores per branch.
    out_scores: dict[str, DirectionScores] = {}
    for b in profile.branches:
        k = accumulators[b.name].shape[0]
        eigvals = b.eigenvalues
        if eigvals is None:
            eigvals = torch.ones(k, dtype=torch.float64)
        else:
            eigvals = eigvals[:k].to(dtype=torch.float64)
        n = max(1, counts[b.name])
        grad_inner_sq = accumulators[b.name] / n
        q = eigvals.cpu() * grad_inner_sq
        out_scores[b.name] = DirectionScores(
            branch_name=b.name,
            eigenvalues=eigvals.cpu(),
            grad_inner_sq=grad_inner_sq.cpu(),
            q=q.cpu(),
        )

    return out_scores


__all__ = ["DirectionScores", "score_directions"]
