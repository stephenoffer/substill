"""F_ASDLoss profile refresh: hot-swap bases without re-allocating projectors."""

from __future__ import annotations

import torch

from substill.api import BranchProfile, TeacherProfile
from substill.losses.subspace import F_ASDLoss


def _mk_profile(hidden=8, k=4):
    V = torch.eye(hidden)
    eig = torch.linspace(5.0, 1.0, hidden)
    bp = BranchProfile(
        name="a",
        kind="block.residual",
        module_path="a",
        principal_components=V,
        eigenvalues=eig,
        behavioral_rank=k,
        variance_rank=hidden,
        channels=hidden,
    )
    return TeacherProfile(branches=[bp])


def test_refresh_from_profile_updates_basis_when_rank_stable():
    prof = _mk_profile(hidden=8, k=4)
    loss_fn = F_ASDLoss(prof, objective="gram", normalize_features=False)

    # Trigger projector creation by running a forward.
    s_hid = torch.randn(2, 3, 6)
    t_hid = torch.randn(2, 3, 8)
    _ = loss_fn({"a": s_hid}, {"a": t_hid})
    assert "a" in loss_fn.projectors
    proj_id_before = id(loss_fn.projectors["a"])

    # Refresh with a rotated basis (same rank).
    torch.manual_seed(1)
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    new_eig = torch.linspace(5.0, 1.0, 8)
    new_bp = BranchProfile(
        name="a",
        kind="block.residual",
        module_path="a",
        principal_components=Q,
        eigenvalues=new_eig,
        behavioral_rank=4,
        variance_rank=8,
        channels=8,
    )
    new_profile = TeacherProfile(branches=[new_bp])
    loss_fn.refresh_from_profile(new_profile)

    # Projector preserved (same rank, no reallocation).
    assert id(loss_fn.projectors["a"]) == proj_id_before
    # Basis buffer changed.
    V_buf = loss_fn._get_v("a")
    assert torch.allclose(V_buf, Q[:, :4].float(), atol=1e-6)


def test_refresh_from_profile_reallocates_when_rank_changes():
    prof = _mk_profile(hidden=8, k=4)
    loss_fn = F_ASDLoss(prof, objective="gram", normalize_features=False)

    s_hid = torch.randn(2, 3, 6)
    t_hid = torch.randn(2, 3, 8)
    _ = loss_fn({"a": s_hid}, {"a": t_hid})
    assert "a" in loss_fn.projectors

    new_bp = BranchProfile(
        name="a",
        kind="block.residual",
        module_path="a",
        principal_components=torch.eye(8),
        eigenvalues=torch.ones(8),
        behavioral_rank=2,  # different rank
        variance_rank=8,
        channels=8,
    )
    loss_fn.refresh_from_profile(TeacherProfile(branches=[new_bp]))
    # Projector is cleared; will be rebuilt on next forward.
    assert "a" not in loss_fn.projectors
    assert loss_fn.branch_ks["a"] == 2
