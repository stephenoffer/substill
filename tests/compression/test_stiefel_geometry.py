"""The Stiefel optimizer must respect the geometry it claims to move on.

`StiefelAdamV` makes four geometric claims. Each is checkable, and each was previously either
wrong or unstated:

1. **The step is a rotation of a stated size.** With the trust region on, ``lr`` is the RMS
   angle ``V`` turns per step -- bounding the subspace rotation, and equalling it when the step
   is horizontal. That makes the knob dimensionless (the same number means the same thing on a
   768-wide teacher and an 8192-wide one), which is what removes the need for the fitted
   constant ``v_lr = min(1e-3, 0.77/d)`` the ambient rule required. The ambient step's rotation
   is *not* controlled: it is proportional to the norm of an Adam direction that, on the real
   benchmark, swings from 0.023 rad at step 0 to 0.0009 rad by step 32.

2. **The rotation does not depend on the teacher's width.** The ambient step's does -- it grows
   like ``sqrt(d)`` -- which is the defect the fitted constant was patching, and it is why that
   constant has the wrong exponent (``1/d`` fitted, ``1/sqrt(d)`` derived).

3. **Momentum is transported.** ``m`` is a tangent vector at ``V``; after the retraction moves
   ``V``, an untransported ``m`` lives in the wrong tangent space and its normal component
   leaks into the next update.

4. **The retraction respects the Grassmann structure.** A subspace has no preferred basis, so
   a retraction that answers differently depending on which basis of the same subspace it is
   handed injects a coordinate choice into the trajectory. The polar factor does not; the thin
   QR does, and its column-sign ambiguity needs a patch that the polar factor never needs.
   (Adam's elementwise second moment breaks equivariance too -- that limitation is *documented*
   rather than hidden, and pinned below.)

Together these make ``V``'s step size a physical quantity (radians of rotation) rather than a
number that has to be re-fitted per model.
"""
from __future__ import annotations

import math

import torch

from substill.compression.restricted import (
    StiefelAdamV,
    _tangent,
    polar_retract,
    qr_retract,
)


def _V(d=64, k=16, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Parameter(polar_retract(torch.randn(d, k)))


def _rms_principal_angle(A: torch.Tensor, B: torch.Tensor) -> float:
    """RMS principal angle (radians) between the column spans of ``A`` and ``B``."""
    s = torch.linalg.svdvals(A.T @ B).clamp(-1, 1)
    return float(s.arccos().pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
def _rotation(d, k, lr, scale, *, trust_region=True, seed=1):
    """The RMS subspace rotation one step produces, for a gradient of the given scale."""
    V = _V(d, k)
    opt = StiefelAdamV([V], lr=lr, trust_region=trust_region)
    before = V.detach().clone()
    torch.manual_seed(seed)
    V.grad = torch.randn_like(V) * scale
    opt.step()
    return _rms_principal_angle(before, V.detach())


def test_trust_region_makes_the_step_independent_of_the_gradient_scale():
    """The gradient's *magnitude* must not change how far the subspace turns.

    This is the property that matters. The ambient step's rotation is proportional to the norm
    of an Adam direction that, on the real benchmark, swings from 27934 at step 0 (the bias
    correction at ``t=1``) down to ~50 by step 32 -- so the subspace turns 0.023 rad on one
    step and 0.0009 on another, and only the LR warmup keeps the first step from throwing the
    basis away entirely. Under the trust region the step length is set by construction, so a
    gradient six orders of magnitude larger produces the *same* rotation.
    """
    lr = 0.01
    angles = [_rotation(64, 16, lr, s) for s in (1e-3, 1.0, 1e3, 1e5)]
    assert max(angles) / min(angles) < 1.01, (
        f"rotation depended on the gradient scale: {angles}")
    assert all(a <= lr * 1.02 for a in angles), (
        f"rotation exceeded the trust region lr={lr}: {angles}")


def test_trust_region_rotation_is_within_the_requested_angle():
    """``lr`` is the RMS tangent step; the realized *subspace* rotation is at most that.

    Equality would need the step to be purely horizontal. A general step also has a vertical
    part -- a re-basing of ``V`` inside its own span, which is idle when the loss sees only the
    subspace (``free_core=False``) but is real motion when a free ``D`` is pinned to ``V``'s
    coordinates. So the honest contract is a *bound*, tight to within the horizontal fraction
    (~93% at ``k/d = 1/4``), not an identity.
    """
    lr = 0.01
    ang = _rotation(64, 16, lr, 1.0)
    assert 0.8 * lr < ang <= 1.02 * lr, f"rotated {ang:.5f} rad for a trust region of {lr}"


def test_trust_region_rotation_is_scale_free_across_teacher_widths():
    """The same ``lr`` must mean the same rotation at every teacher width. That is the claim.

    This is what makes the fitted constant ``v_lr = min(1e-3, 0.77/d)`` unnecessary. Held at a
    fixed compression ratio ``k/d`` (which is what the ratio argument fixes anyway), the
    realized rotation is the same across a 64x change in ``d``.
    """
    lr = 0.01
    angles = {d: _rotation(d, d // 4, lr, 1.0) for d in (64, 256, 1024, 4096)}
    spread = max(angles.values()) / min(angles.values())
    assert spread < 1.02, f"rotation was not width-free across 64x of d: {angles}"


def test_ambient_rotation_grows_like_sqrt_width():
    """The control, and the reason the ambient rule needed a hand-fitted constant at all.

    Without the trust region the step is an Adam direction of ambient norm ~``sqrt(d k)``,
    against a ``V`` of norm ``sqrt(k)``, so the rotation scales like ``sqrt(d)``: the same
    ``lr`` turns a wide teacher's subspace much further than a narrow one's.

    Note the exponent. The repository's fitted rule is ``v_lr ~ 1/d``, but the geometry says
    the compensation should be ``1/sqrt(d)`` -- the ``1/d`` fit came from three teachers whose
    width and *compression ratio* moved together, so it absorbed a ``k/d`` effect into a ``d``
    exponent. The trust region removes the need to get that exponent right at all.
    """
    lr = 1e-3
    angles = {d: _rotation(d, d // 4, lr, 1.0, trust_region=False)
              for d in (256, 1024, 4096)}
    r1 = angles[1024] / angles[256]      # 4x width
    r2 = angles[4096] / angles[1024]     # 4x width
    for r in (r1, r2):
        assert 1.7 < r < 2.3, (
            f"a 4x width increase changed the ambient rotation by {r:.2f}x; "
            f"sqrt(d) predicts 2.0x. angles={angles}")


def test_ambient_step_rotation_depends_on_the_gradient_scale():
    """The control: without the trust region the rotation tracks the gradient, not ``lr``.

    This is the defect the trust region removes, and it is why the ambient rule needed a
    per-teacher constant. Adam normalizes elementwise, so a *uniformly* rescaled gradient is
    largely absorbed -- but a gradient whose scale changes *over time* (exactly what happens
    during warmup, and what Adam's bias correction amplifies at ``t=1``) is not.
    """
    lr = 0.01
    V = _V()
    opt = StiefelAdamV([V], lr=lr, trust_region=False)
    torch.manual_seed(1)
    V.grad = torch.randn_like(V)
    before = V.detach().clone()
    opt.step()
    ang = _rms_principal_angle(before, V.detach())
    # The ambient first step is Adam's bias-corrected m/sqrt(v) ~ elementwise +-1, so the
    # rotation is ~ lr * sqrt(d*k)/sqrt(k) = lr*sqrt(d) -- 8x the requested lr at d=64.
    assert ang > 3 * lr, (
        f"ambient step rotated {ang:.4f} rad for lr={lr}: the point of the test is that this "
        f"is *not* lr, and scales with sqrt(d)")


def test_momentum_is_transported_into_the_new_tangent_space():
    """After the step, ``m`` must be tangent at the *new* ``V``, not the old one."""
    V = _V()
    opt = StiefelAdamV([V], lr=0.05, trust_region=True)
    torch.manual_seed(3)
    for _ in range(3):
        V.grad = torch.randn_like(V)
        opt.step()
    m = opt.state[V]["m"]
    # Tangency at V: V^T m + m^T V == 0  (the symmetric part vanishes).
    sym = (V.detach().T @ m + m.T @ V.detach())
    assert sym.abs().max() < 1e-4, (
        f"momentum is not tangent at the current V (max |sym| = {sym.abs().max():.2e}); "
        f"it was never transported after the retraction")


def test_iterate_stays_on_the_manifold_over_many_steps():
    """Orthonormality must not drift, however long the run."""
    V = _V(d=128, k=48)
    opt = StiefelAdamV([V], lr=0.02, trust_region=True)
    torch.manual_seed(4)
    for _ in range(200):
        V.grad = torch.randn_like(V)
        opt.step()
    err = (V.detach().T @ V.detach() - torch.eye(48)).abs().max()
    assert err < 1e-5, f"V drifted off the Stiefel manifold: ||V^T V - I||_inf = {err:.2e}"


def test_polar_retraction_is_gauge_equivariant_and_qr_is_not():
    """``polar(A R) == polar(A) R``; QR does not commute with a change of representative.

    A subspace has no preferred basis, so a retraction that answers differently depending on
    *which* basis of the same subspace you hand it is quietly injecting a coordinate choice
    into the trajectory. The thin QR does exactly that -- its upper-triangular factor pins a
    canonical, column-order-dependent basis. The polar factor (the metric projection) does
    not. This is also why QR needs the ``diag(R)`` sign patch and polar needs no patch: the
    sign ambiguity is the same defect seen from a different angle.
    """
    torch.manual_seed(5)
    d, k = 64, 16
    A = torch.randn(d, k, dtype=torch.float64)
    R = polar_retract(torch.randn(k, k, dtype=torch.float64))   # a change of representative

    assert (polar_retract(A @ R) - polar_retract(A) @ R).abs().max() < 1e-10
    assert (qr_retract(A @ R) - qr_retract(A) @ R).abs().max() > 1e-2, (
        "qr_retract unexpectedly commuted with a gauge rotation -- if this ever fires, the "
        "rationale documented on polar_retract is stale")
    for ret in (polar_retract, qr_retract):
        Q = ret(A)
        assert torch.allclose(Q.T @ Q, torch.eye(k, dtype=torch.float64), atol=1e-10), (
            f"{ret.__name__} did not land on the Stiefel manifold")


def test_gauge_invariant_mode_is_equivariant_under_a_change_of_representative():
    """``V`` and ``V R`` name the same subspace; with ``gauge_invariant`` they stay that way.

    When the loss depends only on the subspace (``free_core=False``), the optimizer *should*
    take the same subspace-step from either representative. Two things break that, and both
    have to be fixed for the property to hold:

    * Adam's **elementwise** second moment -- a coordinatewise statistic on a quantity with no
      preferred coordinates. ``gauge_invariant=True`` swaps in the scalar ``EMA(||g||_F^2)``.
    * The **QR retraction**, which is not equivariant (see above). ``polar`` is.

    This pins the fixed path and documents the limitation of the default one honestly, rather
    than asserting a Grassmann story the optimizer does not actually implement.

    Run in float64: this is a statement about the *geometry*, and in fp32 Adam's ``t = 1``
    division by a near-zero second moment amplifies rounding enough (to ~2e-3 of drift) to
    swamp the structural signal we are testing for.
    """
    d, k = 48, 12
    dt = torch.float64
    torch.manual_seed(5)
    R = polar_retract(torch.randn(k, k, dtype=dt))    # a change of representative
    grads = [torch.randn(d, k, dtype=dt) for _ in range(5)]

    def run(rotate: bool, **kw):
        torch.manual_seed(6)
        V0 = polar_retract(torch.randn(d, k, dtype=dt))
        V = torch.nn.Parameter(V0 @ R if rotate else V0)
        opt = StiefelAdamV([V], lr=0.02, trust_region=True, **kw)
        for g in grads:
            # The gradient transforms with the representative: dL/d(VR) = (dL/dV) R.
            V.grad = (g @ R) if rotate else g.clone()
            opt.step()
        return V.detach()

    # Fixed path: scalar second moment + polar retraction => the same *subspace* either way.
    got = run(True, gauge_invariant=True, retraction="polar")
    ref = run(False, gauge_invariant=True, retraction="polar") @ R
    assert _rms_principal_angle(got, ref) < 1e-6, (
        "gauge_invariant + polar did not commute with a change of representative")

    # Either defect alone breaks it. These are the documented limitations, not accidents.
    e1 = run(True, gauge_invariant=False, retraction="polar")
    r1 = run(False, gauge_invariant=False, retraction="polar") @ R
    assert _rms_principal_angle(e1, r1) > 1e-3, (
        "elementwise Adam turned out gauge-equivariant -- the StiefelAdamV caveat is stale")

    e2 = run(True, gauge_invariant=True, retraction="qr")
    r2 = run(False, gauge_invariant=True, retraction="qr") @ R
    assert _rms_principal_angle(e2, r2) > 1e-6, (
        "the QR retraction turned out gauge-equivariant -- the polar_retract caveat is stale")


def test_qr_retraction_is_sign_continuous():
    """The sign fix must make the retraction continuous, or momentum flips at random."""
    torch.manual_seed(7)
    A = torch.randn(32, 8)
    Q = qr_retract(A)
    Q2 = qr_retract(A + 1e-7 * torch.randn(32, 8))
    assert (Q - Q2).abs().max() < 1e-4, "retraction is discontinuous: column signs flipped"
    assert torch.allclose(Q.T @ Q, torch.eye(8), atol=1e-5)


def test_tangent_projection_is_a_projection():
    """``_tangent`` must be idempotent and land in the tangent space."""
    torch.manual_seed(8)
    V = _V(40, 10)
    G = torch.randn(40, 10)
    T1 = _tangent(G, V)
    T2 = _tangent(T1, V)
    assert torch.allclose(T1, T2, atol=1e-5), "tangent projection is not idempotent"
    sym = V.T @ T1 + T1.T @ V
    assert sym.abs().max() < 1e-5, "projected gradient is not tangent"
    assert not math.isnan(float(T1.sum()))
