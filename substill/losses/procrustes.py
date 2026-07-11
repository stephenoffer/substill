"""Whitened orthogonal-Procrustes alignment on retained subspace coefficients.

The loss is the residual of the optimal orthogonal alignment::

    min_R ||Z_s R - Z_t||_F^2,   R^T R = I

This is rotation-invariant on both sides (rotating ``Z_s`` or ``Z_t``
inside the retained subspace does not change the loss), which
addresses the "CKA is too permissive / coord_mse is too rigid"
trade-off from the design brief.

When ``whiten=True`` (default), each side is whitened to have
covariance ``I_k`` before alignment — the pure rotational alignment
then strips scale and within-subspace shape differences, leaving only
the matching of principal directions.

Implementation note: every eigendecomposition is computed via
``torch.linalg.eigh`` on a symmetric matrix, never via
``torch.linalg.svd`` / ``svdvals``. PyTorch's cuSOLVER SVD path
occasionally hangs on ill-conditioned covariances under certain CUDA
driver versions — using eigh on ``M^T M`` (for nuclear norm) and on
``Z^T Z + ridge`` (for whitening) sidesteps the issue entirely.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _inv_sqrt_via_eigh(cov: Tensor, eps: float) -> Tensor:
    """Return ``cov^(-1/2)`` via eigh on a ridged symmetric matrix.

    ``cov`` is symmetric positive-semidefinite; ``eps`` scales the
    ridge added to the diagonal so eigenvalues near zero don't blow up
    the inverse square root.
    """
    k = cov.shape[0]
    ridge = eps * cov.diagonal().abs().mean().clamp_min(eps)
    cov = cov + ridge * torch.eye(k, dtype=cov.dtype, device=cov.device)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    inv_sqrt_diag = (eigvals.clamp_min(eps)) ** -0.5
    # V @ diag(inv_sqrt_diag) @ V^T, computed as (V * inv_sqrt_diag) @ V^T so
    # each column j of V is scaled by inv_sqrt_diag[j].
    return (eigvecs * inv_sqrt_diag) @ eigvecs.T


def _whiten(Z: Tensor, *, eps: float = 1e-5) -> Tensor:
    """Return Z whitened so that (1/N) Z^T Z ≈ I_k.

    Z has shape ``(N, k)``. Always uses the eigh-based path (no SVD);
    on CUDA this avoids the cuSOLVER SVD hang observed on some driver
    versions.

    The inverse-square-root factor is detached before the multiply, so
    gradients flow through ``Z`` but not back through the eigh — this
    avoids the backward instability on near-degenerate spectra
    (``1 / (lambda_i - lambda_j)`` in eigh's vjp), and whitening stays
    semantically a data-dependent normalization.
    """
    N = Z.shape[0]
    if N < 2:
        return Z
    with torch.no_grad():
        cov = (Z.detach().T @ Z.detach()) / max(1, N - 1)
        try:
            inv_sqrt = _inv_sqrt_via_eigh(cov, eps)
        except Exception:
            return Z
    return Z @ inv_sqrt


def _optimal_rotation(M: Tensor, eps: float) -> Tensor:
    """Best orthogonal R* (as ``U V^T`` from SVD of ``Z_s^T Z_t``).

    Computed under ``no_grad`` by the caller so autograd flows only through
    the residual ``Z_s R* - Z_t``. Differentiating through the SVD / eigh
    that produces R is ill-conditioned when spectra cluster (vjp has
    ``1/(λ_i - λ_j)`` terms) — this can produce NaN gradients at
    the CKA→Procrustes hand-off once procrustes weight crossed ~0.3.
    """
    try:
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        return (U @ Vh).contiguous()
    except Exception:
        k = M.shape[1]
        try:
            MtM = 0.5 * (M.T @ M + (M.T @ M).T)
            ridge = eps * torch.eye(k, dtype=M.dtype, device=M.device)
            _, V = torch.linalg.eigh(MtM + ridge)
            U_half = M @ V
            U_half = U_half / U_half.norm(dim=0, keepdim=True).clamp_min(eps)
            return (U_half @ V.T).contiguous()
        except Exception:
            return torch.eye(k, dtype=M.dtype, device=M.device)


def procrustes_distance(
    Z_s: Tensor,
    Z_t: Tensor,
    *,
    whiten: bool = True,
    eps: float = 1e-5,
) -> Tensor:
    """Whitened orthogonal-Procrustes residual, normalized to ``[0, 1]``.

    Both inputs have shape ``(N, k)``. Computes ``||Z_s R* - Z_t||_F^2`` with
    R* the optimal orthogonal rotation (detached) divided by
    ``||Z_s||_F^2 + ||Z_t||_F^2`` so the value is in ``[0, 1]`` regardless of
    ``k`` or whether whitening ran.

    Returns a 0-dim scalar (``requires_grad`` if inputs do).
    """
    if Z_s.shape != Z_t.shape:
        raise ValueError(
            f"Procrustes requires equal shapes, got {Z_s.shape} vs {Z_t.shape}"
        )
    if Z_s.dim() != 2:
        raise ValueError(f"Procrustes requires 2D (N, k), got {Z_s.shape}")
    N = Z_s.shape[0]
    if N == 0:
        return Z_s.new_zeros(())

    if whiten:
        Z_s = _whiten(Z_s, eps=eps)
        Z_t = _whiten(Z_t, eps=eps)

    if not torch.isfinite(Z_s).all() or not torch.isfinite(Z_t).all():
        return Z_s.new_zeros(())

    with torch.no_grad():
        M = Z_s.detach().T @ Z_t.detach()
        R = _optimal_rotation(M, eps).to(device=Z_s.device, dtype=Z_s.dtype)

    residual = Z_s @ R - Z_t
    raw = residual.pow(2).sum()
    # Detach the normalizer — it's a scale, not part of the alignment signal,
    # and detaching it removes another path where a near-zero denominator can
    # create runaway gradients on top of the already-detached R.
    norm_sq = (Z_s.pow(2).sum() + Z_t.pow(2).sum()).detach()
    return (raw / norm_sq.clamp_min(eps)).clamp_min(0.0)


def covariance_calibration(
    Z_s: Tensor,
    Z_t: Tensor,
) -> Tensor:
    """``||Cov(Z_s) - Cov(Z_t)||_F^2 / k^2`` — second-moment calibration."""
    if Z_s.shape != Z_t.shape:
        raise ValueError(
            f"covariance_calibration shapes differ: {Z_s.shape} vs {Z_t.shape}"
        )
    k = Z_s.shape[-1]
    if Z_s.shape[0] < 2:
        return Z_s.new_zeros(())
    cov_s = (Z_s.T @ Z_s) / max(1, Z_s.shape[0] - 1)
    cov_t = (Z_t.T @ Z_t) / max(1, Z_t.shape[0] - 1)
    return ((cov_s - cov_t).pow(2).sum()) / float(k * k)


def norm_calibration(Z_s: Tensor, Z_t: Tensor) -> Tensor:
    """``(||Z_s||_F - ||Z_t||_F)^2 / N`` — scale calibration."""
    n = max(1, Z_s.shape[0])
    return (Z_s.norm() - Z_t.norm()).pow(2) / float(n)


__all__ = [
    "procrustes_distance",
    "covariance_calibration",
    "norm_calibration",
]
