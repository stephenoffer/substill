"""Streaming PCA backends recover the top-k subspace of the exact baseline."""

from __future__ import annotations

import math

import torch

from fasd.profiling.streaming_pca import StreamingPCA


def _principal_angle_deg(A, B, k):
    """Max principal angle in degrees between two column-orthonormal bases."""
    M = A[:, :k].T @ B[:, :k]
    s = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    ang = torch.arccos(s) * (180.0 / math.pi)
    return float(ang.max().item())


def _synthetic_batches(C=32, N=4096, k=8, batch=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(C, C, generator=g))
    # Give the first k directions strong variance; rest near zero.
    eig = torch.cat([torch.linspace(10.0, 2.0, k), 0.01 * torch.ones(C - k)])
    for start in range(0, N, batch):
        x = torch.randn(batch, C, generator=g) * eig.sqrt()
        yield x @ Q.T
    # keep Q accessible for comparison
    return Q


def test_streaming_pca_exact_matches_eigh():
    C, k = 16, 4
    pca = StreamingPCA(channels=C, k=k, backend="exact")
    data = []
    for batch in _synthetic_batches(C=C, k=k, N=1024, batch=128, seed=0):
        pca.update(batch)
        data.append(batch)
    basis, _ = pca.top_k()
    assert basis.shape == (C, k)
    # Cross-check with full eigh on the stacked data.
    X = torch.cat(data, dim=0)
    cov = (X - X.mean(0)).T @ (X - X.mean(0)) / max(1, X.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    ref = eigvecs[:, order][:, :k]
    # Exact backend uses raw covariance (not centered), so principal
    # angle shouldn't be exactly zero but should be small on this data.
    ang = _principal_angle_deg(basis, ref, k)
    assert ang < 5.0, f"exact backend top-k should align within 5 deg, got {ang:.2f}"


def test_streaming_pca_randomized_matches_exact():
    C, k = 32, 4
    batches = list(_synthetic_batches(C=C, k=k, N=2048, batch=256, seed=1))

    exact = StreamingPCA(channels=C, k=k, backend="exact")
    for b in batches:
        exact.update(b)
    basis_exact, _ = exact.top_k()

    rand = StreamingPCA(channels=C, k=k, backend="randomized", seed=1)
    for b in batches:
        rand.update(b)
    basis_rand, _ = rand.top_k()

    ang = _principal_angle_deg(basis_exact, basis_rand, k)
    assert ang < 15.0, f"randomized backend should approximate exact, got {ang:.2f} deg"


def test_streaming_pca_oja_runs_and_produces_orthonormal_basis():
    C, k = 16, 4
    oja = StreamingPCA(channels=C, k=k, backend="oja", oja_lr=0.1, seed=2)
    for b in _synthetic_batches(C=C, k=k, N=1024, batch=64, seed=2):
        oja.update(b)
    basis, _ = oja.top_k()
    assert basis.shape == (C, k)
    # Semi-orthogonality: basis^T basis should be close to I.
    I = torch.eye(k)
    err = (basis.T @ basis - I).abs().max().item()
    assert err < 1e-3, f"oja output should be orthonormal, got max err {err}"
