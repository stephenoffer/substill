"""Tests for the RoPE-aware QK circuit basis (CPSD Phase 2-CPI, QK half).

Codifies the de-risk finding: an arbitrary cross-plane PCA basis does NOT commute
with RoPE (post-RoPE score error inflates), while the plane-aligned basis does.
"""
import torch

from fasd.profiling.rope import (
    apply_rope,
    qk_score_residual,
    rope_aware_basis,
    rotate_half,
)


def test_rotate_half_involution_structure():
    x = torch.randn(3, 8)
    # rotate_half applied twice negates (standard RoPE identity).
    assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)


def test_rope_preserves_inner_product_relative_position():
    # RoPE is a rotation: <R_i q, R_j k> depends only on (j-i), and at i=j equals <q,k>.
    torch.manual_seed(0)
    d_h = 16
    q, k = torch.randn(4, d_h), torch.randn(4, d_h)
    pos = torch.arange(4)
    qr = apply_rope(q, pos)
    kr = apply_rope(k, pos)
    # same-position dot products are preserved (rotation by same angle).
    for i in range(4):
        assert torch.allclose((qr[i] * kr[i]).sum(), (apply_rope(q[i:i+1], pos[i:i+1])
                               * apply_rope(k[i:i+1], pos[i:i+1])).sum(), atol=1e-4)


def test_rope_aware_basis_shape_and_plane_structure():
    cov = torch.diag(torch.arange(1, 9).float())  # d_h=8, 4 planes
    V = rope_aware_basis(cov, keep_planes=2)
    assert V.shape == (8, 4)
    # Each column is a unit standard-basis vector.
    assert torch.allclose(V.sum(0), torch.ones(4)) and ((V == 0) | (V == 1)).all()
    gram = V.T @ V
    assert torch.allclose(gram, torch.eye(4), atol=1e-6)


def test_arbitrary_pca_basis_breaks_under_rope_but_plane_aligned_commutes():
    torch.manual_seed(1)
    d_h, keep = 64, 32
    N, T = 4096, 128
    # Dense plane-mixing covariance (so PCA directions are NOT plane-aligned).
    A, _ = torch.linalg.qr(torch.randn(d_h, d_h))
    M = A @ torch.diag(torch.logspace(0, -2, d_h))
    z = torch.randn(N, d_h)
    q_all = z @ M.T + 0.3 * (torch.randn(N, d_h) @ M.T)
    k_all = z @ M.T + 0.3 * (torch.randn(N, d_h) @ M.T)
    cov = torch.cat([q_all, k_all]).T @ torch.cat([q_all, k_all])

    # Arbitrary cross-plane PCA basis.
    evals, evecs = torch.linalg.eigh(cov)
    V_pca = evecs[:, torch.argsort(evals, descending=True)][:, :keep]
    # RoPE-aware plane-aligned basis (same retained dim = keep = 2*keep_planes).
    V_rope = rope_aware_basis(cov, keep_planes=keep // 2)

    q, k = q_all[:T], k_all[:T]
    pos = torch.arange(T)
    pca_norope = qk_score_residual(q, k, V_pca)
    pca_rope = qk_score_residual(q, k, V_pca, positions=pos)
    rope_norope = qk_score_residual(q, k, V_rope)
    rope_rope = qk_score_residual(q, k, V_rope, positions=pos)

    # PCA basis: RoPE substantially inflates the score error (does NOT commute).
    assert pca_rope > 2 * pca_norope, \
        f"expected RoPE to break PCA basis: norope={pca_norope}, rope={pca_rope}"
    # Plane-aligned basis: RoPE barely changes the error (commutes).
    assert rope_rope < 1.25 * rope_norope + 1e-6, \
        f"plane-aligned should commute: norope={rope_norope}, rope={rope_rope}"


def test_full_rank_rope_aware_basis_is_exact_post_rope():
    torch.manual_seed(2)
    d_h = 16
    q, k = torch.randn(8, d_h), torch.randn(8, d_h)
    cov = torch.eye(d_h)
    V_full = rope_aware_basis(cov, keep_planes=d_h // 2)  # all planes -> identity
    resid = qk_score_residual(q, k, V_full, positions=torch.arange(8))
    assert resid < 1e-5
