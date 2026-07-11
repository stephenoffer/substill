"""Tests for circuit-preserving init (CPSD Phase 2-CPI): OV circuit + block-diag basis."""
import torch

from substill.compression.absorbed_init import absorbed_weight
from substill.compression.cpi import block_diagonal_basis, ov_circuit_residual


def test_block_diagonal_basis_shape_and_orthogonality():
    torch.manual_seed(0)
    G, d_h = 2, 8
    bases = torch.stack([torch.linalg.qr(torch.randn(d_h, d_h))[0] for _ in range(G)])
    head_groups = [0, 0, 1, 1]  # 4 heads, 2 per group
    V = block_diagonal_basis(bases, head_groups, keep=4)
    assert V.shape == (4 * d_h, 4 * 4)
    # Columns orthonormal (blocks have disjoint row support so cross-blocks are 0).
    gram = V.T @ V
    assert torch.allclose(gram, torch.eye(V.shape[1]), atol=1e-5)


def test_block_diagonal_per_head_rank():
    torch.manual_seed(1)
    G, d_h = 2, 8
    bases = torch.stack([torch.linalg.qr(torch.randn(d_h, d_h))[0] for _ in range(G)])
    V = block_diagonal_basis(bases, [0, 1], keep=[3, 5])
    assert V.shape == (2 * d_h, 3 + 5)


def test_shared_basis_preserves_ov_circuit_better_than_disjoint():
    # Value activations live in a low-rank subspace; the shared PCA basis should
    # preserve the OV operator while an independent (disjoint) basis breaks it.
    torch.manual_seed(2)
    d, d_h, k = 32, 16, 8
    # Construct W_V so that values occupy a k-dim subspace of the head space.
    U, _ = torch.linalg.qr(torch.randn(d_h, d_h))
    sv = torch.cat([torch.ones(k), 0.02 * torch.ones(d_h - k)])  # energy in top-k
    W_V = (U * sv) @ torch.randn(d_h, d)          # (d_h, d)
    W_O = torch.randn(d, d_h)

    # Shared basis = PCA of value activations.
    X = torch.randn(4096, d)
    v = X @ W_V.T                                  # (N, d_h)
    cov = v.T @ v
    evals, evecs = torch.linalg.eigh(cov)
    V = evecs[:, torch.argsort(evals, descending=True)][:, :k]   # (d_h, k)

    # Disjoint basis: an unrelated random orthonormal k-subspace for the O side.
    V_o_disjoint = torch.linalg.qr(torch.randn(d_h, d_h))[0][:, :k]

    shared = ov_circuit_residual(W_O, W_V, V_v=V, V_o=V)
    disjoint = ov_circuit_residual(W_O, W_V, V_v=V, V_o=V_o_disjoint)
    assert shared < 0.1, f"shared-basis OV residual too high: {shared}"
    assert disjoint > 3 * shared, \
        f"shared ({shared}) should beat disjoint ({disjoint}) substantially"


def test_block_diag_basis_usable_as_absorbed_v_out():
    # The block-diagonal basis is a valid V_out for absorbed_weight.
    torch.manual_seed(3)
    G, d_h = 2, 6
    H = 4
    bases = torch.stack([torch.linalg.qr(torch.randn(d_h, d_h))[0] for _ in range(G)])
    V_out = block_diagonal_basis(bases, [0, 0, 1, 1], keep=3)  # (24, 12)
    d_in = 20
    W_T = torch.randn(H * d_h, d_in)               # (24, 20) fused proj
    V_in = torch.linalg.qr(torch.randn(d_in, 10))[0]
    W_S = absorbed_weight(W_T, V_in, V_out, layout="linear")
    assert W_S.shape == (12, 10)                   # (k_out, k_in)
