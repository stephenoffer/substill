"""Tests for fasd.profiling.gqa_basis.

Headline invariant: a SHARED per-KV-group basis preserves attention scores
strictly better than DISJOINT per-edge bases. This is the test that
demonstrates Sprint 2's contribution: it's not just a correctness test,
it's the experimental gate that justifies the new mechanism.
"""

from __future__ import annotations

import pytest
import torch

from fasd.profiling.gqa_basis import (
    GQAConfig,
    attention_score_residual,
    joint_group_covariance,
    shared_bases_from_covariance,
)


def test_gqa_config_validates_divisibility():
    with pytest.raises(ValueError):
        GQAConfig(num_attention_heads=7, num_key_value_heads=2, head_dim=8)


def test_gqa_config_heads_per_group():
    cfg = GQAConfig(num_attention_heads=8, num_key_value_heads=2, head_dim=4)
    assert cfg.heads_per_group == 4


def test_joint_group_covariance_shape():
    """For B=2, T=8, H=4, G=2, d_h=4: cov has shape (G, d_h, d_h) = (2, 4, 4)."""
    cfg = GQAConfig(num_attention_heads=4, num_key_value_heads=2, head_dim=4)
    q = torch.randn(2, 8, 4 * 4)
    k = torch.randn(2, 8, 2 * 4)
    v = torch.randn(2, 8, 2 * 4)
    cov = joint_group_covariance(q, k, v, cfg)
    assert cov.shape == (2, 4, 4)
    # Symmetric.
    for g in range(2):
        assert torch.allclose(cov[g], cov[g].T, atol=1e-5)
    # Positive semi-definite (eigvals >= 0).
    for g in range(2):
        eigvals = torch.linalg.eigvalsh(cov[g])
        assert (eigvals >= -1e-4).all()


def test_shared_bases_from_covariance_orthonormal():
    """Each per-group basis V_g should satisfy V_g^T V_g = I."""
    torch.manual_seed(0)
    G, d_h = 3, 8
    cov = torch.randn(G, d_h, d_h)
    cov = (cov + cov.transpose(-1, -2)) / 2  # symmetric
    cov = cov @ cov.transpose(-1, -2)  # PSD
    V = shared_bases_from_covariance(cov)
    assert V.shape == (G, d_h, d_h)
    for g in range(G):
        VtV = V[g].T @ V[g]
        assert torch.allclose(VtV, torch.eye(d_h), atol=1e-5)


def test_shared_bases_columns_sorted_by_eigenvalue():
    """First column should correspond to the largest eigenvalue."""
    torch.manual_seed(0)
    d_h = 6
    # Construct a covariance with known spectrum.
    eigvals = torch.tensor([100.0, 50.0, 10.0, 5.0, 1.0, 0.1])
    Q, _ = torch.linalg.qr(torch.randn(d_h, d_h))
    cov = (Q * eigvals).matmul(Q.T).unsqueeze(0)  # (1, d_h, d_h)
    V = shared_bases_from_covariance(cov)
    # First column should be the eigenvector for eigvals[0] = 100.
    v0 = V[0, :, 0]
    cov_v0 = cov[0] @ v0
    eig0_recovered = (v0 @ cov_v0).item()
    assert eig0_recovered == pytest.approx(100.0, rel=0.01)


def test_attention_score_residual_full_rank_basis_is_exact():
    """V[:, :d_h] (full rank) must reproduce attention exactly: residual ≈ 0."""
    torch.manual_seed(0)
    B, H, T, d_h = 2, 4, 8, 16
    q = torch.randn(B, H, T, d_h)
    k = torch.randn(B, H, T, d_h)
    V, _ = torch.linalg.qr(torch.randn(d_h, d_h))  # any orthonormal basis
    res = attention_score_residual(q, k, V)
    assert res < 1e-5, f"full-rank basis should preserve attention exactly, got rel={res:.3e}"


def test_attention_score_residual_truncated_basis_is_lossy():
    """V[:, :s_d_h] with s_d_h < d_h has nonzero residual."""
    torch.manual_seed(0)
    B, H, T, d_h, s_d_h = 1, 2, 4, 8, 4
    q = torch.randn(B, H, T, d_h)
    k = torch.randn(B, H, T, d_h)
    V, _ = torch.linalg.qr(torch.randn(d_h, d_h))
    res = attention_score_residual(q, k, V[:, :s_d_h])
    # Some residual is expected; just check it's non-trivial.
    assert res > 0.05


def test_shared_basis_beats_disjoint_basis_on_synthetic_gqa():
    """The headline test for Sprint 2.

    Build a synthetic GQA scenario where Q heads in group g, K_g, and V_g have
    most of their energy along a common subspace (shared per-group structure
    in the activations). Show that:
      - SHARED basis (joint PCA over q heads in group + k_g + v_g) preserves
        attention scores within tolerance.
      - DISJOINT basis (independent PCA per edge) has measurably worse residual.

    This justifies the entire Sprint 2 fix.
    """
    torch.manual_seed(42)
    B, T, H, G, d_h = 1, 64, 4, 2, 8
    s_d_h = 4  # compress to half
    H_per_G = H // G

    # Build "shared subspace" activations: in each group, q/k/v have most
    # energy along a randomly chosen 4-dim sub-basis of d_h.
    shared_subspaces: list[torch.Tensor] = []
    for _g in range(G):
        # Random orthonormal d_h × d_h basis; pick top-s_d_h as the "true" shared subspace.
        Q_basis, _ = torch.linalg.qr(torch.randn(d_h, d_h))
        shared_subspaces.append(Q_basis[:, :s_d_h])

    # Generate q, k, v activations: most energy in the shared subspace.
    q_act = torch.zeros(B, T, H * d_h)
    k_act = torch.zeros(B, T, G * d_h)
    v_act = torch.zeros(B, T, G * d_h)

    def project_into_subspace(
        d_h_signals: torch.Tensor, subspace: torch.Tensor, noise: float = 0.05
    ) -> torch.Tensor:
        """Map (..., s_d_h) signal into d_h via subspace, plus small noise."""
        in_subspace = d_h_signals @ subspace.T  # (..., d_h)
        noise_t = noise * torch.randn_like(in_subspace)
        return in_subspace + noise_t

    for g in range(G):
        Vsub = shared_subspaces[g]
        # Each query head in group g.
        for h_in_g in range(H_per_G):
            sig = torch.randn(B, T, s_d_h) * 2.0
            slot = (g * H_per_G + h_in_g) * d_h
            q_act[:, :, slot:slot + d_h] = project_into_subspace(sig, Vsub)
        # Key for group g.
        sig_k = torch.randn(B, T, s_d_h) * 2.0
        k_act[:, :, g * d_h:(g + 1) * d_h] = project_into_subspace(sig_k, Vsub)
        # Value for group g (irrelevant for attention score, but needed for joint cov).
        sig_v = torch.randn(B, T, s_d_h) * 2.0
        v_act[:, :, g * d_h:(g + 1) * d_h] = project_into_subspace(sig_v, Vsub)

    cfg = GQAConfig(num_attention_heads=H, num_key_value_heads=G, head_dim=d_h)

    # SHARED basis: joint PCA per group.
    cov_shared = joint_group_covariance(q_act, k_act, v_act, cfg)
    V_shared = shared_bases_from_covariance(cov_shared)  # (G, d_h, d_h)

    # DISJOINT bases: separate PCA on q (all heads pooled), k (all groups pooled),
    # v (all groups pooled). This is approximately what the existing code does.
    q_h = q_act.view(B, T, H, d_h).reshape(-1, d_h)
    k_h = k_act.view(B, T, G, d_h).reshape(-1, d_h)

    def basis_for(x):
        cov = x.T @ x
        cov = 0.5 * (cov + cov.T)
        eigvals, V = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        return V[:, order]

    V_q_disjoint = basis_for(q_h)  # (d_h, d_h)
    V_k_disjoint = basis_for(k_h)

    # Check attention preservation per-group.
    # Truncated to s_d_h dims:
    P_q_disjoint = V_q_disjoint[:, :s_d_h] @ V_q_disjoint[:, :s_d_h].T
    P_k_disjoint = V_k_disjoint[:, :s_d_h] @ V_k_disjoint[:, :s_d_h].T

    q_per_head = q_act.view(B, T, H, d_h).permute(0, 2, 1, 3)  # (B, H, T, d_h)
    k_per_group = k_act.view(B, T, G, d_h).permute(0, 2, 1, 3)  # (B, G, T, d_h)

    # Score original attention per (head, group):
    def attention_residual_for_pair(q_h_, k_g_, P_q, P_k):
        score_orig = q_h_ @ k_g_.transpose(-2, -1)
        score_proj = (q_h_ @ P_q) @ (k_g_ @ P_k).transpose(-2, -1)
        return ((score_orig - score_proj).norm() / score_orig.norm().clamp_min(1e-8)).item()

    res_shared = []
    res_disjoint = []
    for g in range(G):
        P_shared = V_shared[g, :, :s_d_h] @ V_shared[g, :, :s_d_h].T
        for h_in_g in range(H_per_G):
            head_idx = g * H_per_G + h_in_g
            qh = q_per_head[:, head_idx]
            kg = k_per_group[:, g]
            res_shared.append(attention_residual_for_pair(qh, kg, P_shared, P_shared))
            res_disjoint.append(attention_residual_for_pair(qh, kg, P_q_disjoint, P_k_disjoint))

    avg_shared = sum(res_shared) / len(res_shared)
    avg_disjoint = sum(res_disjoint) / len(res_disjoint)

    print(f"[gqa_basis test] shared rel-residual avg = {avg_shared:.4f}")
    print(f"[gqa_basis test] disjoint rel-residual avg = {avg_disjoint:.4f}")

    # Headline assertion: shared basis beats disjoint by a meaningful margin.
    assert avg_shared < avg_disjoint, (
        f"shared rel-residual ({avg_shared:.4f}) should be < disjoint ({avg_disjoint:.4f})"
    )
    # And shared should be small in absolute terms (the synthetic data was
    # constructed so the shared subspace explains nearly all the variance).
    assert avg_shared < 0.2, f"shared rel-residual {avg_shared:.4f} unexpectedly high"


def test_collect_gqa_bases_smoke_on_tiny_llama():
    """Smoke test: run on a tiny HF Llama and verify shapes."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, max_position_embeddings=32,
    )
    model = LlamaForCausalLM(cfg).eval()
    batches = [{"input_ids": torch.randint(0, 64, (1, 8))}]

    from fasd.profiling.gqa_basis import collect_gqa_bases
    bases = collect_gqa_bases(model, batches, device="cpu")
    assert len(bases) == 2  # one per layer
    for _layer_idx, V in bases.items():
        # Shape: (num_kv_heads=2, head_dim=8, head_dim=8)
        assert V.shape == (2, 8, 8)
        # Per-group orthogonality.
        for g in range(2):
            VtV = V[g].T @ V[g]
            assert torch.allclose(VtV, torch.eye(8), atol=1e-4)
