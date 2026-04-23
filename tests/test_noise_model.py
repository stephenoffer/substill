"""Tests for noise-aware rank cutoff (MP, Ledoit-Wolf)."""

from __future__ import annotations

import pytest
import torch

from asd.profiling.svd_analysis import SVDAnalyzer


def test_default_behavior_is_variance_threshold():
    """Default (eps noise, no shrinkage) reproduces classic variance-k
    selection on a clean rank-5 covariance."""
    torch.manual_seed(0)
    C = 64
    V = torch.linalg.qr(torch.randn(C, 5)).Q
    eigs = torch.tensor([10.0, 7.0, 4.0, 2.0, 1.0])
    cov = V @ torch.diag(eigs) @ V.T
    cov = cov + 1e-6 * torch.eye(C)

    analyzer = SVDAnalyzer(variance_threshold=0.95, definition="variance")
    profile = analyzer.analyze("test", cov)
    assert 3 <= profile.effective_rank <= 6


def test_mp_cutoff_falls_back_without_n_effective():
    torch.manual_seed(0)
    C = 32
    V = torch.linalg.qr(torch.randn(C, C)).Q
    eigs = torch.linspace(C, 0.01, C)
    cov = V @ torch.diag(eigs) @ V.T
    cov = 0.5 * (cov + cov.T)

    p_eps = SVDAnalyzer(noise_model="eps").analyze("t", cov)
    p_mp = SVDAnalyzer(noise_model="mp", n_effective=None).analyze("t", cov)
    assert p_eps.effective_rank == p_mp.effective_rank


def test_mp_cutoff_shrinks_rank_with_noise():
    """Rank-4 signal plus isotropic noise, with adequate n_effective:
    MP should reject the noise bulk and give a tighter rank than eps."""
    torch.manual_seed(0)
    C = 40
    V_sig = torch.linalg.qr(torch.randn(C, 4)).Q
    sig_eigs = torch.tensor([50.0, 25.0, 12.0, 5.0])
    signal = V_sig @ torch.diag(sig_eigs) @ V_sig.T
    noise = 0.5 * torch.eye(C)
    cov = signal + noise

    p_eps = SVDAnalyzer(noise_model="eps", eps_relative=1e-6).analyze("t", cov)
    p_mp = SVDAnalyzer(noise_model="mp", n_effective=500).analyze("t", cov)
    assert p_mp.effective_rank <= p_eps.effective_rank


def test_ledoit_wolf_noop_for_wellconditioned():
    """LW shrinkage barely moves rank on a well-conditioned covariance."""
    torch.manual_seed(0)
    C = 16
    V = torch.linalg.qr(torch.randn(C, C)).Q
    eigs = torch.linspace(1.5, 0.5, C)
    cov = V @ torch.diag(eigs) @ V.T

    pa = SVDAnalyzer(shrinkage="none").analyze("t", cov)
    pb = SVDAnalyzer(shrinkage="ledoit_wolf").analyze("t", cov)
    assert abs(pa.effective_rank - pb.effective_rank) <= 1


def test_shrinkage_invalid_raises():
    with pytest.raises(ValueError):
        SVDAnalyzer(shrinkage="bogus")


def test_noise_model_invalid_raises():
    with pytest.raises(ValueError):
        SVDAnalyzer(noise_model="bogus")
