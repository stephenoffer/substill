"""Tests for the fixes that address the critical-review findings.

Each section names the review issue it covers.
"""

from __future__ import annotations

import copy
import warnings

import pytest
import torch

from asd.losses.combined_loss import ASDLoss, _LOSS_NAMES
from asd.losses.relation_loss import RelationalLoss
from asd.losses.sparsity_loss import SparsityPatternLoss
from asd.losses.subspace_loss import SubspaceMatchingLoss
from asd.models.projectors import SubspaceProjectorBank
from asd.models.student import SlimNet
from asd.profiling.activation_capture import CovarianceAccumulator
from asd.profiling.sparsity_analysis import SparsityStats
from asd.profiling.svd_analysis import (
    LayerProfile,
    SVDAnalyzer,
    aggregate_stage_profile,
    group_profiles_by_stage,
    profiles_to_stage_blocks,
    profiles_to_stage_widths,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

def _dummy_sparsity() -> SparsityStats:
    return SparsityStats(
        sparsity_ratio=0.5,
        activation_histogram=torch.ones(64) / 64,
        bin_edges=torch.linspace(0, 3, 65),
        entropy=4.0,
        mean_activation=0.5,
        std_activation=0.3,
    )


def _make_profiles(blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1)) -> list[LayerProfile]:
    """Profiles with multiple blocks per stage, for testing stage aggregation."""
    profiles = []
    for stage_idx, channels in enumerate([256, 512, 1024, 2048]):
        for b in range(blocks_per_stage[stage_idx]):
            rank = max(8, channels // 4 - b * 4)
            sv = torch.sort(torch.rand(channels), descending=True).values
            pc = torch.randn(channels, rank)
            pc, _ = torch.linalg.qr(pc)
            profiles.append(LayerProfile(
                name=f"layer{stage_idx+1}.{b}",
                eigenvalues=sv,
                principal_components=pc,
                effective_rank=rank,
                total_channels=channels,
                compression_ratio=rank / channels,
                sparsity_stats=_dummy_sparsity(),
            ))
    return profiles


# -----------------------------------------------------------------------------
# §2: rank definitions use ceiling (no longer collapse to 1 on heavy tails)
# -----------------------------------------------------------------------------

def test_stable_rank_heavy_tail_no_collapse():
    """Previously round() sent stable_rank → 1 on sharply peaked spectra.

    With ceiling, rank(2.7) = 3 — a clear signal that the tail carries mass.
    """
    ev = torch.tensor([100.0] + [0.5] * 50)  # stable = 1 + 50·0.005 ≈ 1.25
    analyzer = SVDAnalyzer(definition="stable")
    assert analyzer.compute_effective_rank(ev) == 2, \
        f"ceiling should push rank above 1, got {analyzer.compute_effective_rank(ev)}"


def test_participation_rank_uses_ceiling():
    ev = torch.tensor([100.0, 30.0, 10.0])  # sum² / Σ² = 140²/10900 ≈ 1.80
    analyzer = SVDAnalyzer(definition="participation")
    assert analyzer.compute_effective_rank(ev) == 2


def test_entropy_rank_uses_ceiling():
    ev = torch.tensor([10.0, 5.0, 1.0])  # exp(H) ≈ 2.5 → ceil = 3
    analyzer = SVDAnalyzer(definition="entropy")
    assert analyzer.compute_effective_rank(ev) == 3


def test_denoise_threshold_discards_float_noise():
    """Eigenvalues at floating-point noise level should not count."""
    ev = torch.tensor([100.0, 50.0] + [1e-10] * 100)
    analyzer = SVDAnalyzer(definition="participation")
    # Without denoising, the tiny tail would push participation upward. With
    # denoising at eps_relative=1e-6·λ_max=1e-4, the 1e-10 values are zeroed.
    rank = analyzer.compute_effective_rank(ev)
    assert 1 <= rank <= 3, f"noise tail leaked into rank: {rank}"


# -----------------------------------------------------------------------------
# §4 / new API: stage aggregation modes
# -----------------------------------------------------------------------------

def test_aggregate_stage_profile_modes():
    """All three stage-aggregation modes must produce valid LayerProfiles."""
    profiles = _make_profiles(blocks_per_stage=(3, 4, 6, 3))
    stage_map = group_profiles_by_stage(profiles)
    for channels, block_profiles in stage_map.items():
        for mode in ("last", "max_rank", "average"):
            prof = aggregate_stage_profile(block_profiles, mode=mode)
            assert prof.total_channels == channels
            assert prof.principal_components.shape[0] == channels
            assert prof.effective_rank >= 1
            assert prof.principal_components.shape[1] == prof.effective_rank


def test_aggregate_stage_profile_average_uses_all_blocks():
    """Averaged PCs depend on every block — dropping one changes the result."""
    profiles = _make_profiles(blocks_per_stage=(3, 1, 1, 1))
    stage_map = group_profiles_by_stage(profiles)
    blocks = stage_map[256]
    avg_all = aggregate_stage_profile(blocks, mode="average").eigenvalues.clone()
    avg_dropped = aggregate_stage_profile(blocks[:-1], mode="average").eigenvalues.clone()
    assert not torch.allclose(avg_all, avg_dropped), \
        "average aggregation should depend on each block"


def test_aggregate_stage_rejects_mixed_channels():
    profiles = _make_profiles(blocks_per_stage=(1, 1, 1, 1))
    with pytest.raises(ValueError):
        aggregate_stage_profile(profiles, mode="average")


def test_subspace_loss_stage_aggregation_shape():
    profiles = _make_profiles(blocks_per_stage=(2, 2, 2, 2))
    for mode in ("last", "max_rank", "average"):
        loss = SubspaceMatchingLoss(profiles, mode="spatial", stage_aggregation=mode)
        # Component buffers must be the correct shape for each stage.
        for i, ch in enumerate([256, 512, 1024, 2048]):
            components = getattr(loss, f"components_{i}")
            assert components.shape[0] == ch


# -----------------------------------------------------------------------------
# §4: cosine subspace honors SV weighting (was previously silently dropped)
# -----------------------------------------------------------------------------

def test_cosine_subspace_sv_weighting_changes_loss():
    profiles = _make_profiles()
    # Strongly skewed eigenvalues: sqrt-weighting will dampen the largest.
    for p in profiles:
        p.eigenvalues.copy_(torch.linspace(100, 0.01, len(p.eigenvalues)))

    uniform = SubspaceMatchingLoss(profiles, mode="cosine_spatial", sv_weighted=False)
    weighted = SubspaceMatchingLoss(profiles, mode="cosine_spatial",
                                     sv_weighted=True, sv_weighting="sqrt")

    torch.manual_seed(0)
    s = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]
    t = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]

    l_uniform = uniform(s, t).item()
    l_weighted = weighted(s, t).item()
    assert abs(l_uniform - l_weighted) > 1e-5, \
        "SV weighting must now affect cosine-mode loss (previously ignored)"


# -----------------------------------------------------------------------------
# §9: concrete fixes — zeros shape, EMA buffers, use_relation decoupling
# -----------------------------------------------------------------------------

def test_loss_totals_are_scalar_shape():
    profiles = _make_profiles()
    loss_fn = ASDLoss(profiles, alpha=1, beta=0.5, gamma=0.3, delta=1.0)
    s_logits = torch.randn(4, 10, requires_grad=True)
    t_logits = torch.randn(4, 10)
    s_proj = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
              for i, p in enumerate(profiles)]
    s_feat = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
              for i, p in enumerate(profiles)]
    t_feat = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
              for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))
    losses = loss_fn(s_logits, s_proj, s_feat, t_feat, labels, teacher_logits=t_logits)
    assert losses["total"].shape == ()
    assert losses["task"].shape == ()
    assert losses["subspace"].shape == ()


def test_ema_buffers_persist_in_state_dict():
    """Auto-normalize EMA must survive checkpoint save/load."""
    profiles = _make_profiles()
    loss_fn = ASDLoss(profiles, auto_normalize=True, auto_norm_momentum=0.9)
    s_logits = torch.randn(4, 10)
    t_logits = torch.randn(4, 10)
    s_proj = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
              for i, p in enumerate(profiles)]
    s_feat = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
              for i, p in enumerate(profiles)]
    t_feat = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
              for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))
    # Run a few forwards to populate EMAs.
    for _ in range(3):
        loss_fn(s_logits, s_proj, s_feat, t_feat, labels, teacher_logits=t_logits)

    sd = loss_fn.state_dict()
    for name in _LOSS_NAMES:
        assert f"ema_{name}" in sd, f"ema_{name} missing from state_dict"
    assert "ema_count" in sd

    # Round-trip.
    fresh = ASDLoss(profiles, auto_normalize=True, auto_norm_momentum=0.9)
    fresh.load_state_dict(sd)
    for name in _LOSS_NAMES:
        assert torch.equal(getattr(fresh, f"ema_{name}"), getattr(loss_fn, f"ema_{name}"))
    assert int(fresh.ema_count.item()) == int(loss_fn.ema_count.item())


def test_use_relation_not_implicitly_enabled_by_epsilon():
    """Bumping ε with use_relation=False must leave relation disabled."""
    profiles = _make_profiles()
    loss_fn = ASDLoss(profiles, use_relation=False, epsilon=5.0)
    assert loss_fn.use_relation is False
    assert loss_fn.relation_loss is None


def test_kendall_gal_formula_canonical_form():
    """Gradient w.r.t. log_sigma at s=0 and L=1 should be 0 under 0.5·e^−s·L + 0.5·s.

    d/ds [0.5·exp(-s)·L + 0.5·s] = -0.5·exp(-s)·L + 0.5 = 0 when L = 1.
    The old form `exp(-s)·L + s` would give grad = -L + 1 at s=0 — also 0 at
    L=1, but with a 2× scale factor on the regularizer that biases training.
    This test pins the new normalization.
    """
    profiles = _make_profiles()
    loss_fn = ASDLoss(profiles, combination="uncertainty", use_logit_kd=False,
                       alpha=0, beta=0, gamma=0)
    # Manually construct a total = Σ 0.5·exp(-s_i)·L_i + 0.5·s_i with L_i = 1.
    with torch.no_grad():
        loss_fn.log_sigmas.zero_()
    losses = [torch.tensor(1.0, requires_grad=False)] * 3
    total = loss_fn._uncertainty_combine(
        losses[0], losses[1], losses[2],
        torch.zeros(()), torch.zeros(()), torch.zeros(()),
        gamma_scale=1.0,
    )
    # total = 3 · (0.5·1·1 + 0.5·0) = 1.5. Old formula would give 3 · (1 + 0) = 3.
    assert abs(total.item() - 1.5) < 1e-6, f"expected 1.5, got {total.item()}"


# -----------------------------------------------------------------------------
# §3: spatial subsampling in CovarianceAccumulator
# -----------------------------------------------------------------------------

def test_covariance_spatial_subsample_reduces_samples():
    acc = CovarianceAccumulator(num_channels=16, mode="per_pixel", spatial_subsample=2)
    acc.update(torch.randn(4, 16, 8, 8))
    # Subsample keeps every 2nd position → 4 * 4 * 4 = 64 samples per image,
    # times 4 images = 256.
    assert acc.n == 64


def test_covariance_spatial_subsample_preserves_shape():
    acc = CovarianceAccumulator(num_channels=8, mode="per_pixel", spatial_subsample=2)
    for _ in range(3):
        acc.update(torch.randn(4, 8, 8, 8))
    cov = acc.finalize()
    assert cov.shape == (8, 8)
    assert torch.allclose(cov, cov.T, atol=1e-5)


def test_covariance_svd_rejects_bad_subsample():
    with pytest.raises(ValueError):
        CovarianceAccumulator(num_channels=8, spatial_subsample=0)


# -----------------------------------------------------------------------------
# §3: SVDAnalyzer rejects non-PSD covariance instead of silently clamping
# -----------------------------------------------------------------------------

def test_svd_analyzer_rejects_non_psd():
    analyzer = SVDAnalyzer()
    # Strongly indefinite matrix — λ_min/λ_max will be well below the tolerance.
    bad = torch.tensor([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
    with pytest.raises(ValueError, match="not PSD"):
        analyzer.analyze("bad", bad, _dummy_sparsity())


def test_svd_analyzer_tolerates_float_noise():
    """Tiny negative eigenvalues from accumulator float noise should not error."""
    analyzer = SVDAnalyzer()
    # Build a legit PSD matrix, then perturb by < 1e-4 · λ_max.
    A = torch.randn(16, 16)
    psd = A @ A.T
    # Add numerical noise well within the tolerance.
    lam_max = torch.linalg.eigvalsh(psd).max().item()
    noisy = psd - (1e-6 * lam_max) * torch.eye(16)
    profile = analyzer.analyze("ok", noisy, _dummy_sparsity())
    assert profile.effective_rank >= 1


# -----------------------------------------------------------------------------
# §5: sparsity loss sigma adapts to bin width (no more fixed 0.1)
# -----------------------------------------------------------------------------

def test_sparsity_sigma_tracks_bin_width():
    """On narrow-range bins, derived sigma should be small — otherwise the
    histogram kernel washes out structure."""
    profiles = _make_profiles()
    # Narrow teacher bin range (late-stage activation case).
    for p in profiles:
        p.sparsity_stats.bin_edges = torch.linspace(0, 0.1, 65)

    loss = SparsityPatternLoss(profiles, num_bins=64)
    bin_edges = getattr(loss, "teacher_bin_edges_0")
    sigma = loss._stage_sigma(bin_edges)
    # bin_width = 0.1/64 ≈ 0.00156; sigma = 1.5 · bin_width ≈ 0.00234.
    # Previously was max(0.1, …) = 0.1 — 40× larger.
    assert sigma < 0.01, f"sigma should track bin width, got {sigma}"


def test_sparsity_kl_normalized_by_log_bins():
    """KL component should be divided by log(num_bins)."""
    profiles_a = _make_profiles()
    profiles_b = copy.deepcopy(profiles_a)
    loss_64 = SparsityPatternLoss(profiles_a, num_bins=64)
    loss_8 = SparsityPatternLoss(profiles_b, num_bins=8)
    # _log_num_bins: ln(64) = 4.159, ln(8) = 2.079.
    import math
    assert abs(loss_64._log_num_bins - math.log(64)) < 1e-6
    assert abs(loss_8._log_num_bins - math.log(8)) < 1e-6


# -----------------------------------------------------------------------------
# §8: projector BN off by default, count_parameters exposed
# -----------------------------------------------------------------------------

def test_projector_no_bn_by_default():
    bank = SubspaceProjectorBank([16, 32, 64, 128], [8, 16, 32, 64])
    # With use_bn=False the projector is just a single Conv2d (no BN layer).
    for stage in bank.projectors:
        assert len(stage) == 1
        assert isinstance(stage[0], torch.nn.Conv2d)
        # Bias is on since there's no BN to absorb it.
        assert stage[0].bias is not None


def test_projector_with_bn_opts_in():
    bank = SubspaceProjectorBank([16, 32, 64, 128], [8, 16, 32, 64], use_bn=True)
    for stage in bank.projectors:
        assert len(stage) == 2
        assert isinstance(stage[0], torch.nn.Conv2d)
        assert isinstance(stage[1], torch.nn.BatchNorm2d)
        assert stage[0].bias is None  # bias folded into BN


def test_projector_count_parameters_reported():
    bank = SubspaceProjectorBank([48, 96, 160, 320], [64, 128, 256, 512])
    total = bank.count_parameters()
    manual = sum(p.numel() for p in bank.parameters())
    assert total == manual
    assert total > 0


# -----------------------------------------------------------------------------
# SlimNet stem_type option (replaces the CIFAR-only hardcoded stem)
# -----------------------------------------------------------------------------

def test_slimnet_cifar_stem_is_3x3_stride1():
    model = SlimNet(stage_widths=[48, 96, 160, 320], stem_type="cifar")
    conv = model.stem[0]
    assert conv.kernel_size == (3, 3)
    assert conv.stride == (1, 1)
    # No maxpool in CIFAR stem.
    assert len(model.stem) == 3


def test_slimnet_imagenet_stem_is_7x7_stride2_with_pool():
    model = SlimNet(stage_widths=[48, 96, 160, 320], stem_type="imagenet")
    conv = model.stem[0]
    assert conv.kernel_size == (7, 7)
    assert conv.stride == (2, 2)
    # ImageNet stem appends MaxPool after conv+bn+relu.
    assert any(isinstance(m, torch.nn.MaxPool2d) for m in model.stem)


def test_slimnet_rejects_unknown_stem_type():
    with pytest.raises(ValueError):
        SlimNet(stage_widths=[48, 96, 160, 320], stem_type="nonsense")


# -----------------------------------------------------------------------------
# New API: profiles_to_stage_blocks
# -----------------------------------------------------------------------------

def test_profiles_to_stage_blocks_saturation():
    """Rank that plateaus early → fewer blocks; rank that keeps growing → max."""
    # Stage 1 ranks [10, 11, 11, 11]: the third block (index 2) is the first
    # to contribute < 5% relative growth over the previous, so 2 blocks worth
    # of information are used.
    # Stage 2 ranks [10, 20, 40, 80]: never saturates → max_blocks=4.
    profiles = []
    for r in [10, 11, 11, 11]:
        profiles.append(LayerProfile(
            name="s1.x", eigenvalues=torch.ones(64), principal_components=torch.eye(64)[:, :r],
            effective_rank=r, total_channels=64, compression_ratio=r/64,
            sparsity_stats=_dummy_sparsity(),
        ))
    for r in [10, 20, 40, 80]:
        profiles.append(LayerProfile(
            name="s2.x", eigenvalues=torch.ones(128), principal_components=torch.eye(128)[:, :r],
            effective_rank=r, total_channels=128, compression_ratio=r/128,
            sparsity_stats=_dummy_sparsity(),
        ))
    blocks = profiles_to_stage_blocks(profiles, min_blocks=1, max_blocks=4,
                                      saturation_tol=0.05)
    assert blocks == [2, 4], f"expected [2, 4] from saturation analysis, got {blocks}"


def test_profiles_to_stage_blocks_clamps_min():
    """Single-block stages fall back to min_blocks."""
    profiles = [LayerProfile(
        name="s.0", eigenvalues=torch.ones(16), principal_components=torch.eye(16)[:, :4],
        effective_rank=4, total_channels=16, compression_ratio=0.25,
        sparsity_stats=_dummy_sparsity(),
    )]
    blocks = profiles_to_stage_blocks(profiles, min_blocks=2, max_blocks=6)
    assert blocks == [2]


def test_profiles_to_stage_widths_reduction_modes():
    """Different reduction modes should give different widths."""
    profiles = []
    for r in [30, 50, 20]:
        profiles.append(LayerProfile(
            name="s.x", eigenvalues=torch.ones(64), principal_components=torch.eye(64)[:, :r],
            effective_rank=r, total_channels=64, compression_ratio=r/64,
            sparsity_stats=_dummy_sparsity(),
        ))
    for r in [30, 50, 20]:
        profiles.append(LayerProfile(
            name="s2.x", eigenvalues=torch.ones(128), principal_components=torch.eye(128)[:, :r],
            effective_rank=r, total_channels=128, compression_ratio=r/128,
            sparsity_stats=_dummy_sparsity(),
        ))
    w_max = profiles_to_stage_widths(profiles, min_width=16, width_multiple=8,
                                     rank_reduction="max")
    w_last = profiles_to_stage_widths(profiles, min_width=16, width_multiple=8,
                                       rank_reduction="last")
    w_mean = profiles_to_stage_widths(profiles, min_width=16, width_multiple=8,
                                       rank_reduction="mean")
    # max: per-stage max = 50 → rounded to 56. last: 20 → 24. mean: 33.33 → 34 → 40.
    assert w_max == [56, 56]
    assert w_last == [24, 24]
    assert w_mean == [40, 40]


# -----------------------------------------------------------------------------
# §9: trainer restore_best + T_max warning
# -----------------------------------------------------------------------------

def test_trainer_restore_best_loads_snapshot():
    """restore_best() should write the saved best-state back into the live modules."""
    from asd.training.trainer import ASDTrainer
    from asd.models.student import SlimNet
    from asd.models.projectors import SubspaceProjectorBank

    trainer = ASDTrainer.__new__(ASDTrainer)
    trainer.device = "cpu"
    trainer.student = SlimNet(stage_widths=[16, 16, 16, 16], blocks_per_stage=1)
    trainer.projectors = SubspaceProjectorBank([16, 16, 16, 16], [8, 8, 8, 8])

    # Capture a distinctive snapshot (student filled with zeros), then perturb
    # the live weights and verify restore returns the zeros.
    snap = {
        "student": {k: torch.zeros_like(v) for k, v in trainer.student.state_dict().items()},
        "projectors": {k: torch.zeros_like(v) for k, v in trainer.projectors.state_dict().items()},
    }
    trainer._best_state = snap

    with torch.no_grad():
        for p in trainer.student.parameters():
            p.fill_(7.0)
    assert any(float(p.detach().abs().sum()) > 0 for p in trainer.student.parameters())

    restored = trainer.restore_best()
    assert restored is True
    assert all(float(p.detach().abs().sum()) == 0 for p in trainer.student.parameters())


def test_trainer_restore_best_returns_false_when_empty():
    from asd.training.trainer import ASDTrainer
    trainer = ASDTrainer.__new__(ASDTrainer)
    trainer._best_state = None
    assert trainer.restore_best.__func__(trainer) is False


def test_trainer_warns_on_tmax_mismatch():
    """Catching the common cosine-scheduler footgun."""
    from asd.training.trainer import ASDTrainer

    class _FakeCosine:
        T_max = 100

    trainer = ASDTrainer.__new__(ASDTrainer)
    trainer.lr_warmup_epochs = 2
    trainer.lr_scheduler = _FakeCosine()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trainer._warn_if_cosine_tmax_mismatch(num_epochs=100)
    assert any("T_max" in str(rec.message) for rec in w)


def test_trainer_no_warning_when_tmax_matches():
    from asd.training.trainer import ASDTrainer

    class _FakeCosine:
        T_max = 98

    trainer = ASDTrainer.__new__(ASDTrainer)
    trainer.lr_warmup_epochs = 2
    trainer.lr_scheduler = _FakeCosine()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trainer._warn_if_cosine_tmax_mismatch(num_epochs=100)
    assert not any("T_max" in str(rec.message) for rec in w), \
        "T_max warning should only fire on the mismatched case"


# -----------------------------------------------------------------------------
# Relation loss: chunked angle produces the same value as the full-batch form
# -----------------------------------------------------------------------------

def test_relation_angle_chunked_matches_full():
    """Chunking must not change the numerical result of the angle loss."""
    torch.manual_seed(0)
    B, D = 12, 32
    s = torch.randn(B, D)
    t = torch.randn(B, D)

    loss_fn = RelationalLoss(distance_weight=0.0, angle_weight=1.0)
    full = loss_fn.angle_loss(s, t, chunk_size=B)
    chunked = loss_fn.angle_loss(s, t, chunk_size=3)
    assert abs(full.item() - chunked.item()) < 1e-5, \
        f"chunking changed the result: {full} vs {chunked}"


def test_relation_loss_gradient_still_flows():
    """Chunked angle must still backprop to the student."""
    loss_fn = RelationalLoss()
    s = torch.randn(8, 16, requires_grad=True)
    t = torch.randn(8, 16)
    loss_fn(s, t).backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
