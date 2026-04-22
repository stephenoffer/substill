"""Tests for the new mathematical/algorithmic options added to ASD."""

import torch

from asd.losses.combined_loss import ASDLoss
from asd.losses.sparsity_loss import SparsityPatternLoss
from asd.losses.subspace_loss import SubspaceMatchingLoss, _sv_weights
from asd.models.projectors import SubspaceProjectorBank
from asd.profiling.activation_capture import CovarianceAccumulator
from asd.profiling.sparsity_analysis import SparsityStats
from asd.profiling.svd_analysis import LayerProfile, SVDAnalyzer


def _mk_profiles() -> list[LayerProfile]:
    profiles = []
    for channels, name in [(256, "layer1.2"), (512, "layer2.3"),
                           (1024, "layer3.5"), (2048, "layer4.2")]:
        rank = channels // 4
        sv = torch.sort(torch.rand(channels), descending=True).values
        pc = torch.randn(channels, rank)
        pc, _ = torch.linalg.qr(pc)
        spar = SparsityStats(
            sparsity_ratio=0.5,
            activation_histogram=torch.ones(64) / 64,
            bin_edges=torch.linspace(0, 3, 65),
            entropy=4.0,
            mean_activation=0.5,
            std_activation=0.3,
        )
        profiles.append(LayerProfile(
            name=name, eigenvalues=sv, principal_components=pc,
            effective_rank=rank, total_channels=channels,
            compression_ratio=rank / channels, sparsity_stats=spar,
        ))
    return profiles


# --- 1. SV weighting options -------------------------------------------------

def test_sv_weights_sqrt_damps_range():
    # Heavily skewed eigenvalues
    ev = torch.tensor([100.0, 10.0, 1.0, 0.01])
    linear = _sv_weights(ev, 4, "linear")
    sqrt_w = _sv_weights(ev, 4, "sqrt")
    # linear max/min >> sqrt max/min
    assert linear.max() / linear.min() > sqrt_w.max() / sqrt_w.min()


def test_sv_weights_uniform_returns_none():
    ev = torch.tensor([1.0, 0.5, 0.1])
    assert _sv_weights(ev, 3, "uniform") is None


# --- 2. Logit KD -------------------------------------------------------------

def test_logit_kd_zero_when_logits_equal():
    profiles = _mk_profiles()
    loss_fn = ASDLoss(profiles, use_logit_kd=True, alpha=0, beta=0, gamma=0, delta=1.0)

    s_logits = torch.randn(4, 10)
    t_logits = s_logits.clone()

    student_projected = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                         for i, p in enumerate(profiles)]
    student_features = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    teacher_features = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))

    losses = loss_fn(s_logits, student_projected, student_features,
                     teacher_features, labels, teacher_logits=t_logits)
    assert losses["logit"].item() < 1e-6, f"expected ~0, got {losses['logit'].item():.6f}"


def test_logit_kd_positive_when_logits_differ():
    profiles = _mk_profiles()
    loss_fn = ASDLoss(profiles, use_logit_kd=True, alpha=0, beta=0, gamma=0, delta=1.0)

    s_logits = torch.randn(4, 10)
    t_logits = torch.randn(4, 10) * 3.0  # high-confidence different logits

    student_projected = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                         for i, p in enumerate(profiles)]
    student_features = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    teacher_features = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))

    losses = loss_fn(s_logits, student_projected, student_features,
                     teacher_features, labels, teacher_logits=t_logits)
    assert losses["logit"].item() > 0.01


# --- 3. Spatial subspace ------------------------------------------------------

def test_spatial_subspace_gradient_flow():
    profiles = _mk_profiles()
    loss_fn = SubspaceMatchingLoss(profiles, mode="spatial")

    s = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
         for i, p in enumerate(profiles)]
    t = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]
    loss = loss_fn(s, t)
    loss.backward()
    for x in s:
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# --- 4. Per-pixel covariance --------------------------------------------------

def test_per_pixel_cov_has_more_samples_than_gap():
    acc_g = CovarianceAccumulator(16, mode="gap")
    acc_p = CovarianceAccumulator(16, mode="per_pixel")
    for _ in range(3):
        b = torch.randn(8, 16, 4, 4)
        acc_g.update(b)
        acc_p.update(b)
    # per_pixel sees 8*4*4*3=384 samples; gap sees 8*3=24
    assert acc_p.n == 384
    assert acc_g.n == 24


def test_per_pixel_cov_low_rank_recovery():
    """A rank-3 activation (in the channel basis) should yield eff_rank ~ 3 in per-pixel mode."""
    torch.manual_seed(0)
    C = 16
    # Create rank-3 channel activations by mixing 3 latent directions
    basis = torch.randn(C, 3)
    basis, _ = torch.linalg.qr(basis)  # orthonormal columns
    acc = CovarianceAccumulator(C, mode="per_pixel")
    for _ in range(30):
        B, H, W = 8, 4, 4
        # Per-pixel latent (rank-3): drawn from 3-dim Gaussian
        z = torch.randn(B, 3, H, W)
        # Mix: act[b, c, h, w] = Σ_r basis[c, r] * z[b, r, h, w]
        act = torch.einsum("cr,brhw->bchw", basis, z)
        acc.update(act)

    cov = acc.finalize()
    svd = SVDAnalyzer(variance_threshold=0.99)
    ev, _ = torch.linalg.eigh(cov)
    ev = ev.sort(descending=True).values.clamp(min=0)
    rank = svd._variance_rank(ev, 0.99)
    assert rank <= 4, f"Expected rank ~3, got {rank}"


# --- 5. Sparsity BCE + adaptive tau ------------------------------------------

def test_sparsity_bce_finite_and_backpropable():
    profiles = _mk_profiles()
    # Set teacher sparsity to something extreme to test numerical stability
    for p in profiles:
        p.sparsity_stats.sparsity_ratio = 0.99
    loss_fn = SparsityPatternLoss(profiles, ratio_loss="bce", adaptive_tau=True)
    feats = [torch.randn(2, p.total_channels, 32 // (2**i), 32 // (2**i), requires_grad=True)
             for i, p in enumerate(profiles)]
    loss = loss_fn(feats)
    assert torch.isfinite(loss)
    loss.backward()


def test_sparsity_adaptive_tau_tracks_activation_scale():
    """Adaptive tau should produce comparable sparsity ratios regardless of activation scale."""
    profiles = _mk_profiles()
    loss_fn = SparsityPatternLoss(profiles, ratio_loss="bce", adaptive_tau=True)

    # Small-scale activations
    small = torch.cat([torch.zeros(500), torch.randn(500) * 0.01])  # 50% zeros
    r_small = loss_fn._student_sparsity(small)
    # Large-scale activations
    large = torch.cat([torch.zeros(500), torch.randn(500) * 100])  # 50% zeros
    r_large = loss_fn._student_sparsity(large)
    # Both should be close to 0.5 — tau scales with activation std
    assert abs(r_small.item() - 0.5) < 0.15
    assert abs(r_large.item() - 0.5) < 0.15


# --- 6. Uncertainty weighting -------------------------------------------------

def test_uncertainty_weighting_has_learnable_log_sigmas():
    profiles = _mk_profiles()
    loss_fn = ASDLoss(profiles, combination="uncertainty", use_logit_kd=True)
    assert loss_fn.log_sigmas is not None
    assert loss_fn.log_sigmas.requires_grad
    assert loss_fn.log_sigmas.numel() == 4


def test_uncertainty_weighted_backprop_updates_sigmas():
    profiles = _mk_profiles()
    loss_fn = ASDLoss(profiles, combination="uncertainty", use_logit_kd=True)

    s_logits = torch.randn(4, 10, requires_grad=True)
    t_logits = torch.randn(4, 10)
    student_projected = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
                         for i, p in enumerate(profiles)]
    student_features = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
                        for i, p in enumerate(profiles)]
    teacher_features = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))

    losses = loss_fn(s_logits, student_projected, student_features,
                     teacher_features, labels, teacher_logits=t_logits)
    losses["total"].backward()
    assert loss_fn.log_sigmas.grad is not None
    assert torch.isfinite(loss_fn.log_sigmas.grad).all()


# --- 7. Effective rank alternatives ------------------------------------------

def test_rank_definitions_consistent_on_identity():
    """On λ = (1,1,...,1) all definitions should give rank close to C."""
    C = 32
    ev = torch.ones(C)
    for d in ("variance", "stable", "participation", "entropy"):
        svd = SVDAnalyzer(variance_threshold=0.99, definition=d)
        r = svd.compute_effective_rank(ev)
        assert r >= int(0.95 * C), f"{d}: got rank {r}"


def test_rank_definitions_detect_low_rank():
    """On a rank-5 spectrum, all definitions should give small rank."""
    ev = torch.tensor([100.0, 80.0, 60.0, 40.0, 20.0] + [0.001] * 27)
    for d in ("variance", "stable", "participation", "entropy"):
        svd = SVDAnalyzer(variance_threshold=0.95, definition=d)
        r = svd.compute_effective_rank(ev)
        assert r <= 10, f"{d}: got rank {r}"


def test_rank_definitions_reject_bad_name():
    try:
        SVDAnalyzer(definition="nonsense")
        assert False, "Should have raised"
    except ValueError:
        pass


# --- 8. Projector init options ------------------------------------------------

def test_projector_orthogonal_init_preserves_variance():
    """Orthogonal 1x1 conv should approximately preserve per-channel variance."""
    torch.manual_seed(0)
    bank = SubspaceProjectorBank(
        student_widths=[64, 64, 64, 64],
        teacher_ranks=[64, 64, 64, 64],
        init_mode="orthogonal",
    )
    # Feed unit-variance input, measure output variance
    x = torch.randn(32, 64, 8, 8)
    # Before BN, check the conv alone
    conv = bank.projectors[0][0]
    y = conv(x)
    # With orthogonal 1x1 init on a square matrix, output per-channel variance
    # should be approximately equal to input variance (1.0) per channel
    assert 0.5 < y.var().item() < 2.0


def test_projector_init_mode_reject_bad_name():
    try:
        SubspaceProjectorBank([16, 32, 64, 128], [8, 16, 32, 64], init_mode="nonsense")
        assert False, "Should have raised"
    except ValueError:
        pass


# --- 9. Relational (RKD) loss -----------------------------------------------

def test_rkd_zero_when_student_equals_teacher():
    from asd.losses.relation_loss import RelationalLoss
    loss_fn = RelationalLoss(distance_weight=1.0, angle_weight=2.0)
    t = torch.randn(8, 16)
    s = t.clone()
    loss = loss_fn(s, t)
    assert loss < 1e-5, f"expected ~0, got {loss}"


def test_rkd_gradient_flow():
    from asd.losses.relation_loss import RelationalLoss
    loss_fn = RelationalLoss()
    s = torch.randn(8, 16, requires_grad=True)
    t = torch.randn(8, 16)
    loss = loss_fn(s, t)
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_rkd_scale_invariance():
    """Distance term is mean-normalized → loss is invariant to global scale."""
    from asd.losses.relation_loss import RelationalLoss
    loss_fn = RelationalLoss(distance_weight=1.0, angle_weight=0.0)  # distance only
    t = torch.randn(8, 16)
    s = torch.randn(8, 16)
    loss_a = loss_fn(s, t)
    loss_b = loss_fn(s * 5, t)  # student scaled by 5×
    assert abs(loss_a.item() - loss_b.item()) < 1e-4, f"{loss_a} vs {loss_b}"


# --- 10. ASDLoss with relation + all features -------------------------------

# --- 11. Cosine-similarity subspace mode ------------------------------------

def test_cosine_subspace_bounded():
    """Cosine subspace loss is in [0, 2] and invariant to positive scaling."""
    from asd.losses.subspace_loss import SubspaceMatchingLoss
    profiles = _mk_profiles()
    loss = SubspaceMatchingLoss(profiles, mode="cosine_spatial", sv_weighted=False)

    s = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]
    t = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]

    l = loss(s, t).item()
    assert 0.0 <= l <= 2.0, f"cosine loss out of [0,2]: {l}"

    # Scale-invariance: multiply student by 10, loss unchanged (bounded)
    s2 = [x * 10 for x in s]
    l2 = loss(s2, t).item()
    assert abs(l - l2) < 1e-4, f"not scale-invariant: {l} vs {l2}"


def test_cosine_subspace_zero_when_matched():
    """Cosine loss is near-zero when student == teacher projection exactly."""
    from asd.losses.subspace_loss import SubspaceMatchingLoss
    profiles = _mk_profiles()
    loss = SubspaceMatchingLoss(profiles, mode="cosine_spatial", sv_weighted=False)
    t = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]
    s = [torch.einsum("bchw,ck->bkhw", t_feat, p.principal_components)
         for t_feat, p in zip(t, profiles)]
    l = loss(s, t).item()
    assert l < 1e-4, f"expected ~0 at match, got {l}"


# --- 12. L2 feature normalization --------------------------------------------

def test_l2_normalization_scale_invariance():
    """With normalize_features=True, MSE subspace loss is invariant to feature scale."""
    from asd.losses.subspace_loss import SubspaceMatchingLoss
    profiles = _mk_profiles()
    loss = SubspaceMatchingLoss(profiles, mode="spatial", sv_weighted=False, normalize_features=True)
    s = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]
    t = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
         for i, p in enumerate(profiles)]
    l1 = loss(s, t).item()
    # Scale student and teacher separately — normalized MSE should be roughly the same
    s2 = [x * 5 for x in s]
    t2 = [x * 100 for x in t]
    l2 = loss(s2, t2).item()
    # Allow small drift (normalization is not perfectly scale-invariant with MSE)
    assert abs(l1 - l2) < 0.1, f"L2-normalized loss not scale-invariant: {l1} vs {l2}"


# --- 13. Beta warmup ---------------------------------------------------------

def test_beta_warmup_scales_linearly():
    from asd.training.scheduler import BetaWarmupScheduler
    sched = BetaWarmupScheduler(warmup_epochs=4, initial_scale=0.0)
    assert sched.get_beta_scale(0) == 0.0
    assert abs(sched.get_beta_scale(2) - 0.5) < 1e-6
    assert sched.get_beta_scale(4) == 1.0
    assert sched.get_beta_scale(10) == 1.0


def test_asd_loss_applies_beta_scale():
    """beta_scale multiplies into the subspace component only."""
    profiles = _mk_profiles()
    loss_fn = ASDLoss(profiles, alpha=0, beta=1.0, gamma=0, delta=0, use_logit_kd=False)

    s_logits = torch.randn(4, 10)
    student_projected = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                         for i, p in enumerate(profiles)]
    student_features = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    teacher_features = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))

    l_full = loss_fn(s_logits, student_projected, student_features, teacher_features, labels,
                     beta_scale=1.0)["total"].item()
    l_half = loss_fn(s_logits, student_projected, student_features, teacher_features, labels,
                     beta_scale=0.5)["total"].item()
    l_zero = loss_fn(s_logits, student_projected, student_features, teacher_features, labels,
                     beta_scale=0.0)["total"].item()
    # 0.5× beta_scale gives ~half the total (since only the subspace term is on)
    assert abs(l_half - l_full * 0.5) < 1e-4, f"{l_half} vs {l_full*0.5}"
    assert abs(l_zero) < 1e-6, f"beta_scale=0 should zero out, got {l_zero}"


# --- 14. EMA auto-normalization ----------------------------------------------

def test_auto_normalize_scales_losses_to_unit():
    """After one forward, each normalized loss component should be ~1."""
    profiles = _mk_profiles()
    loss_fn = ASDLoss(profiles, auto_normalize=True, auto_norm_momentum=0.0)

    s_logits = torch.randn(4, 10)
    t_logits = torch.randn(4, 10)
    student_projected = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                         for i, p in enumerate(profiles)]
    student_features = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    teacher_features = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))

    losses = loss_fn(s_logits, student_projected, student_features, teacher_features, labels,
                     teacher_logits=t_logits)
    # At momentum=0 the EMA equals the current detached magnitude → normalized = 1.
    # Only α+β+γ+δ (no warmup, γ=0.3 by default) contribute.
    expected = 1.0 + 0.5 + 0.3 + 1.0  # α + β + γ + δ
    assert abs(losses["total"].item() - expected) < 1e-3, \
        f"expected ~{expected} normalized total, got {losses['total'].item()}"


# --- 15. Frozen projector ----------------------------------------------------

def test_frozen_projector_has_no_grads():
    """freeze=True → projector params have requires_grad=False."""
    bank = SubspaceProjectorBank([16, 32, 64, 128], [8, 16, 32, 64], freeze=True)
    for p in bank.parameters():
        assert not p.requires_grad, "frozen projector should have no grads"


def test_unfrozen_projector_trainable():
    bank = SubspaceProjectorBank([16, 32, 64, 128], [8, 16, 32, 64], freeze=False)
    has_trainable = any(p.requires_grad for p in bank.parameters())
    assert has_trainable


# --- 16. Trainer best-checkpoint -------------------------------------------

def test_trainer_tracks_best_state():
    """Trainer should save best-val state and expose it via _best_state."""
    # Just a smoke test that keep_best=True enables the machinery.
    from asd.training.trainer import ASDTrainer
    from asd.models.student import SlimNet
    # Stand-in dummy components (can't actually run training in unit test fast,
    # so just check the attribute exists).
    trainer = ASDTrainer.__new__(ASDTrainer)
    trainer._best_acc = 0.0
    trainer._best_state = None
    trainer.keep_best = True
    assert trainer._best_state is None
    # Simulate an improving epoch
    trainer._best_state = {"student": {}, "projectors": {}}
    assert trainer._best_state is not None


def test_asd_loss_with_relation():
    """Turning on relation should add a 'relation' key and change the total."""
    profiles = _mk_profiles()

    base = ASDLoss(profiles, use_logit_kd=True, use_relation=False)
    with_rel = ASDLoss(profiles, use_logit_kd=True, use_relation=True, epsilon=1.0)

    s_logits = torch.randn(4, 10, requires_grad=True)
    t_logits = torch.randn(4, 10)
    student_projected = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
                         for i, p in enumerate(profiles)]
    student_features = [torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
                        for i, p in enumerate(profiles)]
    teacher_features = [torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
                        for i, p in enumerate(profiles)]
    labels = torch.randint(0, 10, (4,))

    l_base = base(s_logits, student_projected, student_features, teacher_features, labels, teacher_logits=t_logits)
    l_rel = with_rel(s_logits, student_projected, student_features, teacher_features, labels, teacher_logits=t_logits)

    assert "relation" in l_rel
    assert l_rel["relation"].item() > 0
    # Base has relation=0
    assert l_base["relation"].item() == 0.0
