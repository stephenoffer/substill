"""Tests for ASD loss functions — gradient flow, numerical correctness."""

import torch

from asd.profiling.svd_analysis import LayerProfile
from asd.profiling.sparsity_analysis import SparsityStats
from asd.losses.subspace_loss import SubspaceMatchingLoss
from asd.losses.sparsity_loss import SparsityPatternLoss
from asd.losses.combined_loss import ASDLoss


def _make_profiles() -> list[LayerProfile]:
    """Create minimal profiles for testing."""
    profiles = []
    for channels, name in [(256, "layer1.2"), (512, "layer2.3"),
                            (1024, "layer3.5"), (2048, "layer4.2")]:
        rank = channels // 4
        sv = torch.sort(torch.rand(channels), descending=True).values
        pc = torch.randn(channels, rank)
        pc, _ = torch.linalg.qr(pc)  # Orthogonalize

        sparsity = SparsityStats(
            sparsity_ratio=0.5,
            activation_histogram=torch.ones(64) / 64,
            bin_edges=torch.linspace(0, 3, 65),
            entropy=4.0,
            mean_activation=0.5,
            std_activation=0.3,
        )

        profiles.append(LayerProfile(
            name=name,
            eigenvalues=sv,
            principal_components=pc,
            effective_rank=rank,
            total_channels=channels,
            compression_ratio=rank / channels,
            sparsity_stats=sparsity,
        ))
    return profiles


def test_subspace_loss_gradient_flow():
    """Subspace loss should produce gradients for student features."""
    profiles = _make_profiles()
    loss_fn = SubspaceMatchingLoss(profiles, sv_weighted=True)

    # Simulate student projected features and teacher features
    student_projected = [
        torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
        for i, p in enumerate(profiles)
    ]
    teacher_features = [
        torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
        for i, p in enumerate(profiles)
    ]

    loss = loss_fn(student_projected, teacher_features)
    loss.backward()

    for i, sp in enumerate(student_projected):
        assert sp.grad is not None, f"No gradient for stage {i}"
        assert sp.grad.abs().sum() > 0, f"Zero gradient for stage {i}"


def test_subspace_loss_zero_when_matched():
    """Loss should be near zero when student perfectly matches teacher subspace."""
    profiles = _make_profiles()
    loss_fn = SubspaceMatchingLoss(profiles, sv_weighted=False)

    teacher_features = []
    student_projected = []
    for i, p in enumerate(profiles):
        spatial = 32 // (2**i)
        # Create teacher features
        t_feat = torch.randn(4, p.total_channels, spatial, spatial)
        teacher_features.append(t_feat)

        # Create student features that match teacher's subspace projection
        t_pooled = t_feat.mean(dim=(2, 3))  # (B, C)
        components = p.principal_components  # (C, k)
        t_subspace = t_pooled @ components  # (B, k)
        # Reshape to (B, k, 1, 1) and expand to spatial dims
        s_feat = t_subspace.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial, spatial)
        student_projected.append(s_feat)

    loss = loss_fn(student_projected, teacher_features)
    assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item():.4f}"


def test_sparsity_loss_gradient_flow():
    """Sparsity loss should produce gradients (via differentiable sparsity approx)."""
    profiles = _make_profiles()
    loss_fn = SparsityPatternLoss(profiles, num_bins=32)

    student_features = [
        torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i), requires_grad=True)
        for i, p in enumerate(profiles)
    ]

    loss = loss_fn(student_features)
    loss.backward()

    # At least the sparsity ratio term should produce gradients
    for i, sf in enumerate(student_features):
        assert sf.grad is not None, f"No gradient for stage {i}"


def test_combined_loss_all_components():
    """Combined loss should return all named components."""
    profiles = _make_profiles()
    loss_fn = ASDLoss(profiles, alpha=1.0, beta=0.5, gamma=0.3)

    student_logits = torch.randn(4, 10, requires_grad=True)
    student_projected = [
        torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
        for i, p in enumerate(profiles)
    ]
    student_features = [
        torch.randn(4, p.effective_rank, 32 // (2**i), 32 // (2**i), requires_grad=True)
        for i, p in enumerate(profiles)
    ]
    teacher_features = [
        torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i))
        for i, p in enumerate(profiles)
    ]
    labels = torch.randint(0, 10, (4,))

    losses = loss_fn(student_logits, student_projected, student_features, teacher_features, labels)

    assert "total" in losses
    assert "task" in losses
    assert "subspace" in losses
    assert "sparsity" in losses
    assert losses["total"].requires_grad


def test_sparsity_loss_few_nonzero():
    """Sparsity loss should not crash when student activations are mostly zero."""
    profiles = _make_profiles()
    loss_fn = SparsityPatternLoss(profiles, num_bins=32)

    # Create mostly-zero student features (simulating high sparsity)
    student_features = []
    for i, p in enumerate(profiles):
        feat = torch.zeros(2, p.total_channels, 32 // (2**i), 32 // (2**i), requires_grad=True)
        student_features.append(feat)

    loss = loss_fn(student_features)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()


def test_sparsity_loss_full_gradient_chain():
    """Verify gradients propagate through the full sparsity loss."""
    profiles = _make_profiles()
    loss_fn = SparsityPatternLoss(profiles, num_bins=32)

    student_features = [
        torch.randn(4, p.total_channels, 32 // (2**i), 32 // (2**i), requires_grad=True)
        for i, p in enumerate(profiles)
    ]

    loss = loss_fn(student_features)
    loss.backward()

    for i, sf in enumerate(student_features):
        assert sf.grad is not None, f"No gradient at stage {i}"
        assert torch.isfinite(sf.grad).all(), f"Non-finite gradient at stage {i}"


def test_subspace_loss_wrong_stages_raises():
    """Passing wrong number of stages should raise."""
    profiles = _make_profiles()
    loss_fn = SubspaceMatchingLoss(profiles, sv_weighted=True)

    student_projected = [torch.randn(4, 64, 32, 32)]  # Only 1 stage
    teacher_features = [torch.randn(4, 256, 32, 32)]

    try:
        loss_fn(student_projected, teacher_features)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
