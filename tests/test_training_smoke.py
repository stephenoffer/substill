"""Smoke test — verify the full training pipeline runs without errors."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from asd.profiling.svd_analysis import LayerProfile
from asd.profiling.sparsity_analysis import SparsityStats
from asd.models.teacher import TeacherWrapper
from asd.models.student import SlimNet
from asd.models.projectors import SubspaceProjectorBank
from asd.losses.combined_loss import ASDLoss
from asd.training.trainer import ASDTrainer
from asd.training.scheduler import LossWeightScheduler


def _make_profiles() -> list[LayerProfile]:
    """Create minimal profiles for testing."""
    profiles = []
    for channels, name in [(256, "layer1.2"), (512, "layer2.3"),
                            (1024, "layer3.5"), (2048, "layer4.2")]:
        rank = max(16, channels // 8)
        sv = torch.sort(torch.rand(channels), descending=True).values
        pc = torch.randn(channels, rank)
        pc, _ = torch.linalg.qr(pc)

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


def test_training_smoke():
    """Run 2 training steps to verify the full pipeline works."""
    profiles = _make_profiles()
    stage_widths = [32, 64, 128, 256]
    teacher_ranks = [p.effective_rank for p in profiles]

    teacher = TeacherWrapper(profiles=profiles, cifar_stem=True, pretrained=False, num_classes=10)
    student = SlimNet(stage_widths=stage_widths, blocks_per_stage=1, num_classes=10)
    projectors = SubspaceProjectorBank(student_widths=stage_widths, teacher_ranks=teacher_ranks)
    loss_fn = ASDLoss(profiles=profiles, alpha=1.0, beta=0.5, gamma=0.3, num_bins=32)

    params = list(student.parameters()) + list(projectors.parameters())
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    loss_scheduler = LossWeightScheduler(warmup_epochs=1)

    # Tiny synthetic dataset
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = ASDTrainer(
        teacher=teacher,
        student=student,
        projectors=projectors,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_scheduler=loss_scheduler,
        device="cpu",
    )

    # Train 2 epochs — should not crash
    metrics_0 = trainer.train_epoch(loader, epoch=0)
    metrics_1 = trainer.train_epoch(loader, epoch=1)

    assert "total" in metrics_0
    assert "total" in metrics_1

    # Evaluate
    eval_metrics = trainer.evaluate(loader)
    assert "accuracy" in eval_metrics
    assert 0 <= eval_metrics["accuracy"] <= 1
