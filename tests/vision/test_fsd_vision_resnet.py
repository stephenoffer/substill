"""FASD vision arm: Bottleneck channel-narrowing + classifier distillation.

Validates the non-LLM path on a real (tiny) torchvision ResNet: full-width absorbed init
reproduces the teacher exactly, reduced-width builds a strictly-smaller finite student, and
the classification KD loop runs on class logits with a top-1 readout.
"""
from __future__ import annotations

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.models.resnet import Bottleneck, ResNet  # noqa: E402

from substill.vision import (  # noqa: E402
    build_resnet_student,
    channel_variance_scores,
    distill_classifier,
    top1_accuracy,
)


def _tiny_resnet(num_classes=10):
    torch.manual_seed(0)
    return ResNet(Bottleneck, [1, 1, 1, 1], num_classes=num_classes).eval()


def _loader(n=4, B=4, num_classes=10):
    torch.manual_seed(1)
    return [(torch.randn(B, 3, 32, 32), torch.randint(0, num_classes, (B,))) for _ in range(n)]


def _params(m):
    return sum(p.numel() for p in m.parameters())


def test_full_width_reproduces_teacher():
    teacher = _tiny_resnet()
    scores = channel_variance_scores(teacher, _loader(), n_batches=2)
    student, info = build_resnet_student(teacher, scores, width_ratio=1.0)
    assert all(v["s_width"] == v["width"] for v in info.values())
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(teacher(x), student(x), atol=1e-4)


def test_reduced_width_is_smaller_and_finite():
    teacher = _tiny_resnet()
    scores = channel_variance_scores(teacher, _loader(), n_batches=2)
    student, info = build_resnet_student(teacher, scores, width_ratio=0.5)
    # Every Bottleneck's inner width actually shrank.
    assert all(v["s_width"] < v["width"] for v in info.values())
    assert _params(student) < _params(teacher)
    with torch.no_grad():
        out = student(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10) and torch.isfinite(out).all()


def test_distill_classifier_runs_and_reports_top1():
    teacher = _tiny_resnet()
    scores = channel_variance_scores(teacher, _loader(), n_batches=2)
    student, _ = build_resnet_student(teacher, scores, width_ratio=0.5)
    out = distill_classifier(
        teacher, student, _loader(n=6), total_steps=6, lr=1e-3,
        generative_kd="forward_kl", val_loader=_loader(n=2), log_every=0,
    )
    assert len(out["history"]) == 6
    assert all(torch.isfinite(torch.tensor(h["kd"])) for h in out["history"])
    assert 0.0 <= out["student_top1"] <= 1.0
    # top1_accuracy is also callable standalone.
    assert 0.0 <= top1_accuracy(teacher, _loader(n=2)) <= 1.0
