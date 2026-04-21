"""Tests for SlimNet student model construction."""

import torch

from asd.models.student import SlimNet


def test_forward_shape():
    """SlimNet should produce correct output shapes for CIFAR-10."""
    model = SlimNet(stage_widths=[48, 96, 160, 320], num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    logits, features = model(x)

    assert logits.shape == (4, 10)
    assert len(features) == 4
    assert features[0].shape == (4, 48, 32, 32)
    assert features[1].shape == (4, 96, 16, 16)
    assert features[2].shape == (4, 160, 8, 8)
    assert features[3].shape == (4, 320, 4, 4)


def test_parameter_count_small():
    """Student should have far fewer parameters than ResNet50 (23.5M)."""
    model = SlimNet(stage_widths=[48, 96, 160, 320])
    params = model.count_parameters()
    assert params < 5_000_000, f"Student has too many params: {params:,}"


def test_minimum_width():
    """Student should work with minimum widths."""
    model = SlimNet(stage_widths=[16, 16, 16, 16], num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits, features = model(x)
    assert logits.shape == (2, 10)


def test_repr():
    """String representation should include param count."""
    model = SlimNet(stage_widths=[48, 96, 160, 320])
    s = repr(model)
    assert "SlimNet" in s
    assert "stage_widths" in s
    assert "M)" in s  # Should show param count in millions


def test_teacher_forward_shapes_cifar():
    """Teacher with CIFAR stem should produce correct feature shapes."""
    from asd.models.teacher import TeacherWrapper

    teacher = TeacherWrapper(pretrained=False, cifar_stem=True, freeze=True)
    x = torch.randn(2, 3, 32, 32)
    logits, features = teacher(x)

    assert logits.shape == (2, 10)
    assert len(features) == 4
    assert features[0].shape == (2, 256, 32, 32)
    assert features[1].shape == (2, 512, 16, 16)
    assert features[2].shape == (2, 1024, 8, 8)
    assert features[3].shape == (2, 2048, 4, 4)


def test_teacher_student_spatial_alignment():
    """Teacher and student must produce matching spatial dimensions per stage."""
    from asd.models.teacher import TeacherWrapper

    teacher = TeacherWrapper(pretrained=False, cifar_stem=True, freeze=True)
    student = SlimNet(stage_widths=[48, 96, 160, 320])

    x = torch.randn(2, 3, 32, 32)
    _, t_features = teacher(x)
    _, s_features = student(x)

    for i, (t_feat, s_feat) in enumerate(zip(t_features, s_features)):
        assert t_feat.shape[2:] == s_feat.shape[2:], \
            f"Stage {i} spatial mismatch: teacher {t_feat.shape[2:]} vs student {s_feat.shape[2:]}"
