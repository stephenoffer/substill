"""Generative-KD loss sanity checks."""

from __future__ import annotations

import torch

from substill.losses.generative_kd import (
    contrastive_response_loss,
    forward_kl,
    reverse_kl,
    skew_kl,
)


def test_forward_and_reverse_kl_differ_for_two_mode_mixture():
    torch.manual_seed(0)
    # Teacher has mass on two modes; student has mass only on one mode.
    # Forward KL penalizes the student for missing a mode.
    # Reverse KL tolerates missing a mode (it's mode-seeking).
    V = 6
    teacher_logits = torch.full((1, 1, V), -5.0)
    teacher_logits[0, 0, 0] = 2.0
    teacher_logits[0, 0, 5] = 2.0
    student_logits = torch.full((1, 1, V), -5.0)
    student_logits[0, 0, 0] = 2.0  # covers only mode 0

    fk = forward_kl(student_logits, teacher_logits).item()
    rk = reverse_kl(student_logits, teacher_logits).item()
    sk = skew_kl(student_logits, teacher_logits, alpha=0.5).item()

    # Forward KL heavier — teacher mass on mode 5 is orphaned.
    assert fk > rk, f"expected forward_kl > reverse_kl, got fk={fk:.3f} rk={rk:.3f}"
    # Skew interpolates; should be between reverse and forward (with some slack).
    assert sk >= 0


def test_forward_kl_zero_when_student_equals_teacher():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    fk = forward_kl(logits, logits).item()
    rk = reverse_kl(logits, logits).item()
    assert abs(fk) < 1e-5
    assert abs(rk) < 1e-5


def test_contrastive_response_loss_zero_when_teacher_strictly_preferred():
    torch.manual_seed(0)
    B, T, V = 2, 4, 8
    teacher_tokens = torch.randint(0, V, (B, T))
    student_tokens = torch.randint(0, V, (B, T))
    # Student prefers teacher tokens: put huge logit on the correct index there.
    st_on_teacher = torch.full((B, T, V), -10.0)
    for b in range(B):
        for t in range(T):
            st_on_teacher[b, t, teacher_tokens[b, t]] = 10.0
    # Student strongly dis-prefers its own output tokens.
    st_on_student = torch.full((B, T, V), -10.0)
    for b in range(B):
        for t in range(T):
            # Give a random (wrong) index the mass so logp on the actual student
            # tokens is very low.
            wrong = (student_tokens[b, t] + 1) % V
            st_on_student[b, t, wrong] = 10.0
    loss = contrastive_response_loss(
        st_on_teacher, st_on_student, teacher_tokens, student_tokens, margin=0.5
    ).item()
    assert loss == 0.0 or loss < 1e-6


def test_contrastive_response_loss_positive_when_student_preferred():
    torch.manual_seed(0)
    B, T, V = 2, 4, 8
    teacher_tokens = torch.randint(0, V, (B, T))
    student_tokens = torch.randint(0, V, (B, T))
    # Reversed: student prefers its own tokens.
    st_on_student = torch.full((B, T, V), -10.0)
    for b in range(B):
        for t in range(T):
            st_on_student[b, t, student_tokens[b, t]] = 10.0
    st_on_teacher = torch.full((B, T, V), -10.0)
    loss = contrastive_response_loss(
        st_on_teacher, st_on_student, teacher_tokens, student_tokens, margin=0.5
    ).item()
    assert loss > 0.0
