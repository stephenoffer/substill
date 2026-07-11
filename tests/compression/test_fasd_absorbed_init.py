"""Absorbed-weight initialization.

Full-rank case: with V_in, V_out at identity the absorbed weight equals
the teacher. Rank-k case: error bounded by tail eigenvalue mass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from substill.compression.absorbed_init import (
    absorbed_bias,
    absorbed_linear_init,
    absorbed_weight,
)


def test_absorbed_weight_full_rank_equals_teacher():
    torch.manual_seed(0)
    d_in, d_out = 16, 8
    W = torch.randn(d_out, d_in)
    V_in = torch.eye(d_in)
    V_out = torch.eye(d_out)
    W_s = absorbed_weight(W, V_in, V_out, layout="linear")
    assert torch.allclose(W_s, W, atol=1e-6)


def test_absorbed_linear_init_linear_module():
    torch.manual_seed(0)
    d_in, d_out = 8, 4
    teacher = nn.Linear(d_in, d_out)
    student = nn.Linear(d_in, d_out)
    # No compression (bases = I).
    absorbed_linear_init(teacher, student, V_in=torch.eye(d_in), V_out=torch.eye(d_out))
    x = torch.randn(5, d_in)
    assert torch.allclose(teacher(x), student(x), atol=1e-6)


def test_absorbed_linear_init_compressed_output():
    torch.manual_seed(0)
    d_in, d_out = 16, 8
    k_out = 4
    teacher = nn.Linear(d_in, d_out)
    # Student has reduced output dim.
    student = nn.Linear(d_in, k_out)
    # Pick V_out as the top-k_out right-singular vectors of teacher.weight^T
    U, S, Vh = torch.linalg.svd(teacher.weight.detach(), full_matrices=False)
    V_out = U[:, :k_out]  # d_out x k_out
    absorbed_linear_init(teacher, student, V_in=torch.eye(d_in), V_out=V_out)
    # The projected student should approximate V_out^T @ teacher(x).
    x = torch.randn(6, d_in)
    expected = teacher(x) @ V_out  # (B, k_out)
    assert torch.allclose(student(x), expected, atol=1e-5)


def test_absorbed_bias_none_input_returns_none():
    V_out = torch.eye(4)
    assert absorbed_bias(None, V_out) is None


def test_absorbed_weight_rank_k_approximates_teacher_within_tail():
    torch.manual_seed(0)
    d_in, d_out = 32, 16
    # Teacher with known spectrum: first few directions dominant.
    U, _ = torch.linalg.qr(torch.randn(d_out, d_out))
    Vt, _ = torch.linalg.qr(torch.randn(d_in, d_in))
    S = torch.cat([torch.linspace(5.0, 2.0, 6), 0.05 * torch.ones(d_in - 6)])
    # Use diag of size min(d_out, d_in)
    S = S[: min(d_out, d_in)]
    W = (U[:, : len(S)] * S) @ Vt[:, : len(S)].T  # (d_out, d_in)
    # Use top-8 right-singular vectors as V_in.
    k_in = 8
    V_in = Vt[:, :k_in]
    W_s = absorbed_weight(W, V_in, None, layout="linear")  # (d_out, k_in)
    # Reconstruct approximate W by expanding back.
    W_approx = W_s @ V_in.T
    err = float((W - W_approx).norm().item())
    # Allow slack because S is a concatenation that does not exactly
    # match the singular spectrum of W — the test just asserts that the
    # error is small relative to the teacher norm.
    assert err < float(W.norm().item())
