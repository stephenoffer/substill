"""Conv2d absorbed-init: ``W_s = V_out^T W V_in`` lifted over the kernel (Workstream C).

A convolution is linear in its channels for each kernel offset, so absorbing a teacher
conv onto channel subspaces ``V_in`` (input channels) / ``V_out`` (output channels)
preserves the projected operator exactly:

    conv_{W_s}(V_in^T x) == V_out^T · conv_W(V_in V_in^T x)

These tests check the full-rank identity (reproduces the teacher) and the reduced-rank
projected-operator identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fasd.compression.absorbed_init import absorbed_linear_init, absorbed_weight


def _orthonormal(d, k):
    return torch.linalg.qr(torch.randn(d, k))[0][:, :k]


def test_conv2d_full_rank_reproduces_teacher():
    torch.manual_seed(0)
    teacher = nn.Conv2d(6, 8, kernel_size=3, padding=1)
    student = nn.Conv2d(6, 8, kernel_size=3, padding=1)
    V_in = torch.eye(6)
    V_out = torch.eye(8)
    absorbed_linear_init(teacher, student, V_in, V_out)
    x = torch.randn(2, 6, 10, 10)
    with torch.no_grad():
        assert torch.allclose(teacher(x), student(x), atol=1e-5)


def test_conv2d_reduced_rank_matches_projected_operator():
    torch.manual_seed(0)
    in_ch, out_ch, k_in, k_out = 8, 10, 5, 6
    teacher = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
    student = nn.Conv2d(k_in, k_out, kernel_size=3, stride=1, padding=1)
    V_in = _orthonormal(in_ch, k_in)
    V_out = _orthonormal(out_ch, k_out)
    absorbed_linear_init(teacher, student, V_in, V_out)

    # absorbed_weight produces the (k_out, k_in, kh, kw) student kernel.
    W_s = absorbed_weight(teacher.weight.detach(), V_in, V_out, layout="conv2d")
    assert W_s.shape == (k_out, k_in, 3, 3)

    x = torch.randn(2, in_ch, 12, 12)
    P_in = V_in @ V_in.T  # (in_ch, in_ch) channel projector
    with torch.no_grad():
        x_proj = torch.einsum("bihw,ij->bjhw", x, P_in)
        y_full = teacher(x_proj)                                  # (B, out_ch, H, W)
        expected = torch.einsum("bohw,oc->bchw", y_full, V_out)   # project output channels
        x_s = torch.einsum("bihw,ir->brhw", x, V_in)              # compress input channels
        actual = student(x_s)
    assert torch.allclose(actual, expected, atol=1e-4), \
        f"max diff {(actual - expected).abs().max().item():.3e}"
    # Rank-reduced student is strictly smaller in parameters.
    assert sum(p.numel() for p in student.parameters()) < \
        sum(p.numel() for p in teacher.parameters())
