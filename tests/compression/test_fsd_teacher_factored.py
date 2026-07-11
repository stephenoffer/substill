"""Tests for TeacherFactoredLinear (CPSD Phase 2-MT): frozen W_T + Stiefel V."""
import torch
import torch.nn as nn

from substill.compression.factored_linear import TeacherFactoredLinear
from substill.training.stiefel_optim import StiefelAdam, stiefel_param_groups


def _orthonormal(d, k):
    return torch.linalg.qr(torch.randn(d, k))[0]


def test_full_rank_reproduces_teacher_exactly():
    torch.manual_seed(0)
    d_in, d_out = 32, 24
    teacher = nn.Linear(d_in, d_out)
    V_in = _orthonormal(d_in, d_in)   # complete basis -> V V^T = I
    V_out = _orthonormal(d_out, d_out)
    m = TeacherFactoredLinear(teacher.weight.detach(), V_in, V_out,
                              teacher.bias.detach())
    x_full = torch.randn(5, d_in)
    # Student operates in latent space; at full rank latent dim == d_in, and
    # x_S = V_in^T x reproduces teacher output y = V_out (V_out^T W_T V_in)(V_in^T x).
    x_S = x_full @ V_in
    y_S = m(x_S)
    y_teacher_latent = teacher(x_full) @ V_out   # teacher output in V_out coords
    assert torch.allclose(y_S, y_teacher_latent, atol=1e-4)


def test_effective_weight_matches_forward():
    torch.manual_seed(1)
    d_in, d_out, k_in, k_out = 40, 40, 16, 16
    W_T = torch.randn(d_out, d_in)
    b_T = torch.randn(d_out)
    m = TeacherFactoredLinear(W_T, _orthonormal(d_in, k_in),
                              _orthonormal(d_out, k_out), b_T)
    x = torch.randn(7, k_in)
    y_route = m(x)
    W_S, b_S = m.effective_weight(), m.effective_bias()
    y_collapsed = x @ W_S.T + b_S
    assert torch.allclose(y_route, y_collapsed, atol=1e-4)


def test_fold_to_linear_preserves_forward():
    torch.manual_seed(2)
    W_T = torch.randn(48, 48)
    m = TeacherFactoredLinear(W_T, _orthonormal(48, 20), _orthonormal(48, 20))
    lin = m.fold()
    x = torch.randn(6, 20)
    assert torch.allclose(m(x), lin(x), atol=1e-4)


def test_stiefel_params_exposed_and_trainable():
    torch.manual_seed(3)
    W_T = torch.randn(32, 32)
    m = TeacherFactoredLinear(W_T, _orthonormal(32, 12), _orthonormal(32, 12))
    sp = m.stiefel_parameters()
    assert len(sp) == 2 and all(p.requires_grad for p in sp)
    # W_T is a frozen buffer, not a parameter.
    assert "W_T" not in dict(m.named_parameters())
    x = torch.randn(8, 12)
    m(x).pow(2).mean().backward()
    assert m.V_in.grad is not None and m.V_out.grad is not None


def test_free_core_exact_at_init_and_folds():
    torch.manual_seed(7)
    W_T = torch.randn(40, 40)
    m = TeacherFactoredLinear(W_T, _orthonormal(40, 16), _orthonormal(40, 16),
                              free_core=True)
    x = torch.randn(5, 16)
    base = TeacherFactoredLinear(W_T, m.V_in.detach(), m.V_out.detach())  # no free core
    # B_free is zero-init -> identical forward to the no-core module at init.
    assert torch.allclose(m(x), base(x), atol=1e-5)
    assert m.B_free is not None and m.B_free.requires_grad
    # After perturbing the free core, fold still reproduces the forward exactly.
    with torch.no_grad():
        m.B_free.add_(0.1 * torch.randn_like(m.B_free))
    assert torch.allclose(m(x), m.fold()(x), atol=1e-4)
    # B_free is a normal Euclidean parameter (not Stiefel).
    assert all(p is not m.B_free for p in m.stiefel_parameters())


def test_stiefel_optim_keeps_bases_orthonormal():
    torch.manual_seed(4)
    W_T = torch.randn(32, 32)
    model = nn.Sequential(
        TeacherFactoredLinear(W_T, _orthonormal(32, 12), _orthonormal(32, 12))
    )
    groups = stiefel_param_groups(model, base_lr=1e-2)
    opt = StiefelAdam(groups)
    x = torch.randn(16, 12)
    target = torch.randn(16, 12)
    for _ in range(50):
        opt.zero_grad()
        ((model(x) - target) ** 2).mean().backward()
        opt.step()
    for p in model[0].stiefel_parameters():
        gram = p.T @ p
        assert torch.allclose(gram, torch.eye(p.shape[1]), atol=1e-3), \
            f"basis drifted off Stiefel: max dev {(gram - torch.eye(p.shape[1])).abs().max()}"
