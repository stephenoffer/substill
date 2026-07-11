"""Tests for distillation-driven differentiable rank (CPSD Phase 2-DDR)."""
import torch

from substill.compression.diff_rank import DifferentiableRankGate, RankBudgetController


def test_gate_shapes_and_bounds():
    g = DifferentiableRankGate(16)
    z = torch.randn(4, 7, 16)
    out = g(z)
    assert out.shape == z.shape
    gate = g.gate()
    assert gate.shape == (16,) and (gate >= 0).all() and (gate <= 1).all()


def test_init_open_starts_near_full_rank():
    g = DifferentiableRankGate(32, init_open=True)
    assert g.expected_rank().item() > 30  # nearly all columns open


def test_monotone_gate_is_non_increasing():
    g = DifferentiableRankGate(20, monotone=True)
    with torch.no_grad():
        g.alpha.copy_(torch.randn(20))
    gate = g.gate()
    diffs = gate[1:] - gate[:-1]
    assert (diffs <= 1e-6).all(), "monotone gate must be non-increasing"


def test_kd_driven_gate_selects_important_columns_under_budget():
    # Reproduce the de-risk: teacher contributes column i with importance s_i;
    # the budgeted gate should keep the high-importance columns.
    torch.manual_seed(0)
    k = 128
    s = torch.logspace(0, -3, k)              # decaying per-column importance
    X = torch.randn(4096, k)
    target = X * s
    gate = DifferentiableRankGate(k, init_open=True)
    ctrl = RankBudgetController({"e": gate}, {"e": torch.ones(k)},
                                target_params=0.5 * k, lam=0.5)
    opt = torch.optim.Adam(gate.parameters(), lr=5e-2)
    for _ in range(400):
        opt.zero_grad()
        y = gate(X) * s
        kd = ((y - target) ** 2).mean()
        (kd + ctrl.budget_penalty()).backward()
        assert torch.isfinite(gate.alpha.grad).all()  # stability
        opt.step()
    g = gate.gate().detach()
    budget = int(0.5 * k)
    top_true = set(torch.argsort(s, descending=True)[:budget].tolist())
    top_gate = set(torch.argsort(g, descending=True)[:budget].tolist())
    overlap = len(top_true & top_gate) / budget
    assert overlap >= 0.9, f"gate failed to select important columns: overlap={overlap}"
    assert gate.expected_rank().item() <= budget + 5  # respects budget


def test_budget_penalty_zero_when_under_budget():
    gate = DifferentiableRankGate(10, init_open=False)  # ~half open
    ctrl = RankBudgetController({"e": gate}, {"e": torch.ones(10)},
                                target_params=100.0, lam=1.0)  # generous budget
    assert ctrl.budget_penalty().item() == 0.0


def test_harden_returns_integer_rankmap():
    g1 = DifferentiableRankGate(16)
    g2 = DifferentiableRankGate(8)
    with torch.no_grad():
        g1.alpha.copy_(torch.tensor([5.0] * 6 + [-5.0] * 10))   # 6 open
        g2.alpha.copy_(torch.tensor([5.0] * 3 + [-5.0] * 5))    # 3 open
    ctrl = RankBudgetController({"a": g1, "b": g2},
                               {"a": torch.ones(16), "b": torch.ones(8)},
                               target_params=20.0)
    rm = ctrl.harden()
    assert rm == {"a": 6, "b": 3}


def test_cost_weighted_expected_params():
    g = DifferentiableRankGate(4, init_open=False)
    with torch.no_grad():
        g.alpha.copy_(torch.tensor([10.0, 10.0, -10.0, -10.0]))  # 2 open
    cost = torch.tensor([3.0, 3.0, 3.0, 3.0])
    ctrl = RankBudgetController({"e": g}, {"e": cost}, target_params=1.0)
    assert abs(ctrl.expected_params().item() - 6.0) < 1e-2  # 2 open * cost 3
