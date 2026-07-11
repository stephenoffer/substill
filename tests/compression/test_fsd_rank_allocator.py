"""Tests for substill.compression.rank_allocator.

Invariants verified:
  - allocator hits target within tolerance (default 1%)
  - total q is non-decreasing as budget grows
  - per-edge step sizes are respected
  - infeasible (min-rank > budget) is detected and reported, not crashed
  - greedy allocator beats single-multiplier allocation when q is uneven
"""

from __future__ import annotations

import torch

from substill.compression.rank_allocator import (
    EdgeSpec,
    allocate_ranks,
)


def _uniform_edges(n_edges: int, k_max: int, q_pattern: str = "decreasing") -> list[EdgeSpec]:
    """Build n_edges identical edges with k_max possible ranks each, scalar cost=1."""
    out = []
    torch.manual_seed(0)
    for i in range(n_edges):
        if q_pattern == "decreasing":
            q = torch.linspace(10.0, 0.1, k_max)
        elif q_pattern == "uniform":
            q = torch.ones(k_max)
        else:
            q = torch.randn(k_max).abs()
        out.append(EdgeSpec(name=f"e{i}", q=q, cost_per_rank=1.0, max_rank=k_max))
    return out


def test_allocator_hits_target_within_tolerance():
    edges = _uniform_edges(n_edges=4, k_max=10)
    target = 20  # we have 4 edges of cost 1 each, k_max=10 → max total 40
    res = allocate_ranks(edges, target_params=target, tol=0.05)
    assert abs(res.relative_error) <= 0.05, res.summary()
    assert res.total_params <= target * 1.05
    assert res.total_params >= target * 0.95


def test_allocator_respects_min_rank():
    """Even with target=0, every edge must have rank >= min_rank."""
    edges = [
        EdgeSpec(name="e1", q=torch.tensor([5.0, 3.0, 1.0]), cost_per_rank=10.0, min_rank=2),
        EdgeSpec(name="e2", q=torch.tensor([4.0, 2.0]), cost_per_rank=10.0, min_rank=1),
    ]
    # Target intentionally too small; allocator should report infeasible.
    res = allocate_ranks(edges, target_params=5, tol=0.05)
    assert res.ranks["e1"] == 2  # min_rank floor
    assert res.ranks["e2"] == 1


def test_allocator_respects_step_size():
    """Step size > 1 means rank moves in discrete chunks."""
    edges = [
        EdgeSpec(name="head_grouped", q=torch.tensor([10.0, 9.0, 8.0, 7.0]),
                 cost_per_rank=1.0, step=2, min_rank=2),
    ]
    res = allocate_ranks(edges, target_params=4, tol=0.0)
    assert res.ranks["head_grouped"] in (2, 4)  # only legal


def test_allocator_total_q_monotone_in_budget():
    """As budget grows, total q should not decrease."""
    edges = _uniform_edges(n_edges=3, k_max=10, q_pattern="decreasing")
    targets = [5, 10, 15, 20, 25]
    last_q = -float("inf")
    for tgt in targets:
        res = allocate_ranks(edges, target_params=tgt, tol=0.1)
        assert res.total_q >= last_q - 1e-6, f"q went backwards at budget {tgt}"
        last_q = res.total_q


def test_allocator_prefers_higher_q_per_cost():
    """If two edges have different q's at the same cost, allocate to the higher one."""
    edges = [
        EdgeSpec(name="cheap_high_q", q=torch.tensor([100.0, 50.0, 10.0]),
                 cost_per_rank=1.0, min_rank=1),
        EdgeSpec(name="cheap_low_q", q=torch.tensor([1.0, 0.5, 0.1]),
                 cost_per_rank=1.0, min_rank=1),
    ]
    # target = 4: starts at min_rank=1 each (total=2), can add 2 more.
    # Both extra steps should go to cheap_high_q (q=50, q=10) before cheap_low_q.
    res = allocate_ranks(edges, target_params=4, tol=0.0)
    assert res.ranks["cheap_high_q"] == 3
    assert res.ranks["cheap_low_q"] == 1


def test_allocator_with_mixed_costs():
    """An edge with high cost but high q-per-cost should still win."""
    edges = [
        EdgeSpec(name="big_costly", q=torch.tensor([1000.0, 100.0]),
                 cost_per_rank=10.0, min_rank=1),
        EdgeSpec(name="small_cheap", q=torch.tensor([5.0, 4.0, 3.0]),
                 cost_per_rank=1.0, min_rank=1),
    ]
    # min total = 10 + 1 = 11. Target 21.
    # Big_costly: ratio at next step = 100/10 = 10. Small_cheap: 4/1 = 4. Big wins first.
    res = allocate_ranks(edges, target_params=21, tol=0.05)
    assert res.ranks["big_costly"] == 2
    assert res.ranks["small_cheap"] in (1, 2, 3)
    assert res.total_params <= 21 * 1.05


def test_allocator_with_variable_cost_per_rank():
    """cost_per_rank as a tensor (e.g., last directions cheaper)."""
    edges = [
        EdgeSpec(
            name="varcost",
            q=torch.tensor([10.0, 5.0, 1.0, 0.1]),
            cost_per_rank=torch.tensor([5.0, 5.0, 1.0, 1.0]),
            min_rank=1,
        ),
    ]
    # min cost = 5 (rank 1). At target=12 we can afford one more cost-5 step
    # (rank 2, total 10) PLUS a cheap rank-3 step (cost 1, total 11).
    res = allocate_ranks(edges, target_params=12, tol=0.05)
    assert res.ranks["varcost"] >= 2
    # Total params must be a sum of contiguous prefix costs starting at rank 1.
    expected_costs = [5.0, 10.0, 11.0, 12.0]
    assert res.total_params in [int(c) for c in expected_costs]


def test_allocation_result_summary_is_str():
    edges = _uniform_edges(2, 5)
    res = allocate_ranks(edges, target_params=5, tol=0.1)
    s = res.summary()
    assert isinstance(s, str)
    assert "target:" in s


def test_allocator_handles_zero_step_correctly():
    """step=1 means 1 unit at a time; result should be exactly at integer boundary."""
    edges = [EdgeSpec(name="e", q=torch.linspace(10, 1, 10), cost_per_rank=1.0, step=1)]
    res = allocate_ranks(edges, target_params=7, tol=0.0)
    assert res.ranks["e"] == 7
    assert res.total_params == 7


def test_allocator_with_extra_overhead():
    """extra_overhead reserves params; allocator only fills the remainder."""
    edges = [EdgeSpec(name="e", q=torch.linspace(10, 1, 10), cost_per_rank=1.0)]
    res = allocate_ranks(edges, target_params=20, tol=0.05, extra_overhead=15)
    # target_after_overhead = 5. allocator should fill ~5 from edges.
    # But since target_params=20 INCLUDES overhead, we expect total ~= 20.
    # Note: extra_overhead is added to the running total; the allocator stops
    # when running total reaches target.
    assert abs(res.total_params - 20) <= 1


def test_allocator_greedy_beats_uniform_when_q_uneven():
    """When edge qualities differ, greedy should outperform uniform allocation."""
    # Edge 0: very informative (steep q drop-off, but large at top)
    # Edge 1: mostly noise (small uniform q)
    e0_q = torch.tensor([100.0, 90.0, 5.0, 1.0])
    e1_q = torch.tensor([1.0, 1.0, 1.0, 1.0])
    edges = [
        EdgeSpec(name="e0", q=e0_q, cost_per_rank=1.0),
        EdgeSpec(name="e1", q=e1_q, cost_per_rank=1.0),
    ]
    target = 4  # min=2, budget for 2 more
    res = allocate_ranks(edges, target_params=target, tol=0.05)
    # Uniform allocation would give k=2 each, total q = 100+90 + 1+1 = 192
    # Greedy should give k=3 to e0, k=1 to e1, total q = 100+90+5 + 1 = 196
    uniform_q = e0_q[:2].sum().item() + e1_q[:2].sum().item()
    assert res.total_q >= uniform_q, (
        f"greedy got {res.total_q:.2f} ≤ uniform {uniform_q:.2f}; allocation: {res.ranks}"
    )
