"""Stability-adjusted rank caps behavioral rank when bootstrap bases disagree."""

from __future__ import annotations

import torch

from fasd.profiling.stability import stability_adjusted_rank


def test_stability_cap_unstable_bases_produce_lower_rank():
    torch.manual_seed(0)
    C = 8
    # Two bases that agree on the first 2 directions but disagree
    # badly beyond that.
    Q1, _ = torch.linalg.qr(torch.randn(C, C))
    # Keep first two columns, permute the rest.
    Q2 = Q1.clone()
    perm = torch.randperm(C - 2) + 2
    Q2[:, 2:] = Q1[:, perm]
    k_stable, ang = stability_adjusted_rank([Q1, Q2], proposed_rank=6, angle_cap_deg=5.0)
    # The first 2 directions match exactly; the rest are permuted.
    assert k_stable <= 3


def test_stability_cap_perfect_bases():
    torch.manual_seed(1)
    Q, _ = torch.linalg.qr(torch.randn(6, 6))
    k_stable, ang = stability_adjusted_rank([Q, Q.clone()], proposed_rank=6, angle_cap_deg=5.0)
    assert k_stable == 6
    assert ang < 1.0


def test_stability_cap_single_basis():
    torch.manual_seed(2)
    Q, _ = torch.linalg.qr(torch.randn(4, 4))
    k, ang = stability_adjusted_rank([Q], proposed_rank=4)
    assert k == 4
