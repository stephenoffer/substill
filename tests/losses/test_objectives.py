"""Correctness of the KD divergences (scripts/objectives.py).

`scripts/bench.py` shipped a `skew_kl` whose mixture weights were reversed, bounding the
loss by log(1/0.9) and leaving it with ~1.6% of forward-KL's gradient signal. These tests
pin each divergence to its definition so that cannot recur silently.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from scripts.analysis.objectives import divergence


@pytest.fixture
def dists():
    torch.manual_seed(0)
    return torch.randn(8, 32) * 2, torch.randn(8, 32) * 2  # student, teacher logits


def test_identical_distributions_give_zero(dists):
    s, _ = dists
    for kind in ("forward_kl", "reverse_kl", "skl", "srkl", "jsd"):
        assert divergence(s, s.clone(), kind) == pytest.approx(0.0, abs=1e-5), kind


def test_all_divergences_are_nonnegative(dists):
    s, t = dists
    for kind in ("forward_kl", "reverse_kl", "skl", "srkl", "jsd"):
        assert divergence(s, t, kind) >= -1e-6, kind


def test_forward_and_reverse_kl_are_asymmetric(dists):
    s, t = dists
    assert not torch.isclose(divergence(s, t, "forward_kl"),
                             divergence(s, t, "reverse_kl"), atol=1e-3)


def test_forward_kl_matches_manual(dists):
    s, t = dists
    ls, lt = F.log_softmax(s, -1), F.log_softmax(t, -1)
    manual = (lt.exp() * (lt - ls)).sum(-1).mean()
    assert divergence(s, t, "forward_kl") == pytest.approx(float(manual), rel=1e-5)


def test_skew_kl_mixes_toward_the_student_not_the_teacher(dists):
    """DistiLLM: SKL_a(p||q) = KL(p || a*p + (1-a)*q), a=0.1. The mixture is dominated by
    the *student*, so the bound is log(1/a)=2.30. Reversing the weights bounds it at
    log(1/(1-a))=0.105 -- the bug this test exists to prevent."""
    s, t = dists
    a = 0.1
    val = float(divergence(s, t, "skl", skew=a))
    assert val <= math.log(1 / a) + 1e-4
    # and it must be far above the value the reversed mixture would give
    ls, lt = F.log_softmax(s, -1), F.log_softmax(t, -1)
    reversed_mix = torch.logsumexp(
        torch.stack([lt + math.log(1 - a), ls + math.log(a)]), 0)
    reversed_val = float(F.kl_div(reversed_mix, lt, reduction="batchmean", log_target=True))
    assert reversed_val <= math.log(1 / (1 - a)) + 1e-4
    assert val > 5 * reversed_val, (val, reversed_val)


def test_skew_kl_approaches_forward_kl_as_skew_goes_to_zero(dists):
    """SKL_a(p||q) = KL(p || a*p + (1-a)*q). As a -> 0 the mixture becomes q, so the
    divergence becomes forward KL. As a -> 1 the mixture becomes p and it vanishes.
    DistiLLM's a=0.1 therefore sits just off forward KL, bounded by log(1/a)."""
    s, t = dists
    fwd = float(divergence(s, t, "forward_kl"))
    assert float(divergence(s, t, "skl", skew=1e-4)) == pytest.approx(fwd, rel=0.05)
    assert float(divergence(s, t, "skl", skew=0.999)) < 0.01 * fwd


def test_jsd_is_symmetric_at_beta_half(dists):
    s, t = dists
    assert divergence(s, t, "jsd", beta=0.5) == pytest.approx(
        float(divergence(t, s, "jsd", beta=0.5)), rel=1e-5)


def test_jsd_is_bounded_by_log_two(dists):
    s, t = dists
    assert float(divergence(s, t, "jsd", beta=0.5)) <= math.log(2) + 1e-4
