"""Distillation-driven differentiable rank (CPSD Phase 2-DDR).

Replaces the frozen per-edge rank (behavioral-rank threshold, Fisher knapsack)
with a *soft, differentiable* column gate trained against the KD loss under a
global parameter budget. De-risked in ``runs/derisk/optim_derisk.py``: the gate
is stable (finite grads, no Taylor stabilization) and selective (keeps the
high-importance columns, 100% top-overlap under budget).

Novelty note: differentiable rank alone is anticipated (Dobi-SVD 2502.02723,
LLRC 2512.13733), but those optimize a *reconstruction/perplexity* objective.
Here the gate is optimized against the *distillation* loss and shares the
circuit-preserving factors (CPI) and per-expert edges (MoE) — the conjunction
is the contribution (see papers/gap_analysis.md).

Usage::

    gate = DifferentiableRankGate(num_columns=k)      # one per compressible edge
    z = gate(latent)                                  # soft-gated latent (..., k)
    # ... build student forward, compute KD loss ...
    loss = kd + controller.budget_penalty()           # global param budget
    # after training:
    rank = gate.harden()                              # integer rank for the builder

Bases are assumed ordered by descending importance (eigenvalue), so a hardened
gate yields a contiguous top-r prefix in practice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DifferentiableRankGate(nn.Module):
    """Soft per-column gate ``g = sigmoid(alpha / T)`` applied to an ordered latent.

    Parameters
    ----------
    num_columns : int
        Latent width (the maximum / pre-truncation rank of the edge).
    init_open : bool
        If True, start with all gates ≈ open (logit +4); else near 0.5 (logit 0).
    temperature : float
        Sigmoid temperature; anneal toward 0 to sharpen the gate to {0, 1}.
    monotone : bool
        If True, enforce a non-increasing gate (a true prefix rank) by gating on
        the reverse-cumulative product of per-column keep probabilities.
    """

    def __init__(
        self,
        num_columns: int,
        *,
        init_open: bool = True,
        temperature: float = 1.0,
        monotone: bool = False,
    ):
        super().__init__()
        self.num_columns = int(num_columns)
        self.temperature = float(temperature)
        self.monotone = bool(monotone)
        init = 4.0 if init_open else 0.0
        self.alpha = nn.Parameter(torch.full((self.num_columns,), init))

    def gate(self) -> Tensor:
        g = torch.sigmoid(self.alpha / self.temperature)
        if self.monotone:
            # Non-increasing prefix: keep prob of column i is product of
            # per-step "continue" probs up to i. cumprod over ordered columns.
            g = torch.cumprod(g, dim=0)
        return g

    def forward(self, z: Tensor) -> Tensor:
        # z: (..., num_columns). Broadcast-multiply by the soft gate.
        return z * self.gate()

    def expected_rank(self) -> Tensor:
        """Differentiable expected number of open columns (= soft rank)."""
        return self.gate().sum()

    @torch.no_grad()
    def harden(self, threshold: float = 0.5, min_rank: int = 1) -> int:
        """Integer rank = count of open columns (≥ threshold), clamped to ≥ min_rank."""
        return int(max(min_rank, (self.gate() >= threshold).sum().item()))


class RankBudgetController:
    """Holds gates + per-column parameter costs; computes the global budget penalty.

    ``cost[e]`` is a length-``k_e`` tensor of the parameter cost of including each
    column of edge ``e`` (e.g. ``d_in + d_out`` for a low-rank factor column).
    The penalty is a one-sided hinge on total expected params vs the budget
    ``target_params``; ``lam`` weights it against the KD loss.
    """

    def __init__(
        self,
        gates: dict[str, DifferentiableRankGate],
        costs: dict[str, Tensor],
        *,
        target_params: float,
        lam: float = 1.0,
    ):
        if set(gates) != set(costs):
            raise ValueError("gates and costs must have identical edge keys")
        self.gates = gates
        self.costs = costs
        self.target_params = float(target_params)
        self.lam = float(lam)

    def expected_params(self) -> Tensor:
        total = None
        for name, gate in self.gates.items():
            c = self.costs[name].to(gate.alpha.device, gate.alpha.dtype)
            term = (gate.gate() * c).sum()
            total = term if total is None else total + term
        return total if total is not None else torch.zeros(())

    def budget_penalty(self) -> Tensor:
        """One-sided hinge: lam * relu(expected_params - target) / target."""
        over = torch.relu(self.expected_params() - self.target_params)
        return self.lam * over / max(self.target_params, 1.0)

    def anneal_temperature(self, factor: float, floor: float = 0.1) -> None:
        for gate in self.gates.values():
            gate.temperature = max(floor, gate.temperature * factor)

    @torch.no_grad()
    def harden(self, threshold: float = 0.5, min_rank: int = 1) -> dict[str, int]:
        """Final integer rank-map for the builder."""
        return {name: gate.harden(threshold, min_rank) for name, gate in self.gates.items()}


__all__ = ["DifferentiableRankGate", "RankBudgetController"]
