"""Loss-weight warmup schedulers (γ for sparsity, β for subspace)."""

from __future__ import annotations


class LossWeightScheduler:
    """Linear warmup for the sparsity-loss weight (gamma).

    Sparsity matching is meaningless when student activations are random
    at initialization, so we ramp gamma from 0 to 1 over `warmup_epochs`.
    """

    def __init__(self, warmup_epochs: int = 10):
        self.warmup_epochs = warmup_epochs

    def get_gamma_scale(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)


class BetaWarmupScheduler:
    """Linear warmup for the subspace-loss weight (beta).

    At epoch 0, student features are random and subspace MSE against teacher
    projections produces huge, uninformative gradients that destabilize early
    training. Warming β from 0 → target over the first few epochs lets the
    student pick up task CE + logit KD first, then gradually adopt feature
    matching as its representations stabilize.
    """

    def __init__(self, warmup_epochs: int = 3, initial_scale: float = 0.1):
        self.warmup_epochs = warmup_epochs
        self.initial_scale = initial_scale

    def get_beta_scale(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        if epoch >= self.warmup_epochs:
            return 1.0
        t = epoch / self.warmup_epochs
        return self.initial_scale + (1.0 - self.initial_scale) * t
