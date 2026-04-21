"""Loss weight warmup scheduler for the sparsity loss component."""

from __future__ import annotations


class LossWeightScheduler:
    """Linear warmup for the sparsity loss weight (gamma).

    Sparsity matching is meaningless when student activations are random
    at initialization, so we ramp gamma from 0 to 1 over warmup_epochs.
    """

    def __init__(self, warmup_epochs: int = 10):
        self.warmup_epochs = warmup_epochs

    def get_gamma_scale(self, epoch: int) -> float:
        """Return the gamma multiplier for the current epoch (0.0 to 1.0)."""
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)
