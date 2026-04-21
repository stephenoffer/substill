"""Activation sparsity statistics — histogram, entropy, sparsity ratio."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SparsityStats:
    """Sparsity statistics for a single layer's activations."""
    sparsity_ratio: float           # Fraction of activations == 0
    activation_histogram: Tensor    # Normalized histogram (sums to 1)
    bin_edges: Tensor               # Histogram bin edges
    entropy: float                  # Shannon entropy of histogram
    mean_activation: float
    std_activation: float


class SparsityAnalyzer:
    """Compute sparsity statistics from activation samples."""

    def __init__(self, num_bins: int = 64):
        self.num_bins = num_bins

    def analyze(self, sparsity_ratio: float, activation_sample: Tensor) -> SparsityStats:
        """Compute sparsity stats from a subsample of activation values."""
        values = activation_sample.flatten().float()

        mean_val = values.mean().item()
        std_val = values.std().item()

        # Compute histogram of non-zero values
        nonzero_values = values[values != 0]
        if len(nonzero_values) < 2:
            hist = torch.zeros(self.num_bins)
            bin_edges = torch.linspace(0, 1, self.num_bins + 1)
            return SparsityStats(
                sparsity_ratio=sparsity_ratio,
                activation_histogram=hist,
                bin_edges=bin_edges,
                entropy=0.0,
                mean_activation=mean_val,
                std_activation=std_val,
            )

        vmin, vmax = nonzero_values.min().item(), nonzero_values.max().item()
        # Handle edge case where all values are identical
        if vmax - vmin < 1e-8:
            vmin = vmin - 0.5
            vmax = vmax + 0.5

        hist = torch.histc(nonzero_values, bins=self.num_bins, min=vmin, max=vmax)
        bin_edges = torch.linspace(vmin, vmax, self.num_bins + 1)

        # Normalize to probability distribution
        hist = hist / hist.sum().clamp(min=1e-10)

        # Shannon entropy
        entropy = self._shannon_entropy(hist)

        return SparsityStats(
            sparsity_ratio=sparsity_ratio,
            activation_histogram=hist,
            bin_edges=bin_edges,
            entropy=entropy,
            mean_activation=mean_val,
            std_activation=std_val,
        )

    @staticmethod
    def _shannon_entropy(prob: Tensor) -> float:
        """Compute Shannon entropy of a probability distribution."""
        # Filter out zeros to avoid log(0)
        p = prob[prob > 0]
        entropy = -(p * p.log()).sum().item()
        return entropy
