"""SVD decomposition of activation covariance to find principal activation subspace."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .sparsity_analysis import SparsityStats


@dataclass
class LayerProfile:
    """Complete activation profile for one layer."""
    name: str
    eigenvalues: Tensor             # Eigenvalues of activation covariance (descending)
    principal_components: Tensor    # Top-k eigenvectors, shape (C, k)
    effective_rank: int             # k where cumsum(eigenvalues) >= threshold * total
    total_channels: int
    compression_ratio: float        # effective_rank / total_channels
    sparsity_stats: SparsityStats


class SVDAnalyzer:
    """Analyze activation covariance matrices via eigendecomposition."""

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold

    def analyze(
        self,
        name: str,
        covariance: Tensor,
        sparsity_stats: SparsityStats,
    ) -> LayerProfile:
        """Eigendecompose the covariance and compute effective rank."""
        # Symmetric eigendecomposition (covariance is symmetric PSD)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        # Sort descending (eigh returns ascending)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = eigenvalues.clamp(min=0)

        effective_rank = self.compute_effective_rank(eigenvalues)
        total_channels = covariance.shape[0]

        return LayerProfile(
            name=name,
            eigenvalues=eigenvalues,
            principal_components=eigenvectors[:, :effective_rank],
            effective_rank=effective_rank,
            total_channels=total_channels,
            compression_ratio=effective_rank / total_channels,
            sparsity_stats=sparsity_stats,
        )

    def compute_effective_rank(self, eigenvalues: Tensor) -> int:
        """Find smallest k such that top-k eigenvalues capture >= threshold of total variance."""
        total = eigenvalues.sum()
        if total < 1e-10:
            return 1
        cumulative = torch.cumsum(eigenvalues, dim=0)
        ratio = cumulative / total
        # Find first index where ratio >= threshold
        mask = ratio >= self.variance_threshold
        if not mask.any():
            return len(eigenvalues)
        k = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
        return max(1, k)


def profiles_to_stage_widths(
    profiles: list[LayerProfile],
    min_width: int = 16,
    width_multiple: int = 8,
) -> list[int]:
    """Convert layer profiles to per-stage student widths.

    Groups profiles by stage (based on total_channels) and takes the max
    effective rank within each stage. Rounds up to width_multiple.
    """
    # Group by total_channels (stages have same channel count)
    stage_map: dict[int, list[int]] = {}
    for p in profiles:
        stage_map.setdefault(p.total_channels, []).append(p.effective_rank)

    widths = []
    for channels in sorted(stage_map.keys()):
        ranks = stage_map[channels]
        # Use max effective rank in the stage
        rank = max(ranks)
        # Round up to multiple
        width = max(min_width, ((rank + width_multiple - 1) // width_multiple) * width_multiple)
        widths.append(width)

    return widths


def save_profiles(profiles: list[LayerProfile], path: str) -> None:
    """Save profiles to disk."""
    data = {
        "profiles": [
            {
                "name": p.name,
                "eigenvalues": p.eigenvalues,
                "principal_components": p.principal_components,
                "effective_rank": p.effective_rank,
                "total_channels": p.total_channels,
                "compression_ratio": p.compression_ratio,
                "sparsity_stats": {
                    "sparsity_ratio": p.sparsity_stats.sparsity_ratio,
                    "activation_histogram": p.sparsity_stats.activation_histogram,
                    "bin_edges": p.sparsity_stats.bin_edges,
                    "entropy": p.sparsity_stats.entropy,
                    "mean_activation": p.sparsity_stats.mean_activation,
                    "std_activation": p.sparsity_stats.std_activation,
                },
            }
            for p in profiles
        ]
    }
    torch.save(data, path)


def load_profiles(path: str) -> list[LayerProfile]:
    """Load profiles from disk."""
    data = torch.load(path, weights_only=False)
    profiles = []
    for d in data["profiles"]:
        ss = d["sparsity_stats"]
        sparsity_stats = SparsityStats(
            sparsity_ratio=ss["sparsity_ratio"],
            activation_histogram=ss["activation_histogram"],
            bin_edges=ss["bin_edges"],
            entropy=ss["entropy"],
            mean_activation=ss["mean_activation"],
            std_activation=ss["std_activation"],
        )
        profiles.append(LayerProfile(
            name=d["name"],
            eigenvalues=d["eigenvalues"],
            principal_components=d["principal_components"],
            effective_rank=d["effective_rank"],
            total_channels=d["total_channels"],
            compression_ratio=d["compression_ratio"],
            sparsity_stats=sparsity_stats,
        ))
    return profiles
