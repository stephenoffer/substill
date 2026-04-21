"""Visualization utilities for ASD — SVD spectra, compression ratios, training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..profiling.svd_analysis import LayerProfile


def plot_svd_spectrum(profiles: list[LayerProfile], save_path: str | None = None) -> None:
    """Plot eigenvalue spectrum for each layer, showing effective rank cutoff."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SVD Spectrum per Stage (Activation Covariance Eigenvalues)", fontsize=14)

    # Group by stage
    stage_map: dict[int, list[LayerProfile]] = {}
    for p in profiles:
        stage_map.setdefault(p.total_channels, []).append(p)

    for ax, (channels, stage_profiles) in zip(axes.flat, sorted(stage_map.items())):
        for p in stage_profiles:
            sv = p.eigenvalues.cpu().numpy()
            cumulative = sv.cumsum() / sv.sum()
            ax.semilogy(sv, alpha=0.7, label=p.name)
            ax.axvline(x=p.effective_rank, color="red", linestyle="--", alpha=0.3)

        ax.set_title(f"Stage (C={channels})")
        ax.set_xlabel("Component index")
        ax.set_ylabel("Eigenvalue (log scale)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SVD spectrum plot to {save_path}")
    plt.close()


def plot_compression_ratios(profiles: list[LayerProfile], save_path: str | None = None) -> None:
    """Bar chart of compression ratio per layer."""
    names = [p.name for p in profiles]
    ratios = [p.compression_ratio for p in profiles]
    sparsity = [p.sparsity_stats.sparsity_ratio for p in profiles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Compression ratios
    colors = ["#2196F3" if r < 0.3 else "#4CAF50" if r < 0.5 else "#FF9800" for r in ratios]
    ax1.barh(names, ratios, color=colors)
    ax1.set_xlabel("Effective Rank / Total Channels")
    ax1.set_title("Compression Ratio per Layer")
    ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax1.legend()

    # Sparsity ratios
    ax2.barh(names, sparsity, color="#9C27B0")
    ax2.set_xlabel("Fraction of Zero Activations")
    ax2.set_title("Activation Sparsity per Layer")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved compression ratios plot to {save_path}")
    plt.close()


def plot_training_curves(history_path: str, save_path: str | None = None) -> None:
    """Plot training loss components and test accuracy over epochs."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = [r["epoch"] for r in history]
    train_total = [r["train_total"] for r in history]
    train_task = [r["train_task"] for r in history]
    train_subspace = [r["train_subspace"] for r in history]
    train_sparsity = [r["train_sparsity"] for r in history]
    eval_acc = [r["eval_accuracy"] * 100 for r in history]
    lr = [r["lr"] for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ASD Training Curves", fontsize=14)

    # Total loss
    axes[0, 0].plot(epochs, train_total, label="Total", color="black", linewidth=2)
    axes[0, 0].plot(epochs, train_task, label="Task (CE)", alpha=0.7)
    axes[0, 0].plot(epochs, train_subspace, label="Subspace", alpha=0.7)
    axes[0, 0].plot(epochs, train_sparsity, label="Sparsity", alpha=0.7)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test accuracy
    axes[0, 1].plot(epochs, eval_acc, color="#4CAF50", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(epochs, lr, color="#FF5722")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].grid(True, alpha=0.3)

    # Loss components stacked
    axes[1, 1].stackplot(
        epochs, train_task, train_subspace, train_sparsity,
        labels=["Task", "Subspace", "Sparsity"],
        alpha=0.7,
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Loss Components (Stacked)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    plt.close()
