#!/usr/bin/env python3
"""Compare baseline vs improved sweeps on the same model/dataset."""

import argparse
import json
import os

import matplotlib.pyplot as plt


def load(path: str) -> list[dict]:
    with open(path) as f:
        return sorted(json.load(f), key=lambda r: r["threshold"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/sweep/sweep_results.json")
    parser.add_argument("--improved", default="outputs/sweep_improved/sweep_results.json")
    parser.add_argument("--output", default="outputs/paper/baseline_vs_improved.png")
    args = parser.parse_args()

    base = load(args.baseline)
    imp = load(args.improved)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ASD baseline vs algorithmic improvements — ResNet50/CIFAR-10", fontsize=13)

    # Left: compression vs accuracy
    ax = axes[0]
    ax.plot([r["compression"] for r in base], [r["student_acc_best"] * 100 for r in base],
            "o-", color="#999999", linewidth=2, markersize=9, label="Baseline algorithm")
    ax.plot([r["compression"] for r in imp], [r["student_acc_best"] * 100 for r in imp],
            "s-", color="#2E7D32", linewidth=2, markersize=9, label="Improved (ours)")
    t_acc = base[0].get("teacher_acc")
    if t_acc is not None:
        ax.axhline(y=t_acc * 100, color="red", linestyle="--", alpha=0.7, label=f"Teacher {t_acc*100:.2f}%")

    for r in base:
        ax.annotate(f"τ={r['threshold']}", (r["compression"], r["student_acc_best"] * 100),
                    xytext=(5, -12), textcoords="offset points", fontsize=8, color="#555")
    for r in imp:
        ax.annotate(f"τ={r['threshold']}", (r["compression"], r["student_acc_best"] * 100),
                    xytext=(5, 6), textcoords="offset points", fontsize=8, color="#2E7D32")

    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    # Right: per-threshold gain
    ax = axes[1]
    thresholds = sorted(set(r["threshold"] for r in base) & set(r["threshold"] for r in imp))
    gains = []
    for t in thresholds:
        b_r = next((r for r in base if abs(r["threshold"] - t) < 1e-6), None)
        i_r = next((r for r in imp if abs(r["threshold"] - t) < 1e-6), None)
        if b_r and i_r:
            gains.append((t, (i_r["student_acc_best"] - b_r["student_acc_best"]) * 100,
                          i_r["compression"], b_r["compression"]))

    xs = range(len(gains))
    ys = [g[1] for g in gains]
    colors = ["#4CAF50" if g >= 0 else "#F44336" for g in ys]
    bars = ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"τ={g[0]}\n(imp={g[2]:.1f}×,\nbase={g[3]:.1f}×)" for g in gains], fontsize=9)
    ax.set_ylabel("Accuracy delta: improved − baseline (pp)")
    ax.set_title("Per-threshold accuracy gain from improvements")
    ax.grid(alpha=0.3, axis="y")
    for bar, g in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, g + (0.1 if g >= 0 else -0.3),
                f"{g:+.2f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")

    # Text summary
    print("\nBaseline vs Improved:")
    print(f"{'τ':>6} {'base_acc':>10} {'imp_acc':>10} {'Δ (pp)':>10} {'base_params':>12} {'imp_params':>12}")
    for t in thresholds:
        b_r = next((r for r in base if abs(r["threshold"] - t) < 1e-6), None)
        i_r = next((r for r in imp if abs(r["threshold"] - t) < 1e-6), None)
        if b_r and i_r:
            print(f"{t:>6.2f} {b_r['student_acc_best']*100:>9.2f}% {i_r['student_acc_best']*100:>9.2f}% "
                  f"{(i_r['student_acc_best']-b_r['student_acc_best'])*100:>+9.2f} "
                  f"{b_r['student_params']:>12,} {i_r['student_params']:>12,}")


if __name__ == "__main__":
    main()
