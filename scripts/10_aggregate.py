#!/usr/bin/env python3
"""Aggregate all ASD benchmark results into paper-style tables and plots.

Scans outputs/bench_* and outputs/sweep* for results JSON files and produces:
  outputs/paper/results_master.json    — flat list of every (model, dataset, τ) point
  outputs/paper/results_table.md       — markdown table grouped by (model, dataset)
  outputs/paper/compression_vs_acc.png — scatter across all experiments
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def load_all(root: str) -> list[dict]:
    """Load every results.json or sweep_results.json under `root`."""
    patterns = [
        "bench_*/results.json",
        "sweep*/sweep_results.json",
        "dense_sweep*/results.json",
        "seeds_*/results.json",
        "llm/result_t*.json",
        "ablation_*/result.json",
    ]
    results = []
    for pat in patterns:
        for path in glob.glob(os.path.join(root, pat)):
            with open(path) as f:
                data = json.load(f)
            # sweep_results.json stores a list; bench stores a list; llm stores a dict
            if isinstance(data, dict):
                data = [data]
            for r in data:
                r.setdefault("source", path)
                r.setdefault("model", "resnet50")  # sweep_results default
                r.setdefault("dataset", "cifar10")
                results.append(r)
    return results


def make_table(results: list[dict]) -> str:
    """Group by (model, dataset), sort by compression, render markdown."""
    from collections import defaultdict
    grouped = defaultdict(list)
    ablations = []
    for r in results:
        if "variant" in r:
            ablations.append(r)
        else:
            grouped[(r["model"], r["dataset"])].append(r)

    lines = ["# ASD Results — paper-grade matrix\n"]

    for (model, dataset), rs in sorted(grouped.items()):
        lines.append(f"\n## {model} / {dataset}\n")
        has_acc = "student_acc_best" in rs[0]
        if has_acc:
            lines.append("| τ | compression | teacher | student params | student acc | drop (pp) |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            rs = sorted(rs, key=lambda r: r["compression"], reverse=True)
            for r in rs:
                tacc = r.get("teacher_acc", None)
                tacc_s = f"{tacc*100:.2f}%" if tacc is not None else "?"
                lines.append(
                    f"| {r['threshold']:.2f} | {r['compression']:.2f}× | {tacc_s} "
                    f"| {r['student_params']:,} | {r['student_acc_best']*100:.2f}% "
                    f"| {r.get('acc_drop_pp', 0.0):.2f} |"
                )
        else:
            # LLM table (perplexity-based)
            lines.append("| τ | compression | teacher ppl | student params | student ppl |")
            lines.append("|---|---:|---:|---:|---:|")
            rs = sorted(rs, key=lambda r: r["compression"], reverse=True)
            for r in rs:
                lines.append(
                    f"| {r['threshold']:.2f} | {r['compression']:.2f}× "
                    f"| {r.get('teacher_ppl', '?'):.2f} | {r['student_params']:,} "
                    f"| {r.get('student_ppl_best', '?'):.2f} |"
                )

    if ablations:
        lines.append("\n## Ablations (ResNet50 / CIFAR-10 at τ=0.95, 15 epochs)\n")
        lines.append("| Variant | compression | student params | student acc | drop (pp) |")
        lines.append("|---|---:|---:|---:|---:|")
        abl_sorted = sorted(ablations, key=lambda r: -r.get("student_acc_best", 0))
        for r in abl_sorted:
            lines.append(
                f"| {r['variant']} | {r['compression']:.2f}× "
                f"| {r['student_params']:,} | {r['student_acc_best']*100:.2f}% "
                f"| {r['acc_drop_pp']:.2f} |"
            )

    return "\n".join(lines) + "\n"


def plot_scatter(results: list[dict], save_path: str) -> None:
    """Scatter compression vs relative accuracy retention across all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Classification plot: compression vs accuracy drop
    colors = plt.cm.tab10.colors
    label_colors: dict[str, tuple] = {}

    for r in results:
        if "student_acc_best" not in r:
            continue  # skip LLM
        key = f"{r['model']}/{r['dataset']}"
        if key not in label_colors:
            label_colors[key] = colors[len(label_colors) % len(colors)]
        c = label_colors[key]
        ax1.scatter(r["compression"], r["acc_drop_pp"], color=c,
                    s=80, edgecolors="black", linewidths=0.5)

    for key, color in label_colors.items():
        ax1.scatter([], [], color=color, s=80, label=key, edgecolors="black", linewidths=0.5)

    ax1.set_xlabel("Compression ratio")
    ax1.set_ylabel("Accuracy drop (pp)")
    ax1.set_xscale("log")
    ax1.set_title("ASD compression vs accuracy drop (all classification experiments)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Compression vs accuracy
    for r in results:
        if "student_acc_best" not in r:
            continue
        key = f"{r['model']}/{r['dataset']}"
        c = label_colors.get(key, (0.5, 0.5, 0.5))
        ax2.scatter(r["compression"], r["student_acc_best"] * 100, color=c,
                    s=80, edgecolors="black", linewidths=0.5)
    ax2.set_xlabel("Compression ratio")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_xscale("log")
    ax2.set_title("ASD compression vs student accuracy")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/paper")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_all(args.root)
    print(f"Loaded {len(results)} result rows across {len(set((r['model'], r['dataset']) for r in results))} (model, dataset) combos")

    master = os.path.join(args.output_dir, "results_master.json")
    with open(master, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {master}")

    table = make_table(results)
    table_path = os.path.join(args.output_dir, "results_table.md")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Wrote {table_path}")

    plot_path = os.path.join(args.output_dir, "compression_vs_acc.png")
    try:
        plot_scatter(results, plot_path)
        print(f"Wrote {plot_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
