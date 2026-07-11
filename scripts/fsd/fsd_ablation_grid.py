"""Ablation grid driver for the FSD mechanisms.

Runs every combination of mechanisms on/off at a fixed compression and budget,
emits a single CSV summarising the contribution of each one.

Mechanisms (each toggle is one column in the ablation grid):
  - --use-rr-norm           rotation-equivariant normalization
  - --use-fisher-score      Fisher-weighted scoring
  - --use-exact-allocator   greedy q/cost knapsack allocator
  - --use-sparse-block      block-diagonal correction
  - --use-stiefel-optim     trainable Stiefel bases
  - --use-adaptive-skew-kl  adaptive entropy-gap skew-KL
  - --use-plateau-trigger   plateau-driven on-policy ramp

Default is to run only the *cumulative* on/off ladder (each pillar adds to the
previous), not the full 2^7 cross-product. Use ``--full-grid`` for the latter.

Usage::

    python scripts/fsd_ablation_grid.py \
        --teacher gpt2-medium \
        --corpus wikitext \
        --tokens-per-rung 100_000_000 \
        --target-params 1.2e9 \
        --output ablation/results.csv

For Llama-3.2 ablations, swap teacher and bump tokens-per-rung. Each cell takes
~3-7 days on H100×4 for Llama-3.2-3B at 10B tokens, so the cumulative ladder
(7 cells) at 1 seed = ~3-5 weeks. Use a smaller token budget for ablation
(2-5B) and full budget only for the headline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

# Cumulative ladder: each row turns on one more pillar than the previous row.
# This isolates each pillar's contribution as you stack them.
CUMULATIVE_LADDER = [
    # name             flags
    ("baseline",       []),
    ("+rr_norm",       ["--use-rr-norm"]),
    ("+fisher",        ["--use-rr-norm", "--use-fisher-score"]),
    ("+allocator",     ["--use-rr-norm", "--use-fisher-score", "--use-exact-allocator"]),
    ("+sparse",        [
        "--use-rr-norm", "--use-fisher-score", "--use-exact-allocator",
        "--use-sparse-block",
    ]),
    ("+stiefel",       [
        "--use-rr-norm", "--use-fisher-score", "--use-exact-allocator",
        "--use-sparse-block", "--use-stiefel-optim",
    ]),
    ("+adaptive_skewkl", [
        "--use-rr-norm", "--use-fisher-score", "--use-exact-allocator",
        "--use-sparse-block", "--use-stiefel-optim", "--use-adaptive-skew-kl",
    ]),
    ("full_fsd",       [
        "--use-rr-norm", "--use-fisher-score", "--use-exact-allocator",
        "--use-sparse-block", "--use-stiefel-optim", "--use-adaptive-skew-kl",
        "--use-plateau-trigger",
    ]),
]

ALL_PILLAR_FLAGS = [
    "--use-rr-norm",
    "--use-fisher-score",
    "--use-exact-allocator",
    "--use-sparse-block",
    "--use-stiefel-optim",
    "--use-adaptive-skew-kl",
    "--use-plateau-trigger",
]


def run_cell(
    cell_name: str,
    flags: list[str],
    args,
) -> dict:
    """Run one ablation cell, then run eval. Returns metrics dict."""
    out_dir = Path(args.output_dir) / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_train = [
        sys.executable, "scripts/distill_llama32_fsd.py",
        "--teacher", args.teacher,
        "--corpus", args.corpus,
        "--tokens-per-rung", str(args.tokens_per_rung),
        "--student-target-params", str(args.target_params),
        "--seq-len", str(args.seq_len),
        "--per-gpu-batch", str(args.per_gpu_batch),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--output-dir", str(out_dir),
    ] + flags

    print(f"\n[ablation] === Cell: {cell_name} ===")
    print(f"[ablation] flags: {flags}")
    if args.dry_run:
        print(f"[ablation] (dry-run) would execute: {' '.join(cmd_train)}")
        return {"cell": cell_name, "status": "dry_run"}

    rc = subprocess.call(cmd_train)
    if rc != 0:
        return {"cell": cell_name, "status": "train_failed", "returncode": rc}

    # Run eval.
    eval_path = out_dir / "eval.json"
    cmd_eval = [
        sys.executable, "scripts/eval_harness.py",
        "--student-dir", str(out_dir),
        "--tokenizer", args.teacher,
        "--output", str(eval_path),
    ]
    rc = subprocess.call(cmd_eval)
    if rc != 0 or not eval_path.exists():
        return {"cell": cell_name, "status": "eval_failed", "returncode": rc}

    with open(eval_path) as f:
        results = json.load(f)
    if results:
        flat = {k: v for k, v in results[0].items() if isinstance(v, (int, float))}
        flat["cell"] = cell_name
        flat["status"] = "ok"
        return flat
    return {"cell": cell_name, "status": "empty_eval"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--corpus", type=str, default="slimpajama")
    p.add_argument("--tokens-per-rung", type=int, default=2_000_000_000)
    p.add_argument("--target-params", type=float, default=1.2e9)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--per-gpu-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="ablation/")
    p.add_argument("--full-grid", action="store_true",
                   help="Run all 2^7 combinations (expensive). Default: cumulative ladder only.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--csv", type=str, default="ablation/summary.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.full_grid:
        cells = []
        for r in range(len(ALL_PILLAR_FLAGS) + 1):
            for combo in itertools.combinations(ALL_PILLAR_FLAGS, r):
                name = "+".join(c.replace("--use-", "") for c in combo) if combo else "baseline"
                cells.append((name, list(combo)))
    else:
        cells = CUMULATIVE_LADDER

    rows = []
    for cell_name, flags in cells:
        rows.append(run_cell(cell_name, flags, args))

    # Write CSV summary.
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n[ablation] summary written to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
