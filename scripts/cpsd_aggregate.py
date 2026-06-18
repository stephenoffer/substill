#!/usr/bin/env python3
"""Aggregate CPSD experiment result JSONs into a comparison table.

Reads the per-cell JSONs written by fsd_headline_experiment.py (one per
compression x seed), and tabulates final validation PPL per variant, averaged
across seeds, at each compression ratio — so cpsd_mt / cpsd_full can be compared
head-to-head against the F-ASD / FSD baselines.

Usage:
    # after `aws s3 cp ... ./results/` or `aws s3 sync`:
    python scripts/cpsd_aggregate.py results/*.json
    python scripts/cpsd_aggregate.py --csv summary.csv results/*.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

VARIANT_ORDER = ["r0_random", "r1_kd_random", "r2_fasd", "r3_fsd_kd",
                 "r3_fsd_kd_stiefel", "cpsd_mt", "cpsd_full"]
VARIANT_LABEL = {
    "r2_fasd": "F-ASD (absorbed+KD) [baseline]",
    "r3_fsd_kd_stiefel": "FSD (RR-Norm Q Stiefel) [baseline]",
    "cpsd_mt": "CPSD-MT (trained proj. factors) [NOVEL]",
    "cpsd_full": "CPSD-full (+ diff. rank) [NOVEL]",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="result JSON files")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    # (compression, variant) -> list of final_ppl across seeds
    cells = defaultdict(list)
    ratios = defaultdict(list)
    teacher_ppls = []
    for path in args.files:
        with open(path) as f:
            summary = json.load(f)
        C = summary.get("target_compression")
        teacher_ppls.append(summary.get("teacher_ppl"))
        for r in summary.get("results", []):
            if r.get("status") == "failed":
                continue
            v = r["variant"]
            cells[(C, v)].append(r["final_ppl"])
            ratios[(C, v)].append(r.get("compression_ratio", float("nan")))

    comps = sorted({c for (c, _v) in cells})
    tppl = [t for t in teacher_ppls if t]
    print(f"\nTeacher PPL: {sum(tppl)/len(tppl):.2f}" if tppl else "")
    rows = []
    for C in comps:
        print(f"\n=== target compression {C}x ===")
        print(f"{'variant':<42}{'final PPL (mean±std, n)':<26}{'ratio':>8}")
        present = [v for v in VARIANT_ORDER if (C, v) in cells]
        for v in present:
            ppls = cells[(C, v)]
            mean = sum(ppls) / len(ppls)
            std = (sum((p - mean) ** 2 for p in ppls) / len(ppls)) ** 0.5
            rr = ratios[(C, v)]
            rmean = sum(rr) / len(rr) if rr else float("nan")
            label = VARIANT_LABEL.get(v, v)
            print(f"{label:<42}{f'{mean:.2f} ± {std:.2f} (n={len(ppls)})':<26}{rmean:>7.2f}x")
            rows.append((C, v, mean, std, len(ppls), rmean))
        # Verdict: does the best NOVEL beat the best baseline at this C?
        base = [cells[(C, v)] for v in ("r2_fasd", "r3_fsd_kd_stiefel") if (C, v) in cells]
        novel = [cells[(C, v)] for v in ("cpsd_mt", "cpsd_full") if (C, v) in cells]
        if base and novel:
            best_base = min(sum(x) / len(x) for x in base)
            best_novel = min(sum(x) / len(x) for x in novel)
            delta = best_base - best_novel
            verdict = "CPSD WINS" if delta > 0 else "baseline wins"
            print(f"  -> best baseline {best_base:.2f} vs best CPSD {best_novel:.2f}  "
                  f"({verdict}, Δ={delta:+.2f} PPL)")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("compression,variant,final_ppl_mean,final_ppl_std,n_seeds,ratio\n")
            for C, v, mean, std, n, rmean in rows:
                f.write(f"{C},{v},{mean:.4f},{std:.4f},{n},{rmean:.4f}\n")
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
