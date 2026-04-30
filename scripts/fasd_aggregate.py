#!/usr/bin/env python3
"""Aggregate v11 ablation results into a markdown summary.

Lists ``$ANYSCALE_ARTIFACT_STORAGE/fasd/<TAG>/results/*.json``, parses each
result, and prints a grouped table (by compression target, then rung).

Usage:
    python scripts/fasd_aggregate.py v11-pra-apr29
    python scripts/fasd_aggregate.py v11-pra-apr29 --markdown > report.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict


def _list_s3_keys(prefix: str) -> list[str]:
    r = subprocess.run(["aws", "s3", "ls", prefix], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".json"):
            out.append(parts[-1])
    return out


def _read_s3(path: str) -> str:
    r = subprocess.run(["aws", "s3", "cp", "--quiet", path, "-"],
                       capture_output=True, text=True, check=True)
    return r.stdout


def _read_local(path: str) -> str:
    with open(path) as f:
        return f.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--base", default=os.environ.get("ANYSCALE_ARTIFACT_STORAGE", ""))
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    base = args.base.rstrip("/")
    if not base:
        print("ANYSCALE_ARTIFACT_STORAGE not set; pass --base", file=sys.stderr)
        sys.exit(1)

    is_s3 = base.startswith("s3://")
    results_dir = f"{base}/fasd/{args.tag}/results/"

    if is_s3:
        names = _list_s3_keys(results_dir)
        loader = lambda n: _read_s3(f"{results_dir}{n}")
    else:
        if not os.path.isdir(results_dir):
            print(f"no results dir at {results_dir}", file=sys.stderr)
            sys.exit(1)
        names = [n for n in sorted(os.listdir(results_dir)) if n.endswith(".json")]
        loader = lambda n: _read_local(os.path.join(results_dir, n))

    if not names:
        print(f"no result JSONs at {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Group by target_compression.
    groups: dict[str, list[dict]] = defaultdict(list)
    for n in names:
        try:
            r = json.loads(loader(n))
        except Exception as e:
            print(f"skipping {n}: {e}", file=sys.stderr)
            continue
        comp = r.get("target_compression") or r.get("compression_ratio")
        key = f"{float(comp):.1f}x" if comp is not None else "—"
        groups[key].append(r)

    print(f"# F-ASD aggregate — tag `{args.tag}`")
    print()
    print(f"Loaded {len(names)} results from {results_dir}")
    print()

    for comp_key in sorted(groups.keys()):
        rs = groups[comp_key]
        print(f"## Target compression: {comp_key}")
        print()
        print("| Rung | Final PPL | Teacher PPL | Actual cmp | Init PPL | KL fwd | KL rev | Train (s) | PRA every |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(rs, key=lambda x: x.get("rung", "")):
            def fmt(k, spec=".2f", d="—"):
                v = r.get(k)
                try:
                    return format(v, spec)
                except (TypeError, ValueError):
                    return d
            pra = r.get("reabsorb_every_steps") or 0
            print(
                f"| {r.get('rung','?'):<22s} | "
                f"{fmt('final_student_ppl')} | "
                f"{fmt('teacher_ppl')} | "
                f"{fmt('compression_ratio', '.2f')}x | "
                f"{fmt('initial_student_ppl', '.2e')} | "
                f"{fmt('val_kl_forward', '.3f')} | "
                f"{fmt('val_kl_reverse', '.3f')} | "
                f"{fmt('train_time_s', '.0f')} | "
                f"{pra if pra else '—':>3} |"
            )
        print()


if __name__ == "__main__":
    main()
