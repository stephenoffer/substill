"""Does truncating the residual stream on head boundaries matter?

Absorbed init with an identity-truncation basis keeps the first `k` residual
coordinates. GPT-2 lays its attention heads out contiguously along that axis, so if
`k` is a multiple of the teacher's head_dim (64), the student's heads *are* the
teacher's first `k/64` heads -- q, k, v and o intact, attention circuits transferred
whole. If it is not, every head is a fragment of a teacher head glued to a fragment of
the next, and no attention circuit survives.

The repo builds its students with `n_head` fixed at the teacher's 12, so `n_embd=324`
gives head_dim=27 and shatters all twelve heads. `width_pruner.py`'s own docstring says
"Retain attention heads"; the GPT-2 builder does not.

Evidence this matters, from `scripts/axis.py` (matched ~61M, n=3): the three rungs with
head_dim=64 land on a smooth curve (100.4 / 86.6 / 89.9) and the single rung with
head_dim=48 sits ~5 PPL off it (93.2), despite occupying an interior position on that
curve. This script tests the prediction directly at the repo's own operating point:
same parameter count, same compute, `n_embd` moved by 4 so the heads stay whole.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import torch

from scripts.analysis.bench import distill
from scripts.analysis.h2h import load
from substill.compression.seq_absorb import (
    _ffn_basis,
    absorb_gpt2,
    build_narrow_gpt2,
    eval_ppl,
    logit_metric,
    residual_basis,
    residual_second_moment,
)

VOCAB, POS = 50257, 1024


def block_params(k, inner):
    return 4 * k * k + 2 * k * inner + inner + 9 * k


def total_params(k, inner, L):
    return (VOCAB + POS) * k + L * block_params(k, inner) + 2 * k


def solve_inner(k, L, target):
    """FFN width that brings a (k, L) student to `target` parameters."""
    rest = target - (VOCAB + POS) * k - 2 * k - L * (4 * k * k + 9 * k)
    return max(1, round(rest / (L * (2 * k + 1))))


# (label, n_embd, n_head).  head_dim = n_embd / n_head.
ARMS = [
    ("shattered  (head_dim=27)", 324, 12),
    ("whole      (head_dim=64)", 320, 5),
    ("whole      (head_dim=64)", 384, 6),
    ("shattered  (head_dim=32)", 384, 12),
]

# Every width that keeps whole 64-dim teacher heads. `inner` absorbs the parameter
# difference, so these trade attention capacity against FFN capacity at fixed budget.
# Feasible whole-head widths at the 30.0M budget. 448 is excluded because its embedding
# table alone (23.0M) leaves no room for 12 blocks (`inner` collapses to 1); 192 is
# excluded because balancing the budget would need inner=3976 > the teacher's 3072, and
# a student cannot have a wider FFN than the teacher it absorbs from.
HEAD_SWEEP = [(f"whole x{k // 64} heads", k, k // 64) for k in (256, 320, 384)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-params", type=int, default=30_007_116)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--sweep", action="store_true",
                   help="sweep whole-head widths instead of the head-geometry control")
    p.add_argument("--output", default="runs/heads.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    t_params = sum(p.numel() for p in teacher.parameters())
    calib = train[: args.calib_batches]
    L = teacher.config.n_layer
    print(f"teacher PPL={t_ppl:.2f}  head_dim={teacher.config.n_embd // teacher.config.n_head}"
          f"  target={args.target_params:,} params\n", flush=True)

    S = residual_second_moment(teacher, calib, device=args.device)
    M = logit_metric(teacher).to(args.device)

    arms = HEAD_SWEEP if args.sweep else ARMS
    rows = []
    t_inner = teacher.config.n_inner or 4 * teacher.config.n_embd
    for label, k, nh in arms:
        inner = solve_inner(k, L, args.target_params)
        if inner > t_inner:
            raise ValueError(
                f"n_embd={k} needs inner={inner} > teacher's {t_inner} to hit the "
                f"parameter budget; absorbed init cannot widen the FFN.")
        V = residual_basis(S, k, method="identity", M=M).to(args.device)
        ffn = [_ffn_basis(teacher, i, calib, inner, args.device) for i in range(L)]
        for seed in args.seeds:
            args.seed = seed
            torch.manual_seed(seed)
            st = build_narrow_gpt2(teacher, k, inner, n_head=nh).to(args.device)
            absorb_gpt2(teacher, st, V, ffn)
            sp = sum(p.numel() for p in st.parameters())
            ip = eval_ppl(st, val, args.device)
            for p in st.parameters():
                p.requires_grad_(True)
            t0 = time.time()
            torch.manual_seed(seed)
            distill(teacher, st, train, args, args.device, kd="forward_kl",
                    feat=0.0, V=None, Mk=None, steps=args.steps)
            fp = eval_ppl(st, val, args.device)
            print(f"n_embd={k:<4} n_head={nh:<3} head_dim={k // nh:<3} seed={seed}  "
                  f"params={sp/1e6:.2f}M ({t_params/sp:.2f}x)  init={ip:>12,.0f}  "
                  f"final={fp:>8.2f}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({"label": label, "n_embd": k, "n_head": nh, "head_dim": k // nh,
                         "inner": inner, "params": sp, "seed": seed,
                         "init_ppl": ip, "final_ppl": fp})
            del st
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"teacher_ppl": t_ppl, "teacher_params": t_params, "args": vars(args),
         "rows": rows}, indent=2))

    print("\n" + "=" * 74)
    ppl_hdr = f"final PPL (n={len(args.seeds)})"
    print(f"{'n_embd':>7}{'n_head':>8}{'head_dim':>10}{'params':>10}{ppl_hdr:>24}")
    print("-" * 74)
    for _label, k, nh in arms:
        v = [r["final_ppl"] for r in rows if r["n_embd"] == k and r["n_head"] == nh]
        p = [r["params"] for r in rows if r["n_embd"] == k and r["n_head"] == nh][0]
        sd = stats.stdev(v) if len(v) > 1 else 0.0
        star = " <- heads whole" if k % 64 == 0 and k // nh == 64 else ""
        print(f"{k:>7}{nh:>8}{k // nh:>10}{p/1e6:>9.2f}M"
              f"{stats.mean(v):>17.2f} +/- {sd:<5.2f}{star}")
    print(f"\nteacher PPL={t_ppl:.2f}\nwrote {args.output}")


if __name__ == "__main__":
    main()
