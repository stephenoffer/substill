"""Width or depth? The compression axis, at matched parameters and matched compute.

`substill/compression/width_pruner.py` encodes the Minitron doctrine verbatim: "Width-first.
Primary reduction comes from hidden_size." Every student in this repo is built that way.

`docs/init_findings.md` §4d makes the opposite prediction. What survives distillation is
not the teacher's function, its spectrum, or its scale -- it is *weight alignment*: each
student weight being a submatrix of the teacher's weight in the teacher's coordinate
system, so the student's layers compose the way the teacher's do. Permute the rows and
columns of each block matrix -- preserving every singular value and every entry -- and
final PPL goes 161 -> 301.

Narrowing the residual stream destroys alignment on every weight in the network.
Dropping whole layers destroys none: a copied layer is exact. So at a fixed parameter
budget, depth reduction should beat width reduction, and by a lot.

The ladder below holds parameters at ~61M (2x compression) and walks the axis from pure
depth reduction (d=768, 3 layers, zero truncation) to near-pure width reduction (d=512,
11 layers). Layers are taken evenly spaced across the teacher's depth, including the
first and last -- the standard choice, and the one DistilBERT uses.
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

# (n_embd, n_layer). Parameters land within 1.5% of each other; `axis.py --list` prints them.
LADDER = [(768, 3), (640, 6), (576, 8), (512, 11)]


def evenly_spaced(n_teacher: int, n_student: int) -> list[int]:
    """Keep the first and last teacher layers, spread the rest evenly."""
    if n_student >= n_teacher:
        return list(range(n_teacher))
    if n_student == 1:
        return [n_teacher - 1]
    step = (n_teacher - 1) / (n_student - 1)
    return [round(i * step) for i in range(n_student)]


def build(d, L, teacher, calib, args, device, *, absorbed=True):
    inner = 4 * d
    layer_map = evenly_spaced(teacher.config.n_layer, L)
    st = build_narrow_gpt2(teacher, d, inner, n_layer=L).to(device)
    if not absorbed:
        return st, layer_map
    S = residual_second_moment(teacher, calib, device=device)
    V = residual_basis(S, d, method="identity", M=logit_metric(teacher).to(device)).to(device)
    ffn = [_ffn_basis(teacher, tl, calib, inner, device) for tl in layer_map]
    absorb_gpt2(teacher, st, V, ffn, layer_map=layer_map)
    return st, layer_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--random-init", action="store_true",
                   help="also run each rung from a random init, to separate the axis "
                        "effect from the initialization effect")
    p.add_argument("--output", default="runs/axis.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    t_params = sum(p.numel() for p in teacher.parameters())
    calib = train[: args.calib_batches]
    print(f"teacher PPL={t_ppl:.2f}  params={t_params:,}  steps={args.steps}\n", flush=True)

    inits = [True] + ([False] if args.random_init else [])
    rows = []
    for d, L in LADDER:
        for absorbed in inits:
            for seed in args.seeds:
                args.seed = seed
                torch.manual_seed(seed)
                st, lmap = build(d, L, teacher, calib, args, args.device, absorbed=absorbed)
                sp = sum(p.numel() for p in st.parameters())
                ip = eval_ppl(st, val, args.device)
                if args.list:
                    print(f"d={d:<4} L={L:<3} params={sp:,} ({t_params/sp:.2f}x) "
                          f"layers={lmap}", flush=True)
                    break
                for p in st.parameters():
                    p.requires_grad_(True)
                t0 = time.time()
                torch.manual_seed(seed)
                distill(teacher, st, train, args, args.device, kd="forward_kl",
                        feat=0.0, V=None, Mk=None, steps=args.steps)
                fp = eval_ppl(st, val, args.device)
                tag = "absorbed" if absorbed else "random  "
                print(f"d={d:<4} L={L:<3} {tag} seed={seed}  params={sp/1e6:.1f}M  "
                      f"init={ip:>13,.1f}  final={fp:>8.2f}  ({time.time()-t0:.0f}s)",
                      flush=True)
                rows.append({"d": d, "L": L, "absorbed": absorbed, "seed": seed,
                             "params": sp, "init_ppl": ip, "final_ppl": fp,
                             "seconds": time.time() - t0})
                del st
                torch.cuda.empty_cache()
        if args.list:
            continue

    if args.list:
        return
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"teacher_ppl": t_ppl, "teacher_params": t_params, "args": vars(args),
         "rows": rows}, indent=2))

    print("\n" + "=" * 72)
    header = f"{'student':<16}{'params':>10}{'absorbed init':>22}"
    if args.random_init:
        header += f"{'random init':>20}"
    print(header)
    print("-" * 72)
    for d, L in LADDER:
        line = f"d={d}, L={L}".ljust(16)
        r0 = [x for x in rows if x["d"] == d and x["absorbed"]]
        line += f"{r0[0]['params']/1e6:>9.1f}M"
        sd = stats.stdev([x["final_ppl"] for x in r0]) if len(r0) > 1 else 0.0
        line += f"{stats.mean([x['final_ppl'] for x in r0]):>15.2f} +/- {sd:<5.2f}"
        if args.random_init:
            r1 = [x for x in rows if x["d"] == d and not x["absorbed"]]
            sd1 = stats.stdev([x["final_ppl"] for x in r1]) if len(r1) > 1 else 0.0
            line += f"{stats.mean([x['final_ppl'] for x in r1]):>13.2f} +/- {sd1:<5.2f}"
        print(line)
    print(f"\nteacher PPL={t_ppl:.2f}\nwrote {args.output}")


if __name__ == "__main__":
    main()
