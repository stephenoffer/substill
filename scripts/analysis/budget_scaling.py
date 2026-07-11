"""Does absorbed init's advantage survive a real training budget?

Every comparison in this repo -- and in docs/init_findings.md -- lives in the
undertrained regime: students land at 155-440 PPL against a teacher at 50.9. In that
regime an initialization that merely lets the optimizer descend faster looks exactly
like an initialization that reaches a better optimum. The two are different claims,
and only the second justifies "method A beats method B".

This traces final PPL against step count on one continuous run per arm. To make the
intermediate evaluations comparable to each other, the schedule is warmup-then-
*constant* LR rather than the cosine used elsewhere: under a cosine decay an eval at
step 2000 of a 20000-step run is not the same object as the endpoint of a 2000-step
run. The absolute numbers are therefore slightly worse than the cosine runs; the
*gap between arms at a given step* is the quantity of interest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.analysis.h2h import kd_loss, load
from substill.compression.seq_absorb import (
    _ffn_basis,
    absorb_gpt2,
    build_narrow_gpt2,
    eval_ppl,
    logit_metric,
    residual_basis,
    residual_second_moment,
)


def build(arm, teacher, calib, args, device):
    if arm == "random":
        torch.manual_seed(args.seed)
        return build_narrow_gpt2(teacher, args.k, args.inner).to(device)
    S = residual_second_moment(teacher, calib, device=device)
    V = residual_basis(S, args.k, method=arm, M=logit_metric(teacher).to(device)).to(device)
    ffn = [_ffn_basis(teacher, i, calib, args.inner, device)
           for i in range(teacher.config.n_layer)]
    st = build_narrow_gpt2(teacher, args.k, args.inner).to(device)
    absorb_gpt2(teacher, st, V, ffn)
    return st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--k", type=int, default=324)
    p.add_argument("--inner", type=int, default=1068)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--evals", nargs="+", type=int,
                   default=[500, 1000, 2000, 4000, 8000, 14000, 20000])
    p.add_argument("--arms", nargs="+", default=["random", "select", "identity"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--output", default="runs/budget_scaling.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    calib = train[: args.calib_batches]
    print(f"teacher PPL={t_ppl:.2f}  lr={args.lr:.0e} (constant after {args.warmup} "
          f"warmup steps)  max_steps={args.steps}\n", flush=True)
    evals = sorted(set(args.evals))

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            args.seed = seed
            torch.manual_seed(seed)
            st = build(arm, teacher, calib, args, args.device)
            for p in st.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(st.parameters(), lr=args.lr, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / args.warmup))
            g = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(train), generator=g)
            st.train()
            t0 = time.time()
            for step in range(1, max(evals) + 1):
                ids = train[int(order[step % len(order)])]["input_ids"].to(args.device)
                with torch.no_grad():
                    tl = teacher(input_ids=ids).logits
                sl = st(input_ids=ids).logits
                loss = kd_loss(sl[:, :-1], tl[:, :-1], "forward_kl")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
                opt.step()
                sched.step()
                if step in evals:
                    ppl = eval_ppl(st, val, args.device)
                    st.train()
                    print(f"{arm:<10} seed={seed} step={step:<6} PPL={ppl:>8.2f}  "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    rows.append({"arm": arm, "seed": seed, "step": step,
                                 "ppl": ppl, "seconds": time.time() - t0})
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.output).write_text(json.dumps(
                        {"teacher_ppl": t_ppl, "args": vars(args), "rows": rows}, indent=2))
            del st, opt
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    hdr = "".join(f"{s:>9}" for s in evals)
    print(f"{'arm':<12}{hdr}")
    print("-" * 70)
    for arm in args.arms:
        cells = []
        for s in evals:
            v = [r["ppl"] for r in rows if r["arm"] == arm and r["step"] == s]
            cells.append(f"{sum(v)/len(v):>9.1f}" if v else f"{'-':>9}")
        print(f"{arm:<12}{''.join(cells)}")
    print(f"\nteacher PPL={t_ppl:.2f}\nwrote {args.output}")


if __name__ == "__main__":
    main()
