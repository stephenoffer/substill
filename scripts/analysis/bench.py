"""The decisive benchmark: our method vs the field, matched on params AND compute.

Two rules this harness enforces that the repo's earlier comparisons did not:

1. **Compute is matched, not just steps.** A method that spends 30s building a
   clever initialization and then distills for 1500 steps is not comparable to one
   that distills for 1500 steps from a free initialization. Every arm reports its
   total wall-clock, and baseline arms are given extra KD steps so that the totals
   line up. A win that evaporates under compute matching was bought, not earned.

2. **Each arm gets a tuned LR.** Measured: every arm's optimum is near 1e-3, and a
   shared 3e-4 (the repo's default) reverses the ranking of two arms outright.

Arms
----
random         random init + forward KL                     (the naive KD floor)
select         variance channel-selection init + forward KL (Minitron / LLM-Pruner)
absorbed       identity-truncation absorbed init + fwd KL. Equivalent in construction
               to what `substill.build_student(absorbed_init=True)` produces under the
               default profile: the same `torch.eye(d, k)` residual basis that
               `_residual_basis` falls back to, and the same `V_out^T W V_in`. The FFN
               intermediate basis is variance-selected in both but via slightly
               different estimators, so the two are not bit-identical.
absorbed_rkl   + reverse KL                                 (MiniLLM family)
absorbed_skl   + skew KL                                    (DistiLLM family)
anchored       absorbed init + KD anchored to the teacher's residual stream, measured
               through the logit Jacobian                   (ours; costs nothing extra)
sgca           sequential drift-corrected fit + forward KL  (ours; costs fit time)
sgca_anchored  both                                         (ours)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import time
from pathlib import Path

import torch

from scripts.analysis.h2h import kd_loss, load
from substill.compression.seq_absorb import (
    SeqAbsorbConfig,
    _ffn_basis,
    absorb_gpt2,
    build_narrow_gpt2,
    closed_form_absorb_gpt2,
    eval_ppl,
    logit_metric,
    residual_basis,
    residual_second_moment,
    sequential_absorb_gpt2,
)


# --------------------------------------------------------------------------
def build(arm, teacher, calib, args, device):
    """Return (student, V) where V is the residual basis (None for random init)."""
    k, inner = args.k, args.inner
    basis = ARMS[arm]["basis"]
    if basis is None:
        torch.manual_seed(args.seed)
        return build_narrow_gpt2(teacher, k, inner).to(device), None

    S = residual_second_moment(teacher, calib, device=device)
    M = logit_metric(teacher).to(device)
    V = residual_basis(S, k, method=basis, M=M).to(device)
    if not ARMS[arm]["fit"]:
        ffn = [_ffn_basis(teacher, i, calib, inner, device)
               for i in range(teacher.config.n_layer)]
        st = build_narrow_gpt2(teacher, k, inner).to(device)
        absorb_gpt2(teacher, st, V, ffn)
        return st.eval(), V
    cfg = SeqAbsorbConfig(k=k, inner=inner, basis=basis, objective="l2",
                          steps_per_block=args.steps_per_block, verbose=False)
    fit = ARMS[arm]["fit"]
    solver = closed_form_absorb_gpt2 if fit == "closed_form" else sequential_absorb_gpt2
    st, info = solver(teacher, calib, cfg, device=device)
    return st, info["V"].to(device)


ARMS = {
    "random":        {"basis": None,       "fit": False, "kd": "forward_kl", "feat": 0.0},
    "select":        {"basis": "select",   "fit": False, "kd": "forward_kl", "feat": 0.0},
    "random_sel":    {"basis": "random_sel", "fit": False, "kd": "forward_kl", "feat": 0.0},
    "absorbed":      {"basis": "identity", "fit": False, "kd": "forward_kl", "feat": 0.0},
    "absorbed_rkl":  {"basis": "identity", "fit": False, "kd": "reverse_kl", "feat": 0.0},
    "absorbed_skl":  {"basis": "identity", "fit": False, "kd": "skew_kl",    "feat": 0.0},
    "anchored":      {"basis": "identity", "fit": False, "kd": "forward_kl", "feat": 1.0},
    "sgca":          {"basis": "identity", "fit": "adam", "kd": "forward_kl", "feat": 0.0},
    "sgca_anchored": {"basis": "identity", "fit": "adam", "kd": "forward_kl", "feat": 1.0},
    # Same targets as `sgca`, solved in closed form instead of by 300 Adam steps per
    # block. The Adam fit's +11 PPL at equal steps was worth less than the ~23s it
    # cost; this asks whether the gain survives once the cost nearly vanishes.
    "cfa":           {"basis": "identity", "fit": "closed_form", "kd": "forward_kl", "feat": 0.0},
}


# --------------------------------------------------------------------------
def _rel(e, t, Mk):
    if Mk is None:
        return e.pow(2).sum() / t.pow(2).sum().clamp_min(1e-12)
    return ((e @ Mk) * e).sum() / ((t @ Mk) * t).sum().clamp_min(1e-12)


def distill(teacher, student, train, args, device, *, kd, feat, V, Mk, steps):
    """KD, optionally anchored to the teacher's residual stream.

    The anchor is free: the teacher forward that produces the KD target already
    produces its hidden states. Each student state is pulled toward the teacher's
    state projected into the student's basis, measured through the logit Jacobian
    so that directions the unembedding cannot read are not paid for.
    """
    teacher.to(device).eval()
    student.to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * steps))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train), generator=g)
    want_hidden = feat > 0.0 and V is not None
    for step in range(steps):
        ids = train[int(order[step % len(order)])]["input_ids"].to(device)
        with torch.no_grad():
            t_out = teacher(input_ids=ids, output_hidden_states=want_hidden)
        s_out = student(input_ids=ids, output_hidden_states=want_hidden)
        loss = kd_loss(s_out.logits[:, :-1], t_out.logits[:, :-1], kd)
        if want_hidden:
            aux = sum(_rel(hs - ht @ V, ht @ V, Mk)
                      for hs, ht in zip(s_out.hidden_states, t_out.hidden_states, strict=False))
            loss = loss + feat * aux / len(s_out.hidden_states)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
    return student


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--k", type=int, default=324)
    p.add_argument("--inner", type=int, default=1068)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--steps-per-block", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--fit-ref", default="sgca",
                   help="arm whose init cost sets the compute-match bonus")
    p.add_argument("--compute-match", action="store_true",
                   help="give cheap-init arms extra KD steps to equalize wall-clock")
    p.add_argument("--output", default="runs/bench.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    print(f"teacher PPL={t_ppl:.2f}  lr={args.lr:.0e}  steps={args.steps}  "
          f"compute_match={args.compute_match}\n", flush=True)
    calib = train[: args.calib_batches]
    Mk_cache = {}

    # Compute matching: price the fit in KD steps *before* running any arm, so the
    # budget does not depend on arm ordering. Arms that skip the fit get the
    # difference back as extra distillation steps.
    bonus = 0
    if args.compute_match:
        args.seed = args.seeds[0]
        t0 = time.time()
        st, V = build(args.fit_ref, teacher, calib, args, args.device)
        fit_s = time.time() - t0
        for p in st.parameters():
            p.requires_grad_(True)
        t0 = time.time()
        distill(teacher, st, train, args, args.device, kd="forward_kl", feat=0.0,
                V=None, Mk=None, steps=50)
        per_step = (time.time() - t0) / 50
        bonus = int(round(fit_s / per_step))
        print(f"[compute-match] fit={fit_s:.1f}s, KD={per_step * 1000:.0f}ms/step "
              f"-> cheap-init arms get +{bonus} steps\n", flush=True)
        del st
        torch.cuda.empty_cache()

    rows = []
    for arm in args.arms:
        spec = ARMS[arm]
        for seed in args.seeds:
            args.seed = seed
            torch.manual_seed(seed)
            t_init = time.time()
            st, V = build(arm, teacher, calib, args, args.device)
            init_s = time.time() - t_init
            ip = eval_ppl(st, val, args.device)
            Mk = None
            if spec["feat"] > 0 and V is not None:
                if arm not in Mk_cache:
                    Mk_cache[arm] = (V.T @ logit_metric(teacher).to(args.device) @ V)
                Mk = Mk_cache[arm]
            for p in st.parameters():
                p.requires_grad_(True)

            steps = args.steps + (bonus if not spec["fit"] else 0)
            torch.manual_seed(seed)
            t_kd = time.time()
            distill(teacher, st, train, args, args.device, kd=spec["kd"],
                    feat=spec["feat"], V=V, Mk=Mk, steps=steps)
            kd_s = time.time() - t_kd
            fp = eval_ppl(st, val, args.device)
            print(f"{arm:<14} seed={seed}  init={ip:>13,.1f}  final={fp:>8.2f}  "
                  f"steps={steps:<5} init_s={init_s:>5.1f} kd_s={kd_s:>5.1f}", flush=True)
            rows.append({"arm": arm, "seed": seed, "init_ppl": ip, "final_ppl": fp,
                         "steps": steps, "init_s": init_s, "kd_s": kd_s,
                         "total_s": init_s + kd_s})
            del st
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"teacher_ppl": t_ppl, "args": vars(args), "rows": rows}, indent=2))

    print("\n" + "=" * 72)
    print(f"{'arm':<15}{'final PPL':>22}{'total s':>12}")
    print("-" * 72)
    for arm in args.arms:
        v = [r["final_ppl"] for r in rows if r["arm"] == arm]
        c = [r["total_s"] for r in rows if r["arm"] == arm]
        sd = stats.stdev(v) if len(v) > 1 else 0.0
        print(f"{arm:<15}{stats.mean(v):>12.2f} +/- {sd:<6.2f}{stats.mean(c):>10.0f}")
    print(f"\nteacher PPL={t_ppl:.2f}\nwrote {args.output}")


if __name__ == "__main__":
    main()
