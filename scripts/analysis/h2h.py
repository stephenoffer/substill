"""Controlled head-to-head: initialization method x KD objective, matched everything.

Every arm trains the SAME student architecture (identical parameter count) with the
SAME optimizer, schedule, data order, and step budget. Only the named factor varies.
Anything that differs between arms other than the factor under test is a confound,
so the KD loop lives here in one place rather than behind a configurable driver.

Init arms
---------
random        random init (the naive floor; what most KD papers use)
identity      the repo's shipped absorbed init -- first-k residual coords
              (``_residual_basis`` falls back to ``torch.eye`` under the default
              profile mode). Prior-art baseline as actually implemented.
select        variance channel-selection basis + absorbed init (Minitron-style
              importance, LLM-Pruner family)
pca           PCA residual basis + absorbed init (FWSVD/ESPACE family)
pca_fold      PCA + gamma-fold + RMSNorm student, absorbed init (SliceGPT family)
sgca_*        + sequential gap-closing absorption (ours)

Ablations of ours (each removes one ingredient) are listed in ``SGCA_ARMS``.

Superseded: ``scripts/bench.py`` is the harness whose numbers docs/init_findings.md
reports, because it matches compute rather than step count. Keep this script for the
per-ingredient ablations; do not quote its equal-step numbers as head-to-head wins.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from substill.compression.seq_absorb import (
    RMSNorm,
    SeqAbsorbConfig,
    _ffn_basis,
    absorb_gpt2,
    build_narrow_gpt2,
    eval_ppl,
    gamma_fold_gpt2,
    logit_metric,
    residual_basis,
    residual_second_moment,
    sequential_absorb_gpt2,
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).eval()
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def chunk(split):
        ids = tok("\n".join(t for t in raw[split]["text"] if t.strip()),
                  return_tensors="pt").input_ids[0]
        n = ids.numel() // args.seq_len
        ids = ids[: n * args.seq_len].view(n, args.seq_len)
        return [{"input_ids": ids[i:i + args.batch_size]}
                for i in range(0, n, args.batch_size)]

    return teacher, chunk("train"), chunk("validation")[: args.eval_batches]


# --------------------------------------------------------------------------
# init arms
# --------------------------------------------------------------------------
def _plain_absorb(teacher, calib, k, inner, basis, device, *, fold=False, rms=False):
    t = gamma_fold_gpt2(teacher).to(device).eval() if fold else teacher
    S = residual_second_moment(t, calib, device=device)
    V = residual_basis(S, k, method=basis, M=logit_metric(t).to(device)).to(device)
    ffn = [_ffn_basis(t, i, calib, inner, device) for i in range(t.config.n_layer)]
    student = build_narrow_gpt2(t, k, inner).to(device)
    absorb_gpt2(t, student, V, ffn)
    if rms:
        for sb in student.transformer.h:
            sb.ln_1, sb.ln_2 = RMSNorm(k).to(device), RMSNorm(k).to(device)
    return student.eval()


# `sgca` is the full method; each other arm removes exactly one ingredient.
SGCA_ARMS = {
    "sgca":          {},
    "sgca_l2only":   {"objective": "l2"},        # -teacher-graft objective
    "sgca_graftonly": {"objective": "graft"},    # -L2 stabilizer
    "sgca_nodrift":  {"input_mode": "teacher"},  # -drift correction
    "sgca_output":   {"target_mode": "output"},  # -gap closing (classic layerwise recon)
    "sgca_l2metric": {"metric": "l2"},           # -logit-Jacobian metric
    "sgca_var":      {"basis": "select"},        # -logit-aware channel scoring
    "sgca_ident":    {"basis": "identity"},      # -basis entirely (repo's fallback)
}


def build_init(name, teacher, calib, args, device):
    k, inner = args.k, args.inner
    if name == "random":
        torch.manual_seed(args.seed)
        return build_narrow_gpt2(teacher, k, inner).to(device)
    if name in ("identity", "random_sel", "select", "select_gn", "pca"):
        return _plain_absorb(teacher, calib, k, inner, name, device)
    if name == "pca_fold":  # SliceGPT-style: rotation + gamma-fold + RMSNorm student
        return _plain_absorb(teacher, calib, k, inner, "pca", device, fold=True, rms=True)
    if name not in SGCA_ARMS:
        raise ValueError(f"unknown init arm: {name}")
    cfg = SeqAbsorbConfig(k=k, inner=inner, steps_per_block=args.steps_per_block,
                          verbose=False, **SGCA_ARMS[name])
    student, _ = sequential_absorb_gpt2(teacher, calib, cfg, device=device)
    return student


# --------------------------------------------------------------------------
# KD objectives (all on logits; token-mean)
# --------------------------------------------------------------------------
def kd_loss(s_logits, t_logits, kind, tau=1.0):
    # Flatten to (tokens, vocab) so `batchmean` divides by the token count. On the
    # raw (B, T, V) tensor it divides by B alone and silently sums over T, inflating
    # the KD term ~127x relative to any auxiliary loss added to it.
    s_logits = s_logits.reshape(-1, s_logits.size(-1))
    t_logits = t_logits.reshape(-1, t_logits.size(-1))
    ls = F.log_softmax(s_logits / tau, -1)
    lt = F.log_softmax(t_logits / tau, -1)
    if kind == "forward_kl":            # KD (Hinton); teacher || student
        return F.kl_div(ls, lt, reduction="batchmean", log_target=True) * tau * tau
    if kind == "reverse_kl":            # MiniLLM family
        return F.kl_div(lt, ls, reduction="batchmean", log_target=True) * tau * tau
    if kind == "skew_kl":               # DistiLLM family
        # DistiLLM: SKL_a(p||q) = KL(p || a*p + (1-a)*q), a=0.1 -- the mixture is dominated
        # by the STUDENT. This previously mixed 0.9*p_t + 0.1*p_s, bounding the loss by
        # log(1/0.9)=0.105 and leaving it with ~1.6% of forward-KL's gradient signal; the
        # arm it produced (403.33 PPL in runs/bench_matched_s012.json) was measuring a loss
        # that barely trained the model. Use `scripts.objectives.divergence` for new work;
        # this is kept correct so old scripts do not silently mislead.
        a = 0.1
        mix = torch.logsumexp(
            torch.stack([lt + math.log(a), ls + math.log1p(-a)]), 0)
        return F.kl_div(mix, lt, reduction="batchmean", log_target=True) * tau * tau
    raise ValueError(kind)


def distill(teacher, student, train, args, device, *, kd="forward_kl"):
    teacher.to(device).eval()
    student.to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * args.steps))

    def lr_at(s):  # linear warmup -> cosine decay to 0
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, args.steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(args.seed)   # identical data order per arm
    order = torch.randperm(len(train), generator=g)
    for step in range(args.steps):
        b = train[int(order[step % len(order)])]
        ids = b["input_ids"].to(device)
        with torch.no_grad():
            tl = teacher(input_ids=ids).logits
        sl = student(input_ids=ids).logits
        loss = kd_loss(sl[:, :-1], tl[:, :-1], kd)
        if args.ce_weight:
            loss = loss + args.ce_weight * F.cross_entropy(
                sl[:, :-1].reshape(-1, sl.size(-1)), ids[:, 1:].reshape(-1))
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
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--steps-per-block", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ce-weight", type=float, default=0.0)
    p.add_argument("--kd", default="forward_kl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inits", nargs="+",
                   default=["random", "identity", "select", "select_gn", "pca",
                            "pca_fold", "sgca"])
    p.add_argument("--output", default="runs/h2h.json")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"teacher PPL={t_ppl:.2f}  params={t_params:,}  "
          f"seed={args.seed}  kd={args.kd}  steps={args.steps}\n", flush=True)
    calib = train[: args.calib_batches]

    rows = []
    for name in args.inits:
        t0 = time.time()
        torch.manual_seed(args.seed)
        student = build_init(name, teacher, calib, args, args.device)
        p0 = eval_ppl(student, val, args.device)
        sp = sum(p.numel() for p in student.parameters())
        torch.manual_seed(args.seed)
        distill(teacher, student, train, args, args.device, kd=args.kd)
        p1 = eval_ppl(student, val, args.device)
        dt = time.time() - t0
        print(f"{name:<14} init={p0:>14,.1f}   final={p1:>9.2f}   "
              f"params={sp:,}  ({dt:.0f}s)", flush=True)
        rows.append({"init": name, "init_ppl": p0, "final_ppl": p1, "params": sp,
                     "ratio": t_params / sp, "seconds": dt, "seed": args.seed,
                     "kd": args.kd, "steps": args.steps})
        del student
        torch.cuda.empty_cache()

    out = {"teacher": args.teacher, "teacher_ppl": t_ppl, "teacher_params": t_params,
           "k": args.k, "inner": args.inner, "args": vars(args), "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
