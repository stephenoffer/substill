"""Validate the LRD soundness fixes on the real benchmark, one fix at a time.

`docs/learned_restriction.md` reports LRD's win on ``JackFram/llama-160m`` at 3.07x. This
driver re-measures that number against the *fixed* restriction map and isolates what each
fix is worth, on the same teacher, data, geometry, budget and seeds.

Two stages.

**Stage A (``--stage init``)** costs no training: it builds the absorbed-init student under
each combination of fixes and evaluates it. This isolates the two fixes that change the
*initialization* -- the per-norm RMS gain and the write-aware FFN neuron selection -- from
anything training does. The absorbed init is the **shared** starting point of LRD and of the
``pca`` baseline, so an improvement here belongs to both arms and must be given to both; a
fix credited only to LRD would be a confound, not a result.

**Stage B (``--stage train``)** runs the arms to convergence. The baseline is the same
absorbed-init student trained with ordinary AdamW on every weight -- `docs/...`'s ``pca``.
LRD adds the Stiefel coordinate on top of the identical init, so the difference between them
is the ``V`` coordinate and nothing else. ``--v-rule`` switches between the old ambient
Stiefel step (the fitted ``min(1e-3, 0.77/d)`` constant) and the trust-region step, where the
learning rate *is* the RMS principal angle turned per step and no per-teacher constant
exists.

Every arm sees the identical batch sequence for a given seed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from substill.compression.llama_absorb import (
    gamma_fold_llama,
    llama_norm_input_second_moments,
    llama_residual_second_moment,
)
from substill.compression.restricted import (
    RestrictedLlama,
    StiefelAdamV,
    ffn_energy_indices,
)
from substill.compression.seq_absorb import eval_ppl, residual_basis
from substill.losses.generative_kd import forward_kl


# ---------------------------------------------------------------------------
def load(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).eval()
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def chunk(split):
        ids = tok("\n".join(t for t in raw[split]["text"] if t.strip()),
                  return_tensors="pt").input_ids[0]
        n = ids.numel() // args.seq_len
        ids = ids[: n * args.seq_len].view(n, args.seq_len)
        return [{"input_ids": ids[i:i + args.batch_size]}
                for i in range(0, n, args.batch_size)]

    return teacher, chunk("train"), chunk("validation")[: args.eval_batches]


def cosine_warmup(opt, steps, warm_frac=0.1, floor=0.0):
    warm = max(1, int(warm_frac * steps))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        cos = 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        return floor + (1 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def batches(train, steps, seed):
    """The identical batch sequence for every arm at a given seed."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=g)
    for s in range(steps):
        yield train[int(order[s % len(order)])]["input_ids"]


# ---------------------------------------------------------------------------
def balanced_second_moment(folded, calib, device):
    """``E[h h^T]`` pooled over layers with each layer's contribution *trace-normalized*.

    `llama_residual_second_moment` sums the raw second moment of every residual state. A
    transformer's residual norm grows steeply with depth, so that sum is dominated by the last
    few layers and the basis it induces barely sees the embedding: measured on llama-160m at
    3.07x, the PCA basis retains **97.8% of the pooled energy but only ~51% of layer 0's**.

    Normalizing each layer by its own trace before pooling asks instead for the subspace that
    serves every layer equally. Whether that is *better* is an empirical question -- the deep
    layers may deserve the weight they get -- so it is an arm, not a default.
    """
    from substill.compression.llama_absorb import llama_norm_input_second_moments
    nS = llama_norm_input_second_moments(folded, calib, device=device)
    tr = nS.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    return (nS / tr.view(-1, 1, 1)).mean(0)


def build(folded, S, calib, args, *, per_norm: bool, write_aware: bool, free: bool,
          device: str, basis_S=None):
    """The restricted student under a given combination of fixes.

    ``per_norm``    -- each RMSNorm gets the gain its own input distribution loses,
                       instead of one gain pooled over every layer at once.
    ``write_aware`` -- FFN neurons ranked by ``E[a^2] * ||V^T W_down[:, i]||^2`` (what the
                       neuron *writes* into the retained subspace) instead of ``E[a^2]``
                       alone (what it *holds*).
    ``basis_S``     -- the second moment the basis is chosen from (defaults to ``S``).
    """
    B = S if basis_S is None else basis_S
    method = getattr(args, "basis", "pca")
    if method == "gn":
        from substill.compression.llama_absorb import llama_logit_metric
        M = llama_logit_metric(folded).to(device)
        V0 = residual_basis(B, args.hidden, method="gn", M=M).to(device)
    else:
        V0 = residual_basis(B, args.hidden, method=method).to(device)
    idx = ffn_energy_indices(folded, calib, args.interm, device=device,
                             V=V0 if write_aware else None)
    nS = (llama_norm_input_second_moments(folded, calib, device=device)
          if per_norm else None)
    return RestrictedLlama(folded, S, V0, idx, args.n_head, args.n_kv,
                           free=free, norm_S=nS).to(device)


def train_baseline(teacher, student, train, val, *, steps, lr, seed, device):
    """`pca`: ordinary KD on every weight of the absorbed-init student."""
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = cosine_warmup(opt, steps)
    for ids in batches(train, steps, seed):
        ids = ids.to(device)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits[:, :-1]
        loss = forward_kl(student(input_ids=ids).logits[:, :-1], t_logits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
    return student


def _consistency(V, hs_s, t_hidden):
    t_proj = torch.stack([h.float() @ V for h in t_hidden[1:]], dim=0)
    cos = torch.nn.functional.cosine_similarity(hs_s, t_proj, dim=-1)
    return 1.0 - cos.mean()


def train_lrd(teacher, rm, train, *, steps, lr, v_lr, seed, device, aux_w, v_floor,
              trust_region, retraction="polar", aux_stream="prelogit"):
    """LRD: the same student, plus the Stiefel coordinate on the residual-stream basis."""
    rm.train()
    stiefel, euclid = rm.param_groups()
    opt_v = StiefelAdamV(stiefel, lr=v_lr, trust_region=trust_region, retraction=retraction)
    opt_d = torch.optim.AdamW(euclid, lr=lr, weight_decay=0.01)
    sch_v = cosine_warmup(opt_v, steps, floor=v_floor)
    sch_d = cosine_warmup(opt_d, steps)
    for step, ids in enumerate(batches(train, steps, seed)):
        ids = ids.to(device)
        if aux_w > 0:
            with torch.no_grad():
                t_out = teacher(input_ids=ids, output_hidden_states=True)
                t_logits = t_out.logits[:, :-1]
            s_out, hs_s = rm.hidden_and_logits(ids, stream=aux_stream)
            loss = forward_kl(s_out.logits[:, :-1], t_logits)
            lam = aux_w * (1.0 - step / max(1, steps))
            loss = loss + lam * _consistency(rm.V, hs_s, t_out.hidden_states)
        else:
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits[:, :-1]
            loss = forward_kl(rm(ids).logits[:, :-1], t_logits)
        opt_v.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(euclid, 1.0)
        # The trust region already bounds V's step length; clipping it too would only
        # rescale a direction that is about to be renormalized anyway.
        if not trust_region:
            torch.nn.utils.clip_grad_norm_(stiefel, 1.0)
        opt_v.step()
        opt_d.step()
        sch_v.step()
        sch_d.step()
    return rm


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="JackFram/llama-160m")
    p.add_argument("--stage", choices=["init", "train"], default="init")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--interm", type=int, default=1536)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-kv", type=int, default=6)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--v-lr", type=float, default=0.0,
                   help="0 -> the rule for the chosen --v-rule")
    p.add_argument("--v-rule", choices=["ambient", "trust"], default="trust")
    p.add_argument("--retraction", choices=["polar", "qr"], default="polar")
    p.add_argument("--legacy", action="store_true",
                   help="reproduce docs/learned_restriction.md exactly: pooled RMS gain, "
                        "activation-only FFN scoring, ambient Stiefel step, QR retraction")
    p.add_argument("--aux-w", type=float, default=1.0)
    p.add_argument("--aux-stream", choices=["residual", "prelogit"],
                   default="prelogit",
                   help="which student state the aux term matches at the last "
                        "layer; see docs/learned_restriction.md 9f")
    p.add_argument("--v-floor", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32,
                   help="must match scripts/analysis/lrb.py (32) or the PPLs are not "
                        "comparable to docs/learned_restriction.md")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=["pca", "lrd"])
    p.add_argument("--per-norm", type=int, default=1)
    p.add_argument("--basis", default="pca",
                   help="frozen-basis principle: pca (activation SVD) | gn (AIR: "
                        "activation+influence) | whiten (SVD-LLM) | identity")
    p.add_argument("--basis-pool", choices=["pooled", "balanced"], default="pooled",
                   help="how the per-layer second moments are pooled to choose V0; "
                        "'balanced' trace-normalizes each layer first (see 9h)")
    p.add_argument("--write-aware", type=int, default=1)
    p.add_argument("--out", default="")
    args = p.parse_args()
    if args.legacy:
        args.per_norm = 0
        args.write_aware = 0
        args.v_rule = "ambient"
        args.retraction = "qr"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, train, val = load(args)
    folded = gamma_fold_llama(teacher).eval().to(device)
    folded.requires_grad_(False)
    calib = [{"input_ids": b["input_ids"].to(device)} for b in train[: args.calib_batches]]
    S = llama_residual_second_moment(folded, calib, device=device).to(device)
    t_ppl = eval_ppl(folded, val, device)
    print(f"teacher PPL {t_ppl:.2f}   d={folded.config.hidden_size} -> k={args.hidden}",
          flush=True)

    rows = []
    if args.stage == "init":
        # No training: what each fix is worth to the *shared* absorbed init.
        bal = balanced_second_moment(folded, calib, device)
        for basis in ("pooled", "balanced"):
            bS = None if basis == "pooled" else bal
            for per_norm in (0, 1):
                for write_aware in (0, 1):
                    rm = build(folded, S, calib, args, per_norm=bool(per_norm),
                               write_aware=bool(write_aware), free=False, device=device,
                               basis_S=bS)
                    ppl = eval_ppl(rm.fold(), val, device)
                    rows.append({"basis": basis, "per_norm": per_norm,
                                 "write_aware": write_aware, "init_ppl": ppl})
                    print(f"  basis={basis:<8} per_norm={per_norm} "
                          f"write_aware={write_aware}  init PPL {ppl:.2f}", flush=True)
                    del rm
                    torch.cuda.empty_cache()
    else:
        v_lr = args.v_lr or (0.005 if args.v_rule == "trust"
                             else min(1e-3, 0.77 / folded.config.hidden_size))
        basis_S = (balanced_second_moment(folded, calib, device)
                   if args.basis_pool == "balanced" else None)
        for seed in args.seeds:
            for arm in args.arms:
                torch.manual_seed(seed)
                t0 = time.time()
                rm = build(folded, S, calib, args, per_norm=bool(args.per_norm),
                           write_aware=bool(args.write_aware),
                           free=(arm == "lrd"), device=device, basis_S=basis_S)
                init = eval_ppl(rm.fold(), val, device)
                if arm == "pca":
                    student = train_baseline(folded, rm.fold(), train, val,
                                             steps=args.steps, lr=args.lr, seed=seed,
                                             device=device)
                    ang = gap = 0.0
                else:
                    V0 = rm.V.detach().clone()
                    rm = train_lrd(folded, rm, train, steps=args.steps, lr=args.lr,
                                   v_lr=v_lr, seed=seed, device=device, aux_w=args.aux_w,
                                   v_floor=args.v_floor,
                                   trust_region=(args.v_rule == "trust"),
                                   retraction=args.retraction,
                                   aux_stream=args.aux_stream)
                    sv = torch.linalg.svdvals(V0.T @ rm.V.detach()).clamp(-1, 1)
                    ang = float(sv.arccos().max())
                    # How far outside the restriction class D actually took the student.
                    gap = rm.restriction_gap()
                    student = rm.fold()
                ppl = eval_ppl(student, val, device)
                dt = time.time() - t0
                rows.append({"arm": arm, "seed": seed, "init_ppl": init, "ppl": ppl,
                             "max_angle": ang, "restriction_gap": gap, "secs": dt,
                             "v_lr": v_lr, "v_rule": args.v_rule,
                             "retraction": args.retraction, "aux_stream": args.aux_stream,
                             "basis_pool": args.basis_pool, "basis": args.basis,
                             "per_norm": args.per_norm,
                             "write_aware": args.write_aware, "steps": args.steps})
                print(f"  [{arm} seed={seed}] init {init:.2f} -> PPL {ppl:.2f}  "
                      f"(angle {ang:.3f} rad, gap {gap:.3f}, {dt:.0f}s)", flush=True)
                del rm, student
                torch.cuda.empty_cache()

        for arm in args.arms:
            v = [r["ppl"] for r in rows if r["arm"] == arm and math.isfinite(r["ppl"])]
            if v:
                mu = sum(v) / len(v)
                sd = (sum((x - mu) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
                print(f"{arm:>6}: {mu:.2f} +/- {sd:.2f}  (n={len(v)})  {[round(x,2) for x in v]}",
                      flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"teacher_ppl": t_ppl, "args": vars(args), "rows": rows}, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
