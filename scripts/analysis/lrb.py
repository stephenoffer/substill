"""Learned restriction vs. every basis criterion that came before it.

`docs/init_findings.md` §10b closes with six criteria for choosing the residual subspace,
none of which beats plain PCA. Every one of them optimizes a *surrogate*: retained
variance, or logit error under a linearization of the last layer. This script runs the
un-surrogated arm -- the subspace optimized against the **KD loss of the assembled
student**, through the whole network, while never leaving the class of restrictions of the
teacher's operator.

Arms (all at the same student geometry, the same FFN neuron selection, the same whole
heads, the same data order, the same seed; only the residual subspace and how it was
obtained differ):

  identity   V = eye(d, k)                        the `_residual_basis` bug's fallback
  pca        top-k eigenvectors of E[h h^T]       the best criterion known (§10)
  lrb        V trained against KD, from PCA       ours
  lrb_id     V trained against KD, from identity  ours, told nothing about the teacher
  lrb_only   V trained against KD for the whole budget, weights never released

`lrb` spends `--v-steps` steps training only ``V`` (147k degrees of freedom), folds to a
plain `LlamaForCausalLM`, and distils the remaining budget normally. The comparison is
matched on **wall-clock**, not steps: a restricted step costs more than a free one because
it re-materializes ``V^T W V`` every forward, so the baselines are given the difference
back as extra KD steps. `--calibrate` measures both per-step costs before committing.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import time
from pathlib import Path

import torch


def _disable_triton_overrides() -> None:
    """Llama's rotary embedding routes an outer product to a Triton kernel, which needs a
    C compiler at runtime. Dropping the override falls back to eager -- same numerics."""
    try:
        from torch._native import registry
        registry.deregister_op_overrides(disable_dsl_names="triton")
    except Exception:  # noqa: BLE001
        pass


_disable_triton_overrides()

from scripts.analysis.h2h import kd_loss  # noqa: E402
from substill.compression.llama_absorb import (  # noqa: E402
    absorb_llama,
    gamma_fold_llama,
    llama_residual_second_moment,
    rms_gain,
)
from substill.compression.restricted import (  # noqa: E402
    RestrictedLlama,
    StiefelAdamV,
    ffn_energy_indices,
    indices_to_bases,
)
from substill.compression.seq_absorb import eval_ppl, residual_basis  # noqa: E402


# ---------------------------------------------------------------------------
def load(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    # Large teachers (7B+) do not fit a 22 GB card in fp32. Loading the frozen teacher in
    # bf16 halves its storage; the restriction up-casts each small weight slice at the matmul
    # site (`RestrictedLlama._restriction`), so V trains in fp32 and only the teacher's bulk
    # is halved. Default stays fp32 so the small-model numbers are bit-for-bit unchanged.
    tdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}[getattr(args, "teacher_dtype", "float32")]
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=tdtype).eval()
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def chunk(split):
        ids = tok("\n".join(t for t in raw[split]["text"] if t.strip()),
                  return_tensors="pt").input_ids[0]
        n = ids.numel() // args.seq_len
        ids = ids[: n * args.seq_len].view(n, args.seq_len)
        return [{"input_ids": ids[i:i + args.batch_size]}
                for i in range(0, n, args.batch_size)]

    return teacher, chunk("train"), chunk("validation")[: args.eval_batches]


# This driver exists to *reproduce* the published numbers in
# `docs/learned_restriction.md`, so it must keep using the Stiefel step those numbers were
# measured with: a raw ambient Adam step with a sign-fixed QR retraction. `StiefelAdamV` now
# defaults to a trust-region step (the LR *is* the rotation per step) and a polar retraction,
# which are better on every axis (§9c) but are a different optimizer -- silently adopting them
# here would mean this script no longer reproduces what it claims to.
# New work should use the defaults; see `scripts/analysis/lrd_validate.py`.
_LEGACY_STIEFEL = {"trust_region": False, "retraction": "qr"}


def cosine_warmup(opt, steps, warm_frac=0.1, floor=0.0):
    """Linear warmup then cosine decay to ``floor`` (a fraction of peak, default 0).

    ``floor > 0`` keeps a parameter *travelling* through the whole budget instead of
    freezing in the second half -- the knob that matters for ``V``, whose gain the docs
    attribute to *where it travels*, not where it begins.
    """
    warm = max(1, int(warm_frac * steps))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        cos = 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        return floor + (1 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def _batches(train, steps, seed, offset=0):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=g)
    for s in range(steps):
        yield train[int(order[(s + offset) % len(order)])]["input_ids"]


# ---------------------------------------------------------------------------
def train_restriction(teacher, rm, train, *, steps, lr, seed, device, kd="forward_kl",
                      offset=0):
    """Phase 1: descend the KD loss on the Grassmannian. ``V`` is the only parameter."""
    rm.train()
    opt = StiefelAdamV([rm.V], lr=lr, **_LEGACY_STIEFEL)
    sched = cosine_warmup(opt, steps)
    for ids in _batches(train, steps, seed, offset):
        ids = ids.to(device)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits
        loss = kd_loss(rm(ids).logits[:, :-1], t_logits[:, :-1], kd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([rm.V], 1.0)
        opt.step()
        sched.step()
    return rm


def _restriction_consistency(rm, hs_s, t_hidden):
    """Scale-invariant per-layer agreement between the student stream and ``V^T h_T``.

    ``hs_s`` is ``(L, B, T, k)`` (student, from `hidden_and_logits`); ``t_hidden`` is the
    teacher's ``output_hidden_states`` tuple. We project each teacher layer by ``V`` and ask
    the student's stream to *point the same way* -- cosine, so the RMS-gain scale the
    student's norms carry is irrelevant and only the direction (the teacher's computation
    seen through the subspace) is matched. Differentiable in ``V`` on both sides.
    """
    V = rm.V
    t_proj = torch.stack([h.float() @ V for h in t_hidden[1:]], dim=0)   # (L, B, T, k)
    cos = torch.nn.functional.cosine_similarity(hs_s, t_proj, dim=-1)    # (L, B, T)
    return 1.0 - cos.mean()


def train_joint(teacher, rm, train, *, steps, lr, v_lr, seed, device, kd="forward_kl",
                tau=1.0, v_floor=0.0, aux_w=0.0):
    """Ours: descend in *both* coordinates at once.

    The student is ``W_s = V^T W_T V + D`` with ``D`` zero-initialized, so step 0 is exactly
    the absorbed-init student the `pca` baseline starts from, and ``D`` alone spans the same
    function class. The only difference is that gradient descent is additionally offered the
    ``V`` direction -- a single coherent rotation of the residual subspace that all twelve
    layers see at once, and the one direction along which the student stays a restriction of
    the teacher. Freezing ``V`` recovers the baseline exactly (`--arms pca_reparam`).
    """
    rm.train()
    stiefel, euclid = rm.param_groups()
    opt_v = StiefelAdamV(stiefel, lr=v_lr, **_LEGACY_STIEFEL)
    opt_d = torch.optim.AdamW(euclid, lr=lr, weight_decay=0.01)
    sch_v = cosine_warmup(opt_v, steps, floor=v_floor)
    sch_d = cosine_warmup(opt_d, steps)
    for step, ids in enumerate(_batches(train, steps, seed)):
        ids = ids.to(device)
        if aux_w > 0:
            with torch.no_grad():
                t_out = teacher(input_ids=ids, output_hidden_states=True)
                t_logits = t_out.logits
            s_out, hs_s = rm.hidden_and_logits(ids)
            loss = kd_loss(s_out.logits[:, :-1], t_logits[:, :-1], kd, tau=tau)
            # Anneal the auxiliary term to zero: it shapes V's early travel, then hands off
            # to the true KD objective so the final basin is chosen by KD alone.
            lam = aux_w * (1.0 - step / max(1, steps))
            loss = loss + lam * _restriction_consistency(rm, hs_s, t_out.hidden_states)
        else:
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits
            loss = kd_loss(rm(ids).logits[:, :-1], t_logits[:, :-1], kd, tau=tau)
        opt_v.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(euclid, 1.0)
        torch.nn.utils.clip_grad_norm_(stiefel, 1.0)
        if v_lr > 0:
            opt_v.step()
        opt_d.step()
        sch_v.step()
        sch_d.step()
    return rm


def train_amortized(teacher, student, rm, train, *, steps, lr, v_lr, refresh_every,
                    seed, device, kd="forward_kl"):
    """Ours, made cheap: a plain student trained every step; V refreshed rarely.

    The student is an ordinary folded `LlamaForCausalLM`, so its forward is exactly the
    baseline's -- no per-step restriction cost. Every ``refresh_every`` steps we pay for a
    *single* restricted V-step: sync `rm` to the current student (`load_student_residual`),
    take one Riemannian step on ``V``, and fold the improvement back (`write_back`), keeping
    the student's optimizer state attached to its own parameters.

    So the projection is still trained against the true KD loss through the whole network,
    but its cost is amortized ``refresh_every``-fold. This is also the only version that
    could run on a frontier model, where re-projecting every weight each step is infeasible.
    """
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = cosine_warmup(opt, steps)
    v_opt = StiefelAdamV([rm.V], lr=v_lr, **_LEGACY_STIEFEL)
    for step, ids in enumerate(_batches(train, steps, seed)):
        ids = ids.to(device)
        if v_lr > 0 and step > 0 and step % refresh_every == 0:
            rm.load_student_residual(student, rm.V.detach())
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits
            loss = kd_loss(rm(ids).logits[:, :-1], t_logits[:, :-1], kd)
            v_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([rm.V], 1.0)
            v_opt.step()
            rm.write_back(student)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits
        loss = kd_loss(student(input_ids=ids).logits[:, :-1], t_logits[:, :-1], kd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
    return student


def train_free(teacher, student, train, *, steps, lr, seed, device, kd="forward_kl",
               offset=0):
    """Phase 2 / the baselines: ordinary KD on every student weight."""
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = cosine_warmup(opt, steps)
    for ids in _batches(train, steps, seed, offset):
        ids = ids.to(device)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits
        loss = kd_loss(student(input_ids=ids).logits[:, :-1], t_logits[:, :-1], kd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
    return student


# ---------------------------------------------------------------------------
def make_restricted(folded, S, V0, idx, args, device, free=False):
    return RestrictedLlama(folded, S, V0, idx, args.n_head, args.n_kv, free=free).to(device)


def absorbed_student(folded, S, V, idx, device):
    from substill.compression.llama_absorb import build_narrow_llama
    E = indices_to_bases(idx, folded.config.intermediate_size)
    st = build_narrow_llama(folded, V.shape[1], idx[0].numel(),
                            _NH[0], _NKV[0]).to(device)
    absorb_llama(folded, st, V, E, norm_gain=float(rms_gain(S, V)))
    return st


_NH, _NKV = [0], [0]   # set from args in main(); keeps `absorbed_student` a pure helper


# ---------------------------------------------------------------------------
def run_arm(arm, folded, S, idx, train, val, args, seed, device):
    torch.manual_seed(seed)
    t0 = time.time()
    k = args.hidden

    if arm in ("identity", "pca"):
        V = residual_basis(S, k, method=arm).to(device)
        st = absorbed_student(folded, S, V, idx, device)
        init = eval_ppl(st.eval(), val, device)
        st.requires_grad_(True)
        train_free(folded, st, train, steps=args.steps + args.bonus, lr=args.lr,
                   seed=seed, device=device)
        final = eval_ppl(st, val, device)
        return {"arm": arm, "seed": seed, "v_steps": 0, "kd_steps": args.steps + args.bonus,
                    "init_ppl": init, "mid_ppl": init, "final_ppl": final, "secs": time.time() - t0}

    if arm in ("lrb_joint", "pca_reparam", "lrb_joint_id", "lrb_joint_gn"):
        if arm == "lrb_joint_gn":
            # Start LRD from AIR's activation+influence basis (the best *frozen* basis, §4a)
            # instead of PCA. Tests whether a stronger starting subspace widens the win.
            from substill.compression.llama_absorb import llama_logit_metric
            M = llama_logit_metric(folded).to(device)
            V0 = residual_basis(S, k, method="gn", M=M).to(device)
        else:
            start = "identity" if arm == "lrb_joint_id" else "pca"
            V0 = residual_basis(S, k, method=start).to(device)
        rm = make_restricted(folded, S, V0, idx, args, device, free=True)
        init = eval_ppl(rm, val, device)
        # `pca_reparam` freezes V: the same model, the same optimizer, the same data, with
        # the Stiefel coordinate switched off. It must reproduce `pca`, and any gap between
        # it and `lrb_joint` is attributable to that coordinate alone.
        v_lr = 0.0 if arm == "pca_reparam" else args.v_lr
        train_joint(folded, rm, train, steps=args.steps, lr=args.lr, v_lr=v_lr,
                    seed=seed, device=device, kd=args.kd, tau=args.tau,
                    v_floor=args.v_floor, aux_w=args.aux_w)
        st = rm.fold().eval()
        final = eval_ppl(st, val, device)
        sub = float(torch.linalg.svdvals(V0.T @ rm.V.detach()).clamp(-1, 1).arccos().max())
        del rm
        torch.cuda.empty_cache()
        return {"arm": arm, "seed": seed, "v_steps": args.steps, "kd_steps": args.steps,
                    "init_ppl": init, "mid_ppl": init, "final_ppl": final,
                    "max_principal_angle": sub, "secs": time.time() - t0}

    if arm == "lrb_amortized":
        V0 = residual_basis(S, k, method="pca").to(device)
        st = absorbed_student(folded, S, V0, idx, device)
        init = eval_ppl(st.eval(), val, device)
        st.requires_grad_(True)
        rm = make_restricted(folded, S, V0, idx, args, device, free=True)
        train_amortized(folded, st, rm, train, steps=args.steps, lr=args.lr,
                        v_lr=args.v_lr, refresh_every=args.refresh_every, seed=seed,
                        device=device)
        final = eval_ppl(st, val, device)
        sub = float(torch.linalg.svdvals(V0.T @ rm.V.detach()).clamp(-1, 1).arccos().max())
        del rm
        torch.cuda.empty_cache()
        return {"arm": arm, "seed": seed, "v_steps": args.steps // args.refresh_every,
                    "kd_steps": args.steps, "init_ppl": init, "mid_ppl": init, "final_ppl": final,
                    "max_principal_angle": sub, "secs": time.time() - t0}

    start = "pca" if arm in ("lrb", "lrb_only") else "identity"
    V0 = residual_basis(S, k, method=start).to(device)
    rm = make_restricted(folded, S, V0, idx, args, device)
    init = eval_ppl(rm, val, device)

    v_steps = args.steps if arm == "lrb_only" else args.v_steps
    train_restriction(folded, rm, train, steps=v_steps, lr=args.v_lr, seed=seed,
                      device=device)
    mid = eval_ppl(rm, val, device)
    st = rm.fold().eval()
    # `fold` must be function-identical; assert it here too, on the real model.
    with torch.no_grad():
        a = rm(val[0]["input_ids"].to(device)).logits
        b = st(input_ids=val[0]["input_ids"].to(device)).logits
    assert (a - b).abs().max() < 1e-2, f"fold drifted: {(a - b).abs().max():.3e}"
    del rm
    torch.cuda.empty_cache()

    kd_steps = 0 if arm == "lrb_only" else args.steps - args.v_steps
    if kd_steps > 0:
        st.requires_grad_(True)
        train_free(folded, st, train, steps=kd_steps, lr=args.lr, seed=seed,
                   device=device, offset=v_steps)
    final = eval_ppl(st, val, device)
    return {"arm": arm, "seed": seed, "v_steps": v_steps, "kd_steps": kd_steps, "init_ppl": init,
                "mid_ppl": mid, "final_ppl": final, "secs": time.time() - t0}


# ---------------------------------------------------------------------------
def calibrate(folded, S, idx, train, args, device, n=15):
    """Seconds per restricted step and per free step, to set the compute-match bonus."""
    V = residual_basis(S, args.hidden, method="pca").to(device)
    rm = make_restricted(folded, S, V, idx, args, device)
    train_restriction(folded, rm, train, steps=3, lr=1e-9, seed=0, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    train_restriction(folded, rm, train, steps=n, lr=1e-9, seed=0, device=device)
    torch.cuda.synchronize()
    r = (time.time() - t0) / n
    del rm
    torch.cuda.empty_cache()

    st = absorbed_student(folded, S, V, idx, device)
    st.requires_grad_(True)
    train_free(folded, st, train, steps=3, lr=1e-9, seed=0, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    train_free(folded, st, train, steps=n, lr=1e-9, seed=0, device=device)
    torch.cuda.synchronize()
    f = (time.time() - t0) / n
    del st
    torch.cuda.empty_cache()
    return r, f


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="JackFram/llama-160m")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--interm", type=int, default=1536)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-kv", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000, help="total steps for the lrb arms")
    p.add_argument("--v-steps", type=int, default=500, help="phase-1 (restricted) steps")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW lr for free weights")
    p.add_argument("--v-lr", type=float, default=3e-3, help="Stiefel lr for V")
    p.add_argument("--kd", default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "skew_kl"],
                   help="KD divergence for the joint arms")
    p.add_argument("--tau", type=float, default=1.0,
                   help="KD softmax temperature for the joint arms")
    p.add_argument("--v-floor", type=float, default=0.0,
                   help="cosine floor (fraction of peak) for the V learning rate")
    p.add_argument("--aux-w", type=float, default=0.0,
                   help="weight of the annealed restriction-consistency (cosine) aux loss")
    p.add_argument("--teacher-dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="load the frozen teacher in this dtype (bf16 to fit huge models)")
    p.add_argument("--refresh-every", type=int, default=20,
                   help="amortized arm: steps between V refreshes")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--arms", nargs="+",
                   default=["identity", "pca", "pca_reparam", "lrb_joint"])
    p.add_argument("--no-compute-match", action="store_true")
    p.add_argument("--output", default="runs/lrb.json")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    dev = args.device
    teacher, train, val = load(args)          # on CPU
    t_params = sum(p.numel() for p in teacher.parameters())

    # Scale-aware Stiefel LR (see below): v-lr ~ 1/d, capped at 1e-3. `--v-lr 0` requests it.
    if args.v_lr <= 0:
        args.v_lr = min(1e-3, 0.77 / int(teacher.config.hidden_size))
        print(f"[auto v-lr] d={teacher.config.hidden_size} -> v-lr={args.v_lr:.2e}", flush=True)

    # Fold on CPU, then move only the folded model to the GPU. `gamma_fold_llama` deep-copies
    # the teacher; doing that on the GPU would briefly hold *two* copies (26 GB for a 7B bf16
    # teacher -> OOM). RAM is plentiful, so the copy happens there. The fold is
    # function-preserving, so `folded` is the teacher for evaluation purposes too.
    import gc
    folded = gamma_fold_llama(teacher).eval()
    del teacher
    gc.collect()
    folded = folded.to(dev)
    folded.requires_grad_(False)     # the teacher is read-only; no grads, no optimizer state
    torch.cuda.empty_cache()
    t_ppl = eval_ppl(folded, val, dev)        # γ-fold preserves the function, so == teacher PPL

    calib = train[: args.calib_batches]
    S = llama_residual_second_moment(folded, calib, device=dev)
    idx = ffn_energy_indices(folded, calib, args.interm, device=dev)
    _NH[0], _NKV[0] = args.n_head, args.n_kv

    args.bonus = 0
    if not args.no_compute_match:
        r, f = calibrate(folded, S, idx, train, args, dev)
        # A restricted step costs `r`; a free step costs `f`. The lrb arms spend
        # v_steps * r + (steps - v_steps) * f. Give the baselines the difference in
        # free steps so every arm consumes the same wall-clock.
        extra = args.v_steps * (r - f)
        args.bonus = max(0, int(round(extra / f)))
        print(f"calibration: restricted step {r*1000:.1f} ms, free step {f*1000:.1f} ms "
              f"-> baselines get +{args.bonus} steps\n", flush=True)

    print(f"teacher {args.teacher}: PPL={t_ppl:.2f} params={t_params/1e6:.1f}M\n", flush=True)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm(arm, folded, S, idx, train, val, args, seed, dev)
            rows.append(r)
            print(f"{arm:<9} seed={seed}  V-steps={r['v_steps']:<5} KD-steps={r['kd_steps']:<5} "
                  f"init={r['init_ppl']:>11,.0f}  after-V={r['mid_ppl']:>11,.0f}  "
                  f"final={r['final_ppl']:>7.2f}  ({r['secs']:.0f}s)", flush=True)
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"teacher": args.teacher, "teacher_ppl": t_ppl, "args": vars(args), "rows": rows},
        indent=2))

    print("\n" + "=" * 78)
    print(f"{'arm':<10}{'V steps':>9}{'KD steps':>10}{'after-V PPL':>15}"
          f"{'final PPL':>14}{'  s':>8}")
    print("-" * 78)
    for arm in args.arms:
        v = [r["final_ppl"] for r in rows if r["arm"] == arm]
        m = [r["mid_ppl"] for r in rows if r["arm"] == arm]
        s = [r["secs"] for r in rows if r["arm"] == arm]
        one = next(r for r in rows if r["arm"] == arm)
        sd = stats.stdev(v) if len(v) > 1 else 0.0
        print(f"{arm:<10}{one['v_steps']:>9}{one['kd_steps']:>10}{stats.mean(m):>15,.1f}"
              f"{stats.mean(v):>9.2f} +/- {sd:<5.2f}{stats.mean(s):>7.0f}")
    print(f"\nteacher PPL={t_ppl:.2f}   wrote {args.output}")


if __name__ == "__main__":
    main()
