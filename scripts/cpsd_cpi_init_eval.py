"""Concrete-win probe: CPI vs disjoint-basis absorbed init on a REAL GQA Llama.

GPT-2 cannot test CPSD's circuit-preserving init (no GQA/RoPE). This builds a
compressed student of a real GQA+RoPE Llama-family teacher TWO ways — identical
architecture (same cpi_rank_map config), differing ONLY in the attention init:

  - baseline: disjoint per-branch absorbed init (the builders.py:483-485 GQA bug)
  - cpi:      shared-per-group circuit-preserving init (apply_cpi_attention_init)

and measures validation PPL at init and after a short distillation. If CPI beats the
disjoint baseline at matched architecture, that is a concrete win for the novel
circuit-preserving component on the architecture where it engages.

Usage:
    python scripts/cpsd_cpi_init_eval.py --teacher TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
        --steps 200 --output runs/cpi_init.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import fasd
from fasd.builders import build_student
from fasd.compression.cpi import (
    apply_cpi_attention_init,
    apply_ov_align_init,
    cpi_rank_map,
)
from fasd.losses.generative_kd import forward_kl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    p.add_argument("--head-dim-ratio", type=float, default=0.5)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="runs/cpi_init.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load(args):
    if args.teacher == "tiny":
        # Offline smoke: small random GQA+RoPE Llama + synthetic data (no download).
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=args.seq_len)
        teacher = LlamaForCausalLM(cfg).eval()
        torch.manual_seed(0)
        tr = torch.randint(5, 120, (64, args.seq_len))
        va = torch.randint(5, 120, (16, args.seq_len))
        return teacher, tr, va

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float32).eval()
    raw = None
    for ds_id in ("Salesforce/wikitext", "wikitext"):
        try:
            raw = load_dataset(ds_id, "wikitext-2-raw-v1"); break
        except Exception as e:  # noqa: BLE001
            print(f"[cpi]   load_dataset({ds_id!r}) failed: {e}")
    def chunk(split):
        text = "\n".join(t for t in raw[split]["text"] if t.strip())
        ids = tok(text, return_tensors="pt").input_ids[0]
        n = ids.numel() // args.seq_len
        return ids[: n * args.seq_len].view(n, args.seq_len)
    return teacher, chunk("train"), chunk("validation")


def loader(ids, bs, shuffle, seed=0):
    idx = list(range(ids.shape[0]))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        if b:
            yield {"input_ids": ids[b]}


@torch.no_grad()
def eval_ppl(model, val_ids, args):
    model.eval()
    dev = next(model.parameters()).device
    nll, ntok = 0.0, 0
    for batch in list(loader(val_ids, args.batch_size, False))[: args.eval_batches]:
        ids = batch["input_ids"].to(dev)
        lg = model(input_ids=ids).logits
        sl, slab = lg[..., :-1, :].contiguous(), ids[..., 1:].contiguous()
        nll += float(F.cross_entropy(sl.view(-1, sl.size(-1)), slab.view(-1),
                                     reduction="sum").item())
        ntok += slab.numel()
    avg = nll / max(1, ntok)
    return math.exp(min(avg, 20)), avg


def distill(student, teacher, train_ids, args):
    dev = args.device
    student.to(dev).train(); teacher.to(dev).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    while step < args.steps:
        for batch in loader(train_ids, args.batch_size, True, seed=args.seed + step):
            if step >= args.steps:
                break
            ids = batch["input_ids"].to(dev)
            with torch.no_grad():
                tl = teacher(input_ids=ids).logits
            loss = forward_kl(student(input_ids=ids).logits, tl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); step += 1
            if step % 50 == 0:
                print(f"[cpi]   step {step}/{args.steps} loss={loss.item():.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    teacher, train_ids, val_ids = load(args)
    teacher.to(args.device)
    print(f"[cpi] teacher {args.teacher}: H={teacher.config.num_attention_heads} "
          f"G={teacher.config.num_key_value_heads} hidden={teacher.config.hidden_size}")
    t_ppl, _ = eval_ppl(teacher, val_ids, args)
    print(f"[cpi] teacher val PPL = {t_ppl:.2f}")

    calib = list(loader(train_ids, args.batch_size, False))[: args.calib_batches]
    profile = fasd.profile(teacher, calib, n_calib_batches=args.calib_batches,
                           behavioral_calib_batches=min(4, args.calib_batches))
    rm = cpi_rank_map(teacher, profile, head_dim_ratio=args.head_dim_ratio)

    # Three matched-architecture variants: disjoint baseline vs the two CPI bases.
    # cpi_crossplane = max-energy shared basis (best circuit + energy, RoPE-inexact);
    # cpi_planealigned = RoPE-commuting but energy-sacrificing.
    # Matched-architecture variants (same cpi_rank_map config; only attention init differs).
    #   ov_align  = fix OV circuit for free (o_proj input := v_proj's own basis); keep baseline Q/K
    #   crossplane= full shared per-group basis (circuit-preserving, energy-compromising)
    variants = ("baseline_disjoint", "cpi_ov_align", "cpi_crossplane")
    results = {}
    for variant in variants:
        set_seed(args.seed)
        student = build_student(teacher, profile, template="llama", rank_map=rm,
                                absorbed_init=True)
        if variant == "cpi_ov_align":
            apply_ov_align_init(student, teacher, profile, calib)
        elif variant == "cpi_crossplane":
            apply_cpi_attention_init(student, teacher, profile, calib, rope_aware=False)
        sp = sum(p.numel() for p in student.parameters())
        init_ppl, _ = eval_ppl(student.to(args.device), val_ids, args)
        print(f"[cpi] {variant}: {sp/1e6:.1f}M params, init PPL = {init_ppl:.2f}")
        distill(student, teacher, train_ids, args)
        final_ppl, _ = eval_ppl(student, val_ids, args)
        print(f"[cpi] {variant}: final PPL = {final_ppl:.2f}")
        results[variant] = dict(params=int(sp), init_ppl=float(init_ppl),
                                final_ppl=float(final_ppl))
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    b = results["baseline_disjoint"]
    cpi_variants = {k: v for k, v in results.items() if k.startswith("cpi")}
    best_cpi = min(cpi_variants, key=lambda k: cpi_variants[k]["final_ppl"])
    bc = results[best_cpi]
    summary = dict(teacher=args.teacher, teacher_ppl=float(t_ppl),
                   head_dim_ratio=args.head_dim_ratio, steps=args.steps,
                   results=results, best_cpi=best_cpi,
                   final_delta=b["final_ppl"] - bc["final_ppl"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[cpi] === matched-architecture comparison (final PPL) ===")
    for k, v in results.items():
        print(f"  {k:<20} init {v['init_ppl']:.1f}  final {v['final_ppl']:.2f}")
    print(f"  best CPI = {best_cpi}; Δ vs baseline = {summary['final_delta']:+.2f} "
          f"-> {'CPI WINS' if summary['final_delta'] > 0 else 'baseline wins'}")
    print(f"[cpi] wrote {args.output}")


if __name__ == "__main__":
    main()
