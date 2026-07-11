"""Head-to-head, matched-compression comparison: CPSD vs baselines vs competitor foils.

Every variant compresses the SAME teacher to the SAME student width and trains for the
same budget; only the method differs. Emits a summary JSON in the schema
``scripts/cpsd_aggregate.py`` consumes, so multiple seeds/ratios tabulate into one table
with win/tie/loss verdicts.

Variants
--------
- ``r1_kd_random``  : random-init student + KD (the naive competition floor).
- ``r2_fasd``       : absorbed-init + KD, frozen bases (prior-art baseline).
- ``cpsd_mt``       : + manifold-trained (Stiefel) factors [NOVEL].
- ``cpsd_full``     : + distillation-driven differentiable rank (MT+DDR) [NOVEL, ours].
- ``dobi_svd``      : the SAME MT+DDR machinery but the differentiable rank is trained on
                      feature *reconstruction* (KD weight 0), i.e. Dobi-SVD's
                      reconstruction-driven rank. This is the controlled foil that isolates
                      our central claim: *KD-driven* rank vs *reconstruction-driven* rank.

Smoke (CPU, seconds; mechanics only, numbers meaningless):
    python scripts/cpsd_compare.py --smoke
Real (GPT-2 + WikiText-2):
    python scripts/cpsd_compare.py --teacher gpt2 --steps 500 --target-compression 2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import substill


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--target-compression", type=float, default=2.0)
    p.add_argument("--arch-multiplier", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", nargs="+",
                   default=["r1_kd_random", "r2_fasd", "cpsd_mt", "cpsd_full", "dobi_svd"])
    p.add_argument("--output", default="runs/cpsd_compare.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load(args):
    if args.smoke or args.teacher == "tiny":
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(vocab_size=64, n_positions=args.seq_len, n_embd=32,
                         n_layer=2, n_head=4, n_inner=128)
        cfg.pad_token_id = 0
        teacher = GPT2LMHeadModel(cfg).eval()
        torch.manual_seed(0)
        def mk(n):
            out = []
            for _ in range(n):
                t = torch.randint(5, 60, (args.batch_size, args.seq_len))
                out.append({"input_ids": t, "labels": t})
            return out
        return teacher, mk(16), mk(args.eval_batches)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32).eval()
    raw = None
    for ds_id in ("Salesforce/wikitext", "wikitext"):
        try:
            raw = load_dataset(ds_id, "wikitext-2-raw-v1")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[compare] load_dataset({ds_id!r}) failed: {e}")
    if raw is None:
        raise RuntimeError("could not load wikitext-2")

    def chunk(split):
        ids = tok("\n".join(t for t in raw[split]["text"] if t.strip()),
                  return_tensors="pt").input_ids[0]
        n = ids.numel() // args.seq_len
        ids = ids[: n * args.seq_len].view(n, args.seq_len)
        return [{"input_ids": ids[i:i + args.batch_size], "labels": ids[i:i + args.batch_size]}
                for i in range(0, n, args.batch_size)]
    return teacher, chunk("train"), chunk("validation")[: args.eval_batches]


@torch.no_grad()
def eval_ppl(model, val, device):
    model.eval().to(device)
    nll = ntok = 0
    for b in val:
        ids = b["input_ids"].to(device)
        lg = model(input_ids=ids).logits[:, :-1].contiguous()
        lab = ids[:, 1:].contiguous()
        nll += float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.reshape(-1),
                                     reduction="sum"))
        ntok += lab.numel()
    return math.exp(min(20, nll / max(1, ntok)))


# variant -> FSDConfig kwargs. `dobi_svd` shares cpsd_full's config but is run two-phase
# (see run_variant): reconstruction-driven rank SELECTION, then KD fine-tune — the faithful
# Dobi-SVD contrast, isolating *what drives the rank* (reconstruction vs KD).
VARIANTS = {
    "r1_kd_random": {"absorbed_init": False},
    "r2_fasd":      {"absorbed_init": True},
    "cpsd_mt":      {"use_cpsd_factored": True},
    "cpsd_full":    {"use_cpsd_factored": True, "use_diff_rank": True},
    "dobi_svd":     {"use_cpsd_factored": True, "use_diff_rank": True},
}


def _make_pipe(name, teacher, profile, calib, args, total_steps):
    cfg_kw = VARIANTS[name]
    config = substill.FSDConfig(
        arch_multiplier=args.arch_multiplier, generative_kd="forward_kl",
        total_steps=total_steps, lr=args.lr,
        distill_kwargs={"on_policy_start": 2.0, "quantize": False},
        **cfg_kw,
    )
    pipe = substill.FSDPipeline(teacher, config=config)
    pipe.profile, pipe._calib = profile, calib          # reuse the one profile
    pipe.build()
    return pipe, cfg_kw


def run_variant(name, teacher, profile, calib, train, val, args):
    from substill.training.distill import distill
    torch.manual_seed(args.seed)

    if name == "dobi_svd":
        # Faithful Dobi-SVD foil = reconstruction-driven rank, then KD fine-tune.
        # Phase A: train the differentiable-rank gates on feature *reconstruction* only
        # (KD/CE weight 0) to SELECT ranks, then harden+fold. Phase B: KD fine-tune the
        # resulting fixed-rank student — same KD budget contributes, but the rank was NOT
        # chosen by the KD signal. Contrast: cpsd_full co-trains the rank WITH KD.
        recon_steps = max(1, int(0.4 * args.steps))
        kd_steps = max(1, args.steps - recon_steps)
        pipe, _ = _make_pipe(name, teacher, profile, calib, args, recon_steps)
        pipe.train(train, delta=0.0, alpha=0.0, beta=1.0)   # reconstruction-only rank select
        pipe.fold_for_inference()                            # harden recon-selected ranks
        distill(teacher, pipe.student, train, profile=profile,
                generative_kd="forward_kl", total_steps=kd_steps, lr=args.lr,
                on_policy_start=2.0)                          # KD fine-tune the fixed-rank student
        student = pipe.student
    else:
        pipe, cfg_kw = _make_pipe(name, teacher, profile, calib, args, args.steps)
        pipe.train(train)
        if cfg_kw.get("use_cpsd_factored"):
            pipe.fold_for_inference()                        # collapse to plain nn.Linear
        student = pipe.student

    ppl = eval_ppl(student, val, args.device)
    s_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    return {"variant": name, "final_ppl": float(ppl), "status": "ok",
            "compression_ratio": t_params / max(1, s_params),
            "params": int(s_params)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    print(f"[compare] teacher PPL = {t_ppl:.2f}")
    profile = substill.profile(teacher, train[: args.calib_batches])

    results = []
    for name in args.variants:
        try:
            r = run_variant(name, teacher, profile, train[: args.calib_batches],
                            train, val, args)
            print(f"[compare] {name:<14} PPL={r['final_ppl']:.2f}  "
                  f"ratio={r['compression_ratio']:.2f}x")
        except Exception as e:  # noqa: BLE001
            print(f"[compare] {name} FAILED: {e}")
            r = {"variant": name, "status": "failed", "error": str(e)}
        results.append(r)

    summary = {"teacher": args.teacher, "teacher_ppl": float(t_ppl),
               "target_compression": args.target_compression, "seed": args.seed,
               "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[compare] wrote {args.output}")


if __name__ == "__main__":
    main()
