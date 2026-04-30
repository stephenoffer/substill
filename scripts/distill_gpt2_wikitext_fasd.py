#!/usr/bin/env python3
"""End-to-end GPT-2 / WikiText-2 distillation using the ``fasd`` library.

Mirrors :mod:`scripts.distill_gpt2_wikitext` so a reader can diff the
two files to see the F-ASD design delta: branchwise profiling with
behavioral rank selection, absorbed-init student, Procrustes/CKA
schedule, skew-KL logit KD, optional on-policy and quantization
stages.

Usage::

    pip install transformers datasets
    python scripts/distill_gpt2_wikitext_fasd.py --max-steps 200 --batch-size 4

Flags match the plan's design:

    --rank-tol 0.02
    --generative-kd skew_kl  (or forward_kl / reverse_kl)
    --absorbed-init
    --teacher-correction-steps 100
    --on-policy-start 0.5
    --on-policy-ratio 0.5
    --contrastive-weight 0.0
    --quantize
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fasd  # noqa: E402


def get_dataloaders(batch_size: int, seq_len: int):
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    class _WT2(Dataset):
        def __init__(self, split, tokenizer, seq_len):
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [t for t in ds["text"] if t.strip()]
            ids = tokenizer.encode("\n\n".join(texts))
            n = len(ids) // seq_len
            self.tokens = torch.tensor(
                ids[: n * seq_len], dtype=torch.long
            ).view(n, seq_len)

        def __len__(self):
            return self.tokens.shape[0]

        def __getitem__(self, idx):
            t = self.tokens[idx]
            return {
                "input_ids": t,
                "labels": t,
                "attention_mask": torch.ones_like(t),
            }

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    return {
        "train": DataLoader(
            _WT2("train", tok, seq_len), batch_size=batch_size, shuffle=True
        ),
        "val": DataLoader(
            _WT2("validation", tok, seq_len), batch_size=batch_size
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="gpt2", help="HF model id")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank-tol", type=float, default=0.02)
    parser.add_argument(
        "--generative-kd",
        choices=["forward_kl", "reverse_kl", "skew_kl"],
        default="skew_kl",
    )
    parser.add_argument("--absorbed-init", action="store_true", default=True)
    parser.add_argument("--no-absorbed-init", dest="absorbed_init", action="store_false")
    parser.add_argument("--teacher-correction-steps", type=int, default=0)
    parser.add_argument("--on-policy-start", type=float, default=2.0,
                        help="set <1.0 to enable on-policy stage")
    parser.add_argument("--on-policy-ratio", type=float, default=0.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calib-batches", type=int, default=8)
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel

    print(f"Loading teacher: {args.teacher}")
    teacher = GPT2LMHeadModel.from_pretrained(args.teacher)
    teacher.eval()
    teacher.to(args.device)

    loaders = get_dataloaders(args.batch_size, args.seq_len)

    # Use a small subset for calibration to keep profiling fast.
    calib = []
    for i, b in enumerate(loaders["train"]):
        if i >= args.calib_batches:
            break
        calib.append({k: v.to(args.device) for k, v in b.items()})

    print(f"Profiling with {len(calib)} calibration batches...")
    t0 = time.time()
    profile = fasd.profile(
        teacher,
        calib,
        mode="branch",
        rank_tol=args.rank_tol,
        token_weighting="entropy",
        n_calib_batches=len(calib),
        behavioral_calib_batches=min(4, len(calib)),
        device=args.device,
    )
    print(f"Profile done in {time.time() - t0:.1f}s. {len(profile.branches)} branches.")
    for b in profile.branches[:6]:
        print(
            f"  {b.name}: behavioral_rank={b.behavioral_rank}/{b.channels} "
            f"(variance_rank={b.variance_rank})"
        )

    print("Building absorbed-init student...")
    student = fasd.build_student(
        teacher,
        profile,
        absorbed_init=args.absorbed_init,
        template="gpt2",
    )
    student.to(args.device)
    print(
        f"Student: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params "
        f"vs teacher {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M"
    )

    print("Running distill driver...")
    result = fasd.distill(
        teacher,
        student,
        loaders["train"],
        profile=profile,
        val_loader=loaders["val"],
        generative_kd=args.generative_kd,
        total_steps=args.max_steps,
        lr=args.lr,
        teacher_correction_steps=args.teacher_correction_steps,
        on_policy_start=args.on_policy_start,
        on_policy_ratio=args.on_policy_ratio,
        contrastive_weight=args.contrastive_weight,
        quantize=args.quantize,
        device=args.device,
    )
    print(
        f"Student val PPL: {result.best_metric:.2f}   "
        f"Teacher val PPL: {result.teacher_metric:.2f}"
    )


if __name__ == "__main__":
    main()
