#!/usr/bin/env python3
"""F-ASD on a small Llama-architecture teacher.

Uses a tiny random Llama config by default so the script can run on CPU
for smoke testing. Point ``--teacher`` at a real TinyLlama checkpoint
(e.g. ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``) for a real run — will
need GPU and the full ``transformers`` + ``datasets`` stack.

Exercises:
    - Llama branch autodetection (q_proj / k_proj / v_proj / o_proj /
      gate_proj / up_proj / down_proj).
    - Grouped-query attention (``num_key_value_heads`` independent of
      ``num_attention_heads``).
    - Llama absorbed-init via :func:`fasd.build_student`.
    - On-policy rollouts + quantization-aware final stage.

Usage::

    python scripts/distill_tinyllama_fasd.py --max-steps 20 --batch-size 2 --quantize
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fasd  # noqa: E402


def _toy_llama(vocab_size=256, hidden=64, heads=4, kv_heads=2, layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg)


def _random_dataset(n_batches=32, batch_size=2, seq_len=32, vocab=256):
    torch.manual_seed(0)
    data = []
    for _ in range(n_batches):
        tokens = torch.randint(5, vocab - 5, (batch_size, seq_len))
        data.append(
            {
                "input_ids": tokens,
                "labels": tokens.clone(),
                "attention_mask": torch.ones_like(tokens),
            }
        )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=None, help="HF model id; None -> toy Llama")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank-tol", type=float, default=0.05)
    parser.add_argument("--on-policy-start", type=float, default=2.0)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.teacher is None:
        print("Using toy random Llama teacher (for CPU smoke testing).")
        teacher = _toy_llama(hidden=64, heads=4, kv_heads=2, layers=2)
        loader = _random_dataset(
            n_batches=max(32, args.max_steps + 8),
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab=256,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading teacher: {args.teacher}")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher)
        tok = AutoTokenizer.from_pretrained(args.teacher)
        # User should wire a real dataset. For the smoke path we fall back
        # to random tokens sampled from the tokenizer's vocab.
        loader = _random_dataset(
            n_batches=max(32, args.max_steps + 8),
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab=tok.vocab_size,
        )

    teacher.eval()
    teacher.to(args.device)
    loader = [{k: v.to(args.device) for k, v in b.items()} for b in loader]

    calib = loader[:8]

    print("Profiling Llama branches...")
    t0 = time.time()
    profile = fasd.profile(
        teacher,
        calib,
        mode="branch",
        rank_tol=args.rank_tol,
        token_weighting="entropy",
        n_calib_batches=len(calib),
        behavioral_calib_batches=4,
        device=args.device,
    )
    print(f"  {len(profile.branches)} branches, {time.time() - t0:.1f}s")
    for b in profile.branches[:7]:
        print(
            f"  {b.name}: behavioral_rank={b.behavioral_rank}/{b.channels} "
            f"(variance_rank={b.variance_rank})"
        )

    print("Building absorbed-init Llama student...")
    student = fasd.build_student(
        teacher, profile, absorbed_init=True, template="llama"
    )
    student.to(args.device)
    print(
        f"Student: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M vs "
        f"teacher {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M"
    )

    print("Running distill driver...")
    result = fasd.distill(
        teacher,
        student,
        loader,
        profile=profile,
        total_steps=args.max_steps,
        lr=args.lr,
        on_policy_start=args.on_policy_start,
        quantize=args.quantize,
        device=args.device,
    )
    if result.best_metric is not None:
        print(f"Student PPL: {result.best_metric:.2f}")


if __name__ == "__main__":
    main()
