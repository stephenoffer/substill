"""Compress a tiny GPT-2 with the full FSDPipeline, end to end, on CPU.

Builds a small GPT-2 teacher from scratch (no download), profiles it on a handful
of random batches, constructs a half-width student initialized from the teacher's
own activation subspace, and runs a few distillation steps. Prints the parameter
counts and confirms the student produces finite logits.

Run:
    python examples/quickstart_pipeline.py
"""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import logging as hf_logging

import substill

hf_logging.set_verbosity_error()


def toy_teacher() -> GPT2LMHeadModel:
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=64, n_layer=2, n_head=4,
                     n_inner=256, bos_token_id=0, eos_token_id=0)
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg).eval()


def random_batches(n: int, batch: int = 2, seq: int = 8, vocab: int = 64) -> list[dict]:
    torch.manual_seed(0)
    batches = []
    for _ in range(n):
        tokens = torch.randint(5, vocab - 1, (batch, seq))
        batches.append({"input_ids": tokens, "labels": tokens,
                        "attention_mask": torch.ones(batch, seq, dtype=torch.long)})
    return batches


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    teacher = toy_teacher()
    calib, train = random_batches(8), random_batches(8)

    pipe = substill.FSDPipeline(teacher, config=substill.FSDConfig(
        arch_multiplier=0.5,      # student is half the teacher's width
        total_steps=5,
        lr=5e-4,
        distill_kwargs={"teacher_correction_steps": 0, "quantize": False},
    ))
    result = pipe.run(calib, train)
    student = result.student

    with torch.no_grad():
        student.eval()
        logits = student(**random_batches(1)[0]).logits

    print(f"teacher parameters: {count_params(teacher):>10,}")
    print(f"student parameters: {count_params(student):>10,}")
    print(f"student logits finite: {bool(torch.isfinite(logits).all())}")


if __name__ == "__main__":
    main()
