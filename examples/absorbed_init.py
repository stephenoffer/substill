"""Compare absorbed init against random init before any training.

Absorbed init fills the student's linears with the teacher's weights projected
into the retained activation subspace, so the student starts close to the
teacher's function. This script profiles a tiny GPT-2, builds two students of
identical shape — one absorbed, one randomly initialized — and reports the KD
divergence of each to the teacher on a held-out batch. Absorbed init typically
starts with the lower divergence.

Run:
    python examples/absorbed_init.py
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


@torch.no_grad()
def kd_to_teacher(student: torch.nn.Module, teacher: torch.nn.Module, batch: dict) -> float:
    teacher.eval()
    student.eval()
    t_logits = teacher(**batch).logits
    s_logits = student(**batch).logits
    return float(substill.forward_kl(s_logits, t_logits))


def main() -> None:
    teacher = toy_teacher()
    profile = substill.profile(teacher, random_batches(8))
    holdout = random_batches(1)[0]

    absorbed = substill.build_student(teacher, profile, arch_multiplier=0.5, absorbed_init=True)
    random_student = substill.build_student(teacher, profile, arch_multiplier=0.5,
                                        absorbed_init=False)

    print(f"KD(absorbed -> teacher): {kd_to_teacher(absorbed, teacher, holdout):.3f}")
    print(f"KD(random   -> teacher): {kd_to_teacher(random_student, teacher, holdout):.3f}")


if __name__ == "__main__":
    main()
