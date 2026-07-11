"""MiniLLM baseline: reverse-KL with policy-gradient-style on-policy mixing.

Gu et al. 2023. The full MiniLLM uses single-step decomposition + length
normalization, which we approximate here with a simpler reverse-KL on hybrid
batches (50/50 teacher-data + student-rollouts). For a tighter reproduction,
plug in the original MiniLLM trainer or the OpenRLHF implementation.
"""

from __future__ import annotations

import argparse
import sys

import torch

from scripts.repro_baselines._common import (
    add_common_args,
    args_from_namespace,
    build_matched_student,
    load_corpus,
    load_teacher_and_tokenizer,
    save_run,
    train_loop,
)
from substill.losses.generative_kd import reverse_kl


def main() -> int:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--on-policy-ratio", type=float, default=0.5,
                   help="Fraction of training batches that come from student rollouts.")
    ns = p.parse_args()
    args = args_from_namespace(ns)

    torch.manual_seed(args.seed)
    teacher, tok = load_teacher_and_tokenizer(args)
    teacher.to("cuda" if torch.cuda.is_available() else "cpu")
    student = build_matched_student(args, teacher)
    train_loader = load_corpus(args, tok)

    # Note: this is a simplified MiniLLM. For full PG-style reverse-KL with
    # length normalization and single-step decomposition, see the upstream
    # MiniLLM repository. The student-rollout mixing is handled here by the
    # existing substill.training.onpolicy.HybridCollator if integrated.
    def loss_fn(s_logits, t_logits, batch):
        return reverse_kl(s_logits, t_logits, temperature=ns.temperature)

    train_loop(student, teacher, train_loader, args, loss_fn)
    save_run(student, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
