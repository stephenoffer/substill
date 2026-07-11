"""DistiLLM baseline: skew-KL distillation.

Ko et al. 2024. Per-token skew-KL with fixed alpha (typically 0.1) and
mode-seeking direction (student → mix).
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
from substill.losses.generative_kd import skew_kl


def main() -> int:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--direction", type=str, default="student_to_mix",
                   choices=["student_to_mix", "teacher_to_mix"])
    ns = p.parse_args()
    args = args_from_namespace(ns)

    torch.manual_seed(args.seed)
    teacher, tok = load_teacher_and_tokenizer(args)
    teacher.to("cuda" if torch.cuda.is_available() else "cpu")
    student = build_matched_student(args, teacher)
    train_loader = load_corpus(args, tok)

    def loss_fn(s_logits, t_logits, batch):
        return skew_kl(s_logits, t_logits, alpha=ns.alpha,
                       direction=ns.direction, temperature=ns.temperature)

    train_loop(student, teacher, train_loader, args, loss_fn)
    save_run(student, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
