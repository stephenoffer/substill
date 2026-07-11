"""GKD baseline: generalised JSD on student-generated sequences.

Agarwal et al. 2024. On-policy distillation; the student generates sequences
during training, teacher provides supervision on those sequences. JSD-α
interpolates between forward-KL (α=0) and reverse-KL (α=1).

This is a simplified rolling implementation. For tight reproduction, replace
``train_loop`` with one that draws batches half from the corpus and half from
fresh student rollouts each step (using
:mod:`substill.training.onpolicy.HybridCollator`).
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from scripts.repro_baselines._common import (
    add_common_args,
    args_from_namespace,
    build_matched_student,
    load_corpus,
    load_teacher_and_tokenizer,
    save_run,
    train_loop,
)


def jsd_alpha(s_logits, t_logits, alpha: float = 0.1, temperature: float = 1.0):
    """Generalised JSD: interpolate between forward-KL (α=0) and reverse-KL (α=1)."""
    t_logp = F.log_softmax(t_logits / temperature, dim=-1)
    s_logp = F.log_softmax(s_logits / temperature, dim=-1)
    t_p = t_logp.exp()
    s_p = s_logp.exp()
    mix = alpha * s_p + (1.0 - alpha) * t_p
    log_mix = mix.clamp_min(1e-12).log()
    forward_term = (t_p * (t_logp - log_mix)).sum(dim=-1).mean()
    reverse_term = (s_p * (s_logp - log_mix)).sum(dim=-1).mean()
    return (1.0 - alpha) * forward_term + alpha * reverse_term


def main() -> int:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)
    ns = p.parse_args()
    args = args_from_namespace(ns)

    torch.manual_seed(args.seed)
    teacher, tok = load_teacher_and_tokenizer(args)
    teacher.to("cuda" if torch.cuda.is_available() else "cpu")
    student = build_matched_student(args, teacher)
    train_loader = load_corpus(args, tok)

    def loss_fn(s_logits, t_logits, batch):
        return jsd_alpha(s_logits, t_logits, alpha=ns.alpha, temperature=ns.temperature)

    train_loop(student, teacher, train_loader, args, loss_fn)
    save_run(student, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
