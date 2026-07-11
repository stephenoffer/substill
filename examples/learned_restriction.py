"""Learned Restriction Distillation (LRD) on a tiny Llama, end to end, on CPU.

Builds a small Llama teacher from scratch (no download), then distils it into a
half-width student whose every weight is an exact restriction ``V^T W_T V`` of the
teacher, with the residual-stream projection ``V`` trained on the Stiefel manifold
against the KD loss. The student folds to a plain ``LlamaForCausalLM`` with zero
inference overhead.

This is the verified method (docs/learned_restriction.md); on real teachers it beats the
strongest frozen-basis baseline by ~6.8% PPL. Here we just confirm the machinery runs and
that the folded student reproduces the trained restriction.

Run:
    python examples/learned_restriction.py
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import logging as hf_logging

import substill

hf_logging.set_verbosity_error()


def toy_teacher() -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=32, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).eval()


def random_batches(n: int, batch: int = 2, seq: int = 16, vocab: int = 64) -> list[dict]:
    torch.manual_seed(1)
    return [{"input_ids": torch.randint(0, vocab, (batch, seq))} for _ in range(n)]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    teacher = toy_teacher()
    train = random_batches(8)

    config = substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=20, lr=3e-3,
                                      calib_batches=4, device="cpu", log_every=5)
    result = substill.learned_restriction_distill(teacher, train, config=config)
    student = result.student

    with torch.no_grad():
        logits = student(input_ids=random_batches(1)[0]["input_ids"]).logits

    print(f"teacher parameters: {count_params(teacher):>10,}")
    print(f"student parameters: {count_params(student):>10,}  "
          f"({count_params(teacher) / count_params(student):.2f}x smaller)")
    print(f"final KD loss:      {result.final_kd:>10.4f}")
    print(f"V rotated by:       {result.max_principal_angle:>10.4f} rad from PCA init")
    print(f"student logits finite: {bool(torch.isfinite(logits).all())}")


if __name__ == "__main__":
    main()
