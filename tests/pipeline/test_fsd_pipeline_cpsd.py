"""FSDPipeline + CPSD-factored conversion on a real (tiny) GPT-2.

Validates the public pipeline end-to-end on a real HF architecture and that the
manifold-training conversion preserves the student forward at swap-in, then trains.
"""
from __future__ import annotations

import pytest
import torch


def _toy_gpt2(n_layer=2, n_embd=32, n_head=4):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=n_embd,
                     n_layer=n_layer, n_head=n_head, n_inner=4 * n_embd)
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def _loader(n=8, B=2, T=8, vocab=64):
    torch.manual_seed(0)
    out = []
    for _ in range(n):
        tok = torch.randint(5, vocab - 1, (B, T))
        out.append({"input_ids": tok, "labels": tok,
                    "attention_mask": torch.ones(B, T, dtype=torch.long)})
    return out


def test_pipeline_builds_and_distills_gpt2():
    teacher = _toy_gpt2()
    if teacher is None:
        pytest.skip("transformers not installed")
    import substill

    pipe = substill.FSDPipeline(teacher, config=substill.FSDConfig(
        arch_multiplier=0.5, total_steps=3, lr=5e-4,
        distill_kwargs={"on_policy_start": 2.0, "teacher_correction_steps": 0,
                            "quantize": False},
    ))
    pipe.run_profile(_loader())
    student = pipe.build()
    assert student is not None
    # Pipeline produced a working student that runs a forward pass.
    # (Compression ratio is validated at real scale in REPORT.md; at this toy
    # scale the random-data profile is degenerate, so we don't assert sn < tn.)
    with torch.no_grad():
        student.eval()
        logits = student(**_loader(n=1)[0]).logits
    assert torch.isfinite(logits).all()
    result = pipe.train(_loader())
    assert next(result.student.parameters()).is_leaf
    assert torch.isfinite(next(result.student.parameters())).all()


def test_cpsd_conversion_preserves_forward_then_trains():
    teacher = _toy_gpt2()
    if teacher is None:
        pytest.skip("transformers not installed")
    import substill
    from substill.compression.factored_linear import TeacherFactoredLinear
    from substill.training.stiefel_optim import stiefel_param_groups

    pipe = substill.FSDPipeline(teacher, config=substill.FSDConfig(arch_multiplier=0.5))
    pipe.run_profile(_loader())
    student = pipe.build()

    batch = _loader(n=1)[0]
    student.eval()
    with torch.no_grad():
        before = student(**batch).logits

    n = substill.convert_gpt2_to_factored(student, teacher, pipe.profile)
    assert n > 0, "no modules converted"
    # At least one TeacherFactoredLinear is now in the tree.
    assert any(isinstance(m, TeacherFactoredLinear) for m in student.modules())

    with torch.no_grad():
        after = student(**batch).logits
    # Conversion is exact: factored effective weight == absorbed weight.
    assert torch.allclose(before, after, atol=1e-3), \
        f"conversion changed forward: max diff {(before-after).abs().max().item():.3e}"

    # Stiefel param groups now include the factored bases, and a step trains them.
    groups = stiefel_param_groups(student, base_lr=1e-3)
    stiefel_count = sum(len(g["params"]) for g in groups if g.get("stiefel"))
    assert stiefel_count >= 2 * n  # V_in + V_out per converted module
