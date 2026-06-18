"""CPSD-MT conversion on a real (tiny) GQA+RoPE Llama (task: continue / Llama path).

GPT-2 has no GQA/RoPE, so CPSD's circuit-preserving components never engage there.
This validates the Llama MT path: convert absorbed Llama linears to Stiefel-trainable
factored modules, preserving the forward exactly at conversion.
"""
from __future__ import annotations

import pytest
import torch


def _tiny_llama(n_layer=2, hidden=32, heads=4, kv=2):
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return None
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=n_layer, num_attention_heads=heads,
                      num_key_value_heads=kv, max_position_embeddings=32)
    return LlamaForCausalLM(cfg)


def _loader(n=8, B=2, T=8, vocab=64):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(5, vocab - 1, (B, T)),
             "attention_mask": torch.ones(B, T, dtype=torch.long)} for _ in range(n)]


def test_llama_cpsd_conversion_preserves_forward_gqa():
    teacher = _tiny_llama(kv=2)  # GQA: 4 heads, 2 kv groups
    if teacher is None:
        pytest.skip("transformers not installed")
    import fasd
    from fasd.compression.factored_linear import TeacherFactoredLinear
    from fasd.training.stiefel_optim import stiefel_param_groups

    pipe = fasd.FSDPipeline(teacher, config=fasd.FSDConfig(
        arch_multiplier=0.5, template="llama"))
    pipe.run_profile(_loader())
    student = pipe.build()

    batch = _loader(n=1)[0]
    student.eval()
    with torch.no_grad():
        before = student(**batch).logits

    n = fasd.convert_llama_to_factored(student, teacher, pipe.profile, free_core=False)
    assert n > 0
    assert any(isinstance(m, TeacherFactoredLinear) for m in student.modules())
    with torch.no_grad():
        after = student(**batch).logits
    assert torch.allclose(before, after, atol=1e-3), \
        f"conversion changed forward: max diff {(before-after).abs().max().item():.3e}"

    # Stiefel groups now include the factored bases.
    groups = stiefel_param_groups(student, base_lr=1e-3)
    stiefel_n = sum(len(g["params"]) for g in groups if g.get("stiefel"))
    assert stiefel_n >= 2 * n


def test_llama_cpsd_free_core_trains():
    teacher = _tiny_llama(kv=2)
    if teacher is None:
        pytest.skip("transformers not installed")
    import fasd
    from fasd.compression.factored_linear import TeacherFactoredLinear

    pipe = fasd.FSDPipeline(teacher, config=fasd.FSDConfig(
        arch_multiplier=0.5, template="llama"))
    pipe.run_profile(_loader())
    student = pipe.build()
    fasd.convert_llama_to_factored(student, teacher, pipe.profile, free_core=True)

    # free_core params exist and are trainable; the factored forward/backward runs.
    # (StiefelAdam manifold preservation is covered on GPT-2 + synthetic tests; here
    # the toy profile can inflate student dims so V may be (n<k) — not a valid Stiefel
    # point — so we train with plain AdamW, which exercises the Llama factored path.)
    bfree = [m.B_free for m in student.modules()
             if isinstance(m, TeacherFactoredLinear) and m.B_free is not None]
    assert bfree, "no free-core params created"
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    batch = _loader(n=1)[0]
    teacher.eval()
    student.train()
    with torch.no_grad():
        t_logits = teacher(**batch).logits
    s_logits = student(**batch).logits
    loss = fasd.forward_kl(s_logits, t_logits)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(b.grad is not None and torch.isfinite(b.grad).all() for b in bfree)
