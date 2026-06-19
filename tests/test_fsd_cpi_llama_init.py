"""apply_cpi_attention_init: shared-per-group circuit-preserving re-init on Llama.

Validates the integration that fixes the GQA disjoint-basis bug (builders.py:483-485):
the re-init runs with correct shapes and yields finite output. (Whether it beats the
disjoint baseline on PPL is an empirical question tested at real scale on a pretrained
GQA model — see scripts/cpsd_cpi_init_eval.py.)
"""
from __future__ import annotations

import pytest
import torch


def _tiny_llama(hidden=128, heads=4, kv=2, layers=2):
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return None
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=kv, max_position_embeddings=64)
    return LlamaForCausalLM(cfg)


def _loader(n=8, B=2, T=8, vocab=64):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(5, vocab - 1, (B, T)),
             "attention_mask": torch.ones(B, T, dtype=torch.long)} for _ in range(n)]


def test_cpi_attention_init_runs_and_is_finite():
    teacher = _tiny_llama()
    if teacher is None:
        pytest.skip("transformers not installed")
    import fasd
    from fasd.compression.cpi import apply_cpi_attention_init

    loader = _loader()
    from fasd.compression.cpi import cpi_rank_map

    pipe = fasd.FSDPipeline(teacher, config=fasd.FSDConfig(template="llama"))
    pipe.run_profile(loader)
    pipe.config.rank_map = cpi_rank_map(teacher, pipe.profile, head_dim_ratio=0.5)
    student = pipe.build()  # disjoint baseline init, CPI-compatible config (keeps H, G)

    n = apply_cpi_attention_init(student, teacher, pipe.profile, loader, rope_aware=True)
    assert n == teacher.config.num_hidden_layers
    # Student still runs and produces finite logits after the circuit-preserving re-init.
    with torch.no_grad():
        out = student(**_loader(n=1)[0]).logits
    assert torch.isfinite(out).all()


def test_ov_align_init_runs_and_preserves_forward_finiteness():
    teacher = _tiny_llama()
    if teacher is None:
        pytest.skip("transformers not installed")
    import fasd
    from fasd.compression.cpi import apply_ov_align_init, cpi_rank_map

    loader = _loader()
    pipe = fasd.FSDPipeline(teacher, config=fasd.FSDConfig(template="llama"))
    pipe.run_profile(loader)
    pipe.config.rank_map = cpi_rank_map(teacher, pipe.profile, head_dim_ratio=0.5)
    student = pipe.build()
    n = apply_ov_align_init(student, teacher, pipe.profile, loader)
    assert n == teacher.config.num_hidden_layers
    with torch.no_grad():
        assert torch.isfinite(student(**_loader(n=1)[0]).logits).all()


def test_cpi_attention_init_both_rope_modes():
    teacher = _tiny_llama()
    if teacher is None:
        pytest.skip("transformers not installed")
    import fasd
    from fasd.compression.cpi import apply_cpi_attention_init

    from fasd.compression.cpi import cpi_rank_map

    loader = _loader()
    for rope_aware in (True, False):
        pipe = fasd.FSDPipeline(teacher, config=fasd.FSDConfig(template="llama"))
        pipe.run_profile(loader)
        pipe.config.rank_map = cpi_rank_map(teacher, pipe.profile, head_dim_ratio=0.5)
        student = pipe.build()
        apply_cpi_attention_init(student, teacher, pipe.profile, loader, rope_aware=rope_aware)
        with torch.no_grad():
            assert torch.isfinite(student(**_loader(n=1)[0]).logits).all()
