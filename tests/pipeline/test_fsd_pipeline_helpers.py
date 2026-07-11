"""Tests for the FSD trainer helpers (`scripts/distill_llama32_fsd.py`).

These cover the integration layer — the helpers that compose the per-pillar
modules into a working pipeline. We test them against tiny synthetic models
because the actual Llama-3.2 path requires gated weights and an H100.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# The trainer script is intended to be run as a module; for testing we add it
# to sys.path so we can import its helpers directly.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "fsd"
sys.path.insert(0, str(SCRIPTS_DIR))

# Import the helpers we want to test.
from distill_llama32_fsd import inject_sparse_blocks, write_student_arch_json  # noqa: E402

from substill.compression.sparse_block import CorrectedLinear  # noqa: E402


def test_write_student_arch_json_records_llama_config(tmp_path):
    """A Llama-style student must emit hidden_size, intermediate_size, layers, heads, kv-heads."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    student = LlamaForCausalLM(cfg)
    write_student_arch_json(student, tmp_path)

    out = tmp_path / "student_arch.json"
    assert out.exists()
    arch = json.loads(out.read_text())
    assert arch["hidden_size"] == 32
    assert arch["intermediate_size"] == 64
    assert arch["num_hidden_layers"] == 2
    assert arch["num_attention_heads"] == 4
    assert arch["num_key_value_heads"] == 2


def test_write_student_arch_json_records_gpt2_config(tmp_path):
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_inner=128, n_positions=32)
    student = GPT2LMHeadModel(cfg)
    write_student_arch_json(student, tmp_path)
    arch = json.loads((tmp_path / "student_arch.json").read_text())
    # GPT-2's config exposes n_embd / n_layer / n_head / n_inner; our writer maps these.
    assert arch["hidden_size"] == 32
    assert arch["intermediate_size"] == 128
    assert arch["num_hidden_layers"] == 2
    assert arch["num_attention_heads"] == 4


def test_inject_sparse_blocks_replaces_llama_o_proj_and_down_proj():
    """Walk a tiny Llama model and verify o_proj / down_proj become CorrectedLinear."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,  # square so down_proj is eligible
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    student = LlamaForCausalLM(cfg)

    n = inject_sparse_blocks(student)
    # Each of 2 layers contributes one o_proj and one down_proj = 4.
    assert n == 4
    for layer in student.model.layers:
        assert isinstance(layer.self_attn.o_proj, CorrectedLinear)
        assert isinstance(layer.mlp.down_proj, CorrectedLinear)


def test_inject_sparse_blocks_zero_init_preserves_initial_output():
    """At injection time, the corrected linear's correction.weight is zero,
    so student forward equals the original (absorbed-init) forward."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    student = LlamaForCausalLM(cfg).eval()

    input_ids = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        logits_before = student(input_ids).logits.clone()

    n = inject_sparse_blocks(student)
    assert n >= 1

    with torch.no_grad():
        logits_after = student(input_ids).logits

    assert torch.allclose(logits_before, logits_after, atol=1e-5), (
        f"injection should be a no-op at zero correction; max diff = "
        f"{(logits_before - logits_after).abs().max().item():.3e}"
    )


def test_inject_sparse_blocks_skips_non_square_linears():
    """A linear with in_features != out_features must NOT be replaced."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"num_attention_heads": 4})()
            self.q_proj = nn.Linear(16, 16)  # square but not named o_proj/down_proj
            self.gate_proj = nn.Linear(16, 32)  # non-square — must skip
            self.o_proj = nn.Linear(16, 16)  # square — should replace
            self.down_proj = nn.Linear(32, 32)  # not used; isolated to test naming
            self.up_proj = nn.Linear(16, 32)  # non-square; named up_proj — must skip

    m = M()
    n = inject_sparse_blocks(m)
    # Must replace o_proj and the standalone down_proj.
    assert n == 2
    assert isinstance(m.o_proj, CorrectedLinear)
    assert isinstance(m.down_proj, CorrectedLinear)
    # Must leave non-square / wrong-named linears alone.
    assert isinstance(m.q_proj, nn.Linear)
    assert isinstance(m.gate_proj, nn.Linear)
    assert isinstance(m.up_proj, nn.Linear)


def test_inject_sparse_blocks_skips_when_heads_dont_divide():
    """If hidden_size % num_heads != 0, skip the linear — head structure undefined."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            # num_heads=3 doesn't divide hidden_size=16 evenly.
            self.config = type("C", (), {"num_attention_heads": 3})()
            self.o_proj = nn.Linear(16, 16)

    m = M()
    n = inject_sparse_blocks(m)
    assert n == 0
    assert isinstance(m.o_proj, nn.Linear)


def test_inject_sparse_blocks_copies_original_weights():
    """The pre-injection weight must be copied into the new CorrectedLinear's
    inner Linear, so the absorbed init is preserved."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"num_attention_heads": 4})()
            self.o_proj = nn.Linear(16, 16, bias=True)

    torch.manual_seed(0)
    m = M()
    w_orig = m.o_proj.weight.detach().clone()
    b_orig = m.o_proj.bias.detach().clone()

    inject_sparse_blocks(m)
    assert isinstance(m.o_proj, CorrectedLinear)
    assert torch.allclose(m.o_proj.linear.weight, w_orig)
    assert torch.allclose(m.o_proj.linear.bias, b_orig)
