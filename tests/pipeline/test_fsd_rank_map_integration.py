"""Tests for the rank-map → builder integration.

Verifies that ``profile_to_student_config(..., rank_map=...)`` overrides
each branch's stored ``behavioral_rank`` and that ``build_student`` passes
the rank-map through end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from substill.compression.width_pruner import (
    profile_to_student_config,
)


@dataclass
class _FakeBranch:
    name: str
    kind: str
    behavioral_rank: int


@dataclass
class _FakeProfile:
    branches: list


@dataclass
class _FakeTeacherConfig:
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    num_hidden_layers: int = 6


def test_rank_map_overrides_residual_branch():
    """A rank_map entry for the residual branch should override its stored rank."""
    profile = _FakeProfile(branches=[
        _FakeBranch("transformer.h.0.block.residual", "block.residual", 256),
    ])
    cfg = _FakeTeacherConfig()
    out = profile_to_student_config(
        profile, teacher_config=cfg,
        rank_map={"transformer.h.0.block.residual": 64},
    )
    # hidden_size should be derived from the overridden rank, not 256.
    assert out.hidden_size <= 64 or out.hidden_size == 64


def test_rank_map_overrides_ffn_branches():
    """rank_map override should take effect on FFN intermediate, subject to the
    intermediate >= hidden floor in width_pruner."""
    profile = _FakeProfile(branches=[
        _FakeBranch("h.0.ffn.up", "ffn.up", 1024),
        _FakeBranch("h.0.ffn.gate", "ffn.gate", 1024),
        _FakeBranch("h.0.block.residual", "block.residual", 64),
    ])
    cfg = _FakeTeacherConfig()
    # Without rank_map: hidden ~ 64, intermediate ~ 1024.
    out_no_map = profile_to_student_config(profile, teacher_config=cfg)
    assert out_no_map.intermediate_size == 1024
    # With rank_map dropping FFN: hidden=64, intermediate=512 (overridden),
    # subject to min_hidden=64 and head_multiple=8 rounding.
    out_with_map = profile_to_student_config(
        profile, teacher_config=cfg,
        rank_map={"h.0.ffn.up": 512, "h.0.ffn.gate": 512, "h.0.block.residual": 64},
    )
    # FFN override should reduce intermediate from 1024 to 512.
    assert out_with_map.intermediate_size == 512
    # And should not exceed the legacy (no rank_map) value.
    assert out_with_map.intermediate_size < out_no_map.intermediate_size


def test_rank_map_uses_branch_default_when_unset():
    """If a branch is missing from rank_map, behavioral_rank is used."""
    profile = _FakeProfile(branches=[
        _FakeBranch("h.0.block.residual", "block.residual", 192),
        _FakeBranch("h.0.attn.q", "attn.q", 192),
    ])
    cfg = _FakeTeacherConfig()
    # Rank-map covers only attn.q; residual falls back to behavioral_rank=192.
    out = profile_to_student_config(
        profile, teacher_config=cfg,
        rank_map={"h.0.attn.q": 64},
    )
    # hidden_size is driven by residual = 192 (fallback).
    # Rounded up to head_multiple=8 → 192.
    assert out.hidden_size == 192


def test_rank_map_disables_arch_multiplier():
    """When rank_map is provided, arch_multiplier should be ignored.

    Without rank_map: arch_multiplier=2.0 doubles the hidden_size.
    With rank_map: arch_multiplier should not scale further.
    """
    profile = _FakeProfile(branches=[
        _FakeBranch("h.0.block.residual", "block.residual", 64),
    ])
    cfg = _FakeTeacherConfig()
    out_no_map = profile_to_student_config(
        profile, teacher_config=cfg, arch_multiplier=2.0,
    )
    out_with_map = profile_to_student_config(
        profile, teacher_config=cfg, arch_multiplier=2.0,
        rank_map={"h.0.block.residual": 64},
    )
    # With rank_map, scaling is disabled — hidden_size should equal the rank,
    # rounded up to head_multiple = 8 → 64. Without rank_map, it should be 128.
    assert out_with_map.hidden_size == 64
    assert out_no_map.hidden_size == 128


def test_build_student_consumes_rank_map_end_to_end():
    """The full build_student → profile_to_student_config → student path."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    teacher_cfg = GPT2Config(
        vocab_size=128, n_embd=64, n_layer=2, n_head=4, n_inner=128, n_positions=64,
    )
    teacher = GPT2LMHeadModel(teacher_cfg)

    # Synthesise a minimal profile with required branches.
    @dataclass
    class B:
        name: str
        kind: str
        behavioral_rank: int
        principal_components: torch.Tensor
        eigenvalues: torch.Tensor
        slice: tuple | None = None
        module_path: str = ""

    branches = [
        B("transformer.h.0.block.residual", "block.residual", 64,
          torch.eye(64), torch.ones(64), module_path="transformer.h.0"),
        B("transformer.h.1.block.residual", "block.residual", 64,
          torch.eye(64), torch.ones(64), module_path="transformer.h.1"),
        B("transformer.h.0.ffn.up", "ffn.up", 128,
          torch.eye(128), torch.ones(128), module_path="transformer.h.0.mlp.c_fc"),
        B("transformer.h.1.ffn.up", "ffn.up", 128,
          torch.eye(128), torch.ones(128), module_path="transformer.h.1.mlp.c_fc"),
    ]

    @dataclass
    class P:
        branches: list

    profile = P(branches=branches)

    from substill.builders import build_student

    # Build with a rank-map that compresses residual but keeps FFN.
    # Note: width_pruner has min_hidden=64 by default, so smaller ranks floor.
    # To verify the rank_map is consulted, we use a residual rank that *exceeds*
    # the floor and observe the override taking effect on intermediate_size.
    rank_map = {
        "transformer.h.0.block.residual": 32,  # below min_hidden floor
        "transformer.h.1.block.residual": 32,
        "transformer.h.0.ffn.up": 96,  # above hidden floor
        "transformer.h.1.ffn.up": 96,
    }
    student = build_student(
        teacher, profile, absorbed_init=False, template="gpt2",
        rank_map=rank_map,
    )
    # hidden floors at min_hidden=64; FFN intermediate respects override at 96.
    assert student.config.n_embd == 64
    assert student.config.n_inner == 96


def test_rank_map_none_preserves_legacy_behavior():
    """When rank_map=None, the function behaves identically to the pre-change version."""
    profile = _FakeProfile(branches=[
        _FakeBranch("h.0.block.residual", "block.residual", 128),
    ])
    cfg = _FakeTeacherConfig()
    out_no_map = profile_to_student_config(
        profile, teacher_config=cfg, arch_multiplier=1.0, rank_map=None,
    )
    out_legacy = profile_to_student_config(
        profile, teacher_config=cfg, arch_multiplier=1.0,  # no rank_map kwarg
    )
    assert out_no_map.hidden_size == out_legacy.hidden_size
    assert out_no_map.intermediate_size == out_legacy.intermediate_size
