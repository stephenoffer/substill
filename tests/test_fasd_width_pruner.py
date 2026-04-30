"""Width-pruner produces valid transformer configs from a profile."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fasd.api import BranchProfile, TeacherProfile
from fasd.compression.width_pruner import (
    contiguous_layer_mapping,
    plan_progressive_stages,
    profile_to_student_config,
)


@dataclass
class FakeConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    num_hidden_layers: int = 12


def _mk_profile(ranks: dict[str, int], hidden=768, interm=3072):
    branches = []
    # residual
    V = torch.eye(hidden)
    eig = torch.ones(hidden)
    branches.append(
        BranchProfile(
            name="model.layers.0.residual",
            kind="block.residual",
            module_path="model.layers.0",
            principal_components=V,
            eigenvalues=eig,
            behavioral_rank=ranks.get("block.residual", hidden),
            variance_rank=hidden,
            channels=hidden,
        )
    )
    # ffn up
    V2 = torch.eye(interm)
    branches.append(
        BranchProfile(
            name="model.layers.0.ffn.up",
            kind="ffn.up",
            module_path="model.layers.0.mlp.up_proj",
            principal_components=V2,
            eigenvalues=torch.ones(interm),
            behavioral_rank=ranks.get("ffn.up", interm),
            variance_rank=interm,
            channels=interm,
        )
    )
    # attn.k (for GQA test)
    V3 = torch.eye(hidden)
    branches.append(
        BranchProfile(
            name="model.layers.0.attn.k",
            kind="attn.k",
            module_path="model.layers.0.self_attn.k_proj",
            principal_components=V3,
            eigenvalues=torch.ones(hidden),
            behavioral_rank=ranks.get("attn.k", hidden),
            variance_rank=hidden,
            channels=hidden,
        )
    )
    return TeacherProfile(branches=branches)


def test_profile_to_student_config_respects_head_divisibility():
    profile = _mk_profile({"block.residual": 256, "ffn.up": 1024, "attn.k": 128})
    cfg = profile_to_student_config(profile, teacher_config=FakeConfig())
    assert cfg.hidden_size % cfg.num_attention_heads == 0
    assert cfg.intermediate_size >= cfg.hidden_size
    assert cfg.num_hidden_layers == 12  # depth_policy="keep" default
    # KV heads must evenly divide attention heads.
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0


def test_plan_progressive_stages_returns_monotone_chain():
    teacher_cfg = FakeConfig()
    # Target is 6x smaller than teacher.
    from fasd.compression.width_pruner import StudentConfig

    target = StudentConfig(
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=12,
    )
    stages = plan_progressive_stages(teacher_cfg, target, max_single_step=2.0)
    assert len(stages) >= 2
    # Each stage must be smaller or equal to the previous.
    last = teacher_cfg.hidden_size
    for s in stages:
        assert s.hidden_size <= last
        last = s.hidden_size
    assert stages[-1].hidden_size == target.hidden_size


def test_contiguous_layer_mapping_tail():
    m = contiguous_layer_mapping(12, 6, "contiguous_tail")
    assert m == [0, 1, 2, 3, 4, 5]


def test_contiguous_layer_mapping_middle():
    m = contiguous_layer_mapping(12, 6, "contiguous_middle")
    # Keep 3 from start + 3 from end
    assert len(m) == 6
    assert m[:3] == [0, 1, 2]
    assert m[-1] == 11
    # No scattered drops: chosen indices are at the ends.
    assert max(m[:3]) < min(m[3:])
