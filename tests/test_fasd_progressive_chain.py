"""Progressive chain planner shrinks monotonically."""

from __future__ import annotations

from dataclasses import dataclass

from fasd.compression.width_pruner import StudentConfig, plan_progressive_stages


@dataclass
class FakeConfig:
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    num_hidden_layers: int = 8


def test_progressive_chain_short_circuit():
    # No big compression → 1 stage.
    target = StudentConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=8,
    )
    stages = plan_progressive_stages(FakeConfig(), target, max_single_step=3.0)
    assert len(stages) == 1


def test_progressive_chain_long_path():
    target = StudentConfig(
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=4,
    )
    stages = plan_progressive_stages(FakeConfig(), target, max_single_step=2.0)
    assert len(stages) >= 2
    sizes = [s.hidden_size for s in stages]
    assert sizes == sorted(sizes, reverse=True)
    assert stages[-1].hidden_size == 64
