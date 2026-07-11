"""Tests for the `arch_multiplier` knob in `profiles_to_stage_widths`."""

from __future__ import annotations

import pytest
import torch

from substill._asd.profiling.svd_analysis import (
    LayerProfile,
    profiles_to_stage_widths,
)


def _mk_profile(channels: int, rank: int) -> LayerProfile:
    return LayerProfile(
        name=f"layer_{channels}",
        eigenvalues=torch.linspace(float(channels), 0.1, channels),
        principal_components=torch.eye(channels)[:, :rank],
        effective_rank=rank,
        total_channels=channels,
        compression_ratio=rank / channels,
        source="output",
    )


def test_default_multiplier_reproduces_legacy_widths():
    profiles = [_mk_profile(64, 32), _mk_profile(128, 50)]
    widths_legacy = profiles_to_stage_widths(
        profiles, min_width=16, width_multiple=8,
    )
    widths_explicit = profiles_to_stage_widths(
        profiles, min_width=16, width_multiple=8,
        arch_multiplier=1.0,
    )
    assert widths_legacy == widths_explicit


def test_multiplier_scales_widths():
    """With arch_multiplier=2.0, student width should be roughly 2x the
    effective rank (rounded up to width_multiple)."""
    profiles = [_mk_profile(64, 32), _mk_profile(128, 48)]
    widths = profiles_to_stage_widths(
        profiles, min_width=16, width_multiple=8,
        arch_multiplier=2.0,
    )
    # ceil(32 * 2) = 64 rounded up to multiple of 8 = 64
    # ceil(48 * 2) = 96 rounded up to multiple of 8 = 96
    assert widths == [64, 96]


def test_arch_min_enforced():
    """arch_min floors the per-stage width even for tiny ranks."""
    profiles = [_mk_profile(64, 2), _mk_profile(128, 4)]
    widths = profiles_to_stage_widths(
        profiles, min_width=16, width_multiple=8,
        arch_min=64,
    )
    # min_width=16 would give rounding up from 2 → 8, 4 → 8. arch_min=64
    # overrides to 64.
    assert widths == [64, 64]


def test_invalid_multiplier_raises():
    profiles = [_mk_profile(64, 32)]
    with pytest.raises(ValueError):
        profiles_to_stage_widths(
            profiles, arch_multiplier=0.0,
        )
    with pytest.raises(ValueError):
        profiles_to_stage_widths(
            profiles, arch_multiplier=-1.0,
        )


def test_decoupling_keeps_k_loss_unchanged():
    """The effective_rank stored on each profile (k_loss) must not be
    touched by changing arch_multiplier — the loss side reads that, not
    the computed widths."""
    profiles = [_mk_profile(64, 32), _mk_profile(128, 48)]
    # Compute widths with different multipliers.
    _ = profiles_to_stage_widths(profiles, arch_multiplier=1.0)
    _ = profiles_to_stage_widths(profiles, arch_multiplier=2.0)
    _ = profiles_to_stage_widths(profiles, arch_multiplier=0.5)
    # profiles' effective_rank unchanged.
    assert profiles[0].effective_rank == 32
    assert profiles[1].effective_rank == 48
