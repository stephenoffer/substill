#!/usr/bin/env python3
"""Phase 2: Construct SlimNet student architecture from activation profiles."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.models.student import SlimNet
from asd.models.teacher import TeacherWrapper
from asd.profiling.svd_analysis import load_profiles, profiles_to_stage_widths


def main():
    parser = argparse.ArgumentParser(description="Build student architecture from profiles")
    parser.add_argument("--profiles", default="outputs/profiles.pt", help="Path to saved profiles")
    parser.add_argument("--min-width", type=int, default=16)
    parser.add_argument("--width-multiple", type=int, default=8)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    args = parser.parse_args()

    print("Loading activation profiles...")
    profiles = load_profiles(args.profiles)

    # Compute stage widths
    stage_widths = profiles_to_stage_widths(
        profiles,
        min_width=args.min_width,
        width_multiple=args.width_multiple,
    )

    print(f"\n{'=' * 60}")
    print("ACTIVATION RANK ANALYSIS")
    print(f"{'=' * 60}")
    for p in profiles:
        print(f"  {p.name:<15} channels={p.total_channels:>5}  rank={p.effective_rank:>5}  "
              f"ratio={p.compression_ratio:.3f}  sparsity={p.sparsity_stats.sparsity_ratio:.3f}")

    print(f"\n{'=' * 60}")
    print("STUDENT ARCHITECTURE")
    print(f"{'=' * 60}")

    teacher_channels = TeacherWrapper.STAGE_CHANNELS
    print(f"  Teacher channels: {teacher_channels}")
    print(f"  Student widths:   {stage_widths}")

    # Build student
    student = SlimNet(stage_widths, blocks_per_stage=args.blocks_per_stage)

    # Count parameters
    teacher_params = sum(p.numel() for p in TeacherWrapper(pretrained=False).parameters())
    student_params = student.count_parameters()

    print(f"\n  Teacher parameters: {teacher_params:>12,}")
    print(f"  Student parameters: {student_params:>12,}")
    print(f"  Compression:        {teacher_params / student_params:>12.1f}x")

    # Verify forward pass
    print(f"\nVerifying forward pass...")
    x = torch.randn(2, 3, 32, 32)
    logits, features = student(x)
    print(f"  Input:  {x.shape}")
    print(f"  Logits: {logits.shape}")
    for i, feat in enumerate(features):
        print(f"  Stage {i+1}: {feat.shape}")

    print(f"\n{student}")


if __name__ == "__main__":
    main()
