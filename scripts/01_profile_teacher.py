#!/usr/bin/env python3
"""Phase 1: Profile teacher activations — capture covariance, compute SVD, analyze sparsity.

PREREQUISITE: Run scripts/00_finetune_teacher.py first to produce fine-tuned weights.
Profiling an unfine-tuned teacher produces meaningless activation statistics.
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.data.cifar10 import get_cifar10_loaders
from asd.models.teacher import TeacherWrapper
from asd.profiling.activation_capture import ActivationCaptureEngine, get_resnet50_layer_names
from asd.profiling.sparsity_analysis import SparsityAnalyzer
from asd.profiling.svd_analysis import SVDAnalyzer, save_profiles, profiles_to_stage_widths


def main():
    parser = argparse.ArgumentParser(description="Profile ResNet50 teacher activations")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--teacher-weights", default=None, help="Override teacher weights path")
    parser.add_argument("--output", default="outputs/profiles.pt", help="Output path for profiles")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load data
    print("Loading CIFAR-10 calibration data...")
    loaders = get_cifar10_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation="none",  # No augmentation for profiling
        calibration_samples=cfg.profiling.num_calibration_samples,
    )
    calib_loader = loaders["calibration"]
    print(f"  Calibration samples: {len(calib_loader.dataset)}")

    # Load teacher with fine-tuned weights
    weights_path = args.teacher_weights or cfg.teacher.weights_path
    print("Loading ResNet50 teacher...")
    teacher = TeacherWrapper(
        profiles=None,
        cifar_stem=cfg.teacher.cifar_stem,
        pretrained=False,  # We load our own weights
        freeze=True,
    )

    if os.path.exists(weights_path):
        print(f"  Loading fine-tuned weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        teacher.backbone.load_state_dict(state_dict)
    else:
        print(f"  WARNING: Fine-tuned weights not found at {weights_path}")
        print(f"  Run scripts/00_finetune_teacher.py first!")
        print(f"  Falling back to pretrained ImageNet weights with random CIFAR stem.")
        print(f"  Activation profiles will be UNRELIABLE.\n")
        # Rebuild with pretrained weights as fallback
        teacher = TeacherWrapper(
            profiles=None,
            cifar_stem=cfg.teacher.cifar_stem,
            pretrained=True,
            freeze=True,
        )

    # Capture activations
    print("Capturing activations...")
    layer_names = get_resnet50_layer_names()
    print(f"  Hooking {len(layer_names)} layers: {layer_names}")

    cov_mode = cfg.profiling.get("covariance_mode", "per_pixel")
    print(f"  Covariance mode: {cov_mode}")
    engine = ActivationCaptureEngine(
        teacher.backbone, layer_names, covariance_mode=cov_mode,
    )
    accumulators = engine.run(calib_loader, device=device)

    # Analyze each layer
    print("\nAnalyzing activation profiles...")
    svd_analyzer = SVDAnalyzer(
        variance_threshold=cfg.profiling.variance_threshold,
        definition=cfg.profiling.get("rank_definition", "variance"),
    )
    sparsity_analyzer = SparsityAnalyzer(num_bins=cfg.profiling.histogram_bins)

    profiles = []
    print(f"\n{'Layer':<15} {'Channels':>10} {'Eff. Rank':>10} {'Ratio':>10} {'Sparsity':>10} {'Entropy':>10}")
    print("-" * 70)

    for name in layer_names:
        acc = accumulators[name]
        cov = acc.finalize()

        sparsity_stats = sparsity_analyzer.analyze(
            sparsity_ratio=acc.sparsity_ratio,
            activation_sample=acc.get_activation_sample(),
        )

        profile = svd_analyzer.analyze(name, cov, sparsity_stats)
        profiles.append(profile)

        print(
            f"{name:<15} {profile.total_channels:>10} {profile.effective_rank:>10} "
            f"{profile.compression_ratio:>10.3f} {sparsity_stats.sparsity_ratio:>10.3f} "
            f"{sparsity_stats.entropy:>10.3f}"
        )

    # Compute stage widths
    stage_widths = profiles_to_stage_widths(
        profiles,
        min_width=cfg.student.min_width,
        width_multiple=cfg.student.width_multiple,
    )
    print(f"\n{'=' * 70}")
    print(f"Teacher stage channels: {TeacherWrapper.STAGE_CHANNELS}")
    print(f"Student stage widths:   {stage_widths}")
    print(f"Compression ratios:     {[f'{s/t:.2f}' for s, t in zip(stage_widths, TeacherWrapper.STAGE_CHANNELS)]}")

    # Save profiles
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_profiles(profiles, args.output)
    print(f"\nProfiles saved to {args.output}")


if __name__ == "__main__":
    main()
