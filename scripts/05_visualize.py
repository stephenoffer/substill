#!/usr/bin/env python3
"""Generate all visualization plots from profiling and training results."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asd.profiling.svd_analysis import load_profiles
from asd.utils.visualization import (
    plot_svd_spectrum,
    plot_compression_ratios,
    plot_training_curves,
)


def main():
    parser = argparse.ArgumentParser(description="Generate ASD visualization plots")
    parser.add_argument("--profiles", default="outputs/profiles.pt")
    parser.add_argument("--history", default="outputs/training/history.json")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Profile visualizations
    if os.path.exists(args.profiles):
        print("Generating profile visualizations...")
        profiles = load_profiles(args.profiles)
        plot_svd_spectrum(profiles, os.path.join(args.output_dir, "svd_spectrum.png"))
        plot_compression_ratios(profiles, os.path.join(args.output_dir, "compression_ratios.png"))
    else:
        print(f"Profiles not found at {args.profiles}, skipping profile plots")

    # Training visualizations
    if os.path.exists(args.history):
        print("Generating training visualizations...")
        plot_training_curves(args.history, os.path.join(args.output_dir, "training_curves.png"))
    else:
        print(f"History not found at {args.history}, skipping training plots")

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
