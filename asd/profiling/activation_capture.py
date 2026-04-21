"""Hook-based activation capture with memory-efficient covariance accumulation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


class CovarianceAccumulator:
    """Accumulates channel-wise covariance in O(C^2) memory.

    Instead of storing all activation tensors (GBs), we maintain a running
    sum-of-outer-products and mean, then finalize to a covariance matrix.
    """

    def __init__(self, num_channels: int, device: str = "cpu"):
        self.num_channels = num_channels
        self.device = device
        self.n = 0
        self.sum_x = torch.zeros(num_channels, device=device, dtype=torch.float64)
        self.sum_xx = torch.zeros(num_channels, num_channels, device=device, dtype=torch.float64)
        # Running histogram for sparsity analysis
        self.num_zeros = 0
        self.num_total = 0
        self.activation_values: list[Tensor] = []  # small buffer for histogram
        self._hist_budget = 100_000  # max values to store for histogram
        self._hist_stored = 0  # track actual number of stored values

    def update(self, activation: Tensor) -> None:
        """Update with a batch of activations, shape (B, C, H, W) or (B, C)."""
        act = activation.detach().float()

        # Track sparsity (before pooling, on raw activations)
        self.num_zeros += int((act == 0).sum().item())
        self.num_total += act.numel()

        # Store a subsample of values for histogram computation
        if self._hist_stored < self._hist_budget:
            flat = act.flatten()
            # Subsample ~1000 values per batch
            step = max(1, flat.shape[0] // 1000)
            sample = flat[::step].cpu()
            self.activation_values.append(sample)
            self._hist_stored += len(sample)

        # Global average pool if spatial dims present
        if act.dim() == 4:
            act = act.mean(dim=(2, 3))  # (B, C)

        act = act.to(dtype=torch.float64, device=self.device)
        batch_size = act.shape[0]

        self.n += batch_size
        self.sum_x += act.sum(dim=0)
        self.sum_xx += act.T @ act  # (C, C)

    def finalize(self) -> Tensor:
        """Return (C, C) covariance matrix."""
        mean = self.sum_x / self.n
        cov = self.sum_xx / self.n - mean.unsqueeze(1) * mean.unsqueeze(0)
        return cov.float()

    @property
    def mean(self) -> Tensor:
        return (self.sum_x / self.n).float()

    @property
    def sparsity_ratio(self) -> float:
        if self.num_total == 0:
            return 0.0
        return self.num_zeros / self.num_total

    def get_activation_sample(self) -> Tensor:
        """Return concatenated subsample of activation values for histogram."""
        if not self.activation_values:
            return torch.zeros(1)
        return torch.cat(self.activation_values)


class ActivationCaptureEngine:
    """Registers forward hooks on specified layers to accumulate covariance matrices."""

    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.model = model
        self.layer_names = layer_names
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._accumulators: dict[str, CovarianceAccumulator] = {}
        self._initialized: set[str] = set()

    def _get_module(self, name: str) -> nn.Module:
        """Retrieve a submodule by dot-separated name (e.g. 'layer1.0')."""
        parts = name.split(".")
        mod = self.model
        for part in parts:
            if part.isdigit():
                mod = mod[int(part)]
            else:
                mod = getattr(mod, part)
        return mod

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            act = output
            if name not in self._initialized:
                # Lazily create accumulator with correct channel count
                # Covariance is always accumulated on CPU to save GPU memory
                if act.dim() == 4:
                    num_channels = act.shape[1]
                else:
                    num_channels = act.shape[-1]
                self._accumulators[name] = CovarianceAccumulator(num_channels, device="cpu")
                self._initialized.add(name)
            self._accumulators[name].update(act)
        return hook_fn

    def register_hooks(self) -> None:
        for name in self.layer_names:
            module = self._get_module(name)
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def run(self, dataloader: DataLoader, device: str = "cpu") -> dict[str, CovarianceAccumulator]:
        """Run the model on the dataloader and return per-layer covariance accumulators."""
        self.model.eval()
        self.model.to(device)
        self.register_hooks()

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Profiling activations"):
                images = images.to(device)
                self.model(images)

        self.cleanup()
        return self._accumulators

    def cleanup(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def get_resnet50_layer_names() -> list[str]:
    """Return the layer names to hook for ResNet50 (output of each residual block)."""
    names = []
    # layer1: 3 Bottleneck blocks
    for i in range(3):
        names.append(f"layer1.{i}")
    # layer2: 4 Bottleneck blocks
    for i in range(4):
        names.append(f"layer2.{i}")
    # layer3: 6 Bottleneck blocks
    for i in range(6):
        names.append(f"layer3.{i}")
    # layer4: 3 Bottleneck blocks
    for i in range(3):
        names.append(f"layer4.{i}")
    return names


def get_resnet50_stage_layer_names() -> dict[str, list[str]]:
    """Return layer names grouped by stage."""
    return {
        "stage1": [f"layer1.{i}" for i in range(3)],
        "stage2": [f"layer2.{i}" for i in range(4)],
        "stage3": [f"layer3.{i}" for i in range(6)],
        "stage4": [f"layer4.{i}" for i in range(3)],
    }
