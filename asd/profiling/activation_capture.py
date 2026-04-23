"""Hook-based activation capture with memory-efficient covariance accumulation."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


# Valid values for the `activation_source` knob. Naming is shared with the
# config, the saved profile metadata, and the LLM pipeline.
VALID_SOURCES = ("output", "delta", "branch")


class CovarianceAccumulator:
    """Accumulates channel-wise covariance in O(C^2) memory.

    Two aggregation modes determine how spatial activations are reduced before
    the outer product:

    - mode="per_pixel" (default): treat each spatial position as a sample in
      the C-dim channel space. Adjacent pixels share large receptive fields
      and are NOT i.i.d., so the raw (B·H·W) sample count overstates the
      effective sample size and biases the top eigenvalues upward. We
      optionally sub-sample spatial positions to mitigate this —
      `spatial_subsample` keeps every k-th pixel per image (default: all).

    - mode="gap": average over (H, W) first, then accumulate outer products.
      Legacy behavior — measures covariance of spatial means, not of the
      channel distribution at a pixel.

    Accumulation device defaults to the input activation's device (GPU when
    possible) — the previous CPU-only behavior forced a CUDA→CPU copy per
    batch and was a substantial profiling-speed regressor. Pass
    `device="cpu"` to force CPU if GPU memory is tight.

    `source` is a metadata tag ("output", "delta", "branch") describing what
    the accumulator is being fed. The math is source-agnostic; the tag
    travels with the profile so the loss side can refuse to mix a delta
    profile with an output-basis loss.
    """

    def __init__(
        self,
        num_channels: int,
        device: str | None = None,
        mode: str = "per_pixel",
        spatial_subsample: int = 1,
        source: str = "output",
    ):
        if mode not in ("per_pixel", "gap"):
            raise ValueError(f"mode must be 'per_pixel' or 'gap', got {mode!r}")
        if spatial_subsample < 1:
            raise ValueError(f"spatial_subsample must be ≥ 1, got {spatial_subsample}")
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
        self.num_channels = num_channels
        # None → match the activation's device on first update.
        self.device = device
        self.mode = mode
        self.spatial_subsample = spatial_subsample
        self.source = source
        self.n = 0
        self.sum_x: Tensor | None = None
        self.sum_xx: Tensor | None = None
        # Running histogram for sparsity analysis
        self.num_zeros = 0
        self.num_total = 0
        self.activation_values: list[Tensor] = []  # small buffer for histogram
        self._hist_budget = 100_000  # max values to store for histogram
        self._hist_stored = 0  # track actual number of stored values

    def _ensure_buffers(self, act: Tensor) -> None:
        if self.sum_x is not None:
            return
        dev = self.device if self.device is not None else act.device
        self.device = dev
        self.sum_x = torch.zeros(self.num_channels, device=dev, dtype=torch.float64)
        self.sum_xx = torch.zeros(self.num_channels, self.num_channels, device=dev, dtype=torch.float64)

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

        # Reduce spatial dims based on mode
        if act.dim() == 4:
            if self.mode == "gap":
                act = act.mean(dim=(2, 3))  # (B, C)
            else:  # per_pixel
                if self.spatial_subsample > 1:
                    # Stride spatial dims. Halves correlation between adjacent
                    # samples without losing coverage — better-conditioned
                    # covariance estimate at the same compute budget.
                    act = act[:, :, :: self.spatial_subsample, :: self.spatial_subsample]
                # (B, C, H', W') → (B*H'*W', C)
                act = act.permute(0, 2, 3, 1).reshape(-1, act.shape[1])
        elif act.dim() == 3:
            # Transformer-style (B, T, C). Treat each token as a sample.
            act = act.reshape(-1, act.shape[-1])

        self._ensure_buffers(act)
        act = act.to(dtype=torch.float64, device=self.device)
        batch_size = act.shape[0]

        self.n += batch_size
        self.sum_x += act.sum(dim=0)
        self.sum_xx += act.T @ act  # (C, C)

    def finalize(self) -> Tensor:
        """Return (C, C) covariance matrix."""
        if self.sum_x is None or self.n == 0:
            return torch.zeros(self.num_channels, self.num_channels)
        mean = self.sum_x / self.n
        cov = self.sum_xx / self.n - mean.unsqueeze(1) * mean.unsqueeze(0)
        # Symmetrize — float64 accumulation can still drift by a few ULPs and
        # downstream torch.linalg.eigh complains on sufficiently non-Hermitian
        # matrices.
        cov = 0.5 * (cov + cov.T)
        return cov.float().cpu()

    @property
    def mean(self) -> Tensor:
        if self.sum_x is None or self.n == 0:
            return torch.zeros(self.num_channels)
        return (self.sum_x / self.n).float().cpu()

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


def _residual_shortcut(module: nn.Module) -> Callable[[Tensor], Tensor]:
    """Return a function that applies the block's residual shortcut to its input.

    Handles three cases:
    - torchvision BasicBlock / Bottleneck: `module.downsample` is either None
      (identity) or a Sequential. Stride/channel mismatch on the first block
      of stages 2-4 needs the downsample applied before subtracting.
    - SlimNet BasicBlock / Bottleneck: `module.shortcut` is always a
      Sequential — empty for identity, populated for stride/channel change.
      `shortcut(x)` works for both.
    - Anything else (GPT-2 block, generic residual module): fall back to
      identity. Delta subtraction in that case is `output - input`, which
      assumes the block's shortcut is identity in the residual stream —
      true for standard transformer blocks.
    """
    if hasattr(module, "downsample"):
        ds = getattr(module, "downsample")
        if ds is None:
            return lambda x: x
        return ds  # callable (Sequential or Module)
    if hasattr(module, "shortcut"):
        sc = getattr(module, "shortcut")
        # Empty Sequential has no children; forward returns input unchanged.
        return sc
    return lambda x: x


class ActivationCaptureEngine:
    """Registers forward hooks on specified layers to accumulate covariance matrices.

    `source` controls what the hook accumulates:

    - "output" (default): the module's forward output. Legacy behavior.
    - "delta": `output - shortcut(input)` — the residual update Δx_l that
      the block actually computes, stripped of the identity path. For
      standard transformer blocks (identity residual) this is `output -
      input`; for ResNet blocks with downsample, the downsample is applied
      first so shape/stride match.
    - "branch": the same tensor as "output", but the hook is intended to
      be attached to a branch sub-module (e.g., `block.attn` or
      `block.mlp`) rather than the full block. The engine does not rewire
      `layer_names` for you — pass the branch module names directly.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str],
        covariance_mode: str = "per_pixel",
        spatial_subsample: int = 1,
        accumulator_device: str | None = None,
        source: str = "output",
    ):
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
        self.model = model
        self.layer_names = layer_names
        self.covariance_mode = covariance_mode
        self.spatial_subsample = spatial_subsample
        self.accumulator_device = accumulator_device
        self.source = source
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._accumulators: dict[str, CovarianceAccumulator] = {}
        self._initialized: set[str] = set()
        # Per-layer shortcut closure, populated at hook-install time.
        self._shortcuts: dict[str, Callable[[Tensor], Tensor]] = {}

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
        source = self.source

        def hook_fn(module, input, output):
            if source == "delta":
                # input is a tuple; take the first positional arg (the residual
                # input). Apply the module's shortcut so subtraction has
                # matching shape/stride.
                x_in = input[0] if isinstance(input, tuple) else input
                shortcut = self._shortcuts.get(name)
                if shortcut is None:
                    shortcut = _residual_shortcut(module)
                    self._shortcuts[name] = shortcut
                with torch.no_grad():
                    act = output - shortcut(x_in)
            else:
                # "output" and "branch" both use the raw output of whatever
                # module is hooked — "branch" is just a documentation tag
                # meaning the caller should be hooking a branch sub-module.
                act = output

            if name not in self._initialized:
                if act.dim() == 4:
                    num_channels = act.shape[1]
                elif act.dim() == 3:
                    num_channels = act.shape[-1]
                else:
                    num_channels = act.shape[-1]
                self._accumulators[name] = CovarianceAccumulator(
                    num_channels,
                    device=self.accumulator_device,
                    mode=self.covariance_mode,
                    spatial_subsample=self.spatial_subsample,
                    source=source,
                )
                self._initialized.add(name)
            self._accumulators[name].update(act)
        return hook_fn

    def register_hooks(self) -> None:
        for name in self.layer_names:
            module = self._get_module(name)
            # Pre-compute the shortcut closure once per layer so it's stable
            # across batches (avoids re-inspecting the module every call).
            if self.source == "delta":
                self._shortcuts[name] = _residual_shortcut(module)
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


_RESNET_BLOCK_COUNTS = {
    "resnet50": [3, 4, 6, 3],
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
}


def get_resnet_layer_names(backbone: str = "resnet50") -> list[str]:
    """Return the layer names to hook for a ResNet variant (output of each residual block)."""
    counts = _RESNET_BLOCK_COUNTS[backbone]
    names: list[str] = []
    for stage_idx, n_blocks in enumerate(counts, start=1):
        for i in range(n_blocks):
            names.append(f"layer{stage_idx}.{i}")
    return names


def get_resnet_stage_layer_names(backbone: str = "resnet50") -> dict[str, list[str]]:
    """Return layer names grouped by stage."""
    counts = _RESNET_BLOCK_COUNTS[backbone]
    return {
        f"stage{stage_idx}": [f"layer{stage_idx}.{i}" for i in range(n_blocks)]
        for stage_idx, n_blocks in enumerate(counts, start=1)
    }


# Back-compat aliases (keep old call sites working)
def get_resnet50_layer_names() -> list[str]:
    return get_resnet_layer_names("resnet50")


def get_resnet50_stage_layer_names() -> dict[str, list[str]]:
    return get_resnet_stage_layer_names("resnet50")
