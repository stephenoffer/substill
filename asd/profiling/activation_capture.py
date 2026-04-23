"""Hook-based activation capture with memory-efficient covariance accumulation."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


VALID_SOURCES = ("output", "delta", "branch")


class CovarianceAccumulator:
    """Accumulate channel-wise covariance in ``O(C^2)`` memory.

    Two aggregation modes control how spatial activations are reduced
    before the outer product:

    ``mode="per_pixel"`` (default):
        Treat each spatial position as a sample in the C-dim channel
        space. Adjacent pixels share large receptive fields and are
        not i.i.d., so the raw ``B * H * W`` sample count overstates
        the effective sample size and biases the top eigenvalues
        upward. Sub-sampling spatial positions with
        ``spatial_subsample`` keeps every k-th pixel per image.

    ``mode="gap"``:
        Average over ``(H, W)`` first, then accumulate outer
        products. Measures covariance of spatial means rather than
        of the channel distribution at a pixel.

    Accumulation device defaults to the input activation's device.
    Pass ``device="cpu"`` to force CPU if GPU memory is tight.

    ``source`` is a metadata tag (``"output"``, ``"delta"``,
    ``"branch"``) describing what the accumulator is being fed. The
    math is source-agnostic; the tag travels with the profile so the
    loss side can refuse to mix a delta profile with an output-basis
    loss.
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
            raise ValueError(
                f"spatial_subsample must be >= 1, got {spatial_subsample}"
            )
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
        self.num_channels = num_channels
        self.device = device
        self.mode = mode
        self.spatial_subsample = spatial_subsample
        self.source = source
        self.n = 0
        self.sum_x: Tensor | None = None
        self.sum_xx: Tensor | None = None
        self.num_zeros = 0
        self.num_total = 0
        self.activation_values: list[Tensor] = []
        self._hist_budget = 100_000
        self._hist_stored = 0

    def _ensure_buffers(self, act: Tensor) -> None:
        if self.sum_x is not None:
            return
        dev = self.device if self.device is not None else act.device
        self.device = dev
        self.sum_x = torch.zeros(self.num_channels, device=dev, dtype=torch.float64)
        self.sum_xx = torch.zeros(
            self.num_channels, self.num_channels, device=dev, dtype=torch.float64,
        )

    def update(self, activation: Tensor) -> None:
        """Update the accumulator with a batch of activations.

        Accepts shape ``(B, C, H, W)``, ``(B, T, C)``, or ``(B, C)``.
        """
        act = activation.detach().float()

        self.num_zeros += int((act == 0).sum().item())
        self.num_total += act.numel()

        if self._hist_stored < self._hist_budget:
            flat = act.flatten()
            step = max(1, flat.shape[0] // 1000)
            sample = flat[::step].cpu()
            self.activation_values.append(sample)
            self._hist_stored += len(sample)

        if act.dim() == 4:
            if self.mode == "gap":
                act = act.mean(dim=(2, 3))
            else:
                if self.spatial_subsample > 1:
                    act = act[:, :, :: self.spatial_subsample, :: self.spatial_subsample]
                act = act.permute(0, 2, 3, 1).reshape(-1, act.shape[1])
        elif act.dim() == 3:
            act = act.reshape(-1, act.shape[-1])

        self._ensure_buffers(act)
        act = act.to(dtype=torch.float64, device=self.device)
        batch_size = act.shape[0]

        self.n += batch_size
        self.sum_x += act.sum(dim=0)
        self.sum_xx += act.T @ act

    def finalize(self) -> Tensor:
        """Return the ``(C, C)`` covariance matrix."""
        if self.sum_x is None or self.n == 0:
            return torch.zeros(self.num_channels, self.num_channels)
        mean = self.sum_x / self.n
        cov = self.sum_xx / self.n - mean.unsqueeze(1) * mean.unsqueeze(0)
        # Symmetrize: float64 accumulation drifts by a few ULPs and
        # torch.linalg.eigh complains on non-Hermitian inputs.
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
        """Return a concatenated subsample of activation values."""
        if not self.activation_values:
            return torch.zeros(1)
        return torch.cat(self.activation_values)


def _residual_shortcut(module: nn.Module) -> Callable[[Tensor], Tensor]:
    """Return a callable applying the block's residual shortcut.

    Handles:

    - torchvision ``BasicBlock`` / ``Bottleneck``: ``module.downsample``
      is either ``None`` (identity) or a ``Sequential``. Stride or
      channel mismatch on the first block of stages 2-4 needs the
      downsample applied before subtracting.
    - SlimNet ``BasicBlock`` / ``Bottleneck``: ``module.shortcut`` is
      always a ``Sequential``, empty for identity or populated for
      stride/channel change. ``shortcut(x)`` works for both.
    - Anything else (GPT-2 block, generic residual module): identity
      fallback. Delta subtraction is then ``output - input``, which
      assumes the block's shortcut is identity in the residual stream
      (true for standard transformer blocks).
    """
    if hasattr(module, "downsample"):
        ds = getattr(module, "downsample")
        if ds is None:
            return lambda x: x
        return ds
    if hasattr(module, "shortcut"):
        return getattr(module, "shortcut")
    return lambda x: x


class ActivationCaptureEngine:
    """Register forward hooks on specified layers to accumulate covariance.

    ``source`` controls what the hook accumulates:

    ``"output"`` (default):
        The module's forward output.
    ``"delta"``:
        ``output - shortcut(input)``, the residual update the block
        actually computes, stripped of the identity path. For
        transformer blocks (identity residual) this is
        ``output - input``. For ResNet blocks with a downsample, the
        downsample is applied first so shape and stride match.
    ``"branch"``:
        Same tensor as ``"output"``. The tag is intended for hooks
        attached to a branch sub-module (for example ``block.attn``
        or ``block.mlp``). The engine does not rewire
        ``layer_names``; the caller passes the branch module names
        directly.
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
        self._shortcuts: dict[str, Callable[[Tensor], Tensor]] = {}

    def accumulator(self, name: str) -> CovarianceAccumulator | None:
        """Return the accumulator for ``name``, or ``None`` if none fired."""
        return self._accumulators.get(name)

    def _get_module(self, name: str) -> nn.Module:
        """Retrieve a submodule by dot-separated name."""
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
                x_in = input[0] if isinstance(input, tuple) else input
                shortcut = self._shortcuts.get(name)
                if shortcut is None:
                    shortcut = _residual_shortcut(module)
                    self._shortcuts[name] = shortcut
                with torch.no_grad():
                    act = output - shortcut(x_in)
            else:
                act = output

            if name not in self._initialized:
                if act.dim() == 4:
                    num_channels = act.shape[1]
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
            if self.source == "delta":
                self._shortcuts[name] = _residual_shortcut(module)
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def run(
        self, dataloader: DataLoader, device: str = "cpu",
    ) -> dict[str, CovarianceAccumulator]:
        """Run the model on ``dataloader`` and return per-layer accumulators."""
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
    """Return per-residual-block layer names for a ResNet variant."""
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


def get_resnet50_layer_names() -> list[str]:
    return get_resnet_layer_names("resnet50")


def get_resnet50_stage_layer_names() -> dict[str, list[str]]:
    return get_resnet_stage_layer_names("resnet50")
