"""Branchwise hook engine.

Captures activations at every :class:`~fasd.autodetect.BranchSpec` and
either (a) accumulates them into per-branch covariances for profiling,
or (b) exposes them through a dict-like context manager for use during
training.

Covariance accumulation is delegated to
:class:`asd.profiling.activation_capture.CovarianceAccumulator` — no
reimplementation. The new logic here is branch-aware hooking: for a
fused linear like GPT-2 ``c_attn``, the same module is hooked once and
its output is split into Q/K/V slices for three separate accumulators.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from asd.profiling.activation_capture import CovarianceAccumulator

from ..autodetect import BranchSpec


def _get_module(root: nn.Module, path: str) -> nn.Module:
    cur = root
    for p in path.split("."):
        if not hasattr(cur, p):
            raise KeyError(f"module path not found on {type(root).__name__}: {path!r}")
        cur = getattr(cur, p)
    return cur


def _as_output_tensor(out) -> Tensor:
    """Pull the activation tensor out of a module's forward output.

    Transformer block outputs are often tuples ``(hidden, *attn_stuff)``.
    """
    if isinstance(out, Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], Tensor):
        return out[0]
    raise TypeError(f"hook received unexpected output type: {type(out).__name__}")


def _slice_last(x: Tensor, sl: tuple[int, int] | None) -> Tensor:
    if sl is None:
        return x
    a, b = sl
    return x[..., a:b]


def _channel_count(x: Tensor) -> int:
    return int(x.shape[-1])


# -- Profiling engine ---------------------------------------------------


class BranchCaptureEngine:
    """Drive a calibration dataloader and accumulate per-branch covariances.

    Usage::

        engine = BranchCaptureEngine(model, branches)
        accumulators = engine.run(dataloader, device="cuda")
        # accumulators[branch.name] -> CovarianceAccumulator
    """

    def __init__(
        self,
        model: nn.Module,
        branches: Iterable[BranchSpec],
        *,
        accumulator_device: str | None = None,
    ) -> None:
        self.model = model
        self.branches: list[BranchSpec] = list(branches)
        self._device = accumulator_device
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._accumulators: dict[str, CovarianceAccumulator] = {}

        # Group branches by module_path so each module is hooked once.
        self._by_module: dict[str, list[BranchSpec]] = {}
        for b in self.branches:
            self._by_module.setdefault(b.module_path, []).append(b)

    def register_hooks(self) -> None:
        self._handles.clear()
        for module_path, specs in self._by_module.items():
            module = _get_module(self.model, module_path)
            hook = self._make_hook(module, specs)
            self._handles.append(module.register_forward_hook(hook))

    def _make_hook(
        self, module: nn.Module, specs: list[BranchSpec]
    ) -> Callable:
        def hook(mod, inputs, output):
            for spec in specs:
                if spec.hook_point == "input":
                    if not inputs:
                        continue
                    x = inputs[0]
                    if not isinstance(x, Tensor):
                        continue
                elif spec.kind == "block.residual":
                    x = _as_output_tensor(output)
                else:
                    x = _as_output_tensor(output)
                sliced = _slice_last(x, spec.slice)
                acc = self._accumulators.get(spec.name)
                if acc is None:
                    acc = CovarianceAccumulator(
                        num_channels=_channel_count(sliced),
                        device=self._device,
                        source="branch" if spec.kind != "block.residual" else "output",
                    )
                    self._accumulators[spec.name] = acc
                acc.update(sliced.detach())

        return hook

    def cleanup(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def run(
        self,
        dataloader,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, CovarianceAccumulator]:
        self.register_hooks()
        try:
            self.model.eval()
            self.model.to(device)
            with torch.no_grad():
                for batch in dataloader:
                    inputs = self._extract_inputs(batch, device)
                    if isinstance(inputs, dict):
                        self.model(**inputs)
                    elif isinstance(inputs, (tuple, list)):
                        self.model(*inputs)
                    else:
                        self.model(inputs)
        finally:
            self.cleanup()
        return dict(self._accumulators)

    @staticmethod
    def _extract_inputs(batch, device):
        if isinstance(batch, dict):
            return {
                k: (v.to(device) if isinstance(v, Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, (tuple, list)):
            return tuple(b.to(device) if isinstance(b, Tensor) else b for b in batch)
        if isinstance(batch, Tensor):
            return batch.to(device)
        return batch

    def accumulator(self, name: str) -> CovarianceAccumulator:
        return self._accumulators[name]


# -- Capture context manager (training-time hidden collection) ----------


class BranchHiddenCapture:
    """Context manager that collects branch activations during a forward pass.

    After the forward pass, ``self[branch_name]`` holds the captured
    tensor (possibly sliced). Intended for use inside the training loop
    where both teacher and student are run once per batch.
    """

    def __init__(
        self,
        model: nn.Module,
        branches: Iterable[BranchSpec],
        *,
        detach: bool = False,
    ) -> None:
        self.model = model
        self.branches: list[BranchSpec] = list(branches)
        self.detach = detach
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._state: dict[str, Tensor] = {}
        self._by_module: dict[str, list[BranchSpec]] = {}
        for b in self.branches:
            self._by_module.setdefault(b.module_path, []).append(b)

    def __enter__(self) -> "BranchHiddenCapture":
        self._state.clear()
        for module_path, specs in self._by_module.items():
            module = _get_module(self.model, module_path)
            hook = self._make_hook(specs)
            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, specs: list[BranchSpec]) -> Callable:
        def hook(mod, inputs, output):
            for spec in specs:
                if spec.hook_point == "input":
                    if not inputs:
                        continue
                    x = inputs[0]
                    if not isinstance(x, Tensor):
                        continue
                else:
                    x = _as_output_tensor(output)
                sliced = _slice_last(x, spec.slice)
                if self.detach:
                    sliced = sliced.detach()
                self._state[spec.name] = sliced

        return hook

    def __getitem__(self, name: str) -> Tensor:
        return self._state[name]

    def __contains__(self, name: str) -> bool:
        return name in self._state

    def keys(self):
        return self._state.keys()

    def values(self):
        return [self._state[b.name] for b in self.branches if b.name in self._state]

    def items(self):
        return self._state.items()


__all__ = [
    "BranchCaptureEngine",
    "BranchHiddenCapture",
    "CovarianceAccumulator",
]
