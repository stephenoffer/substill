"""Auto-detect distillable layers on known model families.

`autodetect_layers(model)` returns the list of fully-qualified module
names to hook. Covers:

- torchvision ResNets (resnet18/34/50/101/152) and ResNet-like models
  that expose `layer1..layer4` each holding residual blocks.
- HuggingFace GPT-2 (and anything with `model.transformer.h[i]` as the
  block list).
- HuggingFace Llama / Mistral / Qwen-style (anything with
  `model.model.layers[i]`).
- DenseNet, MobileNetV2, VGG: best-effort, hooks the main feature
  sequence.

If the model isn't recognized, raises `NotImplementedError` with a
suggestion to pass `layers=...` explicitly. Users can register their own
detector via `register(family, detector_fn)`.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


_DETECTORS: list[tuple[str, Callable[[nn.Module], list[str] | None]]] = []


def register(family: str, detector: Callable[[nn.Module], list[str] | None]) -> None:
    """Register a layer-detector. `detector(model)` returns a list of
    dotted names, or None if this family doesn't match."""
    _DETECTORS.append((family, detector))


# ---------------------------------------------------------------------------
# Built-in detectors
# ---------------------------------------------------------------------------

def _detect_torchvision_resnet(model: nn.Module) -> list[str] | None:
    """Hook every residual block in torchvision-style ResNet."""
    if not all(hasattr(model, f"layer{i}") for i in (1, 2, 3, 4)):
        return None
    names: list[str] = []
    for i in (1, 2, 3, 4):
        stage = getattr(model, f"layer{i}")
        if not isinstance(stage, nn.Sequential):
            return None
        for j, _ in enumerate(stage):
            names.append(f"layer{i}.{j}")
    return names or None


def _detect_gpt2_like(model: nn.Module) -> list[str] | None:
    """HF GPT-2 has blocks at model.transformer.h[i]."""
    trans = getattr(model, "transformer", None)
    if trans is None:
        return None
    h = getattr(trans, "h", None)
    if h is None or not hasattr(h, "__len__"):
        return None
    return [f"transformer.h.{i}" for i in range(len(h))]


def _detect_llama_like(model: nn.Module) -> list[str] | None:
    """HF Llama/Mistral/Qwen-style blocks at model.model.layers[i]."""
    inner = getattr(model, "model", None)
    if inner is None:
        return None
    layers = getattr(inner, "layers", None)
    if layers is None or not hasattr(layers, "__len__"):
        return None
    return [f"model.layers.{i}" for i in range(len(layers))]


def _detect_decoder_only(model: nn.Module) -> list[str] | None:
    """Generic decoder-only transformer: hunt for a ModuleList named
    `layers` or `h` directly under the model root."""
    for candidate in ("h", "layers", "blocks", "decoder.layers", "encoder.layer"):
        parts = candidate.split(".")
        cur = model
        ok = True
        for p in parts:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and hasattr(cur, "__len__"):
            return [f"{candidate}.{i}" for i in range(len(cur))]
    return None


def _detect_densenet(model: nn.Module) -> list[str] | None:
    features = getattr(model, "features", None)
    if features is None or not isinstance(features, nn.Sequential):
        return None
    names = []
    for n, m in features.named_children():
        if "denseblock" in n.lower() or "transition" in n.lower():
            names.append(f"features.{n}")
    return names or None


def _detect_mobilenet_v2(model: nn.Module) -> list[str] | None:
    features = getattr(model, "features", None)
    if features is None or not isinstance(features, nn.Sequential):
        return None
    # Hook each InvertedResidual / ConvBNActivation block.
    names = []
    for n, _ in features.named_children():
        names.append(f"features.{n}")
    return names or None


register("torchvision_resnet", _detect_torchvision_resnet)
register("gpt2_like", _detect_gpt2_like)
register("llama_like", _detect_llama_like)
register("decoder_only_generic", _detect_decoder_only)
register("densenet", _detect_densenet)
register("mobilenet_v2", _detect_mobilenet_v2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def autodetect_layers(model: nn.Module) -> list[str]:
    """Run every registered detector; return the first non-empty match.

    Raises `NotImplementedError` if none match. If you hit that, the
    right next move is to print `dict(model.named_modules()).keys()`
    and pick the block/stage names yourself, then pass them as
    `asd.profile(model, loader, layers=[...])`.
    """
    errors: list[str] = []
    for family, detector in _DETECTORS:
        try:
            names = detector(model)
        except Exception as e:  # pragma: no cover — best-effort
            errors.append(f"  {family}: raised {type(e).__name__}: {e}")
            continue
        if names:
            return names

    msg = (
        f"autodetect_layers: no detector matched {type(model).__name__}. "
        "Pass `layers=[module_a, module_b, ...]` or "
        "`layers=['transformer.h.0', 'transformer.h.1', ...]` explicitly.\n"
    )
    if errors:
        msg += "Registered detectors that raised:\n" + "\n".join(errors)
    raise NotImplementedError(msg)
