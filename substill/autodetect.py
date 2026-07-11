"""Branch-level layer autodetect for F-ASD.

The key object is :class:`BranchSpec`, which names a specific branch
in a transformer and where to hook it. For decoder-only LLMs we detect:

- ``attn.q``, ``attn.k``, ``attn.v`` — per-head projections (or the
  ``q``/``k``/``v`` slice of a fused ``c_attn`` in GPT-2).
- ``attn.o`` — the attention output projection (``o_proj`` / ``c_proj``).
- ``ffn.up`` / ``ffn.gate`` / ``ffn.down`` — FFN projections.
- ``block.residual`` — the whole block's output, for users who want
  the classic ASD behavior.

:func:`autodetect_branches` tries registered detectors in order and
returns the first non-empty match; :func:`autodetect_layers` returns
the set of module names behind the branches (for compatibility with
old :mod:`asd` call sites).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

BranchKind = Literal[
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.o",
    "ffn.up",
    "ffn.gate",
    "ffn.down",
    "block.residual",
]


@dataclass(frozen=True)
class BranchSpec:
    """Where a branch lives in the model and how to capture it.

    ``name`` is the human-readable identifier used as the key in
    ``TeacherProfile``; it also uniquely identifies the branch across
    teacher and student when their module layouts match. ``module_path``
    is the dotted name of the module to hook. ``slice`` is an optional
    column range on the module's output (used for GPT-2's fused
    ``c_attn``). ``hook_point`` selects input or output capture.
    """

    name: str
    module_path: str
    kind: BranchKind
    slice: tuple[int, int] | None = None
    hook_point: Literal["output", "input"] = "output"


Mode = Literal["branch", "residual"]
Detector = Callable[[nn.Module, Mode], list[BranchSpec] | None]


_DETECTORS: list[tuple[str, Detector]] = []


def register(family: str, detector: Detector) -> None:
    """Register a branch detector for a model family."""
    _DETECTORS.append((family, detector))


# -- GPT-2 --------------------------------------------------------------


def _gpt2_hidden(model: nn.Module) -> int | None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for k in ("n_embd", "hidden_size"):
            v = getattr(cfg, k, None)
            if v is not None:
                return int(v)
    trans = getattr(model, "transformer", None)
    if trans is None:
        return None
    h = getattr(trans, "h", None)
    if h is None or len(h) == 0:
        return None
    c_attn = getattr(h[0].attn, "c_attn", None)
    if c_attn is None:
        return None
    if hasattr(c_attn, "nf"):
        return int(c_attn.nf // 3)
    w = getattr(c_attn, "weight", None)
    if w is None:
        return None
    # GPT-2 Conv1D weight has shape (in, 3*hidden)
    return int(w.shape[-1] // 3)


def _detect_gpt2(model: nn.Module, mode: Mode) -> list[BranchSpec] | None:
    trans = getattr(model, "transformer", None)
    if trans is None:
        return None
    h = getattr(trans, "h", None)
    if h is None or not hasattr(h, "__len__"):
        return None
    hidden = _gpt2_hidden(model)
    if hidden is None:
        return None
    branches: list[BranchSpec] = []
    for i in range(len(h)):
        prefix = f"transformer.h.{i}"
        if mode == "residual":
            branches.append(
                BranchSpec(f"{prefix}.residual", prefix, "block.residual")
            )
            continue
        branches.extend(
            [
                BranchSpec(
                    f"{prefix}.attn.q",
                    f"{prefix}.attn.c_attn",
                    "attn.q",
                    slice=(0, hidden),
                ),
                BranchSpec(
                    f"{prefix}.attn.k",
                    f"{prefix}.attn.c_attn",
                    "attn.k",
                    slice=(hidden, 2 * hidden),
                ),
                BranchSpec(
                    f"{prefix}.attn.v",
                    f"{prefix}.attn.c_attn",
                    "attn.v",
                    slice=(2 * hidden, 3 * hidden),
                ),
                BranchSpec(f"{prefix}.attn.o", f"{prefix}.attn.c_proj", "attn.o"),
                BranchSpec(f"{prefix}.ffn.up", f"{prefix}.mlp.c_fc", "ffn.up"),
                BranchSpec(f"{prefix}.ffn.down", f"{prefix}.mlp.c_proj", "ffn.down"),
            ]
        )
    return branches or None


# -- Llama / Mistral / Qwen ---------------------------------------------


def _detect_llama(model: nn.Module, mode: Mode) -> list[BranchSpec] | None:
    inner = getattr(model, "model", None)
    if inner is None:
        return None
    layers = getattr(inner, "layers", None)
    if layers is None or not hasattr(layers, "__len__"):
        return None
    branches: list[BranchSpec] = []
    needed_attn = ("q_proj", "k_proj", "v_proj", "o_proj")
    needed_mlp = ("gate_proj", "up_proj", "down_proj")
    for i in range(len(layers)):
        block = layers[i]
        prefix = f"model.layers.{i}"
        if mode == "residual":
            branches.append(
                BranchSpec(f"{prefix}.residual", prefix, "block.residual")
            )
            continue
        attn = getattr(block, "self_attn", None)
        mlp = getattr(block, "mlp", None)
        if attn is None or mlp is None:
            return None
        if not all(hasattr(attn, n) for n in needed_attn):
            return None
        if not all(hasattr(mlp, n) for n in needed_mlp):
            return None
        branches.extend(
            [
                BranchSpec(f"{prefix}.attn.q", f"{prefix}.self_attn.q_proj", "attn.q"),
                BranchSpec(f"{prefix}.attn.k", f"{prefix}.self_attn.k_proj", "attn.k"),
                BranchSpec(f"{prefix}.attn.v", f"{prefix}.self_attn.v_proj", "attn.v"),
                BranchSpec(f"{prefix}.attn.o", f"{prefix}.self_attn.o_proj", "attn.o"),
                BranchSpec(f"{prefix}.ffn.gate", f"{prefix}.mlp.gate_proj", "ffn.gate"),
                BranchSpec(f"{prefix}.ffn.up", f"{prefix}.mlp.up_proj", "ffn.up"),
                BranchSpec(f"{prefix}.ffn.down", f"{prefix}.mlp.down_proj", "ffn.down"),
            ]
        )
    return branches or None


# -- Decoder-only generic fallback --------------------------------------


def _detect_decoder_generic(model: nn.Module, mode: Mode) -> list[BranchSpec] | None:
    # Look for any ModuleList of blocks at common paths.
    for candidate in ("h", "layers", "blocks", "decoder.layers"):
        parts = candidate.split(".")
        cur = model
        ok = True
        for p in parts:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and hasattr(cur, "__len__"):
            if mode == "residual":
                return [
                    BranchSpec(f"{candidate}.{i}.residual", f"{candidate}.{i}", "block.residual")
                    for i in range(len(cur))
                ]
            return None
    return None


register("gpt2", _detect_gpt2)
register("llama", _detect_llama)
register("decoder_generic", _detect_decoder_generic)


# -- public API ---------------------------------------------------------


def autodetect_branches(model: nn.Module, *, mode: Mode = "branch") -> list[BranchSpec]:
    """Return the first non-empty branch list across registered detectors.

    Raises ``NotImplementedError`` if none match. Pass branches directly
    to :func:`substill.profile(model, loader, branches=[...])` if the model
    is unrecognized.
    """
    errors: list[str] = []
    for family, detector in _DETECTORS:
        try:
            out = detector(model, mode)
        except Exception as e:  # pragma: no cover
            errors.append(f"  {family}: {type(e).__name__}: {e}")
            continue
        if out:
            return out
    msg = (
        f"autodetect_branches: no detector matched {type(model).__name__} "
        f"in mode={mode!r}. Pass `branches=[BranchSpec(...), ...]` explicitly."
    )
    if errors:
        msg += "\nDetectors that raised:\n" + "\n".join(errors)
    raise NotImplementedError(msg)


def autodetect_layers(model: nn.Module, *, mode: Mode = "branch") -> list[str]:
    """Return the set of distinct module paths backing detected branches."""
    branches = autodetect_branches(model, mode=mode)
    seen: dict[str, None] = {}
    for b in branches:
        seen.setdefault(b.module_path, None)
    return list(seen.keys())


__all__ = [
    "BranchSpec",
    "BranchKind",
    "Mode",
    "autodetect_branches",
    "autodetect_layers",
    "register",
]
