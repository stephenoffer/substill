"""Student builders — turn a profile into a concrete smaller nn.Module.

`asd.build_student(template, profile, arch_multiplier=1.0)` dispatches
to a builder based on the `template` type:

  - str in {"slimnet"}, or a `SlimNet` class → 4-stage ResNet student
    using `profiles_to_stage_widths`.
  - An `nn.Module` whose class name contains "resnet" → same path as
    "slimnet", sized from the profile.
  - `GPT2LMHeadModel` → a reduced-hidden-size GPT-2.

For models not covered, users should build their own student directly
and feed its per-layer hidden widths to `SubspaceLoss(...,
student_widths=[...])`. This helper exists for convenience, not as the
only supported path.
"""

from __future__ import annotations

import math
from typing import Any

import torch.nn as nn

from .models.student import SlimNet
from .profiling.svd_analysis import profiles_to_stage_widths


def _resnet_style_widths(
    profile, *, arch_multiplier: float, arch_min: int | None,
    min_width: int, width_multiple: int,
) -> list[int]:
    """Derive 4 stage widths from a profile whose layer names look like
    'layer{1,2,3,4}.{block_idx}'."""
    return profiles_to_stage_widths(
        profile.profiles,
        min_width=min_width,
        width_multiple=width_multiple,
        arch_multiplier=arch_multiplier,
        arch_min=arch_min,
    )


def _build_slimnet(
    profile, *,
    arch_multiplier: float = 1.0,
    arch_min: int | None = None,
    blocks_per_stage: int = 2,
    num_classes: int = 10,
    block_type: str = "bottleneck",
    stem_type: str = "cifar",
    min_width: int = 16,
    width_multiple: int = 8,
    **_ignored,
) -> SlimNet:
    widths = _resnet_style_widths(
        profile,
        arch_multiplier=arch_multiplier,
        arch_min=arch_min,
        min_width=min_width,
        width_multiple=width_multiple,
    )
    if len(widths) != 4:
        raise ValueError(
            f"SlimNet needs 4 stages; profile grouped into {len(widths)} "
            f"channel counts. Pass `layers=` for the ResNet stage blocks "
            f"only (layer1.*, layer2.*, ..., layer4.*)."
        )
    return SlimNet(
        stage_widths=widths,
        blocks_per_stage=blocks_per_stage,
        num_classes=num_classes,
        block_type=block_type,
        stem_type=stem_type,
    )


def _build_gpt2_reduced(
    profile, *,
    arch_multiplier: float = 1.0,
    arch_min: int | None = None,
    teacher=None,
    student_layers: int | None = None,
    head_multiple: int = 12,
    **_ignored,
) -> nn.Module:
    """Build a GPT-2 student with reduced `n_embd`.

    The new hidden size is `max(per-block effective rank) * mult`, rounded
    up to a multiple of `head_multiple` so the attention factorization
    stays valid.
    """
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as e:
        raise ImportError(
            "transformers not installed — `pip install transformers` to "
            "use GPT-2 builders."
        ) from e
    if teacher is None:
        raise ValueError("_build_gpt2_reduced needs `teacher=` to copy "
                         "config fields (vocab, n_positions, etc.).")

    ranks = [p.effective_rank for p in profile.profiles]
    max_rank = max(ranks) if ranks else 768
    target = max(arch_min or 0, int(math.ceil(max_rank * arch_multiplier)))
    # Round up so head_multiple divides it.
    target = ((target + head_multiple - 1) // head_multiple) * head_multiple
    target = max(target, head_multiple)

    cfg_t = teacher.config
    cfg_s = GPT2Config(
        vocab_size=cfg_t.vocab_size,
        n_positions=cfg_t.n_positions,
        n_embd=target,
        n_layer=student_layers or cfg_t.n_layer,
        n_head=head_multiple,
        activation_function=cfg_t.activation_function,
        resid_pdrop=cfg_t.resid_pdrop,
        embd_pdrop=cfg_t.embd_pdrop,
        attn_pdrop=cfg_t.attn_pdrop,
        bos_token_id=cfg_t.bos_token_id,
        eos_token_id=cfg_t.eos_token_id,
    )
    return GPT2LMHeadModel(cfg_s)


def build(template, profile, **kwargs) -> nn.Module:
    """Dispatcher. See `asd.build_student`."""
    # String tags
    if isinstance(template, str):
        key = template.lower()
        if key in ("slimnet", "resnet"):
            return _build_slimnet(profile, **kwargs)
        if key == "gpt2":
            return _build_gpt2_reduced(profile, **kwargs)
        raise ValueError(f"unknown template string {template!r}")

    # Class objects
    if isinstance(template, type):
        if template is SlimNet:
            return _build_slimnet(profile, **kwargs)
        try:
            from transformers import GPT2LMHeadModel
            if template is GPT2LMHeadModel:
                return _build_gpt2_reduced(profile, **kwargs)
        except ImportError:
            pass
        raise ValueError(f"unknown template class {template.__name__!r}")

    # Instance — infer family
    tname = type(template).__name__.lower()
    if "resnet" in tname:
        return _build_slimnet(profile, **kwargs)
    try:
        from transformers import GPT2LMHeadModel
        if isinstance(template, GPT2LMHeadModel):
            # Allow callers to pass `teacher=` explicitly — if present,
            # it overrides our default of using `template`.
            kwargs.setdefault("teacher", template)
            return _build_gpt2_reduced(profile, **kwargs)
    except ImportError:
        pass

    raise ValueError(
        f"asd.build_student doesn't know how to build from a "
        f"{type(template).__name__}. Build your student manually and "
        f"pass it to asd.SubspaceLoss(profile, student_widths=[...])."
    )
