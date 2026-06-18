"""FSDPipeline: a composable public API for the full CPSD/FSD pipeline.

Previously the end-to-end pipeline lived only inside ``scripts/distill_llama32_fsd.py``
(a 478-line script), so there was no way to run FSD/CPSD programmatically without
copy-pasting. ``FSDPipeline`` wraps the existing, tested stages —
:func:`fasd.profile`, :func:`fasd.build_student`, :func:`fasd.distill` — and threads
the CPSD options through them:

    import fasd
    pipe = fasd.FSDPipeline(teacher, config=fasd.FSDConfig(
        arch_multiplier=0.5, use_cpsd_factored=True, generative_kd="skew_kl"))
    result = pipe.run(calib_loader, train_loader)   # profile -> build -> (convert) -> distill

CPSD post-build conversions (all optional, off by default):
  - ``use_cpsd_factored``: replace absorbed linears with :class:`TeacherFactoredLinear`
    so the bases ``V_in/V_out`` train on the Stiefel manifold against the KD loss
    (the manifold-training novelty). Use :func:`fasd.training.stiefel_optim.StiefelAdam`.
  - ``use_rr_norm``: replace norms with rotation-equivariant ``RRNorm`` (Pillar 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from .api import profile as _profile
from .builders import build_student, gpt2_absorb_targets, llama_absorb_targets
from .compression.factored_linear import TeacherFactoredLinear
from .training.distill import distill
from .util.rr_norm import replace_layernorm_with_rrnorm


# ---------------------------------------------------------------------------
# Module-tree helpers
# ---------------------------------------------------------------------------
def _get_parent_and_attr(root: nn.Module, dotted: str):
    """Return (parent_module, final_attr_or_index) for a dotted module path.

    Handles ModuleList indices (numeric path components).
    """
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _set_module(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent, last = _get_parent_and_attr(root, dotted)
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


# ---------------------------------------------------------------------------
# CPSD post-build conversion: absorbed linears -> trainable Stiefel factors
# ---------------------------------------------------------------------------
def convert_gpt2_to_factored(
    student: nn.Module,
    teacher: nn.Module,
    profile,
    *,
    edges: tuple[str, ...] = ("attn.c_proj", "mlp.c_proj", "mlp.c_fc", "attn.c_attn"),
    free_core: bool = False,
) -> int:
    """Replace absorbed GPT-2 student linears with :class:`TeacherFactoredLinear`.

    Uses :func:`fasd.builders.gpt2_absorb_targets` to recover the exact
    ``(V_in, V_out)`` bases each student linear was absorbed with, then swaps in a
    factored module carrying the frozen teacher weight + those bases as Stiefel
    parameters. The effective weight is unchanged, so the student forward is
    preserved bit-for-bit at conversion; training then adapts the bases.

    Returns the number of modules converted.
    """
    n = 0
    for name, t_mod, _s_mod, V_in, V_out in gpt2_absorb_targets(teacher, student, profile):
        if not any(name.endswith(e) for e in edges):
            continue
        b_t = getattr(t_mod, "bias", None)
        tfl = TeacherFactoredLinear(
            t_mod.weight.detach(), V_in, V_out,
            b_t.detach() if b_t is not None else None,
            layout="conv1d_gpt2", free_core=free_core,
        )
        _set_module(student, name, tfl)
        n += 1
    return n


def convert_llama_to_factored(
    student: nn.Module,
    teacher: nn.Module,
    profile,
    *,
    edges: tuple[str, ...] = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ),
    free_core: bool = True,
) -> int:
    """Replace absorbed Llama-family student linears with :class:`TeacherFactoredLinear`.

    Llama linears are plain ``nn.Linear`` (layout="linear"), so the factored module
    drops in directly. Uses :func:`fasd.builders.llama_absorb_targets` to recover each
    linear's ``(V_in, V_out)`` bases; the effective weight is unchanged at conversion.
    """
    n = 0
    for name, t_mod, _s_mod, V_in, V_out in llama_absorb_targets(teacher, student, profile):
        if not any(name.endswith(e) for e in edges):
            continue
        b_t = getattr(t_mod, "bias", None)
        tfl = TeacherFactoredLinear(
            t_mod.weight.detach(), V_in, V_out,
            b_t.detach() if b_t is not None else None,
            layout="linear", free_core=free_core,
        )
        _set_module(student, name, tfl)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Config + pipeline
# ---------------------------------------------------------------------------
@dataclass
class FSDConfig:
    """Configuration for :class:`FSDPipeline`."""

    # architecture / compression
    arch_multiplier: float = 1.0
    rank_map: dict | None = None
    template: str = "auto"
    depth_policy: str = "keep"
    depth_keep: int | None = None
    absorbed_init: bool = True
    # CPSD post-build options
    use_cpsd_factored: bool = False
    use_rr_norm: bool = False
    # profiling
    profile_kwargs: dict = field(default_factory=dict)
    # distillation
    generative_kd: str = "skew_kl"
    total_steps: int = 200
    lr: float = 5e-5
    distill_kwargs: dict = field(default_factory=dict)


class FSDPipeline:
    """Orchestrates profile -> build -> (CPSD convert) -> distill over a frozen teacher."""

    def __init__(self, teacher: nn.Module, *, config: FSDConfig | None = None):
        self.teacher = teacher
        self.config = config or FSDConfig()
        self.profile = None
        self.student = None

    def run_profile(self, calib_loader):
        self.profile = _profile(self.teacher, calib_loader, **self.config.profile_kwargs)
        return self.profile

    def build(self):
        if self.profile is None:
            raise RuntimeError("call run_profile() before build()")
        c = self.config
        self.student = build_student(
            self.teacher, self.profile,
            arch_multiplier=c.arch_multiplier,
            absorbed_init=c.absorbed_init,
            template=c.template,
            depth_policy=c.depth_policy,
            depth_keep=c.depth_keep,
            rank_map=c.rank_map,
        )
        if c.use_rr_norm:
            d_model = int(getattr(self.student.config, "n_embd",
                                  getattr(self.student.config, "hidden_size", 0)))
            replace_layernorm_with_rrnorm(self.student, d_model=d_model)
        if c.use_cpsd_factored:
            cls = type(self.teacher).__name__
            if "GPT2" in cls:
                convert_gpt2_to_factored(self.student, self.teacher, self.profile)
            elif any(k in cls for k in ("Llama", "Mistral", "Qwen2", "Qwen3ForCausal")):
                convert_llama_to_factored(self.student, self.teacher, self.profile)
            else:
                raise NotImplementedError(
                    f"use_cpsd_factored not wired for {cls}; supported: GPT-2, Llama-family."
                )
        return self.student

    def train(self, train_loader, **overrides):
        if self.student is None:
            raise RuntimeError("call build() before train()")
        c = self.config
        kwargs = {
            "profile": self.profile,
            "generative_kd": c.generative_kd,
            "total_steps": c.total_steps,
            "lr": c.lr,
        }
        kwargs.update(c.distill_kwargs)
        kwargs.update(overrides)
        return distill(self.teacher, self.student, train_loader, **kwargs)

    def run(self, calib_loader, train_loader, **train_overrides):
        self.run_profile(calib_loader)
        self.build()
        return self.train(train_loader, **train_overrides)


__all__ = ["FSDPipeline", "FSDConfig", "convert_gpt2_to_factored",
           "convert_llama_to_factored"]
