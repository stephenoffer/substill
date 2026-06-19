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
from .compression.diff_rank import RankBudgetController
from .compression.factored_linear import (
    GatedFactoredLinear,
    TeacherFactoredLinear,
)
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
    free_core: bool = True,
) -> int:
    """Replace absorbed GPT-2 student linears with :class:`TeacherFactoredLinear`.

    Uses :func:`fasd.builders.gpt2_absorb_targets` to recover the exact
    ``(V_in, V_out)`` bases each student linear was absorbed with, then swaps in a
    factored module carrying the frozen teacher weight + those bases as Stiefel
    parameters. The effective weight is unchanged, so the student forward is
    preserved bit-for-bit at conversion (``B_free`` is zero-initialized); training
    then adapts both the Stiefel bases and the free core.

    ``free_core=True`` (default, matching the Llama path) gives each factored edge a
    full-rank Euclidean core ``B_free`` in the compressed space — without it the edge
    could only *rotate* the frozen teacher weight via the (low-LR) Stiefel bases, too
    few DOF to fit a KD target in a short budget. Measured: with ``free_core=False``
    the manifold-trained variants under-trained and lost to frozen absorbed-init
    (runs/bench_v1, cmp-v2); the free core supplies the missing fitting capacity while
    preserving the zero-overhead inference fold.

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
# DDR: wrap factored edges with differentiable rank gates + a budget controller
# ---------------------------------------------------------------------------
def attach_diff_rank_gates(
    student: nn.Module,
    *,
    target_ratio: float = 0.85,
    lam: float = 1.0,
    temperature: float = 1.0,
    monotone: bool = False,
) -> RankBudgetController | None:
    """Wrap every :class:`TeacherFactoredLinear` in ``student`` with a
    :class:`GatedFactoredLinear` and return a :class:`RankBudgetController` over the
    gates (DDR). The controller's budget is ``target_ratio`` × the factored edges'
    current (fully-open) parameter cost, so the gates must prune ~``(1-ratio)`` of the
    latent columns against the KD loss. Returns ``None`` if there are no factored edges
    (e.g. ``use_cpsd_factored`` was not set).
    """
    targets = [
        (name, mod)
        for name, mod in student.named_modules()
        if isinstance(mod, TeacherFactoredLinear)
    ]
    if not targets:
        return None
    gates, costs = {}, {}
    full_cost = 0.0
    for name, tfl in targets:
        gated = GatedFactoredLinear(tfl, temperature=temperature, monotone=monotone)
        _set_module(student, name, gated)
        gates[name] = gated.gate
        cost = gated.cost()
        costs[name] = cost
        full_cost += float(cost.sum().item())  # gates start fully open (g≈1)
    return RankBudgetController(
        gates, costs, target_params=target_ratio * full_cost, lam=lam
    )


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
    # DDR: distillation-driven differentiable rank. Wraps factored edges with soft
    # column gates trained against the KD loss under a global parameter budget; rank
    # is learned, then hardened + folded for inference. Requires use_cpsd_factored.
    use_diff_rank: bool = False
    diff_rank_target_ratio: float = 0.85   # keep this fraction of factored-edge params
    diff_rank_lambda: float = 1.0          # budget-penalty weight
    diff_rank_anneal: float = 0.97         # per-event temperature decay (sharpens gate)
    # When factored, build a StiefelAdam optimizer automatically so V_in/V_out train on
    # the manifold (the MT novelty). stiefel_lr_ratio/reorth_every tune MT variance.
    stiefel_lr_ratio: float = 0.1
    stiefel_reorth_every: int = 50
    # CPI: shared-per-group circuit-preserving attention init (Llama-family, GQA+RoPE).
    # When set, pins a CPI-compatible config (keep H, G; compress head-dim) and
    # re-initializes attention with the shared-per-group basis, fixing the disjoint bug.
    use_cpi: bool = False
    cpi_head_dim_ratio: float = 0.5
    cpi_rope_aware: bool = True
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
        self._calib = None
        self.rank_controller = None  # set by build() when use_diff_rank

    def run_profile(self, calib_loader):
        # Materialize so we can reuse the calib batches for CPI covariance capture.
        self._calib = list(calib_loader)
        self.profile = _profile(self.teacher, self._calib, **self.config.profile_kwargs)
        return self.profile

    def build(self):
        if self.profile is None:
            raise RuntimeError("call run_profile() before build()")
        c = self.config
        rank_map = c.rank_map
        if c.use_cpi and rank_map is None:
            # Pin a CPI-compatible config (keep H, G; compress head-dim).
            from .compression.cpi import cpi_rank_map
            rank_map = cpi_rank_map(self.teacher, self.profile,
                                    head_dim_ratio=c.cpi_head_dim_ratio)
        self.student = build_student(
            self.teacher, self.profile,
            arch_multiplier=c.arch_multiplier,
            absorbed_init=c.absorbed_init,
            template=c.template,
            depth_policy=c.depth_policy,
            depth_keep=c.depth_keep,
            rank_map=rank_map,
        )
        if c.use_cpi:
            from .compression.cpi import apply_cpi_attention_init
            if self._calib is None:
                raise RuntimeError("use_cpi requires run_profile() to have stored calib data")
            apply_cpi_attention_init(self.student, self.teacher, self.profile,
                                     self._calib, rope_aware=c.cpi_rope_aware)
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
        if c.use_diff_rank:
            if not c.use_cpsd_factored:
                raise ValueError("use_diff_rank requires use_cpsd_factored (it gates the "
                                 "TeacherFactoredLinear edges).")
            self.rank_controller = attach_diff_rank_gates(
                self.student,
                target_ratio=c.diff_rank_target_ratio,
                lam=c.diff_rank_lambda,
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
        # When factored, train V_in/V_out on the Stiefel manifold (MT). Gate alphas and
        # any other Euclidean params land in the standard AdamW group automatically.
        if c.use_cpsd_factored and "optimizer" not in c.distill_kwargs:
            from .training.stiefel_optim import StiefelAdam, stiefel_param_groups
            kwargs["optimizer"] = StiefelAdam(stiefel_param_groups(
                self.student, base_lr=c.lr,
                stiefel_lr_ratio=c.stiefel_lr_ratio,
                reorth_every=c.stiefel_reorth_every,
            ))
        if self.rank_controller is not None:
            kwargs["rank_controller"] = self.rank_controller
            kwargs["rank_anneal"] = c.diff_rank_anneal
        kwargs.update(c.distill_kwargs)
        kwargs.update(overrides)
        return distill(self.teacher, self.student, train_loader, **kwargs)

    def fold_for_inference(self, *, harden: bool = True):
        """Collapse all factored/gated edges back to plain ``nn.Linear`` for zero-overhead
        inference, returning the hardened DDR rank-map (edge name -> integer rank) when
        DDR was used, else ``{}``. Call after :meth:`train`."""
        if self.student is None:
            raise RuntimeError("call build()/train() before fold_for_inference()")
        rank_map: dict[str, int] = {}
        # Two re-queried passes: GatedFactoredLinear owns an inner TeacherFactoredLinear,
        # so folding the parent removes the child's path — never iterate a stale snapshot.
        while True:
            hit = next((nm for nm in self.student.named_modules()
                        if isinstance(nm[1], GatedFactoredLinear)), None)
            if hit is None:
                break
            name, mod = hit
            rank_map[name] = mod.gate.harden() if harden else mod.k_in
            _set_module(self.student, name, mod.fold(harden=harden))
        while True:
            hit = next((nm for nm in self.student.named_modules()
                        if isinstance(nm[1], TeacherFactoredLinear)), None)
            if hit is None:
                break
            name, mod = hit
            _set_module(self.student, name, mod.fold())
        return rank_map

    def run(self, calib_loader, train_loader, **train_overrides):
        self.run_profile(calib_loader)
        self.build()
        return self.train(train_loader, **train_overrides)


__all__ = ["FSDPipeline", "FSDConfig", "convert_gpt2_to_factored",
           "convert_llama_to_factored", "attach_diff_rank_gates"]
