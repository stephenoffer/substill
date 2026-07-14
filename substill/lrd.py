"""Learned Restriction Distillation (LRD) -- the verified public entry point.

LRD is the one construction in this repo that survived an independent re-measurement
of the earlier FSD/CPSD results (``docs/init_findings.md``, ``docs/learned_restriction.md``).
It rests on the single principle that survived every ablation and both architectures
tested:

    **Restrict the teacher's operator; never refit it.** A change of basis
    ``W_s = V^T W_T V`` is the teacher's own weight seen through a subspace, so the
    teacher's layers still compose the way they did -- and that transfers through
    distillation. Refitting each layer to a regression target *replaces* the operator
    and does not transfer.

The compressed student is *initialized* as an exact restriction of the teacher, and the
residual-stream projection ``V`` is trained **on the Stiefel manifold, against the KD
loss, through the whole network** -- jointly with a zero-initialized Euclidean residual
``D`` that starts training at exactly the absorbed-init student a plain baseline begins
from. It folds to a plain ``LlamaForCausalLM`` with zero inference overhead.

Be precise about what the restriction *is* here, because the tempting stronger claim is
false. With ``free_core=False`` the student cannot leave the restriction class: every
reachable point is ``V^T W_T V`` for some ``V``. With ``free_core=True`` -- the default, and
the arm that wins -- ``D`` is unconstrained and can reach any student the baseline can, so
the restriction is **not an invariant of training**. It is the *initialization* (``D = 0``)
and the *coordinate system* (``V`` moves every layer coherently along the one direction that
keeps the student a restriction; ``D`` moves each weight alone). Whether the trained student
stays near the class is an empirical question, and
:meth:`~substill.compression.restricted.RestrictedLlama.restriction_gap` answers it with a
number instead of an assertion. Earlier drafts of this docstring claimed the student "stays
inside the class of restrictions at every step"; it does not, and the claim is withdrawn.

Because the student *is* the teacher restricted to ``V``, its residual stream has a
non-arbitrary target no ordinary KD has -- the teacher's own stream seen through ``V``.
An annealed, scale-invariant **restriction-consistency** term (``aux_w``) asks the two to
point the same way at every layer, handing ``V`` dense per-layer gradient early and then
handing off to pure KD. Combined with a small ``V``-LR floor (``v_floor``), it both lowers
PPL and sharply stabilizes the run.

Measured on ``JackFram/llama-160m`` (WikiText-2, 3.07x, n=3): LRD reaches
**74.36 +/- 0.06** PPL against the strongest frozen-basis baseline's 80.94 +/- 0.90 -- an
**8.1% win** with every seed beating every baseline seed (the plain-KD ``V``-training arm
sits at 75.45 +/- 0.79; the aux term adds -1.1 PPL and cuts seed variance ~13x). See
``docs/learned_restriction.md`` for the full study, scaling, and honest caveats.

Recommended one-call entry point::

    import substill

    result = substill.learned_restriction_distill(
        teacher, train_loader,
        config=substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000),
    )
    student = result.student           # a plain LlamaForCausalLM, zero-overhead

Or drive the phases yourself for inspection / checkpointing::

    lrd = substill.LearnedRestriction(teacher, config)
    lrd.prepare(calib_loader)          # gamma-fold teacher, profile the stream, build V0
    lrd.fit(train_loader)              # descend the KD loss in (V, D) coordinates
    student = lrd.fold()               # collapse to a plain student

Scope: Llama-family decoders (Llama, Mistral, and other bias-free RMSNorm decoders with
``model.layers[*].self_attn.{q,k,v,o}_proj`` and ``mlp.{gate,up,down}_proj``). The
restriction map is architecture-specific; :func:`learned_restriction_distill` validates
the teacher and raises a clear error on an unsupported family.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from .compression.llama_absorb import (
    check_head_geometry,
    gamma_fold_llama,
    llama_balanced_second_moment,
    llama_logit_metric,
    llama_norm_input_second_moments,
    llama_residual_second_moment,
)
from .compression.restricted import RestrictedLlama, StiefelAdamV, ffn_energy_indices
from .compression.seq_absorb import residual_basis
from .losses.generative_kd import forward_kl, reverse_kl, skew_kl

__all__ = [
    "LRDConfig",
    "LRDResult",
    "LearnedRestriction",
    "learned_restriction_distill",
    "plan_restricted_geometry",
]


# ---------------------------------------------------------------------------
# Teacher validation + geometry planning
# ---------------------------------------------------------------------------
def _is_llama_family(teacher: nn.Module) -> bool:
    """Structural check: does ``teacher`` expose the Llama restriction surface?"""
    try:
        layer = teacher.model.layers[0]
        attn, mlp = layer.self_attn, layer.mlp
    except (AttributeError, IndexError, TypeError):
        return False
    needed = (
        getattr(attn, "q_proj", None), getattr(attn, "k_proj", None),
        getattr(attn, "v_proj", None), getattr(attn, "o_proj", None),
        getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None),
        getattr(mlp, "down_proj", None),
    )
    if any(m is None for m in needed):
        return False
    # The restriction absorbs no attention bias; reject biased attention (e.g. Qwen2).
    return all(getattr(m, "bias", None) is None for m in needed)


def _require_llama_family(teacher: nn.Module) -> None:
    if not _is_llama_family(teacher):
        raise NotImplementedError(
            f"learned_restriction_distill supports bias-free RMSNorm decoders of the "
            f"Llama family (Llama, Mistral, ...); {type(teacher).__name__} does not "
            f"expose the required q/k/v/o + gate/up/down projections without bias. "
            f"The restriction map (substill/compression/restricted.py) is architecture-specific."
        )


@dataclass
class RestrictedGeometry:
    """The four numbers that size a restricted student."""

    hidden: int          # residual width k (must be a multiple of the teacher head_dim)
    intermediate: int    # FFN inner width
    n_head: int          # attention heads kept (whole)
    n_kv: int            # key/value heads kept (whole); divides n_head

    def as_dict(self) -> dict:
        return {"hidden": self.hidden, "intermediate": self.intermediate,
                "n_head": self.n_head, "n_kv": self.n_kv}


def plan_restricted_geometry(teacher: nn.Module, width_ratio: float) -> RestrictedGeometry:
    """Choose a restricted student geometry at ``width_ratio`` of the teacher.

    Heads are kept whole (``docs/init_findings.md`` §9 found no head-selection rule beats
    an arbitrary whole-head subset, and shattering heads costs ~7 sigma of PPL), so the
    residual width lands on a multiple of the teacher's head dimension.

    **The grouped-query group size is an invariant, not a free parameter.** Under GQA a
    query head does not stand alone: query head ``i`` was trained to attend against key/value
    head ``i // G``, for the teacher's group size ``G = n_head / n_kv``. The restriction hands
    the student teacher q head ``i`` and teacher kv head ``j`` *verbatim*, and the student
    then re-derives the pairing from **its own** ``G' = n_head' / n_kv'``. So unless
    ``G' == G``, student q head ``i`` attends against a kv head whose keys its query weights
    have never seen -- the student is not the teacher seen through a subspace, it is a
    different operator, and the entire premise of the method is void. Nothing raises, the KD
    loss still descends, and the damage shows up only as unexplained lost quality.

    Choosing ``n_kv`` and *deriving* ``n_head = G * n_kv`` makes ``G' == G`` by construction.
    The cost is that the head count moves in steps of ``G``, so the achievable widths are
    coarser on a GQA teacher than on an MHA one -- an honest constraint of keeping whole
    groups, not a tuning choice. On an MHA teacher ``G == 1`` and this is exactly the old
    behaviour, so every published MHA number is unaffected.
    """
    if not 0.0 < width_ratio <= 1.0:
        raise ValueError(f"width_ratio must be in (0, 1], got {width_ratio}")
    _require_llama_family(teacher)
    c = teacher.config
    d = int(c.hidden_size)
    heads = int(c.num_attention_heads)
    kv_heads = int(getattr(c, "num_key_value_heads", heads))
    head_dim = d // heads
    if heads % kv_heads != 0:
        raise ValueError(
            f"teacher has {heads} query heads over {kv_heads} kv heads, which do not divide; "
            f"the grouped-query structure is not a partition and cannot be restricted."
        )
    group = heads // kv_heads

    # Pick the kv-head count, then derive the query heads as whole groups of it.
    n_kv = min(kv_heads, max(1, round(width_ratio * kv_heads)))
    n_head = group * n_kv
    hidden = n_head * head_dim
    interm = max(1, round(width_ratio * int(c.intermediate_size)))
    return RestrictedGeometry(hidden=hidden, intermediate=interm, n_head=n_head, n_kv=n_kv)




# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------
_KD_FNS = {"forward_kl": forward_kl, "reverse_kl": reverse_kl, "skew_kl": skew_kl}

#: Default trust-region step for ``V``: radians of RMS rotation per step.
#:
#: Unlike the ambient rule it replaces, this is dimensionless -- it is a *rotation*, and the same
#: number means the same motion on a 768-wide teacher and a 2048-wide one. That does **not** make
#: it tuning-free: the *optimal* rotation budget is still a hyperparameter, like any learning
#: rate. What the trust region buys is that a single value now exists that works across scales,
#: and that its failure mode is diagnosable (``LRDResult.max_principal_angle`` runs away toward
#: pi/2).
#:
#: ``0.002`` is the best value measured at **both** 160M and 1.3B (``docs/learned_restriction.md``
#: §11f). An earlier default of ``0.005`` was picked from a 160M sweep where 0.002 and 0.005 tied;
#: the tie broke badly at 1.3B, where 0.005 rotates ``V`` 1.55 rad -- almost orthogonal to its
#: init -- and costs ~40 PPL. Sweeping two scales, not one, is what caught it.
_V_TRUST_DEFAULT = 0.002


@dataclass
class LRDConfig:
    """Configuration for :class:`LearnedRestriction` / :func:`learned_restriction_distill`.

    The student geometry (``hidden``/``intermediate``/``n_head``/``n_kv``) is required;
    build it from a compression ratio with :meth:`for_ratio`.
    """

    # --- student geometry (required) ---
    hidden: int
    intermediate: int
    n_head: int
    n_kv: int
    # --- optimization ---
    steps: int = 2000
    lr: float = 1e-3            # AdamW lr for the Euclidean residual D (the free core)
    v_lr: float = 0.0          # Stiefel step for V; 0.0 selects the default (see below)
    v_trust_region: bool = True
    """Measure ``V``'s step in *rotation* rather than in ambient length.

    With the trust region on, ``v_lr`` **is the RMS angle, in radians, that ``V`` turns per
    step** -- a dimensionless quantity that means the same motion on a 768-wide teacher and a
    2048-wide one. With it off, ``v_lr`` is a raw Adam step in the ambient ``(d, k)`` space
    whose effect on the subspace scales with ``sqrt(d k)``, which is why the ambient rule needed
    the hand-fitted constant ``min(1e-3, 0.77 / d)``.

    **Be precise about what this does and does not buy.** It does *not* make the step
    tuning-free: the best rotation budget is still a hyperparameter, and it is not identical at
    every scale (0.005 is fine at 160M and over-rotates badly at 1.3B). What it buys is that

    * the knob is a **physical quantity** -- radians -- so a value transfers between teachers
      instead of needing a fitted ``1/d`` rule, and a single default (0.002) works at both
      scales measured;
    * its failure mode is **diagnosable**: over-rotation drives
      :attr:`LRDResult.max_principal_angle` toward ``pi/2``, where ``V`` is nearly orthogonal to
      its own initialization. That is a number you can look at. An ambient step size is not.

    See :class:`~substill.compression.restricted.StiefelAdamV`.
    """
    kd: str = "forward_kl"     # forward_kl | reverse_kl | skew_kl
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    v_floor: float = 0.1       # cosine floor for V's LR: keep V travelling to the end
    # --- restriction-consistency auxiliary loss (the 2026-07-11 improvement) ---
    aux_w: float = 1.0         # weight of the annealed cosine consistency term (0 disables)
    aux_until_frac: float = 1.0  # stop paying the aux forward past this fraction of training
    aux_stream: str = "residual"
    """Which student state the aux term matches at the **final** layer.

    ``"residual"`` -- the raw post-layer residual, uniform with every other layer, and what the
    aux term's description says it matches. ``"prelogit"`` -- the state after the student's
    final norm, i.e. what ``lm_head`` actually reads; it carries the student's learned
    per-channel norm gain, and so also supervises that norm.

    The pre-audit code did ``"prelogit"`` *by accident* (it read HuggingFace's
    ``output_hidden_states``, whose last entry is post-norm), so its code and its description
    disagreed and nobody had chosen. Measured at n=3 (``docs/learned_restriction.md`` §9f) the
    two are **statistically tied** -- 74.22 +/- 0.48 for ``residual`` against 74.59 +/- 1.07 for
    ``prelogit``. So the default is ``residual``: it costs nothing measurable, and it is the
    option that means what the method says it means.

    (A single-seed comparison said ``residual`` was 1.3 PPL *worse*. It was noise. The seed
    spread here is +/-0.5 to +/-1.1 PPL, so n=1 cannot resolve a difference this size, and the
    audit nearly shipped a default chosen from one run. See §9f.)
    """
    # --- restriction ---
    basis: str = "pca"         # V0 initializer: pca | identity | gn (logit-metric)
    basis_pool: str = "balanced"
    """How the per-layer residual second moments are pooled *before* the basis is taken from them.

    ``"pooled"`` -- the historical behaviour: sum the raw second moment of every residual state.
    A transformer's residual norm grows steeply with depth, so that sum is dominated by the last
    few layers and the basis it induces barely sees the early ones (on llama-160m it keeps 97.8%
    of the pooled energy but only ~51% of the embedding's).

    ``"balanced"`` -- normalize each layer by its own trace before averaging,
    ``S = mean_l S_l / tr(S_l)``, so every layer gets an equal vote.

    This is the single highest-leverage line in the library. It improves the absorbed init 4.6x
    and buys the **frozen-basis baseline 4.5 PPL** of final quality -- most of what LRD's whole
    Stiefel machinery was reported to be worth. It helps LRD too (74.59 -> 71.68), so the two
    compose; but it means the published margin was measured against a baseline that was weak for
    a reason nobody had checked. See ``docs/learned_restriction.md`` §11.
    """
    free_core: bool = True     # train V *and* D jointly (the verified winning arm)
    per_norm_gain: bool = True
    """Give each of the ``2L+1`` RMSNorms the gain its *own* input distribution loses.

    A single pooled gain is measurably wrong -- 39% too large at layer 0's norms on
    llama-160m at 3.07x, because the pooled second moment is dominated by the high-norm deep
    layers and describes the raw embedding not at all. Costs a ``(2L+1, d, d)`` buffer
    (59 MB on llama-160m, 1.7 GB on a 2560-wide 32-layer teacher); set ``False`` to fall back
    to the pooled gain when that does not fit.
    """
    # --- data / bookkeeping ---
    calib_batches: int = 16
    shuffle: bool = True       # draw training batches in seeded random order (see _cycle_ids)
    seed: int = 0
    device: str | None = None  # None -> teacher's device, else "cuda"/"cpu"
    log_every: int = 0         # >0 prints running KD loss every N steps

    @classmethod
    def for_ratio(cls, teacher: nn.Module, width_ratio: float, **kwargs) -> LRDConfig:
        """Build a config sized at ``width_ratio`` of ``teacher``.

        See :func:`plan_restricted_geometry` for how the student geometry is chosen.
        """
        return cls(**plan_restricted_geometry(teacher, width_ratio).as_dict(), **kwargs)

    def resolved_v_lr(self, teacher: nn.Module) -> float:
        """The Stiefel step, applying the default when ``v_lr <= 0``.

        Under the trust region (the default) this is an angle in radians per step and does
        **not** depend on the teacher: the quantity it controls -- how far ``V`` rotates --
        is already scale-free, so there is nothing left for a ``1/d`` rule to correct.

        Without the trust region we fall back to the historical ambient rule
        ``min(1e-3, 0.77 / d)``. That constant was fitted to three teachers (160M, 1.3B,
        2.7B) and is kept only so the published ambient numbers stay reproducible. It is not
        a law, it does not follow from anything, and it should not be trusted outside the
        range it was fitted in -- which is precisely the reason the trust region exists.
        """
        if self.v_lr > 0:
            return self.v_lr
        if self.v_trust_region:
            return _V_TRUST_DEFAULT
        return min(1e-3, 0.77 / int(teacher.config.hidden_size))


@dataclass
class LRDResult:
    """Return value from :func:`learned_restriction_distill`."""

    student: nn.Module               # a plain, folded LlamaForCausalLM (zero overhead)
    history: list[dict] = field(default_factory=list)
    max_principal_angle: float | None = None   # how far V rotated from its init, radians
    final_kd: float | None = None
    config: LRDConfig | None = None
    restriction_gap: float | None = None
    """``||D||_F / ||V^T W_T V||_F`` at the end of training: how far outside the restriction
    class the student ended up. ``0.0`` with ``free_core=False`` (where it cannot leave);
    with ``free_core=True`` it is a *measurement*, not a guarantee -- see
    :meth:`~substill.compression.restricted.RestrictedLlama.restriction_gap`."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _as_input_ids(batch) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, dict):
        return batch["input_ids"]
    return batch[0]


def _cosine_warmup(opt, steps: int, warm_frac: float, floor: float = 0.0):
    """Linear warmup then cosine decay to ``floor`` (a fraction of peak).

    ``floor > 0`` keeps a parameter *travelling* through the whole budget rather than
    freezing in the second half. For ``V`` that matters: the win is where ``V`` travels,
    not where it starts, so a small floor lets it keep finding a better basin late.
    """
    warm = max(1, int(warm_frac * steps))

    def lr_at(s: int) -> float:
        if s < warm:
            return (s + 1) / warm
        cos = 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        return floor + (1 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def _restriction_consistency(V: Tensor, hs_s: Tensor, t_hidden) -> Tensor:
    """Scale-invariant per-layer agreement between the student stream and ``V^T h_T``.

    The student *is* the teacher restricted to ``V``, so its residual stream should point
    the way the teacher's stream does *seen through* ``V``. ``hs_s`` is the student's
    ``(L, B, T, k)`` **raw** post-layer stack (`RestrictedLlama.hidden_and_logits`);
    ``t_hidden`` is the folded teacher's ``output_hidden_states`` tuple. Cosine (not L2) so the
    RMS gain the student's norms carry is irrelevant -- only the *direction*, the teacher's
    computation seen through the subspace, is matched. Differentiable in ``V`` on both sides.

    Both sides must be the *same kind of state* at every layer, which takes a little care:
    HuggingFace's ``hidden_states[-1]`` is the state after ``model.norm``, not the last layer's
    output. On the **teacher** that is harmless -- after the gamma fold its final norm weight is
    a scalar, and cosine ignores positive scaling, so the direction of ``V^T h_T`` survives. On
    the **student** it is not, because the student's final norm weight is ``g*1 + D`` and ``D``
    makes it per-channel; so `hidden_and_logits` hooks the raw layer outputs instead. See its
    docstring, and `gap_fit_llama`, which documents the same trap.
    """
    t_proj = torch.stack([h.float() @ V for h in t_hidden[1:]], dim=0)   # (L, B, T, k)
    cos = torch.nn.functional.cosine_similarity(hs_s, t_proj, dim=-1)    # (L, B, T)
    return 1.0 - cos.mean()


class LearnedRestriction:
    """Distil a Llama-family teacher by training its residual-stream restriction.

    Three phases, each a public method so callers can inspect or checkpoint between them:

    * :meth:`prepare` -- gamma-fold the teacher (function-preserving), measure the residual
      second moment, select the FFN neurons and the ``V0`` basis, and build the trainable
      :class:`~substill.compression.restricted.RestrictedLlama`.
    * :meth:`fit` -- descend the KD loss on the teacher's own logits, moving ``V`` on the
      Stiefel manifold (and, with ``free_core``, ``D`` in the Euclidean directions), plus
      the annealed restriction-consistency term (``aux_w``) that gives ``V`` dense
      per-layer gradient early on.
    * :meth:`fold` -- collapse ``(V, D)`` into a plain ``LlamaForCausalLM``.
    """

    def __init__(self, teacher: nn.Module, config: LRDConfig):
        _require_llama_family(teacher)
        # A hand-written config can name an (n_head, n_kv) pair that silently re-pairs the
        # teacher's query and kv heads; fail here rather than train a broken restriction.
        check_head_geometry(teacher, config.n_head, config.n_kv)
        self.teacher_src = teacher
        self.config = config
        self.device = torch.device(
            config.device
            or (next(teacher.parameters()).device if any(True for _ in teacher.parameters())
                else "cpu")
        )
        self.folded: nn.Module | None = None    # gamma-folded, frozen teacher
        self.restricted: RestrictedLlama | None = None
        self._V0: Tensor | None = None
        self.teacher_ppl: float | None = None

    # -- phase 1 ------------------------------------------------------------
    def prepare(self, calib_loader) -> LearnedRestriction:
        """Fold the teacher, profile its residual stream, and build the restricted student."""
        c = self.config
        torch.manual_seed(c.seed)
        dev = self.device

        calib = [{"input_ids": _as_input_ids(b).to(dev)}
                 for b in _take(calib_loader, c.calib_batches)]
        if not calib:
            raise ValueError("calib_loader yielded no batches")

        folded = gamma_fold_llama(self.teacher_src).eval().to(dev)
        folded.requires_grad_(False)
        self.folded = folded

        # `S` is the pooled moment: still the fallback for the single RMS gain, and the metric
        # the historical `basis_pool="pooled"` path takes its basis from.
        S = llama_residual_second_moment(folded, calib, device=str(dev))
        # The basis is taken from a moment where every layer gets an equal vote. Summing raw
        # second moments lets the high-norm deep layers drown out the rest (§11).
        basis_S = (llama_balanced_second_moment(folded, calib, device=str(dev))
                   if c.basis_pool == "balanced" else S)
        if c.basis == "gn":
            M = llama_logit_metric(folded).to(dev)
            V0 = residual_basis(basis_S, c.hidden, method="gn", M=M)
        else:
            V0 = residual_basis(basis_S, c.hidden, method=c.basis)
        V0 = V0.to(dev)
        self._V0 = V0.detach().clone()

        # Score neurons by what they write into the *retained* subspace: a neuron writing
        # hard into a direction V0 discards costs the student nothing.
        idx = ffn_energy_indices(folded, calib, c.intermediate, device=str(dev), V=V0)
        norm_S = (llama_norm_input_second_moments(folded, calib, device=str(dev))
                  if c.per_norm_gain else None)
        self.restricted = RestrictedLlama(
            folded, S, V0, idx, c.n_head, c.n_kv, free=c.free_core, norm_S=norm_S
        ).to(dev)
        return self

    # -- phase 2 ------------------------------------------------------------
    def fit(self, train_loader) -> LearnedRestriction:
        """Train the restriction against the KD loss. Returns ``self``."""
        if self.restricted is None:
            raise RuntimeError("call prepare() before fit()")
        c = self.config
        rm, dev = self.restricted, self.device
        teacher = self.folded
        kd_fn = _KD_FNS.get(c.kd)
        if kd_fn is None:
            raise ValueError(f"unknown kd objective {c.kd!r}; choose from {sorted(_KD_FNS)}")
        v_lr = c.resolved_v_lr(self.teacher_src)

        rm.train()
        stiefel, euclid = rm.param_groups()
        opt_v = StiefelAdamV(stiefel, lr=v_lr, trust_region=c.v_trust_region)
        sch_v = _cosine_warmup(opt_v, c.steps, c.warmup_frac, floor=c.v_floor)
        opt_d = sch_d = None
        if euclid:
            opt_d = torch.optim.AdamW(euclid, lr=c.lr, weight_decay=0.01)
            sch_d = _cosine_warmup(opt_d, c.steps, c.warmup_frac)
        aux_until = int(c.aux_until_frac * c.steps)

        history: list[dict] = []
        for step, ids in enumerate(
                _cycle_ids(train_loader, c.steps, dev, seed=c.seed, shuffle=c.shuffle)):
            use_aux = c.aux_w > 0 and step < aux_until
            if use_aux:
                with torch.no_grad():
                    t_out = teacher(input_ids=ids, output_hidden_states=True)
                    t_logits = t_out.logits[:, :-1]
                s_out, hs_s = rm.hidden_and_logits(ids, stream=c.aux_stream)
                loss = kd_fn(s_out.logits[:, :-1], t_logits)
                # Anneal the aux term to zero: it shapes V's early travel, then hands off to
                # the true KD objective so the final basin is chosen by KD alone.
                lam = c.aux_w * (1.0 - step / max(1, c.steps))
                aux = _restriction_consistency(rm.V, hs_s, t_out.hidden_states)
                loss = loss + lam * aux
            else:
                with torch.no_grad():
                    t_logits = teacher(input_ids=ids).logits[:, :-1]
                s_logits = rm(ids).logits[:, :-1]
                loss = kd_fn(s_logits, t_logits)

            opt_v.zero_grad(set_to_none=True)
            if opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
            loss.backward()
            if c.grad_clip:
                # V's step length is already bounded by the trust region (which renormalizes
                # the update anyway), so clipping its gradient there would change nothing
                # except the direction's conditioning.
                if not c.v_trust_region:
                    torch.nn.utils.clip_grad_norm_(stiefel, c.grad_clip)
                if euclid:
                    torch.nn.utils.clip_grad_norm_(euclid, c.grad_clip)
            opt_v.step()
            sch_v.step()
            if opt_d is not None:
                opt_d.step()
                sch_d.step()

            lv = float(loss.detach())
            history.append({"step": step, "kd": lv})
            if c.log_every and (step % c.log_every == 0 or step == c.steps - 1):
                print(f"[lrd] step {step:>5}/{c.steps}  kd={lv:.4f}", flush=True)

        self.history = history
        return self

    # -- phase 3 ------------------------------------------------------------
    def fold(self) -> nn.Module:
        """Return a plain ``LlamaForCausalLM`` with weights ``V^T W_T V (+ D)``."""
        if self.restricted is None:
            raise RuntimeError("call prepare()/fit() before fold()")
        self.folded_student = self.restricted.fold().eval()
        return self.folded_student

    def principal_angle(self) -> float:
        """Largest principal angle (radians) between the trained ``V`` and its init ``V0``."""
        if self.restricted is None or self._V0 is None:
            return float("nan")
        s = torch.linalg.svdvals(self._V0.T @ self.restricted.V.detach()).clamp(-1, 1)
        return float(s.arccos().max())


# ---------------------------------------------------------------------------
# One-call entry point
# ---------------------------------------------------------------------------
def learned_restriction_distill(
    teacher: nn.Module,
    train_loader,
    *,
    config: LRDConfig,
    calib_loader=None,
) -> LRDResult:
    """Distil ``teacher`` into a restricted student and fold it for inference.

    Runs :meth:`LearnedRestriction.prepare` -> :meth:`~LearnedRestriction.fit` ->
    :meth:`~LearnedRestriction.fold`. When ``calib_loader`` is ``None``, the first
    ``config.calib_batches`` batches of ``train_loader`` are reused to profile the
    teacher's residual stream.

    Returns an :class:`LRDResult` whose ``student`` is a plain ``LlamaForCausalLM`` with
    zero inference overhead.
    """
    _require_llama_family(teacher)
    lrd = LearnedRestriction(teacher, config)
    calib = calib_loader if calib_loader is not None else _take(train_loader, config.calib_batches)
    lrd.prepare(calib)
    lrd.fit(train_loader)
    student = lrd.fold()
    hist = getattr(lrd, "history", [])
    return LRDResult(
        student=student,
        history=hist,
        max_principal_angle=lrd.principal_angle(),
        final_kd=hist[-1]["kd"] if hist else None,
        config=config,
        restriction_gap=lrd.restricted.restriction_gap(),
    )


# ---------------------------------------------------------------------------
# Small dataloader helpers
# ---------------------------------------------------------------------------
def _take(loader, n: int) -> list:
    out = []
    for i, b in enumerate(loader):
        if i >= n:
            break
        out.append(b)
    return out


def _cycle_ids(loader, steps: int, device, *, seed: int = 0, shuffle: bool = True,
               buffer_cap: int = 16384):
    """Yield exactly ``steps`` ``input_ids`` tensors, in seeded random order by default.

    A plain, *unshuffled* list is a natural thing to hand this API -- but training the
    projection ``V`` on the residual stream is unusually sensitive to a low-diversity data
    trajectory: walking such a loader in order costs ~20 PPL on the llama-160m benchmark
    versus drawing batches at random (measured 2026-07-11). So the loader is buffered (up to
    ``buffer_cap`` batches, kept on their source device to spare GPU memory) and, with
    ``shuffle``, drawn in a fresh seeded permutation each pass -- matching the research
    driver's ``_batches`` and making the public entry point robust to how the caller's
    loader is built. Pass ``shuffle=False`` when the loader already shuffles every epoch.
    """
    buf: list[Tensor] = []
    truncated = False
    for i, batch in enumerate(loader):
        if i >= buffer_cap:
            truncated = True
            break
        buf.append(_as_input_ids(batch))
    if not buf:
        raise ValueError("train_loader yielded no batches")
    if truncated:
        # Say so. A silent cap reads as "we trained on your corpus" when we trained on a
        # prefix of it, and the difference is invisible in the loss curve.
        warnings.warn(
            f"train_loader has more than buffer_cap={buffer_cap} batches; only the first "
            f"{buffer_cap} are used (shuffling draws from that prefix). Raise buffer_cap to "
            f"cover the whole corpus, or pass a loader that reshuffles each epoch with "
            f"shuffle=False.",
            RuntimeWarning, stacklevel=2)

    if not shuffle:
        for s in range(steps):
            yield buf[s % len(buf)].to(device)
        return

    g = torch.Generator().manual_seed(seed)
    order: list[int] = []
    for s in range(steps):
        if s % len(buf) == 0:
            order = torch.randperm(len(buf), generator=g).tolist()
        yield buf[order[s % len(buf)]].to(device)
