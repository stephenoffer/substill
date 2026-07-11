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

The compressed student is parameterized as an exact restriction of the teacher, and the
residual-stream projection ``V`` is trained **on the Stiefel manifold, against the KD
loss, through the whole network** -- jointly with a zero-initialized Euclidean residual
``D`` that starts training at exactly the absorbed-init student a plain baseline begins
from. The student stays inside the class of restrictions at every step and folds to a
plain ``LlamaForCausalLM`` with zero inference overhead.

Measured on ``JackFram/llama-160m`` (WikiText-2, 3.07x, n=3): LRD reaches
75.45 +/- 0.79 PPL against the strongest frozen-basis baseline's 80.94 +/- 0.90 -- a
6.8% win at ~6 sigma that holds at matched wall-clock and grows with teacher scale
(see ``docs/learned_restriction.md`` for the full study, scaling, and honest caveats).

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
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from .compression.llama_absorb import (
    gamma_fold_llama,
    llama_logit_metric,
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
    residual width lands on a multiple of the teacher's head dimension. The FFN width and
    key/value-head count scale with the same ratio, with ``n_head % n_kv == 0`` enforced
    for the grouped-query attention the student config requires.
    """
    if not 0.0 < width_ratio <= 1.0:
        raise ValueError(f"width_ratio must be in (0, 1], got {width_ratio}")
    _require_llama_family(teacher)
    c = teacher.config
    d = int(c.hidden_size)
    heads = int(c.num_attention_heads)
    kv_heads = int(getattr(c, "num_key_value_heads", heads))
    head_dim = d // heads

    n_head = max(1, round(width_ratio * heads))
    hidden = n_head * head_dim
    # kv heads scale with the ratio, then snap down to a divisor of n_head.
    n_kv = max(1, round(width_ratio * kv_heads))
    n_kv = min(n_kv, n_head)
    while n_head % n_kv != 0:
        n_kv -= 1
    interm = max(1, round(width_ratio * int(c.intermediate_size)))
    return RestrictedGeometry(hidden=hidden, intermediate=interm, n_head=n_head, n_kv=n_kv)


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------
_KD_FNS = {"forward_kl": forward_kl, "reverse_kl": reverse_kl, "skew_kl": skew_kl}


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
    v_lr: float = 0.0          # Stiefel lr for V; 0.0 selects the scale-aware default
    kd: str = "forward_kl"     # forward_kl | reverse_kl | skew_kl
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    # --- restriction ---
    basis: str = "pca"         # V0 initializer: pca | identity | gn (logit-metric)
    free_core: bool = True     # train V *and* D jointly (the verified winning arm)
    # --- data / bookkeeping ---
    calib_batches: int = 16
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
        """The Stiefel LR, applying the scale-aware default when ``v_lr <= 0``.

        The projection LR is the one knob that must scale with the teacher: too high and
        ``V`` over-rotates. The rule ``v_lr = min(1e-3, 0.77 / d)`` tracks it across the
        160M -> 2.7B range measured in ``docs/learned_restriction.md``.
        """
        if self.v_lr > 0:
            return self.v_lr
        return min(1e-3, 0.77 / int(teacher.config.hidden_size))


@dataclass
class LRDResult:
    """Return value from :func:`learned_restriction_distill`."""

    student: nn.Module               # a plain, folded LlamaForCausalLM (zero overhead)
    history: list[dict] = field(default_factory=list)
    max_principal_angle: float | None = None   # how far V rotated from its init, radians
    final_kd: float | None = None
    config: LRDConfig | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _as_input_ids(batch) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, dict):
        return batch["input_ids"]
    return batch[0]


def _cosine_warmup(opt, steps: int, warm_frac: float):
    warm = max(1, int(warm_frac * steps))

    def lr_at(s: int) -> float:
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


class LearnedRestriction:
    """Distil a Llama-family teacher by training its residual-stream restriction.

    Three phases, each a public method so callers can inspect or checkpoint between them:

    * :meth:`prepare` -- gamma-fold the teacher (function-preserving), measure the residual
      second moment, select the FFN neurons and the ``V0`` basis, and build the trainable
      :class:`~substill.compression.restricted.RestrictedLlama`.
    * :meth:`fit` -- descend the KD loss on the teacher's own logits, moving ``V`` on the
      Stiefel manifold (and, with ``free_core``, ``D`` in the Euclidean directions).
    * :meth:`fold` -- collapse ``(V, D)`` into a plain ``LlamaForCausalLM``.
    """

    def __init__(self, teacher: nn.Module, config: LRDConfig):
        _require_llama_family(teacher)
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

        S = llama_residual_second_moment(folded, calib, device=str(dev))
        if c.basis == "gn":
            M = llama_logit_metric(folded).to(dev)
            V0 = residual_basis(S, c.hidden, method="gn", M=M)
        else:
            V0 = residual_basis(S, c.hidden, method=c.basis)
        V0 = V0.to(dev)
        self._V0 = V0.detach().clone()

        idx = ffn_energy_indices(folded, calib, c.intermediate, device=str(dev))
        self.restricted = RestrictedLlama(
            folded, S, V0, idx, c.n_head, c.n_kv, free=c.free_core
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
        opt_v = StiefelAdamV(stiefel, lr=v_lr)
        sch_v = _cosine_warmup(opt_v, c.steps, c.warmup_frac)
        opt_d = sch_d = None
        if euclid:
            opt_d = torch.optim.AdamW(euclid, lr=c.lr, weight_decay=0.01)
            sch_d = _cosine_warmup(opt_d, c.steps, c.warmup_frac)

        history: list[dict] = []
        for step, ids in enumerate(_cycle_ids(train_loader, c.steps, dev)):
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits[:, :-1]
            s_logits = rm(ids).logits[:, :-1]
            loss = kd_fn(s_logits, t_logits)

            opt_v.zero_grad(set_to_none=True)
            if opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
            loss.backward()
            if c.grad_clip:
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


def _cycle_ids(loader, steps: int, device):
    """Yield exactly ``steps`` ``input_ids`` tensors, cycling the loader if it is short."""
    materialized: list[Tensor] = []
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            if not materialized:
                raise ValueError("train_loader yielded no batches") from None
            it = iter(materialized)
            batch = next(it)
        ids = _as_input_ids(batch).to(device)
        if len(materialized) < 4096:      # cache a bounded prefix for cheap cycling
            materialized.append(ids)
        yield ids
