"""Learned restriction: distil a student by training the *projection* alongside the weights.

(The projection *instead of* the weights is the ``free=False`` arm. It is the cleaner object --
a genuine point on the Grassmannian -- but it reaches only ~100 PPL on its own. The arm that
wins trains ``V`` **and** a free weight residual ``D`` jointly; see "The construction" below,
and be careful not to claim more for it than is true.)

`docs/init_findings.md` ends on one principle that survived every experiment and both
architectures tested:

    A change of basis **restricts** the teacher's operator -- ``W_s = V^T W V`` is the
    teacher's weight seen through a subspace, so the teacher's layers still compose the
    way they did. A refit **replaces** the operator with a regression solution that merely
    reproduces the teacher's activations on a calibration set. Restriction transfers
    through distillation; replacement does not.

Six criteria were tried for choosing that subspace -- variance ranking, logit-weighted
variance, ablation importance, coverage, layerwise refit, and the Grassmann-optimal
*logit-error* basis -- and not one beat plain PCA (§10b). But every one of them is a
**surrogate**: each optimizes some proxy of the student's quality (retained variance,
linearized logit error) rather than the quantity actually being minimized, the KD loss of
the assembled student. §10b diagnoses its own failure precisely::

    ``M = W_lm^T W_lm`` is the Jacobian of the *final* layer alone, while the residual
    basis is shared by all twelve -- a direction that barely reaches the logits directly
    may be exactly what layer 3 needs to compute what layer 9 writes.

So the missing arm is the un-surrogated one: **optimize the restriction against the KD
loss, through the whole network.** That is what this module does.

The construction
----------------
A `RestrictedLlama` holds the teacher's (gamma-folded) weights frozen and a single
column-orthonormal ``V in St(d, k)``. Its forward pass materializes, on the fly,

    embed  = W_E V              lm_head = W_lm V
    q,k,v  = W_{qkv}[:rows] V   o       = V^T W_o[:, :rows]
    gate,up= W_{g,u}[idx] V     down    = V^T W_d[:, idx]
    norms  = sqrt(d/k * rho(V)) * 1     (the RMS gain, differentiable in V)

and runs the narrow student.

With ``free=False`` **every reachable point of the parameter space is an exact restriction of
the teacher** -- there is no way to leave the class, by construction, and the trainable object
is a point on the Grassmannian ``G(d, k)``: 147k degrees of freedom for Llama-160m against the
folded student's 30M weights.

**With ``free=True`` -- the arm that actually wins -- that is not true, and saying it anyway
would be the easiest thing in this repository for a reviewer to falsify.** ``D`` is an
unconstrained Euclidean residual on every weight, so ``W_s = V^T W_T V + D`` can reach *any*
student the baseline can. The restriction is then not an invariant of training. It is two
other things, both real and both weaker:

* an **initialization** -- ``D = 0``, so training starts at exactly the absorbed-init student
  the baseline starts from; and
* a **coordinate system** -- the ``V`` direction moves all layers coherently, along the one
  direction that keeps the student a restriction, while ``D`` moves each weight on its own.

Whether the trained student *stays* near the class is then an empirical question, not a
theorem. `restriction_gap` measures it. (Weight decay on ``D`` pulls it back toward zero, i.e.
toward the restriction, which is a regularizer toward the class rather than toward the origin
-- worth knowing, and worth ablating, but it is not a constraint.)

Two properties do hold in both arms:

1. It never linearizes. The gradient ``dL/dV`` is accumulated from every layer that reads
   or writes the residual stream, so a direction layer 3 needs is paid for by layer 3.
2. It folds. `fold()` returns a plain `LlamaForCausalLM` whose weights are ``V^T W V (+ D)``,
   bit-identical in function to the restricted module (`tests/compression/test_restricted.py`),
   so inference carries zero overhead.

Two structural preconditions the map makes on the teacher, both enforced rather than assumed
(`tests/compression/test_lrd_soundness.py`):

* the ``lm_head`` must be **untied** from the input embedding before the gamma fold, or
  folding the final norm into the head silently rescales the embedding (`untie_lm_head`);
* the student must keep the teacher's **grouped-query group size**, or its query heads are
  re-paired with key/value heads they were never trained against (`check_head_geometry`).

Scaling. The trainable state is one ``(d, k)`` matrix regardless of teacher depth or
vocabulary, and the teacher's weights are read-only -- no optimizer state on them. That is
what makes the same recipe run on a 30M student and, in principle, on a frontier decoder,
where materializing ``W_s`` for every edge is affordable but ``|W_T|`` optimizer states are
not.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call

from .llama_absorb import (
    absorb_llama,
    build_narrow_llama,
    check_head_geometry,
    norm_param_names,
)

__all__ = [
    "RestrictedLlama",
    "StiefelAdamV",
    "ffn_energy_indices",
    "polar_retract",
    "qr_retract",
]


# ---------------------------------------------------------------------------
# Stiefel utilities
# ---------------------------------------------------------------------------
def _sym(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def _tangent(G: Tensor, V: Tensor) -> Tensor:
    """Project ``G`` onto ``T_V St(d, k) = {Z : V^T Z + Z^T V = 0}``."""
    return G - V @ _sym(V.transpose(-1, -2) @ G)


def _horizontal(T: Tensor, V: Tensor) -> Tensor:
    """The part of a tangent vector ``T`` that actually moves the *subspace*.

    The Stiefel tangent space at ``V`` splits into two pieces that do very different things::

        T  =  V skew(V^T T)   +   (I - V V^T) T
              [   vertical   ]   [  horizontal  ]

        vertical    spins the basis inside span(V) -- the subspace does not move
        horizontal  tilts the subspace out of span(V) -- the subspace moves

    Only the horizontal part changes ``span(V)``, and therefore only the horizontal part
    changes the model when the model depends on the subspace alone (``free=False``). This is
    not a technicality: a uniformly random tangent direction is ~36% vertical at ``(d, k) =
    (64, 16)``, so a trust region that budgets the *whole* tangent step over-counts the motion
    that matters.

    (With ``free=True`` the vertical part is not idle -- ``D`` is expressed in ``V``'s
    coordinates, so re-basing ``V`` against a fixed ``D`` does change the function -- which is
    why the optimizer budgets the full tangent step rather than the horizontal one alone. See
    `StiefelAdamV`.)
    """
    return T - V @ _skew(V.transpose(-1, -2) @ T)


def _skew(A: Tensor) -> Tensor:
    return 0.5 * (A - A.transpose(-1, -2))


def polar_retract(A: Tensor) -> Tensor:
    """Retract ``A`` onto the Stiefel manifold by the polar factor ``A (A^T A)^{-1/2}``.

    This is the **metric projection**: the closest column-orthonormal matrix to ``A`` in the
    Frobenius norm. Two properties make it the right retraction here, and QR the wrong one.

    **It is gauge-equivariant.** ``polar(A R) = polar(A) R`` for any orthogonal ``R``, exactly.
    A Grassmann point has no preferred basis, so a retraction that answers differently
    depending on which representative of the subspace you hand it is a retraction that
    silently injects a coordinate choice into the trajectory. QR does exactly that --
    ``qr(A R) != qr(A) R`` (measured: 0.195 max abs at ``(64, 16)``) -- because the
    upper-triangular factor pins a canonical, order-dependent basis.

    **It has no sign ambiguity.** ``torch.linalg.qr`` leaves each column of ``Q`` free up to a
    sign, so a QR retraction is *discontinuous*: an arbitrarily small step can flip a column
    and, with it, the sign of that column's momentum. `qr_retract` patches this by pinning the
    signs of ``diag(R)``, but the patch is a symptom -- the polar factor is continuous and
    needs no patch.

    Cost is ``O(d k^2 + k^3)``, the same order as the thin QR it replaces.
    """
    w, U = torch.linalg.eigh(A.transpose(-1, -2) @ A)
    inv_sqrt = U @ torch.diag_embed(w.clamp_min(1e-12).rsqrt()) @ U.transpose(-1, -2)
    return A @ inv_sqrt


def qr_retract(A: Tensor) -> Tensor:
    """Retract ``A`` onto the Stiefel manifold by a sign-fixed thin QR.

    ``torch.linalg.qr`` leaves the sign of each column of ``Q`` free; without the fix the
    retraction is discontinuous and momentum flips sign at random.

    Kept for the published ambient-step numbers and for callers that want the cheaper factor.
    `polar_retract` is the default and the better-behaved choice -- QR is *not*
    gauge-equivariant, and the sign fix here is a patch over a discontinuity the polar factor
    simply does not have.
    """
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.sign(R.diagonal()) + 0.5)


class StiefelAdamV(torch.optim.Optimizer):
    """Adam on the Riemannian gradient of a Stiefel parameter, with a QR retraction.

    `substill.training.stiefel_optim.StiefelAdam` uses Cayley + Adafactor row/col statistics,
    which is the right choice when a model carries dozens of bases. Here there is exactly
    one ``(d, k)`` matrix, so full elementwise second moments are affordable and strictly
    better conditioned.

    Four things this gets right that a naive port of Adam to a manifold does not.

    **The step is measured in rotation, not in ambient length** (``trust_region=True``,
    default). An Adam direction has entries of order 1, so its Frobenius norm grows like
    ``sqrt(d k)``: the *same* ``lr`` rotates a wide teacher's subspace much further per step
    than a narrow one's, which is why an ambient ``lr`` has to be re-tuned for every teacher
    size and why this repo previously carried the fitted constant ``v_lr = min(1e-3, 0.77/d)``
    -- three teachers, one magic number, no reason to extrapolate. Normalizing the update to
    ``||D||_F = sqrt(k)`` makes ``lr`` **the RMS angle, in radians, that ``V`` turns per step**:
    to first order the principal angles satisfy ``theta_j ~ lr * sigma_j(D)``, so
    ``rms_j(theta_j) = lr * ||D||_F / sqrt(k) = lr``.

    Precisely: ``lr`` bounds the *subspace* rotation and equals it when the step is horizontal.
    A tangent step splits into a **horizontal** part (which tilts ``span(V)``) and a
    **vertical** part (which merely re-bases ``V`` inside its own span -- see `_horizontal`).
    Only the horizontal part rotates the subspace, so the realized rotation is
    ``lr * ||horizontal|| / ||step||``: exactly ``lr`` when the loss depends on ``span(V)``
    alone (``free_core=False``, where the gradient *is* purely horizontal), and ~0.93 ``lr`` at
    ``k/d = 1/4`` for a general gradient. Measured: constant across a 64x change in ``d``, and
    unchanged by a gradient six orders of magnitude larger
    (`tests/compression/test_stiefel_geometry.py`).

    This also subsumes gradient clipping for ``V``: the step length is bounded by construction,
    so no spike can throw the basis.

    **Momentum is transported.** ``m`` is a tangent vector at ``V``, but the retraction moves
    ``V``; leaving ``m`` where it was makes it a vector in the wrong tangent space, and its
    normal component then leaks into the next update. After each step ``m`` is projected onto
    the new tangent space -- the standard (projection) vector transport for the Stiefel
    manifold. ``v`` is a magnitude, not a direction, so it is not transported.

    **The retraction is the polar factor, not QR.** ``polar`` is the metric projection and is
    exactly gauge-equivariant; QR is neither, and its column-sign ambiguity needs a patch that
    the polar factor simply does not (see `polar_retract`). ``retraction="qr"`` reproduces the
    published ambient numbers.

    **Tangency survives preconditioning.** Adam's elementwise rescaling does not preserve
    tangency, so the direction is re-projected *after* dividing by ``sqrt(v)``, not before.

    One honest limitation. The elementwise second moment ``v`` is not equivariant under the
    gauge ``V -> V R``, ``R in O(k)`` -- the trajectory depends on which representative of the
    subspace we hold. With ``free_core=True`` (the default) the loss is not gauge-invariant
    either (``D`` is expressed in ``V``'s coordinates), so there is no symmetry to break and
    this is merely a choice of coordinates. With ``free_core=False`` the loss *is* a function
    on the Grassmannian alone, and then this preconditioner is formally a gauge-dependent
    approximation. Set ``gauge_invariant=True`` for the scalar second moment
    ``v <- EMA(||g||_F^2)``, which is equivariant, at the usual cost of a coarser
    preconditioner.
    """

    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.99), eps: float = 1e-8,
                 trust_region: bool = True, gauge_invariant: bool = False,
                 retraction: str = "polar"):
        if retraction not in ("polar", "qr"):
            raise ValueError(f"retraction must be 'polar' or 'qr', got {retraction!r}")
        super().__init__(list(params), {"lr": lr, "betas": betas, "eps": eps,
                                        "trust_region": trust_region,
                                        "gauge_invariant": gauge_invariant,
                                        "retraction": retraction})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            eps, gauge = group["eps"], group["gauge_invariant"]
            retract = polar_retract if group["retraction"] == "polar" else qr_retract
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = (torch.zeros((), device=p.device, dtype=p.dtype) if gauge
                               else torch.zeros_like(p))
                st["t"] += 1
                g = _tangent(p.grad, p)                     # Riemannian gradient
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                if gauge:
                    st["v"].mul_(b2).add_(g.pow(2).sum(), alpha=1 - b2)
                else:
                    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = st["m"] / (1 - b1 ** st["t"])
                vh = st["v"] / (1 - b2 ** st["t"])
                # Re-project: Adam's rescaling does not preserve tangency.
                d = _tangent(mh / (vh.sqrt() + eps), p)
                if group["trust_region"]:
                    # ||d||_F = sqrt(k)  =>  lr is the RMS angle, in radians, that V turns
                    # per step. When the loss depends only on span(V) (free_core=False) the
                    # gradient is purely horizontal and that is exactly the *subspace*
                    # rotation; with a free core, part of the step is vertical (a re-basing
                    # of V against a fixed D, which is real motion), and lr bounds the total.
                    k = p.shape[-1]
                    d = d * (k ** 0.5 / d.norm().clamp_min(eps))
                p.copy_(retract(p - group["lr"] * d))
                # Vector transport: m lived in the old tangent space; V has moved.
                st["m"].copy_(_tangent(st["m"], p))
        return loss


# ---------------------------------------------------------------------------
# FFN intermediate selection (fixed; SiLU is elementwise so only *selection* commutes)
# ---------------------------------------------------------------------------
@torch.no_grad()
def ffn_energy_indices(teacher, calib, interm: int, device="cuda",
                       V: Tensor | None = None) -> list[Tensor]:
    """Indices of the ``interm`` FFN neurons that write the most into the residual stream.

    A neuron is one coordinate of ``silu(gate) * up``, i.e. one column of ``down_proj``. A
    *rotation* of that space would not commute with SiLU; a selection does, exactly -- which
    is why the FFN inner dimension is selected rather than rotated.

    **What a neuron is worth is what it writes, not what it holds.** Neuron ``i`` contributes
    ``a_i * W_down[:, i]`` to the residual stream, so dropping it perturbs the stream by that
    vector; its expected squared cost is ``E[a_i^2] * ||W_down[:, i]||^2``. Ranking on the
    activation energy ``E[a_i^2]`` alone -- as this function used to -- treats a loud neuron
    whose output column is near zero (it writes nothing) as more important than a quiet one
    whose output column is large (it writes a lot). That is exactly backwards, and it is a
    ranking error, so it survives any rescaling of the calibration set.

    Passing the residual basis ``V`` measures the write *where the student will feel it*: the
    student only sees the component inside the subspace, so the norm that matters is
    ``||V^T W_down[:, i]||``, not ``||W_down[:, i]||``. A neuron writing hard into a direction
    ``V`` discards costs the student nothing. ``V=None`` falls back to the full-space norm.

    (The criterion is still greedy -- it ignores correlation between neurons, so a pair of
    near-duplicate neurons is double-counted. Fixing that means a subset-selection problem,
    not a ranking; it is out of scope here and noted in ``docs/learned_restriction.md``.)
    """
    teacher = teacher.to(device).eval()
    L = teacher.config.num_hidden_layers
    d_int = teacher.config.intermediate_size
    acc = [torch.zeros(d_int, dtype=torch.float64, device=device) for _ in range(L)]

    def mk(li):
        def hook(_m, inp, _o):
            acc[li] += (inp[0].detach().double() ** 2).reshape(-1, d_int).sum(0)
        return hook

    hooks = [layer.mlp.down_proj.register_forward_hook(mk(li))
             for li, layer in enumerate(teacher.model.layers)]
    for b in calib:
        teacher(input_ids=b["input_ids"].to(device))
    for h in hooks:
        h.remove()

    out = []
    for li, a in enumerate(acc):
        if interm >= d_int:
            out.append(torch.arange(d_int, device=device))
            continue
        W_d = teacher.model.layers[li].mlp.down_proj.weight.detach().to(device).double()
        write = W_d if V is None else V.to(device).double().T @ W_d      # (d or k, d_int)
        score = a * write.pow(2).sum(0)                                  # E[a_i^2]*||write_i||^2
        out.append(torch.argsort(score, descending=True)[:interm])
    return out


def indices_to_bases(idx: list[Tensor], d_int: int) -> list[Tensor]:
    """The ``(d_int, interm)`` selection matrices ``E`` that `absorb_llama` expects."""
    out = []
    for ix in idx:
        E = torch.zeros(d_int, ix.numel(), device=ix.device)
        E[ix, torch.arange(ix.numel(), device=ix.device)] = 1.0
        out.append(E)
    return out


# ---------------------------------------------------------------------------
# The restricted student
# ---------------------------------------------------------------------------
class _Out:
    """Minimal stand-in for a HF `CausalLMOutput`, so `eval_ppl` needs no branch."""

    __slots__ = ("logits",)

    def __init__(self, logits: Tensor):
        self.logits = logits


class RestrictedLlama(nn.Module):
    """A narrow Llama whose every weight is ``V^T W_T V``, with ``V`` the only parameter.

    Parameters
    ----------
    folded : LlamaForCausalLM
        The teacher, already gamma-folded (`gamma_fold_llama`). Kept frozen.
    S : Tensor
        ``E[h h^T]`` of the teacher's residual stream, used for the differentiable RMS
        gain. Frozen.
    V0 : Tensor
        ``(d, k)`` column-orthonormal starting point (PCA, identity, random -- the arm).
    idx : list[Tensor]
        Per-layer FFN neuron indices from `ffn_energy_indices`.
    n_head, n_kv : int
        Student head counts. Heads are kept whole and taken in order, the arbitrary choice
        `docs/init_findings.md` §9a-9b found no rule beats.
    free : bool
        Add a zero-initialized Euclidean residual ``D`` to every student weight, so the
        model becomes ``W_s = V^T W_T V + D``. This is the *same function class* as an
        ordinary absorbed-init student -- ``D`` alone can reach any weight -- expressed in a
        different **coordinate system**, and carrying ``d * k`` extra parameters for ``V``
        itself (295k on Llama-160m, ~1% of the student's 30M; the claim of an *identical*
        parameter count, made in earlier drafts, was wrong).

        Moving ``V`` moves all twelve layers coherently, in the one direction that keeps the
        student a restriction of the teacher; moving ``D`` is the ordinary per-weight
        direction. Since ``D`` starts at zero, training begins at exactly the absorbed-init
        student the baseline starts from, so a comparison against it (with the Stiefel LR set
        to zero -- the ``pca_reparam`` control) isolates the added coordinate and nothing
        else.

        Note what this does **not** say: with ``free=True`` the student is *not* confined to
        the restriction class during training. See the module docstring and
        `restriction_gap`.
    norm_S : Tensor, optional
        ``(2L+1, d, d)`` second moments at each RMSNorm's input, from
        `llama_norm_input_second_moments`. Gives every norm the gain its *own* input
        distribution loses. Without it all ``2L+1`` norms share one gain pooled over every
        layer at once, which is measurably wrong -- 39% too large at layer 0's norms on
        llama-160m at 3.07x -- because the pooled moment is dominated by the high-norm deep
        layers and describes the raw embedding not at all.
    """

    def __init__(self, folded, S: Tensor, V0: Tensor, idx: list[Tensor],
                 n_head: int, n_kv: int, *, free: bool = False,
                 norm_S: Tensor | None = None):
        super().__init__()
        check_head_geometry(folded, n_head, n_kv)
        self.teacher = [folded]              # hidden from `.parameters()` / `.to()`
        d, k = V0.shape
        self.d, self.k = d, k
        self.idx = idx
        self.n_head, self.n_kv = n_head, n_kv
        self.free = free

        tc = folded.config
        self.head_dim = tc.hidden_size // tc.num_attention_heads
        self.q_rows = n_head * self.head_dim
        self.kv_rows = n_kv * self.head_dim
        self.interm = idx[0].numel()

        self.V = nn.Parameter(V0.detach().clone().float())
        self.V.is_stiefel = True
        self.register_buffer("S", S.detach().float())
        self.register_buffer("trS", S.detach().float().diagonal().sum())

        # Per-norm gains: the scale each RMSNorm's *own* input distribution loses under the
        # truncation. `norm_S` is the (2L+1, d, d) stack from
        # `llama_norm_input_second_moments`; without it every norm falls back to the single
        # pooled gain, which is measurably wrong (39% at layer 0 on llama-160m).
        self.norm_names = norm_param_names(folded)
        self.per_norm = norm_S is not None
        if self.per_norm:
            if norm_S.shape[0] != len(self.norm_names):
                raise ValueError(
                    f"norm_S has {norm_S.shape[0]} second moments but the teacher has "
                    f"{len(self.norm_names)} norms (2*{tc.num_hidden_layers}+1)")
            nS = norm_S.detach().float()
            self.register_buffer("norm_S", nS)
            self.register_buffer("norm_trS", nS.diagonal(dim1=-2, dim2=-1).sum(-1))

        # A parameterless skeleton with the student's exact geometry. Its own weights are
        # never used or trained -- `functional_call` substitutes ours on every forward, so
        # `fold()` writing into a fresh copy of the same config is exact.
        self.skeleton = build_narrow_llama(folded, k, self.interm, n_head, n_kv)
        self.skeleton.requires_grad_(False)
        self._hollow_skeleton()

        if free:
            self.D = nn.ParameterDict(
                {self._key(n): nn.Parameter(torch.zeros_like(t))
                 for n, t in self._restriction(detach=True).items()})
            vocab = int(tc.vocab_size)
            self.D_emb = nn.Parameter(torch.zeros(vocab, k))
            self.D_lm = nn.Parameter(torch.zeros(vocab, k))

    @staticmethod
    def _key(name: str) -> str:
        return name.replace(".", "__")

    @torch.no_grad()
    def _hollow_skeleton(self) -> None:
        """Free the skeleton's weight storage. Not one byte of it is ever read.

        The skeleton exists only to supply the student's *module graph* -- its shapes, its RoPE
        buffers, its forward code. Every weight it needs is handed to it by `functional_call`
        on each forward (`restricted_params`), and the two it is *not* handed --
        ``embed_tokens`` and ``lm_head`` -- are never called at all: the embedding is passed in
        as ``inputs_embeds``, and the logits are lifted through the *teacher's* head.

        So the randomly-initialized weights `LlamaForCausalLM(cfg)` allocates are dead on
        arrival, and at any real scale they are not cheap: a 1024-wide student of a 1.3B teacher
        carries ~400M of them, ~1.6 GB, which is enough on its own to put the restricted forward
        over a 22 GB card. Replacing each with a zero-element tensor keeps the module graph and
        drops the storage.

        Buffers (RoPE's ``inv_freq``) are left alone -- those *are* read.
        """
        for p in self.skeleton.parameters():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    # -- the restriction map ------------------------------------------------
    def gain(self) -> Tensor:
        """Return the single pooled gain ``sqrt(d/k * rho(V))``. Differentiable in ``V``."""
        rho = torch.einsum("di,dc,ci->", self.V, self.S, self.V) / self.trS.clamp_min(1e-12)
        return (self.d / self.k) ** 0.5 * rho.clamp_min(1e-12).sqrt()

    def gains(self) -> dict[str, Tensor]:
        """The gain each RMSNorm needs, keyed by student parameter name.

        Norm ``l`` sees its own distribution: its input has second moment ``S_l``, of which
        the student retains the fraction ``rho_l(V) = tr(V^T S_l V) / tr(S_l)``. The student
        normalizes by ``rms(V^T h) = ||V^T h||/sqrt(k)`` where the absorbed weights expect
        ``V^T h * sqrt(d)/||h||``, so the scale that norm must restore is
        ``sqrt(d/k * rho_l(V))`` -- a *different* number at each of the ``2L+1`` norms.

        All ``2L+1`` traces come from one shared projector ``P = V V^T``:
        ``tr(V^T S_l V) = <S_l, P>``. So the whole vector costs one ``(d,k)`` product plus a
        contraction, not ``2L+1`` separate ones, and stays differentiable in ``V``.
        """
        if not self.per_norm:
            return dict.fromkeys(self.norm_names, self.gain())
        P = self.V @ self.V.T                                     # (d, d), rank-k projector
        rho = (self.norm_S * P).sum((-2, -1)) / self.norm_trS.clamp_min(1e-12)
        g = ((self.d / self.k) ** 0.5) * rho.clamp_min(1e-12).sqrt()
        return dict(zip(self.norm_names, g.unbind(0), strict=True))

    def _restriction(self, detach: bool = False) -> dict[str, Tensor]:
        """``V^T W_T V`` for every student weight, as differentiable functions of ``V``."""
        V = self.V.detach() if detach else self.V
        t = self.teacher[0]
        gs = self.gains()
        if detach:
            gs = {n: g.detach() for n, g in gs.items()}
        ones = torch.ones(self.k, device=V.device, dtype=V.dtype)
        p: dict[str, Tensor] = {"norm.weight": gs["norm.weight"] * ones}
        for i, tl in enumerate(t.model.layers):
            a, m = tl.self_attn, tl.mlp
            ix = self.idx[i]
            pre = f"layers.{i}."
            p[pre + "input_layernorm.weight"] = gs[pre + "input_layernorm.weight"] * ones
            p[pre + "post_attention_layernorm.weight"] = (
                gs[pre + "post_attention_layernorm.weight"] * ones)
            # `.float()` is a no-op when the teacher is already fp32 (verified path), and
            # casts a single small weight slice up from bf16 when a huge teacher was loaded
            # in half precision -- so the matmul stays fp32-accurate while the teacher's bulk
            # storage is halved. Peak transient is one layer's weights, not the whole model.
            p[pre + "self_attn.q_proj.weight"] = a.q_proj.weight[: self.q_rows].float() @ V
            p[pre + "self_attn.k_proj.weight"] = a.k_proj.weight[: self.kv_rows].float() @ V
            p[pre + "self_attn.v_proj.weight"] = a.v_proj.weight[: self.kv_rows].float() @ V
            p[pre + "self_attn.o_proj.weight"] = V.T @ a.o_proj.weight[:, : self.q_rows].float()
            p[pre + "mlp.gate_proj.weight"] = m.gate_proj.weight[ix].float() @ V
            p[pre + "mlp.up_proj.weight"] = m.up_proj.weight[ix].float() @ V
            p[pre + "mlp.down_proj.weight"] = V.T @ m.down_proj.weight[:, ix].float()
        return p

    def restricted_params(self) -> dict[str, Tensor]:
        """The student's block weights: the restriction, plus the free residual if any."""
        p = self._restriction()
        if self.free:
            p = {n: w + self.D[self._key(n)] for n, w in p.items()}
        return p

    # -- forward ------------------------------------------------------------
    def forward(self, input_ids: Tensor):
        """Logits of the student, wrapped like a HF output so `eval_ppl` works.

        The embedding and the unembedding are *never materialized*: ``W_E V`` is a
        ``(vocab, k)`` matrix, but only the batch's rows are needed, and
        ``h (W_lm V)^T = (h V^T) W_lm^T`` lifts back through the teacher's own head. This
        keeps the per-step cost independent of vocabulary size -- the property that lets
        the same code run on a frontier decoder.
        """
        t = self.teacher[0]
        # `.float()` no-ops on an fp32 teacher and up-casts a bf16 one (see `_restriction`).
        emb = t.model.embed_tokens.weight[input_ids].float() @ self.V     # (B, T, k)
        if self.free:
            emb = emb + self.D_emb[input_ids]
        h = functional_call(self.skeleton.model, self.restricted_params(),
                            kwargs={"inputs_embeds": emb}).last_hidden_state
        logits = (h @ self.V.T) @ t.lm_head.weight.T.float()
        if self.free:
            logits = logits + h @ self.D_lm.T
        return _Out(logits)

    def hidden_and_logits(self, input_ids: Tensor, *, stream: str = "residual"):
        """Like :meth:`forward`, but also return the per-layer stream the aux loss matches.

        Returns ``(logits, hs)`` with ``hs`` a ``(L, B, T, k)`` stack in the same ``k``-space as
        ``V^T h_T``, differentiable in both ``V`` (via the restriction) and ``D``. Used by the
        restriction-consistency auxiliary loss (`docs/learned_restriction.md` §2b).

        ``stream`` selects **what the last layer contributes**, and the choice is a real one:

        * ``"residual"`` -- every entry is the *raw* post-layer residual. This is what the aux
          term's own description says it matches ("the student's residual stream should point
          the way the teacher's does, seen through ``V``"), and it is uniform across layers.
        * ``"prelogit"`` -- the last entry is the state *after* the student's final norm, i.e.
          the input ``lm_head`` actually reads. Not raw -- it carries the student's learned
          per-channel norm gain ``g*1 + D`` -- but it is still the honest restriction statement
          *at that point in the network*, since the teacher's corresponding state is its own
          ``lm_head`` input, and it additionally supervises the final norm.

        The pre-audit code did ``"prelogit"`` **by accident**: it read both streams off
        HuggingFace's ``output_hidden_states``, whose last entry is post-``model.norm`` rather
        than the last layer's output (the same trap `gap_fit_llama` documents in a comment). So
        the code and its description disagreed, and nobody had chosen. Both are now reachable,
        and the choice is made on measured PPL rather than on which one the docstring happened
        to claim -- see `docs/learned_restriction.md` §9f.
        """
        if stream not in ("residual", "prelogit"):
            raise ValueError(f"stream must be 'residual' or 'prelogit', got {stream!r}")
        t = self.teacher[0]
        emb = t.model.embed_tokens.weight[input_ids].float() @ self.V
        if self.free:
            emb = emb + self.D_emb[input_ids]

        params = self.restricted_params()
        raw: list[Tensor] = []

        def grab(_m, _i, o):
            raw.append(o[0] if isinstance(o, tuple) else o)

        hooks = [layer.register_forward_hook(grab) for layer in self.skeleton.model.layers]
        try:
            out = functional_call(self.skeleton.model, params,
                                  kwargs={"inputs_embeds": emb})
        finally:
            for h_ in hooks:
                h_.remove()

        h = out.last_hidden_state                     # already post-`model.norm`
        logits = (h @ self.V.T) @ t.lm_head.weight.T.float()
        if self.free:
            logits = logits + h @ self.D_lm.T

        hs = list(raw)
        if stream == "prelogit":
            hs[-1] = h
        return _Out(logits), torch.stack(hs, dim=0)

    # -- optimization -------------------------------------------------------
    def param_groups(self):
        """``(stiefel, euclidean)`` -- ``V`` needs a retraction, ``D`` does not."""
        euc = [p for n, p in self.named_parameters()
               if p.requires_grad and not n.endswith("V") and not n.startswith("skeleton")]
        return [self.V], euc

    # -- diagnostics --------------------------------------------------------
    @torch.no_grad()
    def restriction_gap(self) -> float:
        """How far outside the restriction class the student currently sits, in ``[0, inf)``.

        ``||D||_F / ||V^T W_T V||_F`` over all block weights: 0 when the student *is* the
        exact restriction ``V^T W_T V``, and order 1 once the free residual is as large as
        the restriction it was added to.

        This exists because the headline claim needs to be stated truthfully. With
        ``free=False`` the student is a restriction *by construction* -- every reachable point
        of the parameter space is ``V^T W_T V`` for some ``V``, and no gradient can leave the
        class. But the arm that actually wins is ``free=True``, where ``D`` is an
        unconstrained Euclidean residual on every weight and can therefore reach **any**
        student; the class is not an invariant of training, only of the initialization
        (``D = 0``) and of the coordinate system. Claiming otherwise -- as this module's
        docstring and ``docs/learned_restriction.md`` both once did -- is false, and a
        reviewer would find it in a minute.

        Measuring the gap converts the false claim into a real question with a number
        attached: *does* the winning student stay near the restriction class, or does ``D``
        carry it far away and the restriction merely supply a good starting point? Report it;
        do not assume it.
        """
        base = self._restriction(detach=True)
        if not self.free:
            return 0.0
        num = sum(float(self.D[self._key(n)].pow(2).sum()) for n in base)
        den = sum(float(w.pow(2).sum()) for w in base.values())
        return (num / max(den, 1e-12)) ** 0.5

    # -- amortized refresh --------------------------------------------------
    @torch.no_grad()
    def load_student_residual(self, student, V: Tensor) -> None:
        """Set ``V`` and choose ``D`` so this module *equals* ``student`` exactly.

        With ``D = W_student - V^T W_T V`` (per weight), the restricted module reproduces
        ``student`` bit-for-bit, but now expressed in the ``(V, D)`` coordinates. A gradient
        step on ``V`` from here therefore starts at the student the cheap loop has trained,
        not at a stale checkpoint -- so the projection is refined *around the current
        weights*, and everything the Euclidean loop learned is preserved inside ``D``.

        This is what makes the expensive restricted forward a rare event: the student is a
        plain, cheap `LlamaForCausalLM` between refreshes, and only the periodic V-step
        pays the projection cost. On a frontier model, re-projecting every weight each step
        is infeasible; re-projecting once every M steps is not.
        """
        assert self.free, "amortized refresh needs free=True (the residual D)"
        self.V.copy_(V.to(self.V))
        base = self._restriction(detach=True)
        sp = dict(student.model.named_parameters())
        for n, dw in self.D.items():
            plain = n.replace("__", ".")
            dw.copy_(sp[plain].detach() - base[plain])
        self.D_emb.copy_(student.model.embed_tokens.weight.detach() - self._emb_restriction())
        self.D_lm.copy_(student.lm_head.weight.detach() - self._lm_restriction())

    def _emb_restriction(self) -> Tensor:
        return self.teacher[0].model.embed_tokens.weight.detach().float() @ self.V.detach()

    def _lm_restriction(self) -> Tensor:
        return self.teacher[0].lm_head.weight.detach().float() @ self.V.detach()

    @torch.no_grad()
    def write_back(self, student) -> None:
        """Fold the current ``(V, D)`` into ``student`` in place, keeping its param objects.

        The optimizer driving ``student`` keeps its moment estimates attached to the very
        same `nn.Parameter` tensors; only their ``.data`` moves, by the small amount one
        V-step changed the restriction. (Contrast the Periodic Re-Absorption negative
        result in `papers/gap_analysis.md`, whose bug was *resetting* optimizer state on a
        fresh module; here the state is never detached from its parameter.)

        Writes the restriction directly into the student's tensors -- no fresh model is
        allocated, so a refresh costs one restricted forward/backward, not a rebuild.
        """
        base = self._restriction(detach=True)
        sp = dict(student.model.named_parameters())
        for n, dw in self.D.items():
            sp[n.replace("__", ".")].data.copy_(base[n.replace("__", ".")] + dw)
        student.model.embed_tokens.weight.data.copy_(self._emb_restriction() + self.D_emb)
        student.lm_head.weight.data.copy_(self._lm_restriction() + self.D_lm)

    # -- deployment ---------------------------------------------------------
    @torch.no_grad()
    def fold(self):
        """Return a plain `LlamaForCausalLM` with weights ``V^T W_T V (+ D)``.

        Function-identical to this module (pinned by `tests/test_restricted.py`), with a
        normal parameter set that inference serves with zero overhead.
        """
        t = self.teacher[0]
        V = self.V.detach()
        st = build_narrow_llama(t, self.k, self.interm, self.n_head, self.n_kv).to(V.device)
        E = indices_to_bases(self.idx, t.config.intermediate_size)
        gains = {n: float(g) for n, g in self.gains().items()}
        absorb_llama(t, st, V, E, norm_gain=gains)
        if self.free:
            sd = dict(st.model.named_parameters())
            for n, dw in self.D.items():
                sd[n.replace("__", ".")].data.add_(dw)
            st.model.embed_tokens.weight.data.add_(self.D_emb)
            st.lm_head.weight.data.add_(self.D_lm)
        return st
