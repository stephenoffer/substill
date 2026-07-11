"""Learned restriction: distil a student by training the *projection*, not the weights.

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

and runs the narrow student. **Every reachable point of the parameter space is an exact
restriction of the teacher** -- there is no way to leave the class, by construction. The
trainable object is a point on the Grassmannian ``G(d, k)``: 147k degrees of freedom for
Llama-160m, against the folded student's 30M weights.

Three properties make this the right shape for the principle:

1. It is the **only** parameterization tried here that optimizes the true objective while
   remaining inside the class that transfers. The layerwise fits (§4, §4c, §10a) optimize
   a true objective *outside* the class. PCA and the Grassmann basis stay inside the class
   but optimize a surrogate.
2. It never linearizes. The gradient ``dL/dV`` is accumulated from every layer that reads
   or writes the residual stream, so a direction layer 3 needs is paid for by layer 3.
3. It folds. `fold()` returns a plain `LlamaForCausalLM` whose weights are ``V^T W V``,
   bit-identical in function to the restricted module (`tests/test_restricted.py`), so
   inference carries zero overhead and phase 2 can release the weights and distil
   normally.

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

from .llama_absorb import absorb_llama, build_narrow_llama, rms_gain

__all__ = [
    "RestrictedLlama",
    "StiefelAdamV",
    "ffn_energy_indices",
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


def qr_retract(A: Tensor) -> Tensor:
    """Retract ``A`` onto the Stiefel manifold by a sign-fixed thin QR.

    ``torch.linalg.qr`` leaves the sign of each column of ``Q`` free; without the fix the
    retraction is discontinuous and momentum flips sign at random.
    """
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.sign(R.diagonal()) + 0.5)


class StiefelAdamV(torch.optim.Optimizer):
    """Adam on the Riemannian gradient of a Stiefel parameter, with a QR retraction.

    `substill.training.stiefel_optim.StiefelAdam` uses Cayley + Adafactor row/col statistics,
    which is the right choice when a model carries dozens of bases. Here there is exactly
    one ``(d, k)`` matrix, so full elementwise second moments are affordable and strictly
    better conditioned; the retraction is QR rather than Cayley because ``k`` is a large
    fraction of ``d`` (the ``2k x 2k`` Cayley shortcut buys nothing at ``k = d/2``).

    The second-moment state lives in the ambient space, so the update direction is
    re-projected to the tangent space *after* preconditioning -- Adam's rescaling does not
    preserve tangency.
    """

    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.99), eps: float = 1e-8):
        super().__init__(list(params), {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                g = _tangent(p.grad, p)
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = st["m"] / (1 - b1 ** st["t"])
                vh = st["v"] / (1 - b2 ** st["t"])
                d = _tangent(mh / (vh.sqrt() + group["eps"]), p)
                p.copy_(qr_retract(p - group["lr"] * d))
        return loss


# ---------------------------------------------------------------------------
# FFN intermediate selection (fixed; SiLU is elementwise so only *selection* commutes)
# ---------------------------------------------------------------------------
@torch.no_grad()
def ffn_energy_indices(teacher, calib, interm: int, device="cuda") -> list[Tensor]:
    """Indices of the ``interm`` highest-energy FFN neurons per layer.

    A neuron is one coordinate of ``silu(gate) * up``, i.e. one column of ``down_proj``.
    A *rotation* of that space would not commute with SiLU; a selection does, exactly.
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

    return [torch.arange(d_int, device=device) if interm >= d_int
            else torch.argsort(a, descending=True)[:interm]
            for a in acc]


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
        model becomes ``W_s = V^T W_T V + D``. That is the *same function class and the same
        parameter count* as an ordinary absorbed-init student -- ``D`` alone can reach any
        weight -- but a different **coordinate system** on it. Moving ``V`` moves all twelve
        layers coherently, in the one direction that keeps the student a restriction of the
        teacher; moving ``D`` is the ordinary per-weight direction. Since ``D`` starts at
        zero, training begins at exactly the absorbed-init student the baseline starts from,
        so a comparison against it isolates the added Stiefel coordinate and nothing else.
    """

    def __init__(self, folded, S: Tensor, V0: Tensor, idx: list[Tensor],
                 n_head: int, n_kv: int, *, free: bool = False):
        super().__init__()
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

        # A parameterless skeleton with the student's exact geometry. Its own weights are
        # never used or trained -- `functional_call` substitutes ours on every forward, so
        # `fold()` writing into a fresh copy of the same config is exact.
        self.skeleton = build_narrow_llama(folded, k, self.interm, n_head, n_kv)
        self.skeleton.requires_grad_(False)

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

    # -- the restriction map ------------------------------------------------
    def gain(self) -> Tensor:
        """Return ``sqrt(d/k * rho(V))``, the scale the truncated stream loses.

        Expressed as a scalar the student's (affine-free, post-fold) RMSNorms can
        carry. Differentiable in ``V``.
        """
        rho = torch.einsum("di,dc,ci->", self.V, self.S, self.V) / self.trS.clamp_min(1e-12)
        return (self.d / self.k) ** 0.5 * rho.clamp_min(1e-12).sqrt()

    def _restriction(self, detach: bool = False) -> dict[str, Tensor]:
        """``V^T W_T V`` for every student weight, as differentiable functions of ``V``."""
        V = self.V.detach() if detach else self.V
        t = self.teacher[0]
        g = self.gain().detach() if detach else self.gain()
        ones = torch.ones(self.k, device=V.device, dtype=V.dtype)
        p: dict[str, Tensor] = {"norm.weight": g * ones}
        for i, tl in enumerate(t.model.layers):
            a, m = tl.self_attn, tl.mlp
            ix = self.idx[i]
            pre = f"layers.{i}."
            p[pre + "input_layernorm.weight"] = g * ones
            p[pre + "post_attention_layernorm.weight"] = g * ones
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

    def hidden_and_logits(self, input_ids: Tensor):
        """Like :meth:`forward`, but also return the per-layer residual stream.

        Returns ``(logits, hs)`` where ``hs`` is a ``(L, B, T, k)`` stack of the student's
        post-layer hidden states -- the narrow residual stream, in the same ``k``-space as
        ``V^T h_T``. Used by the restriction-consistency auxiliary loss, which asks the
        student's stream to point the way the teacher's stream does *seen through* ``V``
        (`docs/learned_restriction.md` §2b). Every hidden state stays differentiable in
        both ``V`` (via the restriction) and ``D``.
        """
        t = self.teacher[0]
        emb = t.model.embed_tokens.weight[input_ids].float() @ self.V
        if self.free:
            emb = emb + self.D_emb[input_ids]
        out = functional_call(self.skeleton.model, self.restricted_params(),
                              kwargs={"inputs_embeds": emb, "output_hidden_states": True})
        h = out.last_hidden_state
        logits = (h @ self.V.T) @ t.lm_head.weight.T.float()
        if self.free:
            logits = logits + h @ self.D_lm.T
        hs = torch.stack(out.hidden_states[1:], dim=0)      # (L, B, T, k), post-layer
        return _Out(logits), hs

    # -- optimization -------------------------------------------------------
    def param_groups(self):
        """``(stiefel, euclidean)`` -- ``V`` needs a retraction, ``D`` does not."""
        euc = [p for n, p in self.named_parameters()
               if p.requires_grad and not n.endswith("V") and not n.startswith("skeleton")]
        return [self.V], euc

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
        absorb_llama(t, st, V, E, norm_gain=float(rms_gain(self.S, V)))
        if self.free:
            sd = dict(st.model.named_parameters())
            for n, dw in self.D.items():
                sd[n.replace("__", ".")].data.add_(dw)
            st.model.embed_tokens.weight.data.add_(self.D_emb)
            st.lm_head.weight.data.add_(self.D_lm)
        return st
