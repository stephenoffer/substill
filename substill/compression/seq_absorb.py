"""Sequential drift-corrected absorption -- and the measurements that refute it.

What this module was built to test
----------------------------------
The plain absorbed init ``W_s = V_out^T W_T V_in`` fits each student linear *in
isolation*, assuming the student's input equals the teacher's input projected into
the retained subspace (``h_s = V^T h_t``). That holds only at the embedding layer.
From block 1 onward the student's residual stream has drifted, and because every
block is fit against a *teacher* input it never sees that drift. The error
compounds with depth: on GPT-2 at 4.15x, per-block relative drift
``||h_s - V^T h_t|| / ||V^T h_t||`` grows 0.78 -> 2.0 by block 6 (error larger than
signal), and initial PPL is 7e7 -- far worse than random init (5.4e4).

``sequential_absorb_gpt2`` fixes exactly that: blocks are fit in forward order,
each against the teacher's projected residual state ``V^T h_{l+1}`` but from the
student's own drifted input ``x_l``, re-propagating after every block. It works, as
a function approximation: drift drops to ~0.28 and stays flat, initial PPL falls
from 7e7 to 1.4e3, and it generalizes (held-out drift matches fit drift to <0.01,
and 12x more calibration data changes nothing).

What the measurements say
-------------------------
It does not help the thing we actually care about. On GPT-2 -> 30.0M params
(4.15x), WikiText-2, 3 seeds, each arm at its own tuned LR, **matched on
wall-clock**:

    random init      + KD      438.9 +/- 3.5
    variance-select  + KD      180.6 +/- 2.7
    identity-truncate+ KD      161.0 +/- 1.9      <- plain absorbed init wins
    sequential fit   + KD      170.0 +/- 2.8      <- this module

The fit's apparent 6% gain at equal *step* count (169.1 vs 180.0) is entirely
explained by the ~23s it spends fitting; return those seconds to the baseline as
extra distillation steps and the gain inverts.

``closed_form_absorb_gpt2`` then removes the cost objection: each sublayer is linear
given its input, so the same targets have an exact ridge solution. It is the *worst*
arm of all -- 190.1 +/- 2.1 vs absorbed init's 173.3 +/- 1.3, while consuming *less*
wall-clock (124.5s vs 154.3s). Solving the layerwise objective better makes the
distilled student worse. The 300 Adam steps were never "the fit"; they were an
early-stopping regularizer, and the regularization was doing the work.

Four findings worth keeping, all counter to the intuition that motivated this:

1. **Refitting a compressed student's weights to approximate the teacher better makes
   the distilled student worse -- on every architecture tested.** Final quality is
   non-monotone in how well the layerwise objective is solved: not at all 180.0, partially
   169.1, exactly 190.1 (GPT-2, 1500 steps). It replicates on an RMSNorm, untied-embedding
   model: ``gap_fit_llama`` buys a 22x better init (895 vs 19,514 PPL) and loses 61% of
   final quality (129.84 vs 80.44) at equal wall-clock.

   Do **not** confuse this with the *basis* result. Choosing the residual basis by an
   information criterion is inverted **only on GPT-2**: on Llama, PCA beats identity
   truncation by 16% and init PPL orders the bases exactly as final PPL does. That
   inversion is caused by LayerNorm's centering plus the tied lm_head; see
   ``substill/compression/llama_absorb.py``. Changing the basis restricts the teacher's
   operator (alignment survives); refitting replaces it (alignment does not).

2. **Selecting what to keep, by any criterion we tried, loses to an arbitrary choice.**
   Residual coordinates ranked by variance or by logit-weighted variance lose to plain
   truncation (180.6 / 202.8 vs 161.0). Attention heads ranked by the KL their ablation
   costs the teacher lose to "the first five" (154.2 vs 149.1), even though head
   importance spans 296x within one layer -- and selecting heads for *coverage* of the
   teacher's head functions is worse still (155.5), which refutes the obvious
   redundancy explanation for that. Solving the layerwise weight fit exactly loses to not
   fitting at all (190.1 vs 180.0). See ``head_importance`` / ``head_similarity`` and
   docs/init_findings.md 9a-9b. We cannot say why. On Llama, importance-ranked selection
   merely *ties* an arbitrary subset (90.82 vs 90.77) rather than losing to it -- the
   surviving claim across both architectures is that ranking never *beats* arbitrary.

3. **RETRACTED (was: "the graft objective is harmful").** ``objective="graft"`` splices
   the student's block output back into the teacher's stream and backprops the real logit
   KL through the frozen teacher tail. We reported it at 192.1 vs the L2 surrogate's 169.1.
   That gap was a bug of ours: HF's ``hidden_states[-1]`` is already ``ln_f(...)``, and we
   applied ``ln_f`` to it again, so every graft target was a distribution the teacher never
   produces. With correct targets the two are **tied**. Graft is not harmful, just
   3x slower for nothing.
   Pinned by ``tests/test_seq_absorb.py::test_graft_targets_are_not_double_normalized``.

4. **What absorbed init carries is weight *alignment*, not spectrum**
   (``scripts/why_absorbed.py``, n=3). Permuting each block matrix's rows and columns
   preserves its singular values and its entire entry multiset exactly, and costs
   161.0 -> 301.3 PPL. Replacing the matrices with random ones carrying the teacher's
   exact singular values gives 314.0; replacing them with Gaussians matched only in
   per-matrix standard deviation gives 298.9. Those three are indistinguishable, so the
   spectrum is worth nothing beyond one scalar per matrix. Absorbed init works because
   each student weight is a *submatrix of the teacher's weight in the teacher's
   coordinate system*, so the student's layers compose the way the teacher's do. That
   is exactly the property a layerwise refit trades away.

Everything here is an *initialization*: the output is a stock ``GPT2LMHeadModel``
with plain ``nn.Linear``/``Conv1D`` weights and zero inference overhead. The module
is retained because these are reproducible negative results with a working harness,
not because the mechanism is recommended. ``scripts/bench.py`` regenerates the table;
``docs/init_findings.md`` has the full write-up.

Use ``absorb_gpt2`` (no fit). ``sequential_absorb_gpt2`` and
``closed_form_absorb_gpt2`` exist so the negative results stay reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

BasisMethod = Literal["identity", "random_sel", "select", "select_gn", "pca", "gn", "whiten"]

# Empirical note (GPT-2, k=324, measured):
# a rotation basis is a *worse* residual basis than a coordinate selection, even
# though it retains more variance (98.7% vs 96.7%) and less logit error (0.12% vs
# 0.34% relative). GPT-2's residual stream carries a large mean/outlier direction;
# the top principal component aligns with it, and LayerNorm -- which normalizes
# across coordinates -- then collapses onto that one coordinate, so initial PPL
# diverges (>1e16) where a selection basis gives ~5e3. SliceGPT escapes this by
# making the stream mean-zero and converting LN to RMSNorm, but that fold is not
# available here: GPT-2 ties lm_head to wte, so the mean-removal that fixes the
# stream also corrupts the unembedding. Selection bases are therefore the default.


# ---------------------------------------------------------------------------
# Residual-stream statistics
# ---------------------------------------------------------------------------
@torch.no_grad()
def residual_second_moment(
    teacher: nn.Module,
    calib: list[dict],
    *,
    device: str | torch.device = "cuda",
    include_final: bool = True,
) -> Tensor:
    """Uncentered second moment ``E[h h^T]`` pooled over every residual state.

    The residual basis is necessarily *shared* across depth (the skip connections
    add states from different blocks), so the statistic must be pooled over all
    of ``hidden_states``, not taken from one branch.
    """
    teacher = teacher.to(device).eval()
    d = int(teacher.config.n_embd)
    acc = torch.zeros(d, d, dtype=torch.float64, device=device)
    n = 0
    for b in calib:
        ids = b["input_ids"].to(device)
        hs = teacher(input_ids=ids, output_hidden_states=True).hidden_states
        states = hs if include_final else hs[:-1]
        for h in states:
            x = h.reshape(-1, d).double()
            acc += x.T @ x
            n += x.shape[0]
    return (acc / max(n, 1)).float()


@torch.no_grad()
def logit_metric(teacher: nn.Module) -> Tensor:
    """Gauss-Newton metric of the logits w.r.t. the final residual state.

    For a tied head, ``logits = W_emb h`` so ``J = W_emb`` and ``J^T J =
    W_emb^T W_emb``. Normalized to unit mean eigenvalue so it composes with the
    second moment without rescaling the eigenproblem.
    """
    W = teacher.transformer.wte.weight.detach().float()  # (vocab, d)
    M = W.T @ W
    M = 0.5 * (M + M.T)
    return M / M.diagonal().mean().clamp_min(1e-12)


def residual_basis(
    S: Tensor,
    k: int,
    *,
    method: BasisMethod = "pca",
    M: Tensor | None = None,
    ridge: float = 1e-3,
) -> Tensor:
    """Return a ``(d, k)`` column-orthonormal residual-stream basis.

    - ``identity``   first ``k`` coordinates (the historical fallback; best measured)
    - ``random_sel`` an arbitrary ``k``-coordinate subset (control for ``identity``)
    - ``select``     the ``k`` highest-variance coordinates
    - ``select_gn``  highest ``diag(S) * diag(M)`` -- variance as the unembedding reads it
    - ``pca``        top-``k`` eigenvectors of ``S``
    - ``gn``         top-``k`` eigenvectors of ``M^{1/2} S M^{1/2}`` pulled back through
                     ``M^{-1/2}`` and re-orthonormalized: the subspace best preserving
                     ``h`` *through the logit Jacobian* ``M``
    - ``whiten``     the ``S``-metric variant used by whitened-SVD compressors

    Ranked by final PPL after matched distillation, the order is the *reverse* of
    every information-retention statistic: identity (161.0) < select (180.6) <
    select_gn (202.8) << pca/gn (diverge). The rotations diverge because LayerNorm
    normalizes across coordinates and GPT-2's top principal component is its residual
    mean/outlier direction. Choosing this basis by an information criterion is, on the
    evidence, the wrong thing to do; see the module docstring and docs/init_findings.md.
    """
    d = S.shape[0]
    if k >= d:
        return torch.eye(d, dtype=S.dtype, device=S.device)

    if method == "identity":
        return torch.eye(d, k, dtype=S.dtype, device=S.device)

    if method == "random_sel":
        # Control for `identity`: a representative-but-arbitrary coordinate subset.
        # If this matches `identity`, the lesson is "any unbiased subset works, and
        # importance ranking hurts", not "the first k coordinates are special".
        g = torch.Generator(device="cpu").manual_seed(0)
        top = torch.randperm(d, generator=g)[:k].to(S.device)
        E = torch.zeros(d, k, dtype=S.dtype, device=S.device)
        E[top, torch.arange(k, device=S.device)] = 1.0
        return E

    if method in ("select", "select_gn"):
        # Rank coordinates by the variance they carry (``select``), or by the
        # variance they carry *as the unembedding reads it* (``select_gn``:
        # diag(M) * diag(S), the per-coordinate contribution to logit error).
        score = S.diagonal().clone()
        if method == "select_gn":
            if M is None:
                raise ValueError("method='select_gn' requires the metric M")
            score = score * M.diagonal()
        top = torch.argsort(score, descending=True)[:k]
        E = torch.zeros(d, k, dtype=S.dtype, device=S.device)
        E[top, torch.arange(k, device=S.device)] = 1.0
        return E

    S = 0.5 * (S + S.T)
    if method == "pca":
        evals, evecs = torch.linalg.eigh(S.double())
        return evecs[:, torch.argsort(evals, descending=True)[:k]].float().contiguous()

    if method in ("gn", "whiten"):
        A = M if method == "gn" else S
        if A is None:
            raise ValueError("method='gn' requires the metric M")
        A = 0.5 * (A.double() + A.double().T)
        A = A + ridge * A.diagonal().mean() * torch.eye(d, dtype=A.dtype, device=A.device)
        ea, Ua = torch.linalg.eigh(A)
        ea = ea.clamp_min(1e-12)
        A_half = Ua @ torch.diag(ea.sqrt()) @ Ua.T
        C = A_half @ S.double() @ A_half
        C = 0.5 * (C + C.T)
        ec, Uc = torch.linalg.eigh(C)
        top = torch.argsort(ec, descending=True)[:k]
        # Pull back through A^{-1/2}, then re-orthonormalize (QR) so the residual
        # add and the tied head still see an orthonormal frame.
        A_inv_half = Ua @ torch.diag(ea.rsqrt()) @ Ua.T
        V = A_inv_half @ Uc[:, top]
        Q, _ = torch.linalg.qr(V)
        return Q.float().contiguous()

    raise ValueError(f"unknown basis method: {method!r}")


# ---------------------------------------------------------------------------
# Student construction
# ---------------------------------------------------------------------------
def grassmann_logit_basis(S: Tensor, M: Tensor, k: int, *, steps: int = 1200,
                          lr: float = 2e-3, init: Tensor | None = None) -> Tensor:
    """The rank-``k`` residual subspace minimizing the projection's *logit* error.

    PCA minimizes ``E||(I - P)h||^2`` -- error in the residual stream. What a compressed
    student actually pays is ``E||W_lm (I - P) h||^2``: error measured through the
    unembedding. Those differ, and no ranking or eigendecomposition of ``S`` produces the
    second, because

        f(P) = tr(M (I-P) S (I-P)),   P = V V^T,   M = W_lm^T W_lm

    is quadratic in ``P``, not linear -- so ``argmin f`` is a genuine Grassmann-manifold
    problem rather than a top-k eigenproblem. Solve it directly: Adam on ``V`` with a QR
    retraction onto the Stiefel manifold each step, started from PCA (so it can only
    improve on it). Cheap: the iterates are ``d x k``.

    Measured (llama-160m, k=384 of 768): PCA gives 0.0192 relative logit error, this gives
    **0.0130** -- 32% less -- while retaining slightly *less* variance (0.9756 vs 0.9773).
    The two objectives genuinely disagree, which is what makes it a usable test of which
    one distillation cares about. On GPT-2 the same optimization is pointless
    (0.0012 -> 0.0009): the tied head leaves PCA already near-optimal.
    """
    dev = S.device
    S64, M64 = S.double(), M.double()
    den = torch.trace(M64 @ S64).clamp_min(1e-12)
    eye = torch.eye(S.shape[0], dtype=torch.float64, device=dev)
    V0 = init if init is not None else residual_basis(S, k, method="pca")
    V = V0.to(dev).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)
    for _ in range(steps):
        P = eye - V @ V.T
        loss = torch.trace(M64 @ P @ S64 @ P) / den
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            Q, _ = torch.linalg.qr(V)
            V.copy_(Q)
    return V.detach().float().contiguous()


def relative_logit_error(S: Tensor, M: Tensor, V: Tensor) -> float:
    """Fraction of logit energy the projection onto ``span(V)`` destroys.

    Computes ``tr(M (I-P) S (I-P)) / tr(M S)`` -- the quantity
    :func:`grassmann_logit_basis` minimizes.
    """
    S64, M64, V64 = S.double(), M.double(), V.double()
    P = torch.eye(S.shape[0], dtype=torch.float64, device=S.device) - V64 @ V64.T
    return float(torch.trace(M64 @ P @ S64 @ P) / torch.trace(M64 @ S64).clamp_min(1e-12))


class RMSNorm(nn.Module):
    """Scale-only normalizer.

    Unlike LayerNorm it does not subtract the channel mean, which is what breaks
    under a basis rotation: the student's "mean over its own k coordinates"
    corresponds to no teacher-side quantity.
    """

    def __init__(self, k: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(k))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@torch.no_grad()
def gamma_fold_gpt2(teacher: nn.Module) -> nn.Module:
    """Fold each block LayerNorm's affine into the consuming linear.

    Absorbs ``(gamma, beta)`` into the linear that consumes it, leaving an
    affine-free norm. Function-preserving.

    Without this, projecting a LayerNorm into a rotated basis requires
    approximating the non-diagonal ``V^T diag(gamma) V`` by its diagonal -- the
    step that makes a PCA residual basis diverge. After folding there is no gamma
    left to project: it lives inside ``c_attn`` / ``c_fc``, where the absorbed
    projection handles it exactly.

    ``ln_f`` is deliberately left alone: GPT-2 ties ``lm_head`` to ``wte``, so
    folding it would untie the head and cost a full embedding matrix.
    """
    import copy

    t = copy.deepcopy(teacher)
    for blk in t.transformer.h:
        for ln, lin in ((blk.ln_1, blk.attn.c_attn), (blk.ln_2, blk.mlp.c_fc)):
            g, b = ln.weight.detach().clone(), ln.bias.detach().clone()
            # Conv1D: y = x @ W + bias, so (g*x_hat + b) @ W = x_hat @ (diag(g) W) + b @ W
            lin.bias.data.add_(b @ lin.weight.data)
            lin.weight.data.mul_(g.unsqueeze(1))
            ln.weight.data.fill_(1.0)
            ln.bias.data.zero_()
    return t


@torch.no_grad()
def build_narrow_gpt2(teacher: nn.Module, k: int, inner: int, *,
                      n_head: int | None = None, n_layer: int | None = None):
    """Build a stock narrow ``GPT2LMHeadModel``.

    Has residual width ``k``, FFN width ``inner``, and optionally fewer layers
    than the teacher.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    t = teacher.config
    if n_head is None:
        n_head = int(t.n_head)
        while n_head > 1 and k % n_head != 0:
            n_head -= 1
    cfg = GPT2Config(
        vocab_size=int(t.vocab_size),
        n_positions=int(getattr(t, "n_positions", 1024)),
        n_embd=int(k),
        n_layer=int(n_layer or t.n_layer),
        n_head=int(n_head),
        n_inner=int(inner),
        activation_function=getattr(t, "activation_function", "gelu_new"),
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=float(getattr(t, "layer_norm_epsilon", 1e-5)),
    )
    return GPT2LMHeadModel(cfg)


@torch.no_grad()
def _ffn_basis(teacher: nn.Module, layer: int, calib, inner: int, device) -> Tensor:
    """Channel-selection basis for one block's FFN intermediate.

    GELU is elementwise, so selecting coordinates commutes with it while a
    rotation does not; the intermediate basis therefore stays a selection even
    though the residual basis is a rotation.
    """
    blk = teacher.transformer.h[layer]
    d_inner = blk.mlp.c_fc.weight.shape[1]
    if inner >= d_inner:
        return torch.eye(d_inner, device=device)
    acc = torch.zeros(d_inner, dtype=torch.float64, device=device)
    n = 0
    for b in calib:
        ids = b["input_ids"].to(device)
        hs = teacher(input_ids=ids, output_hidden_states=True).hidden_states
        h = hs[layer]
        z = blk.mlp.c_fc(blk.ln_2(h + blk.attn(blk.ln_1(h))[0]))
        z = z.reshape(-1, d_inner).double()
        acc += (z * z).sum(0)
        n += z.shape[0]
    top = torch.argsort(acc / max(n, 1), descending=True)[:inner]
    E = torch.zeros(d_inner, inner, device=device)
    E[top, torch.arange(inner, device=device)] = 1.0
    return E


@torch.no_grad()
def head_importance(teacher: nn.Module, calib: list[dict], *, device="cuda") -> Tensor:
    """``(n_layer, n_head)`` importance: the KL a head's removal costs the teacher.

    Head ``h`` of a layer writes to coordinates ``[64h, 64h+64)`` of the merged
    attention output, i.e. to one contiguous column block of ``attn.c_proj``. Zeroing
    that block ablates exactly that head. We measure the mean KL from the intact
    teacher's next-token distribution -- so importance is scored on the objective
    distillation will actually optimize, not on activation magnitude.
    """
    teacher = teacher.to(device).eval()
    n_layer, n_head = teacher.config.n_layer, teacher.config.n_head
    head_dim = teacher.config.n_embd // n_head

    ref = []
    for b in calib:
        ids = b["input_ids"].to(device)
        ref.append(F.log_softmax(teacher(input_ids=ids).logits, -1))

    scores = torch.zeros(n_layer, n_head)
    for li in range(n_layer):
        W = teacher.transformer.h[li].attn.c_proj.weight  # (d_in=768, d_out=768)
        for h in range(n_head):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            saved = W.data[sl, :].clone()
            W.data[sl, :] = 0.0
            tot = 0.0
            for b, lp in zip(calib, ref, strict=True):
                ids = b["input_ids"].to(device)
                out = F.log_softmax(teacher(input_ids=ids).logits, -1)
                tot += float(F.kl_div(out, lp, reduction="batchmean", log_target=True))
            W.data[sl, :] = saved
            scores[li, h] = tot / max(len(calib), 1)
    return scores


@torch.no_grad()
def head_similarity(teacher: nn.Module, calib: list[dict], *, device="cuda") -> Tensor:
    """``(n_layer, n_head, n_head)`` cosine similarity between heads' *contributions*.

    Head ``h`` contributes ``ctx[:, dh*h : dh*(h+1)] @ W_cproj[dh*h : dh*(h+1), :]`` to
    the residual stream. Two heads are redundant when those contributions point the same
    way on the same tokens. This is precisely what ablation importance cannot see: delete
    either of two redundant heads and little is lost, because the other still does the
    job -- so both look *unimportant* even though the pair is essential.
    """
    teacher = teacher.to(device).eval()
    n_layer, n_head = teacher.config.n_layer, teacher.config.n_head
    d = teacher.config.n_embd
    head_dim = d // n_head

    ctxs: dict[int, list[Tensor]] = {li: [] for li in range(n_layer)}

    def mk(li):
        def hook(_mod, inp, _out):
            ctxs[li].append(inp[0].detach().reshape(-1, d))
        return hook

    hooks = [blk.attn.c_proj.register_forward_hook(mk(li))
             for li, blk in enumerate(teacher.transformer.h)]
    for b in calib:
        teacher(input_ids=b["input_ids"].to(device))
    for h in hooks:
        h.remove()

    sims = torch.zeros(n_layer, n_head, n_head)
    for li in range(n_layer):
        ctx = torch.cat(ctxs[li], 0)                       # (N, d)
        W = teacher.transformer.h[li].attn.c_proj.weight   # (d_in, d_out)
        contrib = torch.stack([
            (ctx[:, h * head_dim:(h + 1) * head_dim]
             @ W[h * head_dim:(h + 1) * head_dim, :]).flatten()
            for h in range(n_head)])                       # (n_head, N*d)
        cn = F.normalize(contrib, dim=1)
        sims[li] = (cn @ cn.T).abs().cpu()
    return sims


def _greedy_coverage(sim: Tensor, n_keep: int) -> list[int]:
    """Facility-location: pick ``S`` maximizing ``sum_h max_{s in S} sim(h, s)``.

    The set that best *represents* every head, rather than the set of individually
    hardest-to-lose heads. The objective is submodular, so greedy is (1-1/e)-optimal.
    """
    n = sim.shape[0]
    keep: list[int] = []
    best = torch.zeros(n)
    for _ in range(n_keep):
        gains = torch.tensor([
            -1.0 if c in keep else float(torch.maximum(best, sim[:, c]).sum())
            for c in range(n)])
        keep.append(int(torch.argmax(gains)))
        best = torch.maximum(best, sim[:, keep[-1]])
    return keep


def _greedy_diverse(sim: Tensor, n_keep: int) -> list[int]:
    """Max-min: repeatedly add the head least similar to everything already chosen."""
    keep = [int(torch.argmin(sim.mean(1)))]  # start from the least typical head
    while len(keep) < n_keep:
        worst = sim[:, keep].max(dim=1).values
        worst[torch.tensor(keep)] = float("inf")
        keep.append(int(torch.argmin(worst)))
    return keep


def head_selection_bases(scores: Tensor, k: int, head_dim: int, *,
                         mode: Literal["important", "first", "random",
                                       "coverage", "diverse"] = "important",
                         sims: Tensor | None = None,
                         seed: int = 0) -> list[Tensor]:
    """Per-layer ``(d, k)`` selection of whole head blocks.

    ``first`` / ``random`` ignore ``scores`` entirely. ``important`` takes the top of
    ``scores``. ``coverage`` and ``diverse`` need ``sims`` from :func:`head_similarity`
    and select for *representativeness* instead of individual importance.

    Columns are ordered by teacher head index, not by score, so the student's heads keep
    their original relative order.
    """
    n_layer, n_head = scores.shape
    d = n_head * head_dim
    n_keep = k // head_dim
    if n_keep * head_dim != k:
        raise ValueError(f"k={k} is not a multiple of head_dim={head_dim}")
    if mode in ("coverage", "diverse") and sims is None:
        raise ValueError(f"mode={mode!r} requires `sims` from head_similarity()")
    g = torch.Generator().manual_seed(seed)
    out = []
    for li in range(n_layer):
        if mode == "first":
            keep = list(range(n_keep))
        elif mode == "random":
            keep = torch.randperm(n_head, generator=g)[:n_keep].tolist()
        elif mode == "coverage":
            keep = _greedy_coverage(sims[li], n_keep)
        elif mode == "diverse":
            keep = _greedy_diverse(sims[li], n_keep)
        else:
            keep = torch.argsort(scores[li], descending=True)[:n_keep].tolist()
        E = torch.zeros(d, k)
        for j, h in enumerate(sorted(keep)):
            E[h * head_dim:(h + 1) * head_dim, j * head_dim:(j + 1) * head_dim] = \
                torch.eye(head_dim)
        out.append(E)
    return out


def _gpt2_block_parts(block: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(mid, out)`` of a GPT-2 block: the post-attention and post-MLP states."""
    mid = x + block.attn(block.ln_1(x))[0]
    out = mid + block.mlp(block.ln_2(mid))
    return mid, out


def _graft(z_student: Tensor, h_teacher: Tensor, V: Tensor) -> Tensor:
    """Lift the student's k-dim state into the teacher's d-dim residual stream.

    The retained subspace is taken from the student, the discarded complement from
    the teacher. This is the counterfactual "what do the student's errors cost,
    holding everything the student never modelled at its true value" -- feeding the
    teacher tail a state with the complement simply zeroed would put it far
    off-distribution (the complement carries most of the stream's energy) and the
    resulting logits would say nothing about the block.
    """
    return z_student @ V.T + (h_teacher - (h_teacher @ V) @ V.T)


def _teacher_tail_logits(teacher: nn.Module, h: Tensor, start: int) -> Tensor:
    """Run frozen teacher blocks ``start..L-1``, then ``ln_f`` and the head."""
    for blk in teacher.transformer.h[start:]:
        h = _gpt2_block_parts(blk, h)[1]
    return teacher.lm_head(teacher.transformer.ln_f(h))


@torch.no_grad()
def absorb_gpt2(teacher, student, V: Tensor, ffn_bases: list[Tensor],
                layer_map: list[int] | None = None,
                head_bases: list[Tensor] | None = None) -> None:
    """Plain absorbed init ``V_out^T W V_in`` for every weight of a narrow GPT-2.

    This is the *starting point* the sequential fit refines (and, on its own, the
    prior-art baseline). GPT-2's ``Conv1D`` stores ``(d_in, d_out)`` and computes
    ``x @ W``, so the absorbed form is ``V_in^T W V_out``.

    ``layer_map[i]`` is the teacher layer that student layer ``i`` is absorbed from;
    it defaults to the identity. ``ffn_bases`` is indexed by *student* layer.

    ``head_bases[i]`` is a ``(d, k)`` basis on the *attention* space of student layer
    ``i`` -- the output space of ``c_attn`` (per q/k/v block) and the input space of
    ``attn.c_proj``. That space is distinct from the residual stream; it only happens
    to share its dimension. Passing ``None`` reuses ``V``, which is what the code did
    implicitly and which pins the student to the teacher's *first* ``k/head_dim``
    heads in every layer. Supplying a per-layer selection of whole head blocks lets
    each layer keep its own most useful heads instead.
    """
    dev = V.device
    V = V.float()
    t, s = teacher, student
    if layer_map is None:
        layer_map = list(range(len(s.transformer.h)))
    if head_bases is None:
        head_bases = [V] * len(s.transformer.h)

    s.transformer.wte.weight.data.copy_(t.transformer.wte.weight.detach().to(dev) @ V)
    s.transformer.wpe.weight.data.copy_(t.transformer.wpe.weight.detach().to(dev) @ V)

    for i, sb in enumerate(s.transformer.h):
        tb = t.transformer.h[layer_map[i]]
        E = ffn_bases[i].to(dev)
        H = head_bases[i].to(dev).float()
        Vqkv = torch.block_diag(H, H, H)
        _copy_ln(tb.ln_1, sb.ln_1, V)
        _copy_ln(tb.ln_2, sb.ln_2, V)
        sb.attn.c_attn.weight.data.copy_(V.T @ tb.attn.c_attn.weight.detach().to(dev) @ Vqkv)
        sb.attn.c_attn.bias.data.copy_(Vqkv.T @ tb.attn.c_attn.bias.detach().to(dev))
        sb.attn.c_proj.weight.data.copy_(H.T @ tb.attn.c_proj.weight.detach().to(dev) @ V)
        sb.attn.c_proj.bias.data.copy_(V.T @ tb.attn.c_proj.bias.detach().to(dev))
        sb.mlp.c_fc.weight.data.copy_(V.T @ tb.mlp.c_fc.weight.detach().to(dev) @ E)
        sb.mlp.c_fc.bias.data.copy_(E.T @ tb.mlp.c_fc.bias.detach().to(dev))
        sb.mlp.c_proj.weight.data.copy_(E.T @ tb.mlp.c_proj.weight.detach().to(dev) @ V)
        sb.mlp.c_proj.bias.data.copy_(V.T @ tb.mlp.c_proj.bias.detach().to(dev))
    _copy_ln(t.transformer.ln_f, s.transformer.ln_f, V)


@torch.no_grad()
def _copy_ln(src: nn.Module, dst: nn.Module, V: Tensor) -> None:
    """Project a LayerNorm's diagonal affine into the student basis.

    ``gamma`` is a diagonal operator; its projection onto ``V`` keeps only the
    diagonal of ``V^T diag(gamma) V``. Exact for a selection basis, a first-order
    approximation for a rotation -- which the sequential fit then corrects.
    """
    g, b = src.weight.detach().to(V.device), src.bias.detach().to(V.device)
    dst.weight.data.copy_((V.pow(2) * g.unsqueeze(1)).sum(0))
    dst.bias.data.copy_(V.T @ b)


# ---------------------------------------------------------------------------
# The sequential drift-corrected fit
# ---------------------------------------------------------------------------
InputMode = Literal["student", "teacher"]
TargetMode = Literal["state", "output"]
Metric = Literal["l2", "logit"]


@dataclass
class SeqAbsorbConfig:
    """Configuration for the sequential and closed-form absorption fits."""

    k: int
    inner: int
    # "identity" (plain coordinate truncation) measurably beats every importance-
    # ranked selection after distillation. Ranking coordinates by variance -- or by
    # variance seen through the unembedding -- concentrates the student's residual
    # stream on GPT-2's outlier channels, and LayerNorm, which normalizes *across*
    # coordinates, then sees a degenerate scale distribution. Arbitrary truncation
    # keeps a representative mix.
    basis: BasisMethod = "identity"
    fit: bool = True

    # --- the three ablatable ingredients of the fit -----------------------
    # input_mode="student": fit block l on the student's own drifted stream
    #   (drift correction). "teacher": fit on V^T h_t, i.e. every block sees a
    #   pristine input and never learns about upstream error -- the behavior of
    #   plain absorbed init, but with fitting, so the two can be compared.
    input_mode: InputMode = "student"
    # target_mode="state": match the post-block residual *state* V^T h_{l+1}, so a
    #   sublayer may spend capacity cancelling inherited error ("gap closing").
    #   "output": match the sublayer's own output V^T f_t(h_l), the classic
    #   layerwise-reconstruction target, which forces each block to reproduce its
    #   own contribution and leaves upstream error untouched.
    target_mode: TargetMode = "state"
    # metric="logit": measure the residual error through the logit Jacobian
    #   (V^T W_emb^T W_emb V), so the fit spends capacity on directions the
    #   unembedding actually reads. "l2": plain Euclidean.
    metric: Metric = "logit"
    # objective="l2": the block's own residual error (a surrogate).
    # objective="graft": splice the student block's output back into the teacher's
    #   residual stream -- student coordinates from the student, the discarded
    #   complement from the teacher -- run the *frozen teacher tail* on it, and
    #   minimize the KL to the teacher's own logits. This scores each block by the
    #   logit damage its error actually causes, at the cost of a tail pass, instead
    #   of assuming every residual direction matters equally.
    # objective="both": graft KL + `l2_weight` * the L2 surrogate.
    #
    # Graft is *worse* than the plain L2 surrogate, and 3x slower.
    # The reason is in `_graft`: it feeds the teacher tail the student's retained
    # coordinates but the *teacher's* discarded complement, so each block is rewarded
    # for producing outputs that only work when paired with information the deployed
    # student does not have. Optimizing the true KD loss locally is not the same as
    # optimizing it globally. Left in as a documented negative; default is "l2".
    objective: Literal["l2", "graft", "both"] = "l2"
    l2_weight: float = 1.0
    # Trust region around the absorbed init: `prox` * sum ||theta - theta_absorbed||^2
    # / ||theta_absorbed||^2, per block.
    #
    # An unconstrained block fit reaches a far better *function* approximation than
    # plain absorbed init (lower init PPL and per-block drift) yet distills to a far
    # worse model -- worse even than random init. The fit buys function accuracy
    # by destroying the teacher's weight geometry, which is what distillation actually
    # exploits. `prox` interpolates: prox -> inf is absorbed init, prox = 0 is the
    # unconstrained fit. The useful setting is in between.
    prox: float = 0.0
    # Which block parameters the fit may move. "all" refits every weight matrix.
    # "affine" moves only the 1-D parameters -- every bias plus the LayerNorm
    # gamma/beta -- roughly 4k numbers per block instead of 1.1M. An affine-only
    # fit can cancel the drift's systematic shift and rescale without touching a
    # single weight matrix, so the teacher's weight geometry survives intact.
    fit_params: Literal["all", "affine"] = "all"
    # Ridge coefficient for `closed_form_absorb_gpt2`, relative to the mean diagonal
    # of the Gram matrix (so it is scale-free across layers).
    ridge_lambda: float = 1e-4

    # Fold LayerNorm affines into the consuming linear (SliceGPT's gamma-fold), so
    # a rotated residual basis needs no diagonal approximation of V^T diag(g) V.
    gamma_fold: bool = False
    # "rms" drops LayerNorm's centering in the student, which has no teacher-side
    # counterpart once the basis is a rotation. Only useful with a rotation basis.
    student_norm: Literal["ln", "rms"] = "ln"

    steps_per_block: int = 300
    lr: float = 1e-3
    mid_weight: float = 0.5          # weight on the post-attention residual target
    batch_seqs: int = 8              # sequences per fit minibatch
    # Fraction of calibration sequences withheld from the fit, used only to report
    # the fit's generalization gap. A block fit has ~1e6 free parameters; with a
    # small calibration set it will memorize, and train-drift will diverge from
    # holdout-drift. This is the diagnostic for that.
    holdout_frac: float = 0.25
    fit_ln_f: bool = True
    lnf_steps: int = 200
    verbose: bool = True


def _rel_mse(pred: Tensor, target: Tensor, Mk: Tensor | None = None) -> Tensor:
    """Relative error, optionally in the metric ``Mk`` (k x k, PSD)."""
    e = pred - target
    if Mk is None:
        return e.pow(2).sum() / target.pow(2).sum().clamp_min(1e-12)
    # ((x @ Mk) * x).sum() rather than einsum: the einsum backward dispatches to a
    # triton outer-product kernel, which needs a C compiler at runtime.
    num = ((e @ Mk) * e).sum()
    den = ((target @ Mk) * target).sum().clamp_min(1e-12)
    return num / den


def sequential_absorb_gpt2(
    teacher: nn.Module,
    calib: list[dict],
    cfg: SeqAbsorbConfig,
    *,
    device: str | torch.device = "cuda",
):
    """Build and drift-correct a narrow GPT-2 student. Returns ``(student, info)``.

    Blocks are fit in forward order against ``V^T h_{l+1}`` (teacher stream) from
    the student's own ``x_l`` (drifted stream). After block ``l`` is fit, the
    student stream is re-propagated through the *fitted* block, so block ``l+1``
    sees the drift it will actually face at inference.
    """
    teacher = teacher.to(device).eval()
    if cfg.gamma_fold:
        teacher = gamma_fold_gpt2(teacher).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    S = residual_second_moment(teacher, calib, device=device)
    M_full = logit_metric(teacher).to(device)
    V = residual_basis(S, cfg.k, method=cfg.basis, M=M_full).to(device)

    ffn_bases = [
        _ffn_basis(teacher, i, calib, cfg.inner, device)
        for i in range(teacher.config.n_layer)
    ]

    student = build_narrow_gpt2(teacher, cfg.k, cfg.inner).to(device)
    absorb_gpt2(teacher, student, V, ffn_bases)
    if cfg.student_norm == "rms":
        if not cfg.gamma_fold:
            raise ValueError("student_norm='rms' requires gamma_fold=True: the "
                             "teacher's beta has nowhere to go otherwise.")
        for sb in student.transformer.h:
            sb.ln_1 = RMSNorm(cfg.k).to(device)
            sb.ln_2 = RMSNorm(cfg.k).to(device)
    student.eval()

    info: dict = {"basis": cfg.basis, "block_loss": []}
    if not cfg.fit:
        return student, info

    # Teacher residual stream on the calibration set, projected once.
    with torch.no_grad():
        H = []  # H[l] = (N, T, d) teacher stream entering block l
        for b in calib:
            ids = b["input_ids"].to(device)
            hs = teacher(input_ids=ids, output_hidden_states=True).hidden_states
            H.append([h.detach() for h in hs])
        H = [torch.cat([hb[li] for hb in H], 0) for li in range(len(H[0]))]
        ids_all = torch.cat([b["input_ids"] for b in calib], 0).to(device)
        # `hidden_states[-1]` is ln_f(block_{L-1}(...)), NOT the last block's raw output
        # -- see tests/test_seq_absorb.py::test_hidden_states_last_is_post_final_norm.
        # H[0..L-1] are genuine pre-block states; H[L] is already normalized. Passing it
        # through ln_f again (as `_teacher_logprobs` and `_fit_ln_f` used to) applies the
        # norm twice and corrupts every target derived from it.
        h_final = _gpt2_block_parts(teacher.transformer.h[-1], H[-2])[1]

    N = H[0].shape[0]
    n_hold = int(cfg.holdout_frac * N)
    perm = torch.randperm(N, device=device)
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
    if fit_idx.numel() == 0:
        fit_idx, hold_idx = perm, perm[:0]

    # Student stream: exact at the embedding layer (wte, wpe absorbed with V).
    with torch.no_grad():
        pos = torch.arange(ids_all.shape[1], device=device)
        X = student.transformer.wte(ids_all) + student.transformer.wpe(pos)

    # Logit-Jacobian metric pushed into student coordinates: a residual error e
    # perturbs the logits by W_emb V e, so its cost is e^T (V^T W^T W V) e.
    Mk = (V.T @ M_full @ V).contiguous() if cfg.metric == "logit" else None

    use_graft = cfg.objective in ("graft", "both")
    # Each block enables exactly the parameters it is allowed to fit; everything
    # else stays frozen so autograd never builds their gradients.
    for p in student.parameters():
        p.requires_grad_(False)

    def _teacher_logprobs(idx):
        # Recomputed per minibatch from the cached final hidden state: caching the
        # full (N, T, vocab) tensor would cap the calibration set, which is exactly
        # the knob that matters here.
        with torch.no_grad():
            return F.log_softmax(
                teacher.lm_head(teacher.transformer.ln_f(h_final[idx])), -1)

    for li, (tb, sb) in enumerate(
        zip(teacher.transformer.h, student.transformer.h, strict=True)
    ):
        with torch.no_grad():
            mid_t, out_t = _gpt2_block_parts(tb, H[li])
            x_in = X if cfg.input_mode == "student" else H[li] @ V
            if cfg.target_mode == "state":
                y_mid, y_out = mid_t @ V, out_t @ V
            else:  # match each sublayer's own contribution, not the stream
                y_mid, y_out = (mid_t - H[li]) @ V, (out_t - mid_t) @ V

        params = [p for p in sb.parameters()
                  if cfg.fit_params == "all" or p.dim() == 1]
        anchor = [p.detach().clone() for p in params] if cfg.prox > 0 else None
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps_per_block)

        last = float("nan")
        nf = fit_idx.numel()
        for _ in range(cfg.steps_per_block):
            idx = fit_idx[torch.randint(0, nf, (min(cfg.batch_seqs, nf),), device=device)]
            xb = x_in[idx]
            m_s, o_s = _gpt2_block_parts(sb, xb)
            state = o_s
            if cfg.target_mode == "output":
                m_s, o_s = m_s - xb, o_s - m_s
            loss = xb.new_zeros(())
            if cfg.objective in ("l2", "both"):
                w = cfg.l2_weight if cfg.objective == "both" else 1.0
                loss = loss + w * (
                    _rel_mse(o_s, y_out[idx], Mk)
                    + cfg.mid_weight * _rel_mse(m_s, y_mid[idx], Mk)
                )
            if use_graft:
                lg = _teacher_tail_logits(teacher, _graft(state, out_t[idx], V), li + 1)
                loss = loss + F.kl_div(
                    F.log_softmax(lg, -1), _teacher_logprobs(idx),
                    reduction="batchmean", log_target=True,
                )
            if anchor is not None:
                num = sum((p - a).pow(2).sum() for p, a in zip(params, anchor, strict=True))
                den = sum(a.pow(2).sum() for a in anchor).clamp_min(1e-12)
                loss = loss + cfg.prox * num / den
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            last = float(loss.detach())

        for p in params:
            p.requires_grad_(False)

        # Re-propagate the student stream through the *fitted* block, so block
        # l+1 is fit on exactly the drift it will face at inference.
        with torch.no_grad():
            X = torch.cat(
                [_gpt2_block_parts(sb, X[i : i + cfg.batch_seqs])[1]
                 for i in range(0, N, cfg.batch_seqs)], 0
            )
            state_t = out_t @ V

            def _drift(sel, X=X, state_t=state_t):
                if sel.numel() == 0:
                    return float("nan")
                e, t = X[sel] - state_t[sel], state_t[sel]
                return float(e.pow(2).sum().sqrt() / t.pow(2).sum().sqrt().clamp_min(1e-12))

            drift = _drift(torch.arange(N, device=device))
            d_fit, d_hold = _drift(fit_idx), _drift(hold_idx)
        info["block_loss"].append({"block": li, "loss": last, "drift": drift,
                                   "drift_fit": d_fit, "drift_holdout": d_hold})
        if cfg.verbose:
            print(f"  [seq-absorb] block {li:>2}  fit_loss={last:.4f}  "
                  f"drift={drift:.4f}  (fit {d_fit:.4f} / holdout {d_hold:.4f})",
                  flush=True)

    if cfg.fit_ln_f:
        _fit_ln_f(student, teacher, h_final, X, fit_idx, cfg, device)
    student.eval()
    info["V"] = V.cpu()
    return student, info


def _fit_ln_f(student, teacher, h_last_t: Tensor, X: Tensor, fit_idx, cfg, device) -> None:
    """Fit the final LayerNorm's affine against the teacher's *logits*.

    ``lm_head`` is tied to ``wte`` and cannot be refit without adding parameters,
    so ``ln_f`` is the only free surface between the compressed stream and the
    logits. Matching logits (not hidden states) here is what the KD loss will see.
    """
    lnf = student.transformer.ln_f
    for p in lnf.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(lnf.parameters(), lr=cfg.lr)
    nf = fit_idx.numel()
    for _ in range(cfg.lnf_steps):
        idx = fit_idx[torch.randint(0, nf, (min(cfg.batch_seqs, nf),), device=device)]
        # Targets per minibatch: caching the full (N, T, vocab) tensor is prohibitive.
        with torch.no_grad():
            tgt = teacher.lm_head(teacher.transformer.ln_f(h_last_t[idx]))
        logits = student.lm_head(lnf(X[idx]))
        loss = F.kl_div(
            F.log_softmax(logits, -1), F.log_softmax(tgt, -1),
            reduction="batchmean", log_target=True,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for p in lnf.parameters():
        p.requires_grad_(False)
    if cfg.verbose:
        print(f"  [seq-absorb] ln_f logit-KL = {float(loss):.4f}", flush=True)


# ---------------------------------------------------------------------------
# Closed-form gap-closing absorption
# ---------------------------------------------------------------------------
@torch.no_grad()
def _ridge(X: Tensor, Y: Tensor, lam: float) -> tuple[Tensor, Tensor]:
    """Solve ``min_W,b ||[X 1] [W; b] - Y||^2 + lam ||W||^2``. Returns ``(W, b)``."""
    N, din = X.shape
    Xb = torch.cat([X, X.new_ones(N, 1)], 1).double()
    A = Xb.T @ Xb
    A[:din, :din] += (
        lam * torch.eye(din, dtype=A.dtype, device=A.device) * A.diagonal()[:din].mean()
    )
    S = torch.linalg.solve(A, Xb.T @ Y.double())
    return S[:-1].float(), S[-1].float()


@torch.no_grad()
def closed_form_absorb_gpt2(
    teacher: nn.Module,
    calib: list[dict],
    cfg: SeqAbsorbConfig,
    *,
    device: str | torch.device = "cuda",
):
    """The gap-closing sequential fit, solved exactly instead of by Adam.

    Every sublayer of a GPT-2 block is linear given its input, so each fit is a ridge
    regression rather than a 300-step optimization. Same targets as
    :func:`sequential_absorb_gpt2` with ``objective="l2"``, ``target_mode="state"``,
    ``input_mode="student"``:

    * ``c_attn``      -> the teacher's projected q/k/v, from the student's own ``ln_1`` output
    * ``attn.c_proj`` -> the *gap* ``V^T h_mid_t - x_s``: whatever the attention block must
      emit for the student's stream to land on the teacher's, given where it started
    * ``c_fc``        -> the teacher's projected pre-GELU intermediate
    * ``mlp.c_proj``  -> the gap ``V^T h_out_t - mid_s``

    The point is cost. The Adam fit reaches the same place in ~23s, which buys the
    baseline ~500 extra distillation steps and thereby loses; this reaches it in a few
    seconds. Returns ``(student, info)``.
    """
    teacher = teacher.to(device).eval()
    S = residual_second_moment(teacher, calib, device=device)
    M_full = logit_metric(teacher).to(device)
    V = residual_basis(S, cfg.k, method=cfg.basis, M=M_full).to(device)
    ffn = [_ffn_basis(teacher, i, calib, cfg.inner, device)
           for i in range(teacher.config.n_layer)]

    student = build_narrow_gpt2(teacher, cfg.k, cfg.inner).to(device)
    absorb_gpt2(teacher, student, V, ffn)
    student.eval()

    ids_all = torch.cat([b["input_ids"] for b in calib], 0).to(device)
    with torch.no_grad():
        # `teacher.transformer`, not `teacher`: the LM head would materialize a
        # (N, T, 50257) logit tensor we never use.
        per = [teacher.transformer(input_ids=b["input_ids"].to(device),
                                   output_hidden_states=True).hidden_states
               for b in calib]
        H = [torch.cat([p[li] for p in per], 0) for li in range(len(per[0]))]
        del per
        pos = torch.arange(ids_all.shape[1], device=device)
        X = student.transformer.wte(ids_all) + student.transformer.wpe(pos)

    Vqkv = torch.block_diag(V, V, V)
    lam = cfg.ridge_lambda
    info: dict = {"basis": cfg.basis, "block_loss": [], "V": V.cpu()}

    for li, (tb, sb) in enumerate(
        zip(teacher.transformer.h, student.transformer.h, strict=True)
    ):
        h_t = H[li]
        mid_t, out_t = _gpt2_block_parts(tb, h_t)

        # --- attention: c_attn against the teacher's projected qkv ---------
        a_s = sb.ln_1(X)
        qkv_t = tb.attn.c_attn(tb.ln_1(h_t)) @ Vqkv
        W, b = _ridge(a_s.reshape(-1, cfg.k), qkv_t.reshape(-1, 3 * cfg.k), lam)
        sb.attn.c_attn.weight.copy_(W)
        sb.attn.c_attn.bias.copy_(b)

        # --- attention: c_proj against the residual-stream gap -------------
        # Setting c_proj to the identity (it is square, k -> k) exposes the merged
        # attention context, which is what c_proj actually consumes.
        w_save, b_save = sb.attn.c_proj.weight.clone(), sb.attn.c_proj.bias.clone()
        sb.attn.c_proj.weight.copy_(torch.eye(cfg.k, device=device))
        sb.attn.c_proj.bias.zero_()
        ctx = sb.attn(sb.ln_1(X))[0]
        sb.attn.c_proj.weight.copy_(w_save)
        sb.attn.c_proj.bias.copy_(b_save)
        gap = mid_t @ V - X
        W, b = _ridge(ctx.reshape(-1, cfg.k), gap.reshape(-1, cfg.k), lam)
        sb.attn.c_proj.weight.copy_(W)
        sb.attn.c_proj.bias.copy_(b)

        mid_s = X + sb.attn(sb.ln_1(X))[0]

        # --- MLP: c_fc against the teacher's projected pre-GELU ------------
        E = ffn[li].to(device)
        b_s = sb.ln_2(mid_s)
        z_t = tb.mlp.c_fc(tb.ln_2(mid_t)) @ E
        W, b = _ridge(b_s.reshape(-1, cfg.k), z_t.reshape(-1, cfg.inner), lam)
        sb.mlp.c_fc.weight.copy_(W)
        sb.mlp.c_fc.bias.copy_(b)

        # --- MLP: c_proj against the residual-stream gap -------------------
        z_s = sb.mlp.act(sb.mlp.c_fc(b_s))
        gap = out_t @ V - mid_s
        W, b = _ridge(z_s.reshape(-1, cfg.inner), gap.reshape(-1, cfg.k), lam)
        sb.mlp.c_proj.weight.copy_(W)
        sb.mlp.c_proj.bias.copy_(b)

        X = _gpt2_block_parts(sb, X)[1]
        state_t = out_t @ V
        drift = float((X - state_t).pow(2).sum().sqrt()
                      / state_t.pow(2).sum().sqrt().clamp_min(1e-12))
        info["block_loss"].append({"block": li, "loss": drift, "drift": drift})
        if cfg.verbose:
            print(f"  [closed-form] block {li:>2}  drift={drift:.4f}", flush=True)

    return student, info


@torch.no_grad()
def eval_ppl(model, val, device) -> float:
    """Token-level perplexity of ``model`` over ``val``, returning inf on divergence."""
    model.eval().to(device)
    nll = ntok = 0
    for b in val:
        ids = b["input_ids"].to(device)
        lg = model(input_ids=ids).logits[:, :-1]
        lab = ids[:, 1:]
        nll += float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.reshape(-1),
                                     reduction="sum"))
        ntok += lab.numel()
    # Report divergence as inf rather than clamping: a clamp at exp(20) silently
    # collapsed every broken cell of the init grid onto one identical number,
    # which reads as "these arms tied" when in fact they all produced NaN.
    mean_nll = nll / max(ntok, 1)
    if not math.isfinite(mean_nll) or mean_nll > 40.0:
        return float("inf")
    return math.exp(mean_nll)


__all__ = [
    "SeqAbsorbConfig",
    "sequential_absorb_gpt2",
    "closed_form_absorb_gpt2",
    "grassmann_logit_basis",
    "relative_logit_error",
    "head_importance",
    "head_similarity",
    "head_selection_bases",
    "residual_basis",
    "residual_second_moment",
    "logit_metric",
    "build_narrow_gpt2",
    "absorb_gpt2",
    "eval_ppl",
]
