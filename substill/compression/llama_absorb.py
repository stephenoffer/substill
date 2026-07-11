"""Absorbed init for Llama-family students -- the control for the GPT-2 findings.

`docs/init_findings.md` §2 reports that on GPT-2 a PCA rotation of the residual stream
*diverges* (initial PPL >1e16) and that variance-ranked coordinate selection loses to
plain truncation. The mechanism proposed there is specific to two GPT-2 properties:

  * **LayerNorm centers across coordinates.** GPT-2's residual stream is dominated by a
    single coordinate carrying 73.7% of its variance, so the top principal component
    aligns with it and the normalized rotated stream collapses onto one axis.
  * **`lm_head` is tied to `wte`.** That blocks the mean-removal fold (SliceGPT,
    2401.15024) which would otherwise make the rotation legitimate.

`JackFram/llama-160m` has GPT-2's exact shape -- hidden 768, 12 layers, 12 heads,
head_dim 64 -- but **RMSNorm** (no centering) and **untied embeddings**. If the mechanism
is right, rotations should stop diverging here and the inversion should weaken. If
identity truncation still wins, the phenomenon is deeper than LayerNorm and none of §2's
explanation survives.

That is why this module exists: to make the GPT-2 results falsifiable. What it found
(docs/init_findings.md 10, 10a):

  * **The basis inversion is a GPT-2 artifact.** On Llama the ordering is conventional --
    final PPL tracks retained variance, and PCA beats identity truncation 80.94 +/- 0.90 to
    96.22 +/- 1.11 (n=3, 3.07x). So the profiled subspace this library was designed around
    is worth ~16% on its real targets, and only looked worthless on the GPT-2 testbed.
  * **The layerwise-refit inversion is NOT an artifact.** ``gap_fit_llama`` gives a 22x
    better initialization (895 vs 19,514 PPL) and a 61% worse distilled model
    (129.84 +/- 1.54 vs 80.44 +/- 0.40) at equal wall-clock. Same direction as GPT-2.

The two together isolate the mechanism. Changing the basis *restricts* the teacher's
operator -- ``V_out^T W V_in`` is still the teacher's weight, seen through a subspace, so
its layers compose as before. Refitting *replaces* the operator with a regression solution
that merely reproduces the teacher's activations on a calibration set. Restriction
transfers through distillation; replacement does not, on either architecture.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "gamma_fold_llama",
    "build_narrow_llama",
    "absorb_llama",
    "llama_residual_second_moment",
    "rms_gain",
    "gap_fit_llama",
    "llama_logit_metric",
]


@torch.no_grad()
def llama_logit_metric(teacher: nn.Module) -> Tensor:
    """Return ``W_lm^T W_lm``, the Gauss-Newton metric of the logits.

    The metric is taken w.r.t. the final residual state. Unlike GPT-2's, this head
    is untied, so it is a genuinely separate operator from the input embedding.
    Normalized to unit mean diagonal.
    """
    W = teacher.lm_head.weight.detach().float()
    M = W.T @ W
    M = 0.5 * (M + M.T)
    return M / M.diagonal().mean().clamp_min(1e-12)


@torch.no_grad()
def gamma_fold_llama(teacher: nn.Module) -> nn.Module:
    """Fold every RMSNorm's diagonal gain into the linear that consumes it.

    Function-preserving. Afterwards every norm has ``weight == 1``, so projecting a
    student's norm into a *rotated* basis needs no diagonal approximation of
    ``V^T diag(g) V`` -- the gain lives inside q/k/v (and gate/up, and ``lm_head``),
    where the absorbed projection handles it exactly.

    Unlike GPT-2, the final norm can be folded here: ``lm_head`` is untied, so scaling
    it does not corrupt the input embedding.
    """
    t = copy.deepcopy(teacher)
    for layer in t.model.layers:
        g = layer.input_layernorm.weight.detach().clone()
        for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            lin.weight.data.mul_(g.unsqueeze(0))       # W (out, in): scale input columns
        layer.input_layernorm.weight.data.fill_(1.0)

        g2 = layer.post_attention_layernorm.weight.detach().clone()
        for lin in (layer.mlp.gate_proj, layer.mlp.up_proj):
            lin.weight.data.mul_(g2.unsqueeze(0))
        layer.post_attention_layernorm.weight.data.fill_(1.0)

    g3 = t.model.norm.weight.detach().clone()
    t.lm_head.weight.data.mul_(g3.unsqueeze(0))
    t.model.norm.weight.data.fill_(1.0)
    return t


@torch.no_grad()
def llama_residual_second_moment(teacher, calib, *, device="cuda") -> Tensor:
    """``E[h h^T]`` pooled over every residual state (all ``hidden_states``)."""
    teacher = teacher.to(device).eval()
    d = int(teacher.config.hidden_size)
    acc = torch.zeros(d, d, dtype=torch.float64, device=device)
    n = 0
    for b in calib:
        hs = teacher(input_ids=b["input_ids"].to(device),
                     output_hidden_states=True).hidden_states
        for h in hs:
            x = h.reshape(-1, d).double()
            acc += x.T @ x
            n += x.shape[0]
    return (acc / max(n, 1)).float()


@torch.no_grad()
def rms_gain(S: Tensor, V: Tensor) -> float:
    """Constant to put in the student's RMSNorm weights.

    The student normalizes by ``rms(V^T h) = ||V^T h|| / sqrt(k)``; the absorbed weights
    were derived assuming the input is ``V^T (h / rms(h))``, i.e. ``V^T h * sqrt(d)/||h||``.
    The two differ by ``sqrt(d/k) * ||V^T h|| / ||h||``, which is ``sqrt(d/k) * sqrt(rho)``
    for retained energy fraction ``rho``. That factor is a per-token scalar only to the
    extent that ``rho`` varies token to token; its mean is exactly what a scale-only norm
    can absorb, so we put it there. Applied identically to every basis, so it cannot
    favor one.
    """
    d, k = V.shape
    rho = float(torch.trace(V.T @ S @ V) / torch.trace(S).clamp_min(1e-12))
    return (d / k) ** 0.5 * rho ** 0.5


@torch.no_grad()
def build_narrow_llama(teacher, k: int, interm: int, n_head: int, n_kv: int):
    """Construct an empty narrowed Llama with the given hidden/FFN/head geometry."""
    from transformers import LlamaConfig, LlamaForCausalLM

    t = teacher.config
    head_dim = int(t.hidden_size) // int(t.num_attention_heads)
    cfg = LlamaConfig(
        vocab_size=int(t.vocab_size),
        hidden_size=int(k),
        intermediate_size=int(interm),
        num_hidden_layers=int(t.num_hidden_layers),
        num_attention_heads=int(n_head),
        num_key_value_heads=int(n_kv),
        head_dim=head_dim,
        max_position_embeddings=int(getattr(t, "max_position_embeddings", 2048)),
        rms_norm_eps=float(getattr(t, "rms_norm_eps", 1e-6)),
        rope_theta=float(getattr(t, "rope_theta", 10000.0)),
        hidden_act=getattr(t, "hidden_act", "silu"),
        attention_bias=False,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(cfg)


@torch.no_grad()
def absorb_llama(teacher, student, V: Tensor, interm_bases: list[Tensor],
                 *, norm_gain: float = 1.0) -> None:
    """``W_s = V_out^T W V_in`` for every weight of a narrow Llama.

    ``teacher`` must already be gamma-folded. Attention heads are kept whole: the student
    inherits the teacher's first ``n_head`` heads (and first ``n_kv`` kv-heads), which is
    the arbitrary choice `docs/init_findings.md` §9a-9b found no rule beats.
    """
    dev = V.device
    V = V.float()
    t, s = teacher, student
    tc, sc = t.config, s.config
    head_dim = int(tc.hidden_size) // int(tc.num_attention_heads)
    nh, nkv = int(sc.num_attention_heads), int(sc.num_key_value_heads)
    q_rows, kv_rows = nh * head_dim, nkv * head_dim

    s.model.embed_tokens.weight.data.copy_(t.model.embed_tokens.weight.detach().to(dev).float() @ V)
    s.lm_head.weight.data.copy_(t.lm_head.weight.detach().to(dev).float() @ V)

    for i, (tl, sl) in enumerate(zip(t.model.layers, s.model.layers, strict=True)):
        E = interm_bases[i].to(dev).float()          # (t_interm, s_interm) selection
        ta, sa = tl.self_attn, sl.self_attn
        # q/k/v: (out, in) -> keep leading head rows, project input columns onto V
        sa.q_proj.weight.data.copy_(ta.q_proj.weight.detach().to(dev).float()[:q_rows] @ V)
        sa.k_proj.weight.data.copy_(ta.k_proj.weight.detach().to(dev).float()[:kv_rows] @ V)
        sa.v_proj.weight.data.copy_(ta.v_proj.weight.detach().to(dev).float()[:kv_rows] @ V)
        # o_proj: (d, n_head*head_dim) -> project output onto V, keep the same head cols
        sa.o_proj.weight.data.copy_(V.T @ ta.o_proj.weight.detach().to(dev).float()[:, :q_rows])

        tm, sm = tl.mlp, sl.mlp
        sm.gate_proj.weight.data.copy_(E.T @ tm.gate_proj.weight.detach().to(dev).float() @ V)
        sm.up_proj.weight.data.copy_(E.T @ tm.up_proj.weight.detach().to(dev).float() @ V)
        sm.down_proj.weight.data.copy_(V.T @ tm.down_proj.weight.detach().to(dev).float() @ E)

        # Norms are affine-free after the gamma fold; `norm_gain` restores the scale the
        # truncated stream loses (see `rms_gain`).
        sl.input_layernorm.weight.data.fill_(norm_gain)
        sl.post_attention_layernorm.weight.data.fill_(norm_gain)
    s.model.norm.weight.data.fill_(norm_gain)


# ---------------------------------------------------------------------------
# Sequential gap-closing fit, solved in closed form
# ---------------------------------------------------------------------------
@torch.no_grad()
def _ridge_nobias(X: Tensor, Y: Tensor, lam: float) -> Tensor:
    """``argmin_W ||X W - Y||^2 + lam ||W||^2``, returned as ``(out, in)`` for `nn.Linear`.

    Llama's projections have no bias, so no intercept column is added.
    """
    Xd = X.double()
    A = Xd.T @ Xd
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A, Xd.T @ Y.double()).T.float()


@torch.no_grad()
def gap_fit_llama(teacher, student, V: Tensor, calib, *, ridge: float = 1e-4,
                  device="cuda", verbose: bool = False) -> list[float]:
    """Refit each block's two residual *writers* to close the drift, sequentially.

    `absorb_llama` initializes every student linear in isolation, assuming its input is
    the teacher's input projected onto ``V``. That holds only at the embedding layer; from
    block 1 on, the student's residual stream has drifted, and the error compounds.

    Here `o_proj` and `down_proj` -- the only two weights that write to the residual
    stream -- are re-solved in forward order against the *gap*: whatever the sublayer must
    emit for the student's stream to land on the teacher's, **given where the student's
    own stream actually starts**. Each is a ridge regression, because each sublayer is
    linear in its own input. Blocks 0..l-1 are already refit when block l is solved, so
    the drift block l sees is the drift it will face at inference.

    On GPT-2 the analogous fit is *harmful* (docs/init_findings.md §4, §4c): there,
    approximating the teacher better makes the distilled student worse. §10 shows that
    inversion is a LayerNorm+tied-embedding artifact, so on an RMSNorm model the fit
    should help. That is the prediction `scripts/llama_gapfit.py` tests.

    Returns the per-block relative residual drift after fitting. Mutates ``student``.
    """
    teacher, student = teacher.to(device).eval(), student.to(device).eval()
    V = V.float().to(device)
    L = teacher.config.num_hidden_layers
    ids = torch.cat([b["input_ids"] for b in calib], 0).to(device)

    # Teacher targets: the projected residual state before and after each sublayer.
    #
    # `hidden_states[-1]` is the state *after* `model.norm`, not the last layer's output
    # (verified in tests/test_llama_absorb.py). Using it as block L-1's target silently
    # asks that block to emit a normalized state. Hook the layers instead, so every
    # target is a raw residual state.
    t_attn: dict[int, Tensor] = {}
    t_out: dict[int, Tensor] = {}
    hooks = []
    for li, tl in enumerate(teacher.model.layers):
        hooks.append(tl.self_attn.o_proj.register_forward_hook(
            lambda _m, _i, o, li=li: t_attn.__setitem__(li, o.detach())))
        hooks.append(tl.register_forward_hook(
            lambda _m, _i, o, li=li: t_out.__setitem__(
                li, (o[0] if isinstance(o, tuple) else o).detach())))
    H = teacher(input_ids=ids, output_hidden_states=True).hidden_states
    for h in hooks:
        h.remove()

    drifts = []
    for li in range(L):
        sl = student.model.layers[li]
        cap: dict[str, Tensor] = {}
        h1 = sl.self_attn.o_proj.register_forward_hook(
            lambda _m, i, _o, cap=cap: cap.__setitem__("ctx", i[0].detach()))
        S = student.model(input_ids=ids, output_hidden_states=True).hidden_states
        h1.remove()
        X = S[li]                                  # student stream entering block li

        # --- o_proj: emit the gap to the teacher's post-attention state -------
        mid_t = (H[li] + t_attn[li]) @ V
        ctx = cap["ctx"]
        d = ctx.shape[-1]
        sl.self_attn.o_proj.weight.data.copy_(
            _ridge_nobias(ctx.reshape(-1, d), (mid_t - X).reshape(-1, V.shape[1]), ridge))

        # --- down_proj: emit the gap to the teacher's post-MLP state ----------
        mid_s = X + sl.self_attn.o_proj(ctx)
        b = sl.post_attention_layernorm(mid_s)
        z = sl.mlp.act_fn(sl.mlp.gate_proj(b)) * sl.mlp.up_proj(b)
        out_t = t_out[li] @ V
        sl.mlp.down_proj.weight.data.copy_(
            _ridge_nobias(z.reshape(-1, z.shape[-1]),
                          (out_t - mid_s).reshape(-1, V.shape[1]), ridge))

        after = mid_s + sl.mlp.down_proj(z)
        drift = float((after - out_t).pow(2).sum().sqrt()
                      / out_t.pow(2).sum().sqrt().clamp_min(1e-12))
        drifts.append(drift)
        if verbose:
            print(f"  [gap-fit] block {li:>2}  drift={drift:.4f}", flush=True)
    return drifts
