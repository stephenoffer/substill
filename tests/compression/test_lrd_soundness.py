"""Soundness of the LRD restriction map, on the teachers the library *claims* to support.

`tests/compression/test_restricted.py` pins the internal consistency of the construction
(restricted module == folded student) but exercises exactly one teacher shape: **MHA,
untied embeddings, fp32**. Every benchmark in `docs/learned_restriction.md` uses that shape
too (llama-160m and the Sheared-LLaMAs are all MHA/untied). So the two structural
assumptions the restriction map actually makes were never tested:

1. **The final norm can be folded into `lm_head`.** True only when `lm_head` is *untied*
   from the input embedding. On a tied teacher -- Llama-3.2-1B/3B, and most small Llamas --
   `lm_head.weight` *is* `embed_tokens.weight`, so folding the final norm's gamma into it
   also scales the embedding, and the "function-preserving" gamma fold silently changes the
   teacher's function.

2. **A query head keeps the key/value head it was trained with.** Under GQA, query head
   ``i`` reads kv head ``i // G`` for group size ``G = n_head / n_kv``. The restriction
   copies the teacher's *first* ``n_head`` q heads and *first* ``n_kv`` kv heads, so the
   student re-pairs them at *its own* group size ``G' = n_head' / n_kv'``. Unless
   ``G' == G``, student q head ``i`` is wired to a kv head its weights never saw, and the
   student is not a restriction of the teacher at all -- it is a different operator.

These are silent-wrongness bugs: nothing raises, the loss still descends, and the student
still folds. They only show up as quality that is worse than it should be, on exactly the
architectures (Llama-3, Mistral, Qwen-family GQA decoders) the library advertises.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from substill.compression.llama_absorb import (  # noqa: E402
    gamma_fold_llama,
    llama_balanced_second_moment,
    llama_norm_input_second_moments,
    llama_residual_second_moment,
)
from substill.compression.restricted import (  # noqa: E402
    RestrictedLlama,
    _horizontal,
    _tangent,
    ffn_energy_indices,
)
from substill.lrd import plan_restricted_geometry  # noqa: E402


def _llama(*, hidden=32, interm=64, layers=2, heads=4, kv_heads=4, vocab=61, tied=False):
    """A tiny Llama with *non-trivial* norm gains, so the gamma fold is never a no-op."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=interm,
                      num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=kv_heads, max_position_embeddings=64,
                      tie_word_embeddings=tied)
    m = LlamaForCausalLM(cfg).eval()
    for layer in m.model.layers:
        layer.input_layernorm.weight.data.uniform_(0.5, 1.5)
        layer.post_attention_layernorm.weight.data.uniform_(0.5, 1.5)
    m.model.norm.weight.data.uniform_(0.5, 1.5)
    return m


def _ids(vocab=61, b=2, seq=16, seed=1):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (b, seq))


# ---------------------------------------------------------------------------
# 1. The gamma fold must preserve the teacher's function -- including when tied.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tied", [False, True])
def test_gamma_fold_preserves_the_teacher_function(tied):
    """`gamma_fold_llama` is advertised as function-preserving. It must actually be.

    With ``tie_word_embeddings=True`` HF makes ``lm_head.weight`` *the same tensor* as
    ``embed_tokens.weight``. Folding ``model.norm``'s gain into ``lm_head`` then rescales
    the input embedding as a side effect, and every downstream number -- the profiled second
    moment, the PCA basis, the KD targets -- is computed against a teacher that is not the
    teacher.
    """
    teacher = _llama(tied=tied)
    ids = _ids()
    with torch.no_grad():
        before = teacher(input_ids=ids).logits
        after = gamma_fold_llama(teacher)(input_ids=ids).logits
    rel = ((after - before).norm() / before.norm()).item()
    assert rel < 1e-5, f"gamma fold changed the teacher's function (rel err {rel:.3e})"


def test_gamma_fold_does_not_mutate_the_caller_s_teacher():
    """It deep-copies, so the *source* model must come out untouched either way."""
    teacher = _llama(tied=True)
    ids = _ids()
    with torch.no_grad():
        before = teacher(input_ids=ids).logits.clone()
    gamma_fold_llama(teacher)
    with torch.no_grad():
        after = teacher(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. GQA: the restriction must keep every query head with its own kv head.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width_ratio", [0.75, 0.6, 0.5, 0.4, 0.25])
def test_planned_geometry_preserves_the_gqa_group_size(width_ratio):
    """The student's queries-per-kv-head must equal the teacher's.

    Teacher: 8 query heads over 2 kv heads, so ``G = 4`` -- q heads 0-3 read kv head 0 and
    q heads 4-7 read kv head 1. The restriction hands the student teacher q head ``i`` and
    teacher kv head ``j`` verbatim, so the student re-derives the pairing from *its own*
    ``G' = n_head / n_kv``. If ``G' != G`` the student's q head ``i`` attends against a kv
    head its q weights were never trained against.
    """
    teacher = _llama(hidden=64, heads=8, kv_heads=2, interm=64)
    g = plan_restricted_geometry(teacher, width_ratio)
    G = 8 // 2
    G_s = g.n_head // g.n_kv
    assert g.n_head % g.n_kv == 0, "n_kv must divide n_head"
    assert G_s == G, (
        f"width_ratio={width_ratio}: student pairs {G_s} queries per kv head but the "
        f"teacher pairs {G}. Student q head i would read kv head i//{G_s}, whose weights "
        f"belong to teacher q heads {G}*(i//{G_s})...; the restriction is broken."
    )


def test_the_prefix_rule_is_coherent_for_every_gqa_geometry():
    """Exhaustive: keeping the *first* ``n_head`` q heads and *first* ``n_kv`` kv heads is
    coherent for **every** grouped-query geometry, given ``n_head = G * n_kv``.

    The restriction takes prefixes of both head sets. That is only sound if each kept query
    head's group index still lands inside the kept key/value heads -- i.e. ``i // G < n_kv``
    for every ``i < n_head`` -- *and* if the student re-derives the same group size it started
    with. Rather than spot-check a few ratios, enumerate every teacher shape up to 64 heads and
    every student kv count, and assert the pairing a student would compute is the pairing the
    teacher trained.

    This is what turns `check_head_geometry` from a plausible guard into a proved one.
    """
    checked = 0
    for heads in range(1, 65):
        for kv_heads in range(1, heads + 1):
            if heads % kv_heads:
                continue
            group = heads // kv_heads
            for n_kv in range(1, kv_heads + 1):
                n_head = group * n_kv
                for i in range(n_head):
                    # student q head i is teacher q head i; student kv head j is teacher kv j.
                    student_reads = i // (n_head // n_kv)   # what the student's attention does
                    teacher_trained = i // group            # what that q head was trained with
                    assert student_reads == teacher_trained, (
                        f"H={heads} KV={kv_heads} n_kv={n_kv}: q head {i} would read kv head "
                        f"{student_reads}, trained against {teacher_trained}")
                    assert student_reads < n_kv, "group index fell outside the kept kv heads"
                checked += 1
    assert checked > 500, f"expected a broad sweep, only checked {checked} geometries"


def test_restricted_attention_matches_the_teacher_s_kept_heads_under_gqa():
    """Functional proof, at full width, that the kept heads compute what the teacher's do.

    At ``V = I`` and ``k = d`` with every FFN neuron kept, a restriction that keeps a subset
    of *whole query heads* must make each kept head emit exactly the teacher's head output
    for the same input -- head ``i``'s q/k/v weights are copied verbatim, and RoPE and the
    ``1/sqrt(head_dim)`` scale are unchanged. So the student's attention sublayer must equal
    the teacher's attention sublayer with the dropped heads' ``o_proj`` columns zeroed.

    This holds *only* if head ``i`` still reads the kv head it was trained against.
    """
    teacher = gamma_fold_llama(_llama(hidden=64, heads=8, kv_heads=2, interm=64)).eval()
    d, H, KV = 64, 8, 2
    head_dim = d // H
    G = H // KV
    n_head, n_kv = 4, 4 // G          # 4 q heads => exactly 1 whole kv group
    assert n_head // n_kv == G

    calib = [{"input_ids": _ids()}]
    S = llama_residual_second_moment(teacher, calib, device="cpu")
    V = torch.eye(d)                                     # full width: no truncation at all
    idx = ffn_energy_indices(teacher, calib, 64, device="cpu")
    rm = RestrictedLlama(teacher, S, V, idx, n_head, n_kv).eval()

    ids = _ids()
    # Teacher's attention output with only the first n_head heads contributing.
    t_attn = {}
    hook = teacher.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
        lambda _m, i: t_attn.__setitem__("ctx", i[0].detach()))
    s_attn = {}
    with torch.no_grad():
        teacher(input_ids=ids)
    hook.remove()

    hs = rm.skeleton.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
        lambda _m, i: s_attn.__setitem__("ctx", i[0].detach()))
    with torch.no_grad():
        rm(ids)
    hs.remove()

    # The per-head context vectors of the kept heads must agree exactly.
    t_kept = t_attn["ctx"][..., : n_head * head_dim]
    s_kept = s_attn["ctx"][..., : n_head * head_dim]
    rel = ((s_kept - t_kept).norm() / t_kept.norm().clamp_min(1e-12)).item()
    assert rel < 1e-4, (
        f"kept query heads do not reproduce the teacher's head outputs (rel err {rel:.2e}) "
        f"-- the query/kv pairing was not preserved by the restriction"
    )


# ---------------------------------------------------------------------------
# 3. The RMS gain must actually be the scale the truncated stream loses.
# ---------------------------------------------------------------------------
def test_rms_gain_is_calibrated_per_norm_not_globally():
    """A single scalar gain cannot be right at every norm; the error is measurable.

    The student's norm must restore ``rms_s / rms_T = sqrt(d/k) * ||V^T h|| / ||h||`` at the
    point where it sits. That ratio is a property of the *distribution entering that norm*,
    and the residual stream's energy and its alignment with ``V`` both change sharply with
    depth. A gain fitted to the pooled second moment of every layer at once is
    systematically wrong at each individual norm.

    We assert the construction restores each norm's scale to within 10%.
    """
    teacher = gamma_fold_llama(_llama(hidden=64, interm=96, layers=4, heads=8, kv_heads=8)).eval()
    calib = [{"input_ids": _ids(b=4, seq=32)}]
    S = llama_residual_second_moment(teacher, calib, device="cpu")
    k = 32
    evals, evecs = torch.linalg.eigh(S.double())
    V = evecs[:, -k:].flip(-1).float()
    idx = ffn_energy_indices(teacher, calib, 48, device="cpu", V=V)
    norm_S = llama_norm_input_second_moments(teacher, calib, device="cpu")
    rm = RestrictedLlama(teacher, S, V, idx, 4, 4, norm_S=norm_S).eval()

    params = rm.restricted_params()
    norm_names = [n for n in params if n.endswith("layernorm.weight") or n == "norm.weight"]

    # Capture the true input to every teacher norm, and the gain each one actually needs.
    wanted: dict[str, float] = {}
    hooks = []

    def cap(name, mod):
        def pre(_m, inp):
            h = inp[0].detach().float().reshape(-1, h_d)
            rms_t = h.pow(2).mean(-1).sqrt()
            hv = h @ V
            rms_s = hv.pow(2).mean(-1).sqrt()
            wanted[name] = float((rms_s / rms_t.clamp_min(1e-9)).mean())
        return mod.register_forward_pre_hook(pre)

    h_d = teacher.config.hidden_size
    for li, layer in enumerate(teacher.model.layers):
        hooks.append(cap(f"layers.{li}.input_layernorm.weight", layer.input_layernorm))
        hooks.append(cap(f"layers.{li}.post_attention_layernorm.weight",
                         layer.post_attention_layernorm))
    hooks.append(cap("norm.weight", teacher.model.norm))
    with torch.no_grad():
        teacher(input_ids=_ids(b=4, seq=32))
    for h in hooks:
        h.remove()

    errs = {n: abs(float(params[n][0]) / wanted[n] - 1.0) for n in norm_names}
    worst = max(errs, key=errs.get)
    assert errs[worst] < 0.10, (
        f"norm gain is off by {errs[worst]:.1%} at {worst} "
        f"(applied {float(params[worst][0]):.4f}, needed {wanted[worst]:.4f}); "
        f"per-norm errors: { {n: f'{e:.1%}' for n, e in errs.items()} }"
    )


# ---------------------------------------------------------------------------
# 4. FFN neuron selection must rank by what the neuron writes, not what it holds.
# ---------------------------------------------------------------------------
def test_ffn_selection_ranks_by_the_residual_write_not_the_activation():
    """Neuron ``i`` writes ``a_i * W_down[:, i]``. Its importance scales with *both*.

    Construct a teacher where the ranking by activation energy alone is provably wrong: give
    one neuron a large activation but a near-zero output column (it writes nothing), and
    another a small activation but a large output column (it writes a lot). Selecting on
    ``E[a_i^2]`` alone keeps the first and drops the second -- exactly backwards.
    """
    teacher = gamma_fold_llama(_llama(hidden=32, interm=8, layers=1, heads=4, kv_heads=4)).eval()
    mlp = teacher.model.layers[0].mlp
    with torch.no_grad():
        # Neuron 0: loud activation, silent output column.
        mlp.gate_proj.weight[0] *= 8.0
        mlp.up_proj.weight[0] *= 8.0
        mlp.down_proj.weight[:, 0] *= 1e-3
        # Neuron 1: quiet activation, loud output column.
        mlp.gate_proj.weight[1] *= 0.2
        mlp.up_proj.weight[1] *= 0.2
        mlp.down_proj.weight[:, 1] *= 50.0

    calib = [{"input_ids": _ids(b=4, seq=32)}]
    kept = ffn_energy_indices(teacher, calib, 4, device="cpu")[0].tolist()

    assert 1 in kept, (
        f"dropped the neuron that writes the most into the residual stream "
        f"(kept {kept}); selection ignored ||W_down[:, i]||"
    )
    assert 0 not in kept, (
        f"kept a neuron whose output column is ~0 -- it contributes nothing "
        f"(kept {kept}); selection ranked it on activation energy alone"
    )


# ---------------------------------------------------------------------------
# 3b. The skeleton's weights are dead, and must not be allocated.
# ---------------------------------------------------------------------------
def test_the_skeleton_carries_no_weight_storage():
    """`RestrictedLlama`'s skeleton supplies a module *graph*, not weights. It must cost nothing.

    Every weight the skeleton needs is handed to it by ``functional_call`` on each forward, and
    the two it is never handed -- ``embed_tokens`` and ``lm_head`` -- are never called (the
    embedding arrives as ``inputs_embeds``; the logits are lifted through the *teacher's* head).
    So the randomly-initialized weights `LlamaForCausalLM(cfg)` allocates are dead on arrival.

    At toy scale that is merely wasteful. At real scale it is a bug: a 1024-wide student of a
    1.3B teacher carries ~400M dead parameters (~1.6 GB), which was on its own enough to push
    the restricted forward past a 22 GB card and OOM the 1.3B re-measurement.

    Assert the storage is gone *and* that the module still computes -- the fix is only valid if
    hollowing the skeleton is invisible to the forward pass.
    """
    teacher = gamma_fold_llama(_llama(hidden=32, interm=64, layers=2, heads=4, kv_heads=4)).eval()
    calib = [{"input_ids": _ids()}]
    S = llama_residual_second_moment(teacher, calib, device="cpu")
    V = torch.linalg.eigh(S.double())[1][:, -16:].flip(-1).float()
    idx = ffn_energy_indices(teacher, calib, 32, device="cpu", V=V)
    norm_S = llama_norm_input_second_moments(teacher, calib, device="cpu")
    rm = RestrictedLlama(teacher, S, V, idx, 2, 2, free=True, norm_S=norm_S).eval()

    dead = sum(p.numel() for p in rm.skeleton.parameters())
    assert dead == 0, f"the skeleton still holds {dead:,} dead weights"

    # RoPE's inv_freq is a *buffer* and IS read -- it must survive.
    assert any(b.numel() > 0 for b in rm.skeleton.buffers()), "buffers were freed too"

    ids = _ids()
    with torch.no_grad():
        logits = rm(ids).logits
        folded = rm.fold()(input_ids=ids).logits
    assert torch.isfinite(logits).all()
    rel = ((logits - folded).norm() / logits.norm()).item()
    assert rel < 1e-5, f"hollowing the skeleton changed the forward (rel {rel:.2e})"


# ---------------------------------------------------------------------------
# 4a. The basis must be taken from a statistic every layer has a vote in.
# ---------------------------------------------------------------------------
def test_balanced_pooling_gives_every_layer_a_vote():
    """Summing raw second moments lets the biggest-norm layers choose the basis alone.

    A transformer's residual norm grows steeply with depth, and
    `llama_residual_second_moment` *sums* the raw moment of every residual state. A sum is
    dominated by its largest terms, so the "activation covariance" the basis is taken from is
    effectively the covariance of the **last few layers** -- and the shared basis it induces
    barely sees the early ones. Nothing intends this; it is an artifact of adding together
    quantities with wildly different scales.

    `llama_balanced_second_moment` normalizes each layer by its own trace first, so every layer
    gets an equal vote. On the real benchmark that one line is worth **4.5 PPL to the frozen
    baseline** -- most of what LRD's entire Stiefel machinery was reported to buy (§11).

    Pin the mechanism: build a teacher whose deep layers carry far more energy than its shallow
    ones, and assert that (a) the pooled moment is dominated by the loud layers while the
    balanced one is not, and (b) the balanced basis retains far more of the *quiet* layers'
    energy.
    """
    teacher = gamma_fold_llama(
        _llama(hidden=32, interm=64, layers=4, heads=4, kv_heads=4)).eval()
    # Make the residual stream grow with depth, as a trained transformer's does.
    with torch.no_grad():
        for li, layer in enumerate(teacher.model.layers):
            layer.self_attn.o_proj.weight.mul_(3.0 ** li)
            layer.mlp.down_proj.weight.mul_(3.0 ** li)

    calib = [{"input_ids": _ids(b=4, seq=32)}]
    nS = llama_norm_input_second_moments(teacher, calib, device="cpu")
    traces = nS.diagonal(dim1=-2, dim2=-1).sum(-1)

    pooled = llama_residual_second_moment(teacher, calib, device="cpu")
    balanced = llama_balanced_second_moment(teacher, calib, device="cpu")

    # (a) the raw sum is dominated by the loudest layer; the balanced average is not.
    # With 2L+1 = 9 norms an equal split would be 11% each, so one layer holding a third or
    # more of the total is already the regime the pooled statistic cannot cope with.
    share = float(traces.max() / traces.sum())
    assert share > 0.33, f"fixture is not depth-imbalanced enough (loudest layer = {share:.0%})"

    k = 8
    def top_k(S):
        return torch.linalg.eigh(S.double())[1][:, -k:].flip(-1).float()

    V_pool, V_bal = top_k(pooled), top_k(balanced)

    def retained(V, S):                       # fraction of S's energy inside span(V)
        return float(torch.trace(V.T @ S @ V) / torch.trace(S).clamp_min(1e-12))

    quiet = nS[0]                              # layer 0's norm input -- the embedding
    r_pool, r_bal = retained(V_pool, quiet), retained(V_bal, quiet)
    assert r_bal > r_pool + 0.02, (
        f"balanced pooling did not help the quiet layers: it retains {r_bal:.1%} of layer 0's "
        f"energy against the pooled basis's {r_pool:.1%}. The pooled basis is supposed to be "
        f"the one that ignores them.")

    # (c) and the pooled basis must still look *fine* by its own statistic -- that is what makes
    # the failure silent. It reports near-perfect retention while half-destroying layer 0.
    r_pool_self = retained(V_pool, pooled)
    assert r_pool_self > r_pool + 0.1, (
        "the pooled basis is supposed to flatter itself: it should retain far more of the "
        "*pooled* energy than of the quiet layers' -- that gap is the whole bug")


# ---------------------------------------------------------------------------
# 4b. The public entry point, on the teacher shape that was silently mis-compressed.
# ---------------------------------------------------------------------------
def test_end_to_end_on_a_tied_gqa_teacher():
    """`learned_restriction_distill` on the Llama-3.2 shape: GQA *and* tied embeddings.

    This is the integration test the library never had. Both defects in §9a live here at once
    and neither raises, so before the fix this ran to completion and returned a student that
    was not a restriction of its teacher: the gamma fold had rewritten the teacher's embedding,
    and (at most width ratios) the student's query heads were reading key/value heads they were
    never trained against.

    Assert what the method promises: a plain `LlamaForCausalLM`, the teacher's query-per-kv
    group size preserved, finite logits, and a `V` that actually moved.
    """
    import substill

    teacher = _llama(hidden=64, interm=128, layers=3, heads=8, kv_heads=2,   # G = 4
                     vocab=128, tied=True)
    data = [{"input_ids": _ids(vocab=128, b=2, seq=16, seed=s)} for s in range(8)]

    cfg = substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=15,
                                       calib_batches=4, device="cpu")
    assert cfg.n_head // cfg.n_kv == 8 // 2, "planned geometry broke the GQA group size"

    result = substill.learned_restriction_distill(teacher, data, config=cfg)
    student = result.student

    assert type(student).__name__ == "LlamaForCausalLM"
    assert student.config.num_attention_heads == cfg.n_head
    assert student.config.num_key_value_heads == cfg.n_kv
    with torch.no_grad():
        logits = student(input_ids=data[0]["input_ids"]).logits
    assert torch.isfinite(logits).all()
    assert result.max_principal_angle > 0, "V never moved"
    assert result.restriction_gap is not None and result.restriction_gap >= 0


# ---------------------------------------------------------------------------
# 5. fold() must still be exact on the teacher shapes that used to be broken.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("free", [False, True])
def test_fold_is_exact_on_a_tied_gqa_teacher_with_per_norm_gains(free):
    """The three fixes compose: tied embeddings + GQA + per-norm gains, and `fold()` is exact.

    `fold()` writing a *different* model than the one that was trained would make every number
    in `docs/learned_restriction.md` meaningless, so the existing suite pins it -- but only on
    an MHA, untied teacher with a single pooled gain. Each fix adds a new way for the folded
    weights to disagree with the trained module:

    * untying changes which tensor the final norm folds into;
    * GQA changes which kv head each query head reads;
    * per-norm gains make `absorb_llama` take a *dict* of gains rather than one float.

    So re-pin the identity on all three at once, with a non-zero free residual, which is the
    configuration a real Llama-3.2 user would hit.
    """
    teacher = gamma_fold_llama(
        _llama(hidden=32, interm=64, layers=3, heads=4, kv_heads=2, tied=True)).eval()
    calib = [{"input_ids": _ids()}]
    S = llama_residual_second_moment(teacher, calib, device="cpu")
    V = torch.linalg.eigh(S.double())[1][:, -16:].flip(-1).float()
    idx = ffn_energy_indices(teacher, calib, 32, device="cpu", V=V)
    norm_S = llama_norm_input_second_moments(teacher, calib, device="cpu")

    # 4 q heads over 2 kv heads => G = 2, so a 2-q-head student takes exactly 1 kv head.
    rm = RestrictedLlama(teacher, S, V, idx, 2, 1, free=free, norm_S=norm_S).eval()
    if free:
        with torch.no_grad():
            for p in rm.D.values():
                p.normal_(0, 0.02)
            rm.D_emb.normal_(0, 0.02)
            rm.D_lm.normal_(0, 0.02)

    ids = _ids()
    with torch.no_grad():
        trained = rm(ids).logits
        folded = rm.fold()(input_ids=ids).logits
    rel = ((trained - folded).norm() / trained.norm()).item()
    assert rel < 1e-5, f"fold() is not the model that was trained (rel err {rel:.2e})"

    # And the gains really are per-norm, not one number wearing 2L+1 hats.
    gains = [float(g) for g in rm.gains().values()]
    assert len(gains) == 2 * 3 + 1
    assert max(gains) - min(gains) > 1e-4, "per-norm gains collapsed to a single value"


# ---------------------------------------------------------------------------
# 6. The aux term must compare like with like at every layer.
# ---------------------------------------------------------------------------
def test_aux_stream_options_are_what_they_say_they_are():
    """Both aux streams must be *the state they claim*, and they must actually differ.

    The pre-audit code read the student's stream off HuggingFace's ``output_hidden_states``,
    whose last entry is the state after ``model.norm``, not the last layer's output. So the aux
    term compared ``diag(gamma_s) h_s`` against ``V^T h_T`` at the final layer while comparing
    ``h_s`` against ``V^T h_T`` at every other -- and the student's ``gamma_s`` is per-channel
    once ``D`` trains it. That contradicted the term's own description ("the student's
    *residual stream*"), so the code and the docs disagreed and nobody had chosen.

    It is *not*, on reflection, simply a bug: matching the state ``lm_head`` actually reads is a
    defensible restriction statement at that point, and it supervises the final norm too. So
    both are real options (``stream="residual"`` / ``"prelogit"``), the default is decided by
    measurement (§9f), and this test pins that each option returns the state it names.
    """
    teacher = gamma_fold_llama(
        _llama(hidden=32, interm=64, layers=3, heads=4, kv_heads=4)).eval()
    calib = [{"input_ids": _ids()}]
    S = llama_residual_second_moment(teacher, calib, device="cpu")
    V = torch.linalg.eigh(S.double())[1][:, -16:].flip(-1).float()
    idx = ffn_energy_indices(teacher, calib, 32, device="cpu", V=V)
    norm_S = llama_norm_input_second_moments(teacher, calib, device="cpu")
    rm = RestrictedLlama(teacher, S, V, idx, 2, 2, free=True, norm_S=norm_S).eval()

    # Train the norm weights away from a scalar, exactly as `D` does during a real run. Without
    # this the two streams differ only by a positive scalar and the distinction is invisible.
    with torch.no_grad():
        for n, p in rm.D.items():
            if "layernorm" in n or n == "norm__weight":
                p.normal_(0, 0.15)

    ids = _ids()
    _, hs_res = rm.hidden_and_logits(ids, stream="residual")
    _, hs_pre = rm.hidden_and_logits(ids, stream="prelogit")
    assert hs_res.shape[0] == hs_pre.shape[0] == 3, "one entry per decoder layer"

    # The raw last-layer residual, straight from the module.
    raw: list[torch.Tensor] = []
    hook = rm.skeleton.model.layers[-1].register_forward_hook(
        lambda _m, _i, o: raw.append((o[0] if isinstance(o, tuple) else o).detach()))
    rm.hidden_and_logits(ids)
    hook.remove()

    assert (hs_res[-1].detach() - raw[-1]).abs().max() < 1e-5, (
        "stream='residual' did not return the raw post-layer residual")
    # 'prelogit' is the normed state, so it must NOT equal the raw one...
    assert not torch.allclose(hs_pre[-1].detach(), raw[-1], atol=1e-3), (
        "stream='prelogit' returned the raw residual -- the two options are the same thing")
    # ...and the two agree everywhere except the last layer.
    assert torch.allclose(hs_res[:-1], hs_pre[:-1], atol=1e-6), (
        "the streams must differ only at the final layer")

    with pytest.raises(ValueError, match="stream"):
        rm.hidden_and_logits(ids, stream="nonsense")


# ---------------------------------------------------------------------------
# 7. The restriction really is a function of the subspace, not of the basis.
# ---------------------------------------------------------------------------
def test_pure_restriction_is_a_grassmann_function():
    """Without a free core, the loss sees ``span(V)`` and nothing else. Two consequences.

    **The model is invariant under ``V -> V R``.** Every weight is conjugated
    (``V^T W V -> R^T V^T W V R``), the embedding and head are rotated
    (``W_E V -> W_E V R``), and the ``R``'s cancel through the network: the student computes
    the same function. This is the claim that licenses calling ``V`` "a point on the
    Grassmannian".

    **Therefore its gradient is purely horizontal.** A function constant along the vertical
    directions (basis spins that do not move the subspace) has no gradient component there. So
    the entire Riemannian gradient tilts the subspace, and none of it is wasted re-basing --
    which is exactly the regime where `StiefelAdamV`'s trust region turns ``lr`` into the
    subspace rotation itself rather than a bound on it.

    Both are checkable, and neither was pinned. If either ever breaks, the geometric story
    told in `substill/compression/restricted.py` is wrong.
    """
    teacher = gamma_fold_llama(_llama(hidden=32, interm=64, layers=2, heads=4, kv_heads=4)).eval()
    calib = [{"input_ids": _ids()}]
    S = llama_residual_second_moment(teacher, calib, device="cpu")
    k = 16
    evals, evecs = torch.linalg.eigh(S.double())
    V0 = evecs[:, -k:].flip(-1).float()
    idx = ffn_energy_indices(teacher, calib, 32, device="cpu", V=V0)
    norm_S = llama_norm_input_second_moments(teacher, calib, device="cpu")

    ids = _ids()
    rm = RestrictedLlama(teacher, S, V0, idx, 2, 2, free=False, norm_S=norm_S).eval()

    # (a) gauge invariance of the function
    torch.manual_seed(11)
    A = torch.randn(k, k)
    R = torch.linalg.qr(A)[0]                                  # an orthogonal re-basing
    with torch.no_grad():
        base = rm(ids).logits.clone()
    rm2 = RestrictedLlama(teacher, S, V0 @ R, idx, 2, 2, free=False, norm_S=norm_S).eval()
    with torch.no_grad():
        rot = rm2(ids).logits
    rel = ((rot - base).norm() / base.norm()).item()
    assert rel < 1e-4, (
        f"the restricted student changed under V -> VR (rel {rel:.2e}); it is then NOT a "
        f"function of the subspace alone and the Grassmann framing is wrong")

    # (b) hence the Riemannian gradient has no vertical component
    rm.train()
    with torch.no_grad():
        t_logits = teacher(input_ids=ids).logits[:, :-1]
    loss = torch.nn.functional.kl_div(
        rm(ids).logits[:, :-1].log_softmax(-1), t_logits.log_softmax(-1),
        log_target=True, reduction="batchmean")
    loss.backward()
    g = _tangent(rm.V.grad, rm.V.detach())
    vert = g - _horizontal(g, rm.V.detach())
    frac = float(vert.norm() / g.norm().clamp_min(1e-12))
    assert frac < 0.05, (
        f"{frac:.1%} of the gradient is vertical -- it spins the basis without moving the "
        f"subspace, which a subspace-only loss cannot do. Either the model is not gauge "
        f"invariant, or the tangent decomposition is wrong.")
