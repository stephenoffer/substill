# What actually determines a compressed student's quality after distillation

**Date:** 2026-07-10
**Setup:** GPT-2 (124.4M) → 30.0M-param student (4.15×), WikiText-2, seq-len 128,
batch 4, A10G. Every arm shares the student architecture (`n_embd=324`,
`n_inner=1068`, 12 layers, 12 heads), the optimizer (AdamW, weight-decay 0.01,
grad-clip 1.0), the schedule (10% linear warmup → cosine), the data order, and the
seed. Teacher validation PPL **50.86**. Reproduce with `scripts/analysis/bench.py`.

This document exists because several conclusions in `docs/cpsd.md` and `README.md`
were reached under a configuration that silently differed from the one they
describe. The results below supersede them where they conflict.

> **Scope warning, added after §10.** §§1–9 are GPT-2 only. §10 repeats the residual-basis
> comparison on an RMSNorm, untied-embedding model of identical shape and finds the
> **opposite** result: there, PCA is the best basis and the conventional ordering holds.
> §2's proposed mechanism is thereby confirmed *and* §2's finding is shown not to
> generalize. Every GPT-2-only claim below should be read as provisional until repeated on
> a Llama-family model. §10 says which ones have been.

> ## ⚠ Correction, added 2026-07-13 by the soundness audit
>
> **§10b's headline — "six ways of choosing the basis were tried and not one beats plain PCA"
> — is *true*, and it is true for a reason this document does not give.**
>
> All six criteria were computed from the **same** activation covariance, built by
> `llama_residual_second_moment` as a *raw sum* over every residual state. A transformer's
> residual energy grows by ~10⁵ from the embedding to the final norm (measured: **130,711×** on
> llama-160m), and a sum is dominated by its largest terms. So the covariance every criterion
> was reading is effectively the covariance of the **last few layers**: the basis it induces
> keeps 97.8% of the pooled energy and only **50.9% of the embedding's**.
>
> Varying the *criterion* across six settings while holding that *statistic* fixed was rigorous
> about the wrong axis. Six careful experiments over a broken input give six careful wrong
> answers, and their agreement reads as robustness.
>
> Give every layer an equal vote (`S = mean_ℓ S_ℓ / tr(S_ℓ)`, one line) and **every criterion
> improves by ~5–6 PPL at once** — 80.9 → 75.0 on llama-160m at 3.07×, n=6 — while their
> *ranking* stays unchanged: activation SVD 74.96, activation+influence (AIR) 75.00,
> activation-whitened (SVD-LLM) 74.97, a dead heat. So the finding survives in a sharper form:
>
> > **The choice of criterion does not matter. The covariance they all share does.**
>
> The corrected statistic is worth more than every basis-selection criterion in the 2024–2026
> literature put together — and more than half of what LRD's Stiefel machinery was reported to
> buy. See `learned_restriction.md` §11.

---

## 0. The bug underneath the old numbers

`fasd.profile()` defaults to `mode="branch"`, which enumerates `attn.*` and `ffn.*`
branches and **no `block.residual` branch**. `fasd/builders.py::_residual_basis`
searches the profile for `block.residual`, finds nothing, and falls through to:

```python
return torch.eye(t_hidden, s_hidden)
```

So every absorbed-init student ever built through `FSDPipeline`, every CPSD result
included, used **identity truncation**: keep the first `s_hidden` residual
coordinates, discard the rest. The profile's residual statistics were never used.
`fasd/compression/cpi.py` hits the same fallback.

Verified by `tests/test_seq_absorb.py::test_profile_default_yields_identity_residual_basis`.
`_residual_basis` now warns loudly instead of falling through in silence.

At `k=324` that basis retains **10.8%** of the residual stream's variance and
incurs **59%** relative logit error (`‖W(I−VVᵀ)h‖² / ‖Wh‖²`). For comparison, PCA
retains 98.7% of variance at 0.12% logit error.

**And on GPT-2 it is the best-performing basis we tested** (§2). That, too, is an
artifact of GPT-2's LayerNorm and tied head. Swap to a model with RMSNorm and untied
embeddings and the profiled subspace is worth **16%** over this fallback, exactly as the
original design intended (§10). The bug is real, and costly on the architectures this
library targets.

## 1. Absorbed init compounds error catastrophically, and does not care

Per-block relative drift `‖h_s − Vᵀh_t‖ / ‖Vᵀh_t‖`, measured on the student that
`fasd.build_student(arch_multiplier=0.5, absorbed_init=True)` returns
(`scripts/analysis/diag_init_error.py`):

| block | 1 | 2 | 3 | 4 | 5 | 6 | … | 11 |
|---|---|---|---|---|---|---|---|---|
| drift | 0.78 | 0.81 | 1.17 | 1.54 | 1.97 | 2.00 | … | 1.90 |

By block 3 the error exceeds the signal. That student's initial PPL is **2.28×10⁶**;
the equivalent construction in `scripts/analysis/bench.py` (identical residual basis, slightly
different FFN-intermediate estimator) gives **6.99×10⁷**. Both are 40–1,300× worse than
a *random* student at 5.4×10⁴. The exact value is not meaningful: a single mis-scaled
output direction dominates it.

Yet after distillation this initialization produces the best model in the table.
Initialization fidelity is not what distillation exploits **on GPT-2**; §4d shows what is.
On Llama, init PPL orders the bases exactly as final PPL does (§10). So even this is
architecture-specific.

## 2. Ranking channels by importance is worse than not ranking them

Final PPL after distillation, n=3 seeds, mean ± sd, all at lr=1e-3. The two budgets are
reported separately because they come from different runs; the *ordering* is identical
in both, and no arm changes rank across them.

| residual basis | variance retained | rel. logit error | 1500 steps | 1999 steps |
|---|---|---|---|---|
| PCA rotation | 98.7% | 0.0012 | **diverges** (>10¹⁶ at init) | — |
| Grassmann-optimal rotation | 98.5% | 0.0009 | (not run; rotations diverge) | — |
| logit-weighted selection (`select_gn`) | — | — | 202.84 ± 2.05 | — |
| variance selection (`select`) | 96.7% | 0.0034 | 207.32 (n=1) | 180.57 ± 2.74 |
| **identity truncation** | **10.8%** | **0.59** | **179.99 ± 2.39** | **161.00 ± 1.85** |

Two mechanisms are at work.

Rotations break LayerNorm. GPT-2's residual stream carries a large mean / outlier
direction. The top principal component aligns with it, and LayerNorm normalizes *across*
coordinates, so the rotated stream collapses onto that one coordinate. SliceGPT
(2401.15024) escapes this by making the stream mean-zero and converting LN → RMSNorm.
That fold is unavailable here: GPT-2 ties `lm_head` to `wte`, so the mean-removal that
repairs the stream corrupts the unembedding. We implemented and verified the γ-fold
(function-preserving). It does not rescue the rotation.

Importance ranking concentrates outliers. A handful of coordinates dominate GPT-2's
residual stream. The single largest carries **73.7% of total variance**, the top 4 carry
86.9%, and the max/median channel-variance ratio is **9,136×**. Ranking by variance
therefore hands LayerNorm, which normalizes *across* coordinates, a stream where every
coordinate is an outlier:

| basis | mean variance of kept channels | max/mean |
|---|---|---|
| `select` (top-variance) | 188.7 | 247× |
| `identity` (first 324) | 21.2 | 167× |
| `random_sel` (arbitrary 324) | 12.7 | 109× |

An arbitrary subset keeps a representative scale distribution. The importance-ranked one
does not. That same dominant coordinate is why the top principal component aligns with it
and the rotation collapses. All of this is consistent with "Variance Is Not Importance"
(2604.20682), which reports the same for compression more generally.

Note the ordering is *inverted* against both variance retained and logit error. Neither
statistic predicts post-distillation quality **here**.

**§10 tests both mechanisms and confirms the first.** Swap in a model of identical shape
that uses RMSNorm and untied embeddings and the rotation stops diverging, wins by 16%, and
the ordering becomes conventional. So the inversion is caused by LayerNorm's centering plus
the tied head, exactly as claimed. The second mechanism needs amending. Llama-160m's stream is
*also* outlier-dominated (top coordinate 61.7% of variance) and shows no inversion, so
outliers alone are not sufficient; they are only harmful once a *centering* normalizer
sees them. What survives both architectures is narrower. Importance ranking never *beats*
an arbitrary subset: worse on GPT-2, tied on Llama.

## 3. A single learning rate silently reverses the ranking

Final PPL, 1500 steps, seed 0 (`runs/lr_sweep_s0.json`):

| init | lr=3e-5 | lr=1e-4 | lr=3e-4 | lr=1e-3 |
|---|---|---|---|---|
| random | 1325.3 | 797.4 | 562.4 | **507.6** |
| identity | 592.5 | 375.8 | 241.2 | **180.6** |
| select | 749.1 | 418.8 | 263.5 | **207.3** |
| sequential fit (`select_gn` basis, graft objective) | 1199.8 | 948.9 | 723.3 | **572.7** |

Every arm's optimum is at the edge of the grid (1e-3), 20× the repo's default `lr=5e-5`,
and the gaps between arms shrink by ~3× as the LR approaches its optimum. Comparisons
of initializations at one shared LR are not informative.

The last row is the *worst* fit variant (§2 shows `select_gn` is a bad basis; §4a shows
`graft` is a bad objective). The best variant, `identity` basis with the L2 objective,
reverses sign with the LR:

| | sequential fit | absorbed init | verdict |
|---|---|---|---|
| `lr=3e-4`, 2000 steps (`runs/h2h_s0.json`) | 436.91 | 210.73 | fit loses 2× |
| `lr=1e-3`, 1500 steps (`runs/prox_identity_s0.json`) | 171.56 | 180.55 | fit wins 5% |

Note the fit is *better* at 1500 steps with the larger LR than at 2000 steps with the
smaller one, so this is the LR, not the budget. One shared LR would have supported
either conclusion. (Compute matching, §4, then settles it against the fit.)

## 4. Our proposed mechanism does not survive compute matching

`sequential_absorb_gpt2` fits each block in forward order against the teacher's
projected residual state, from the student's own drifted input. It does what it
claims:

- per-block drift → **0.28**, flat with depth (vs 2.0 and growing)
- initial PPL **7×10⁷ → 1.4×10³**
- it *generalizes*: held-out drift matches fit drift to <0.01, and 12× more
  calibration data (64 → 768 sequences) moves initial PPL by <1%. Not overfitting.

And it loses. Wall-clock-matched, 3 seeds (`runs/bench_matched_s012.json`): the
fit costs 22.7s, which buys the baseline 499 extra distillation steps.

| arm | steps | final PPL | init s | KD s | total s |
|---|---|---|---|---|---|
| random init + forward KL | 1999 | 438.87 ± 3.51 | 0.5 | 90.2 | 91 |
| variance-select + forward KL | 1999 | 180.57 ± 2.74 | 3.6 | 90.0 | 94 |
| **identity-truncation absorbed + forward KL** | 1999 | **161.00 ± 1.85** | 3.6 | 89.8 | **93.4** |
| sequential fit + forward KL | 1500 | 170.00 ± 2.80 | 22.8 | 67.6 | 90.4 |
| sequential fit + stream anchor | 1500 | 167.76 ± 2.20 | 22.5 | 74.1 | **96.6** |

The 6%-at-equal-steps result was compute, not method.

The matching is not exact. `absorbed` ends up with 93.4s against `sgca`'s 90.4s, a 3.3%
advantage *to the baseline*. That residual does not carry the conclusion: `sgca_anchored`
consumed **96.6s, more than `absorbed`'s 93.4s**, and still lost by 6.8 PPL. Whichever way
the few remaining seconds are assigned, the fit does not pay for itself.

### 4a. RETRACTED: the graft objective is neutral, not harmful

**This section previously claimed that optimizing the true KD loss locally is worse than a
crude L2 surrogate (192.09 ± 1.60 vs 169.12 ± 3.36). That was our bug, not a result.**

`objective="graft"` splices the student block's output back into the teacher's residual
stream (student coordinates from the student, discarded complement from the teacher), runs
the frozen teacher tail, and backprops the actual logit KL. Its teacher targets were
computed as `log_softmax(lm_head(ln_f(hidden_states[-1])))`. But HuggingFace's
`hidden_states[-1]` is *already* `ln_f(...)` of the last block's output, so `ln_f` was
applied **twice**. Every graft target was a distribution the teacher never produces. The
same double-norm corrupted `_fit_ln_f`'s targets in the `l2` arm.

Pinned by `tests/test_seq_absorb.py::test_hidden_states_last_is_post_final_norm` and
`::test_graft_targets_are_not_double_normalized`. Re-run with correct targets, n=3, equal
steps (`runs/factorial_fixed_s012.json`):

| fit | before (buggy targets) | after (fixed) |
|---|---|---|
| none (absorbed init) | 179.99 ± 2.39 | 180.24 ± 2.38 |
| L2 surrogate | 169.12 ± 3.36 | 170.14 ± 3.36 |
| **teacher-graft (true KD loss)** | **192.09 ± 1.60** | **170.27 ± 2.09** |

The graft objective is **statistically tied** with the L2 surrogate. It is not harmful; it
is merely not worth its 3× cost. The `l2` arm barely moved (its ln_f targets improved,
initial PPL 1,444 → 1,111, with no effect on final PPL), so §4's compute-matched conclusion
is unaffected.

The lesson we drew from the old number, that "optimizing the true KD loss locally is not
optimizing it globally", was a plausible story fitted to a bug. We had a mechanism for it:
the graft hands each block the teacher's discarded complement, information the deployed
student never has. The mechanism may even be true. It was not what those numbers showed.

### 4b. Stream anchoring also only buys compute

Anchoring pulls each student residual state toward the teacher's state projected
into the student's basis. The teacher's hidden states are free (the KD forward pass
already produced them), but the extra backward is not: 51ms/step vs 46ms/step. Give
the plain-KD control those seconds back as 2190 steps against anchored's 1999
(n=2, `runs/anchor_sweep_s01.json`):

| arm | final PPL |
|---|---|
| **plain KD, wall-clock-matched control** | **155.72** |
| anchor λ=1, logit-Jacobian metric | 158.75 |
| anchor λ=0.3, logit metric | 159.49 |
| anchor λ=1, Euclidean metric | 163.76 |

One real sub-finding survives: measuring the anchor through the logit Jacobian
beats plain Euclidean at every λ tested (158.75 vs 163.76 at λ=1). The metric is
right; the mechanism still is not worth its compute.

### 4c. Solving the layerwise objective *exactly* is the worst of all

If the Adam fit's +11 PPL at equal steps was real and only its 23s cost sank it, then
solving the same objective in closed form should win. Each GPT-2 sublayer is linear
given its input, so every target the fit chases is a ridge regression:
`closed_form_absorb_gpt2` recovers the exact minimizer in one pass
(`tests/test_seq_absorb.py` checks it is exact at full width).

It is the *worst* arm we ran (n=3, `runs/bench_cfa_s012.json`), and it loses while being
*given* a wall-clock advantage:

| arm | steps | final PPL | init PPL | total s |
|---|---|---|---|---|
| **absorbed (no fit)** | 1648 | **173.27 ± 1.30** | 7.0×10⁷ | 154.3 |
| closed-form exact fit | 1500 | 190.13 ± 2.11 | 3.5×10³ | 124.5 |

Line the three fit qualities up at 1500 steps:

| how well the layerwise objective is solved | final PPL |
|---|---|
| not at all (absorbed init) | 179.99 ± 2.39 |
| partially (300 Adam steps/block) | **169.12 ± 3.36** |
| exactly (closed-form ridge) | 190.13 ± 2.11 |

Final quality is **non-monotone** in fit quality, and the exact solution is worse than no
fit at all. The 300 Adam steps were never "the fit". They were an *early-stopping
regularizer*, and the regularization, not the fitting, was doing the work. (An explicit
L2 trust region around the absorbed weights does not reproduce it: §4's `prox` sweep
found final PPL monotone *decreasing* in fit freedom within the Adam budget.)

This is the sharpest statement of the theme. Better approximation of the teacher's
function at initialization, by any measure we tried (initial PPL, per-block drift,
exactness of the layerwise solve), produces a worse distilled student.

### 4d. What absorbed init actually carries

If absorbed init's advantage is not function fidelity (§1), what is it? Each arm below
destroys one property of the *block weight matrices* and keeps the rest. All arms keep
the absorbed embeddings (16.6M of the student's 30.0M parameters), so the contrast
against `random`, which randomizes those too, bounds the embeddings' own contribution.
n=3, 1999 steps, lr=1e-3 (`runs/why_absorbed_s012.json`, `scripts/analysis/why_absorbed.py`):

| arm | what survives in the block weights | init PPL | final PPL |
|---|---|---|---|
| `absorbed` | everything: function, spectrum, entries, scale, alignment | 7.0×10⁷ | **161.00 ± 1.85** |
| `permuted` | exact spectrum + exact entry multiset + scale | 3.5×10⁵ | 301.25 ± 1.35 |
| `spectral` | exact singular values + scale | 9.4×10⁵ | 313.97 ± 5.36 |
| `scaled_random` | per-matrix scale only | 1.3×10⁶ | 298.93 ± 5.97 |
| `random` | nothing (embeddings randomized too) | 5.3×10⁴ | 438.87 ± 3.51 |

Read the middle three rows together: `permuted` preserves every singular value **and**
every weight entry exactly; `spectral` preserves the singular values; `scaled_random`
preserves only a scalar per matrix. They are indistinguishable (301.3 / 314.0 / 298.9,
sd ≈ 1–6). **Preserving the teacher's spectrum buys nothing over preserving one number
per matrix.** Every low-rank/SVD-flavored justification for absorbed init that appeals
to singular-value structure is, at this compression ratio, appealing to the wrong thing.

The entire 298.9 → 161.0 gap is the *alignment* of weights to coordinates: the teacher's
function. The 438.9 → 298.9 gap is the absorbed embeddings plus correct per-matrix scale.

That resolves the §1 paradox. Absorbed init preserves the teacher's computational graph
up to a truncation of the residual stream. Initial PPL cannot see this: one badly-scaled
output direction destroys perplexity while leaving intact the internal circuitry that
distillation goes on to refine. Initial loss is the wrong instrument, which is why
optimizing it (§4) makes things worse.

## 5. What we did *not* establish

**We did not beat MiniLLM, GKD, or DistiLLM**, and the arms that appeared to were worse
than useless. `scripts/analysis/bench.py`'s `skew_kl` (403.33 ± 5.09) computed
`KL(p_t ‖ 0.9·p_t + 0.1·p_s)`, but DistiLLM defines `SKL_a(p‖q) = KL(p ‖ a·p + (1−a)·q)`
with `a = 0.1`, i.e. a mixture dominated by the **student**. Our version was bounded
above by log(1/0.9) = 0.105 and carried **1.6%** of forward-KL's gradient signal at a
typical teacher/student gap: that arm measured a loss which was barely training the
model. `reverse_kl` (314.95 ± 4.98) is a real divergence but not MiniLLM, which samples
on-policy. Both are now fixed and reimplemented in `scripts/analysis/objectives.py` (pinned by
`tests/test_objectives.py`), and §12 reports the corrected comparison. **The 403.33 and
314.95 numbers above are void.**

We did not test beyond GPT-2 / WikiText-2 / 4.15×. Findings 2 and 4 are specific to a
LayerNorm model with tied embeddings at one compression ratio.

And the `identity` result is not a recommendation to truncate arbitrarily. It is evidence
that the basis-selection literature's proxies (variance, Fisher, logit sensitivity) do not
predict post-distillation quality for this architecture. A `random_sel` control (arbitrary
permuted subset) distinguishes "any unbiased subset works" from "the first k coordinates
are special"; see `runs/`.

## 6. Honest status of the novelty claim

Mechanism by mechanism:

| mechanism | status |
|---|---|
| drift-corrected sequential absorption | works as an approximator; **loses at matched compute** (§4). Anticipated by SAES-SVD (2602.03051) and SVD-LLM (2403.07378) for the closed-form training-free case. |
| closed-form gap-closing absorption | **worst arm measured** (§4c), while cheaper than the baseline. |
| logit-Jacobian (Gauss-Newton) basis | **buys ~nothing**: PCA 0.0012 → Grassmann-optimal 0.0009 relative logit error. Anticipated by Fisher-aligned subspaces (2601.07197). |
| teacher-graft layerwise KD objective | **measurably harmful** (§4a). Novel as far as we can tell; that is not a recommendation. |
| stream anchoring in the logit metric | **loses** to a wall-clock-matched plain-KD control (158.75 vs 155.72, §4b). Feature distillation is FitNets (1412.6550) regardless. |

Every mechanism proposed here was either anticipated, or measured to be neutral or
harmful. **There is no conclusive win over prior distillation methods in this work**, and
this document should not be read as claiming one.

What this pass does contribute:

1. **Two bugs, and both fixes are worth real PPL.** The residual basis was never the
   profiled subspace (§0). On the RMSNorm models this library targets, using the profiled
   subspace properly (γ-fold + PCA + RMS gain) is worth **16%** (§10). And the student's
   attention heads were shattered by a rounding rule that contradicted its own docstring
   (§9); fixing it is worth **7.2% PPL at matched parameters and ~11% less wall-clock**.
2. A **benchmark rig** that matches compute rather than steps and tunes the LR per arm.
   Either correction alone reverses at least one published ranking in `docs/cpsd.md`
   (§3, §4).
3. Seven **reproducible negative results** (§2, §4, §4b, §4c, §9a, §9b, §10b), one
   **retraction of our own** (§4a, killed by a double-normalization bug we found and
   fixed), and one **positive mechanistic finding** (§4d: absorbed init carries weight
   alignment, not spectrum), now confirmed across two architectures by §10a.

Six different criteria have been tried against plain reconstruction-optimal projection,
across two architectures (table in §10b). **Not one beats it.** The prescription that
survives every experiment in this document is: restrict the teacher's operator onto its
maximum-variance subspace, keep whole circuits (heads, layers), and change nothing else.

Two claims survive §10's architecture check:

> **Refitting a compressed student's weights to approximate the teacher better makes the
> distilled student worse.** Not at all: 180.2. Partially: 170.1. Exactly: 190.1 (GPT-2).
> And on Llama, a 22× better initialization costs 61% of final quality at equal
> wall-clock (§10a). This is architecture-general.

The reason (§4d, §10a): changing the residual **basis** *restricts* the teacher's operator.
`V_out^T W V_in` is still the teacher's weight seen through a subspace, so its layers
compose as before, and a better subspace is straightforwardly better. **Refitting** replaces
the operator with a regression solution that merely reproduces the teacher's activations on
a calibration set. Restriction transfers through distillation; replacement does not.

And, weakened but intact:

> **Ranking what to keep by importance never beats an arbitrary subset.** On GPT-2 it is
> measurably *worse* (residual coordinates, §2; attention heads, §9a; and selecting heads
> for coverage instead is worse still, §9b). On Llama it is exactly *tied* (§10). It is
> never better. What produces a real gain there is a **rotation**, which no ranking of
> coordinates can express.

The GPT-2 claim that initial loss is an *inverted* proxy **across bases** does not
generalize. On the Llama testbed it is an ordinary well-behaved proxy
(§10). That much belonged to LayerNorm and tied embeddings. But across *refits* the
inversion holds on both architectures (§10a). So "init fidelity is inverted" was two
different claims wearing one name, and only one of them was an artifact.

The structural account is that the student inherits whole teacher submatrices in teacher
coordinates (§4d) and whole circuits: heads (§9), layers (§9). That account predicted the
`d=576` anomaly before that rung was run, located the head-geometry bug (§9), and now has
cross-architecture support (§10a). It is the one idea here that has paid rent more than once.

But it also predicted that keeping the *important* heads would help, and that was wrong
(§9a). The obvious repair, importance measures teacher-side redundancy so select for
coverage instead, is measurable without training, matches the redundancy data exactly,
and is *also* wrong (§9b).

So: a principle that says "restrict the teacher's operator, do not re-derive it, and do not
rank its parts", with one architecture-general confirmation (§10a), one prediction that
located a real bug (§9), two refuted corollaries (§9a, §9b), one prediction of ours that a
scope check refuted (§10), and one of our own published results retracted after we found
the bug under it (§4a). No technique built on any of it beats prior distillation methods.

## 7. Reproducing

The surviving reproduction entry points live under `scripts/analysis/`. A number
of one-off exploratory sweeps (the per-section head-selection, gap-fit, LR, and
factorial scans) were prototypes and are not retained; their results are recorded
in the sections above.

```bash
# the drift table of §1
PYTHONPATH=. python scripts/analysis/diag_init_error.py

# the width/depth axis of §9, and the head-geometry fix it located
PYTHONPATH=. python -m scripts.analysis.axis --steps 2000 --seeds 0 1 2 --random-init
PYTHONPATH=. python -m scripts.analysis.heads --steps 2000 --seeds 0 1 2
PYTHONPATH=. python -m scripts.analysis.heads --sweep --steps 2000 --seeds 0 1 2

# does any of it generalize? (§10) -- RMSNorm + untied embeddings, GPT-2's exact shape
PYTHONPATH=. python -m scripts.analysis.llama_basis --steps 2000 --seeds 0 1 2 \
    --bases identity random_sel select pca

# the logit-optimal Grassmann basis vs plain PCA (§10b)
PYTHONPATH=. python -m scripts.analysis.llama_basis --steps 2000 --seeds 0 1 2 --bases pca grassmann

# the compute-matched head-to-head of §4  (the table to cite)
PYTHONPATH=. python -m scripts.analysis.bench --compute-match --steps 1500 --lr 1e-3 --seeds 0 1 2

# what absorbed init actually carries (permute / spectrum / scale ablations)
PYTHONPATH=. python -m scripts.analysis.why_absorbed --seeds 0 1 2

# does the init advantage survive a real budget?
PYTHONPATH=. python -m scripts.analysis.budget_scaling --arms random identity select --steps 20000
```

`runs/` is gitignored; every number quoted above is reproduced in this document so the
tables survive without the artifacts.

## 8. Is the advantage just optimization speed?

Every arm above is far from convergence (155–440 PPL against a teacher at 50.9). An
initialization that merely lets the optimizer descend faster is indistinguishable, at a
fixed short budget, from one that reaches a better optimum. Only the second justifies
"method A beats method B".

`scripts/analysis/budget_scaling.py` traces PPL against step count at constant LR (so an eval at
step *n* means the same thing in every run). Both arms, seed 0, same schedule and data
order (`runs/budget_paired_s0.json`):

| steps | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| random init | 749.2 | 611.2 | 454.3 | 374.0 | 288.6 |
| **absorbed init** | **313.4** | **230.4** | **167.6** | **138.6** | **119.5** |
| ratio | 2.39× | 2.65× | 2.71× | 2.70× | 2.41× |

Over a **16× budget range the ratio does not close**. At 8000 steps random init (288.6)
has not reached what absorbed init achieves by step 500 (313.4 — nearly). The absolute
gap narrows (436 → 169 PPL) but the multiplicative gap is flat.

So absorbed init is not merely a faster descent: it is a genuinely better basin at every
budget we can afford. That is the one claim in this repo that survives scrutiny intact.
It is also prior art (FWSVD 2207.00112, Weight Subcloning 2312.09299, Minitron
2407.14679), not a contribution of this codebase.

Note it holds for the *identity-truncation* basis, which no prior work proposes, and not
for the profiled subspace the repo believes it is using.

## 9. The compression axis, and a broken attention geometry

§4d says what survives distillation is the teacher's weight *alignment*. That predicts
something about *where* to take parameters from. `scripts/analysis/axis.py` holds parameters at
~61M (2×) and walks from pure depth reduction (no residual truncation at all) to
near-pure width reduction, with a random-init control on every rung (n=3, 2000 steps,
lr=1e-3, `runs/axis_s012.json`):

| student | params | absorbed init | random init | gain |
|---|---|---|---|---|
| d=768, L=3 | 60.6M | 100.41 ± 0.65 | 492.20 ± 16.87 | 4.90× |
| **d=640, L=6** | 62.4M | **86.59 ± 0.59** | 491.34 ± 14.49 | 5.67× |
| d=576, L=8 | 61.4M | 93.19 ± 0.94 | 487.42 ± 5.19 | 5.23× |
| d=512, L=11 | 60.9M | 89.89 ± 0.79 | 494.62 ± 18.71 | 5.50× |

Two things. First, the random-init column is **flat** (487–495): these architectures are
interchangeable on their own. Everything separating the rungs is how well absorption
transfers into them. Second, neither extreme wins. "Width-first" (the doctrine
`fasd/compression/width_pruner.py` encodes, citing Minitron 2407.14679) and depth-first
are both wrong here; the optimum is interior. The deeper, narrower rungs also consume
*more* FLOPs per step (`L·d²` runs 1.77M → 2.88M), so they are not compute-starved
relative to `d=768, L=3`.

**And `d=576` sits ~5 PPL off the curve its neighbours define**, at 8× the seed noise.
The reason is not the width/depth trade-off. GPT-2 lays its attention heads out
contiguously along the residual axis with head_dim = 64. Identity truncation keeps the
first `k` coordinates, so when `k` is a multiple of 64 the student's heads *are* the
teacher's first `k/64` heads: q, k, v, o transferred whole, attention circuits intact.
When it is not, every head is a fragment of one teacher head glued to a fragment of the
next.

| rung | head_dim | on the curve? |
|---|---|---|
| d=768 (12 heads) | 64 | yes |
| d=640 (10 heads) | 64 | yes |
| d=512 (8 heads) | 64 | yes |
| **d=576 (12 heads)** | **48** | **no — 5 PPL worse** |

This is §4d one level down. Alignment lives in the *circuits* the weight matrices compose
into, not in the matrices taken in isolation.

`scripts/analysis/heads.py` tests it directly at the repo's own operating point. The repo builds
students with `n_head` pinned to the teacher's 12, so `n_embd=324` gives **head_dim=27**
and shatters all twelve heads, even though `width_pruner.py`'s docstring states "Retain
attention heads". n=3, 2000 steps, lr=1e-3, `runs/heads_s012.json`:

| n_embd | n_head | head_dim | params | final PPL |
|---|---|---|---|---|
| 384 | 6 | **64 (whole)** | 30,004,920 | **158.41 ± 1.51** |
| 384 | 12 | 32 (shattered) | 30,004,920 | 172.04 ± 2.30 |
| **320** | **5** | **64 (whole)** | 30,006,128 | **149.10 ± 2.01** |
| 324 (repo default) | 12 | 27 (shattered) | 30,007,116 | 160.74 ± 1.64 |

The first two rows are the control: same width, same *bit-identical* parameter count,
differing only in how the 384 coordinates are grouped into heads. **13.6 PPL, ~7σ.**

And the whole-head width sweep at the same budget (`runs/heads_sweep_s012.json`) has an
interior optimum, because `n_embd` and the FFN width trade against each other:

| n_embd | heads | inner | final PPL |
|---|---|---|---|
| 256 | 4 | 2226 | 179.03 ± 4.02 |
| **320** | **5** | **1124** | **149.10 ± 2.01** |
| 384 | 6 | 346 | 158.41 ± 1.51 |

### 9a. But *which* heads? Importance ranking loses again

Head geometry fixed, the choice of which `k/64` heads each layer inherits is free. The
attention space and the residual stream only shared a basis by accident (`absorb_gpt2`
now takes `head_bases`). Head importance is famously non-uniform, and measured by the KL
that ablating a head costs the teacher, it spans **296× within layer 0 alone** (337.3 to
1.14). Surely keeping the important ones helps.

It does not. All arms sit at `n_embd=320`, 5 heads, 30,006,128 parameters; the *only*
difference is which five of the twelve teacher heads each layer inherits (n=3, 2000 steps,
`runs/head_select_s012.json`; `mode="first"` is bit-identical to the previous behavior):

| head choice | init PPL | final PPL |
|---|---|---|
| **first five (arbitrary)** | 1.1×10⁸ | **149.10 ± 2.01** |
| five at random (arbitrary) | 6.1×10⁴ | 151.75 ± 1.09 |
| five most important (KL ablation) | 6.5×10⁴ | 154.15 ± 1.44 |

Importance-ranked selection is ~3σ worse than an arbitrary choice; the two arbitrary
choices are indistinguishable. And the init-PPL column is inverted yet again: the winner
starts at 111 *million* perplexity, the losers at ~60 thousand.

This is the **third independent replication** of one phenomenon, now across three
different structural units:

| what is selected | importance criterion | result vs arbitrary |
|---|---|---|
| residual coordinates (§2) | variance, logit-weighted variance | **worse** (180.6 / 202.8 vs 161.0) |
| layerwise weight fit (§4c) | exactness of the reconstruction | **worse** (190.1 vs 180.0) |
| attention heads (§9a) | KL cost of ablation | **worse** (154.2 vs 149.1) |

Three different objects, three different importance measures, three inversions. Whatever
distillation extracts from an initialization, it is not "the parts the teacher would miss
most". A plausible reading: ablation importance scores what the *teacher* cannot lose,
which is a statement about redundancy (the top-5 heads may duplicate each other's
function), whereas a student that will be retrained anyway is better served by *coverage*.
We did not test that reading.

### 9b. The redundancy explanation, pre-registered and falsified

§9a offered a reading: ablation importance scores what the *teacher* cannot lose, which is
a statement about redundancy. Delete either of two heads that duplicate each other and
little is lost, so both look unimportant, while a pair that jointly carries something
unique both look important. A student that gets retrained anyway might be better served by
*coverage* of the teacher's head functions than by individually-important heads.

That is testable without training. `head_similarity` measures the cosine similarity
between heads' actual contributions to the residual stream
(`ctx[:, dh·h : dh·(h+1)] @ W_cproj[dh·h : dh·(h+1), :]`). Mean pairwise similarity within
each selected set, GPT-2, 12 layers:

| set | mean pairwise similarity |
|---|---|
| important-5 | 0.126 (most redundant) |
| first-5 | 0.093 |
| coverage-5 (facility-location) | 0.068 |
| diverse-5 (max-min) | 0.029 (least redundant) |

The important heads *are* the most redundant, exactly as the reading predicts. So we
pre-registered: `coverage` and `diverse` should beat `first`'s 149.10.

They do not (n=3, `runs/head_cov_s012.json`; all arms 30,006,128 parameters):

| head choice | redundancy | init PPL | final PPL |
|---|---|---|---|
| **first (arbitrary)** | 0.093 | 1.1×10⁸ | **149.10 ± 2.01** |
| diverse (max-min) | 0.029 | 1.1×10⁶ | 149.69 ± 1.38 |
| random (arbitrary) | — | 6.1×10⁴ | 151.75 ± 1.09 |
| important (ablation KL) | 0.126 | 6.5×10⁴ | 154.15 ± 1.44 |
| coverage (facility-location) | 0.068 | 2.1×10⁵ | 155.45 ± 1.29 |

Redundancy ascends 0.029 → 0.068 → 0.093 → 0.126 while PPL goes 149.7 → **155.5** →
149.1 → 154.2. `coverage` is *less* redundant than `first` and 6.4 PPL worse. The
relation is not monotone, and the redundancy explanation is refuted.

What the five arms do say, descriptively: `first` and `diverse` are tied and best,
`random` is a hair behind, and the two rules that select *systematically for a property*
(highest ablation KL, facility-location centrality) are both ~5–6 PPL worse. No selection
rule beats an arbitrary choice.

There is no gain available here, then. `head_bases` stays in the API, defaulted to the
arbitrary choice, so all of this stays reproducible.

We do not have an explanation for §9a. The obvious one is measured and wrong.

### The fix

`profile_to_student_config` now rounds `hidden_size` to the teacher's **head_dim**
(64) rather than its head *count* (12), and drops whole heads. `preserve_head_dim=False`
restores the old behavior; an explicit `rank_map` (e.g. `cpi_rank_map`, which requires
`H_s == H_t`) still governs its own geometry. Rounding is to the *nearest* head boundary,
not up, so a requested rank of 325 gives 320 rather than inflating to 384.

On GPT-2 this changes the default student at `arch_multiplier=0.5` from
`n_embd=336, n_head=12, head_dim=28` to `n_embd=320, n_head=5, head_dim=64`, and at
`arch_multiplier=1.0` from `660/12/55` to `640/10/64`. The latter is exactly the best rung
of the §9 axis table, found independently.

**This is the one change in this pass that makes the method measurably better:
7.2% lower PPL at matched parameters and ~11% less wall-clock.** It is not a novel
technique. Retaining whole attention heads is standard practice (Michel et al. 2019,
Minitron 2407.14679), and `width_pruner.py` already claimed to do it. It is a bug fix
that the alignment finding predicted and then located.

## 10. The inversion is a GPT-2 artifact (and §2's mechanism is confirmed)

Everything above runs on GPT-2. §2 blamed the inversion on two GPT-2 properties:
**LayerNorm centers across coordinates**, and **`lm_head` is tied to `wte`** (which blocks
the mean-removal fold that would make a rotation legitimate). That is a falsifiable claim,
so we falsified it.

`JackFram/llama-160m` has GPT-2's exact shape (hidden 768, 12 layers, 12 heads,
head_dim 64) with **RMSNorm** (no centering) and **untied embeddings**. `fasd/compression/
llama_absorb.py` gives it the same treatment: a function-preserving γ-fold (verified to
preserve logits, and to reproduce the teacher bit-for-bit at full width), a whole-head
student, an RMS-gain correction applied identically to every arm, and only the residual
basis `V` varying. n=3, 2000 steps, lr=1e-3, 3.07×, teacher PPL 28.66
(`runs/llama_basis_s012.json`):

| residual basis | var retained | init PPL | final PPL |
|---|---|---|---|
| identity truncation | 0.707 | 393,850 | 96.22 ± 1.11 |
| arbitrary subset (`random_sel`) | 0.905 | 25,668 | 90.77 ± 1.11 |
| variance selection (`select`) | 0.939 | 58,292 | 90.82 ± 1.12 |
| **PCA rotation** | **0.977** | **19,514** | **80.94 ± 0.90** |

Compare GPT-2 at 4.15×:

| residual basis | var retained | init PPL | final PPL |
|---|---|---|---|
| **identity truncation** | **0.108** | 7.0×10⁷ | **161.00 ± 1.85** |
| variance selection | 0.967 | 4,898 | 180.57 ± 2.74 |
| PCA rotation | 0.987 | >10¹⁶ | **diverges** |

The two tables are opposites. On Llama the ordering is **perfectly conventional**: final
PPL tracks retained variance monotonically, init PPL tracks final PPL, and the PCA
rotation, which diverges on GPT-2, is the *best* basis, by 16% over identity truncation.
On GPT-2 every one of those relations is inverted or broken.

So §2's mechanism holds and **§2's finding does not generalize.** The inversion belongs to
LayerNorm + tied embeddings, not to distillation.

Two refinements are worth keeping.

Outlier dominance is not the cause. Llama-160m's residual stream is *also*
outlier-dominated: its top coordinate carries **61.7%** of the variance (GPT-2: 73.7%),
max/median 3,520× (GPT-2: 9,136×). Same pathology, opposite outcome. What differs is that
RMSNorm never centers across those coordinates, and an untied `lm_head` lets the γ-fold
absorb the final norm's gain, so a rotated stream is representable.

Importance ranking still buys nothing. On Llama, `select` (0.939 variance retained, 90.82)
is statistically **tied** with an arbitrary subset (`random_sel`, 0.905 retained, 90.77).
It merely stops *hurting*. The entire Llama gain comes from the rotation, which no ranking
of coordinates can produce. That much does survive both architectures: on GPT-2
importance-ranked selection is worse than arbitrary, on Llama it is equal, and nowhere is
it better.

### Practical consequence

For RMSNorm, untied-embedding models (Llama, Mistral, Qwen, i.e. everything the repo
actually targets) the right absorbed init is **γ-fold + PCA rotation + RMS-gain
correction**. It is worth **16%** over the identity truncation the `_residual_basis`
bug (§0) silently shipped. That is a larger gain than the head-geometry fix (§9), and it
means the repo's *original intent* (use the profiled subspace) was right for its target
architectures and wrong only for the GPT-2 testbed it was measured on.

`scripts/analysis/llama_basis.py` reproduces the table. `tests/test_llama_absorb.py` pins the fold
and the full-width exactness.

### 10a. But the *fit* inversion is not an artifact, and that separates two phenomena

§10 makes a prediction about §4. If the inversion is a LayerNorm artifact, then the
drift-corrected fit that *fails* on GPT-2 should *succeed* on Llama, where initial loss is
an honest proxy again. `gap_fit_llama` re-solves each block's two residual writers
(`o_proj`, `down_proj`) in forward order against the gap between the student's own drifted
stream and the teacher's projected one: a ridge regression per sublayer, costing 2.7s
against the 23s that sank the GPT-2 version. Per-block drift after fitting: 0.09–0.17, flat
with depth. Pre-registered; the control gets +43 steps to repay the 2.7s
(`runs/llama_gapfit_s012.json`, n=3):

| arm | steps | init PPL | final PPL | total s |
|---|---|---|---|---|
| **absorbed (γ-fold + PCA + RMS gain)** | 2043 | 19,514 | **80.44 ± 0.40** | 118 |
| + sequential closed-form gap fit | 2000 | **895** | 129.84 ± 1.54 | 119 |

**Wrong again.** A 22× better initialization, at equal wall-clock, gives a 61% worse
distilled model, on the very architecture where the *basis* ordering is conventional.

So there are **two different phenomena**, and §10 separated them:

| | GPT-2 (LayerNorm, tied) | Llama (RMSNorm, untied) |
|---|---|---|
| change the residual **basis** | inverted (identity beats PCA) | **conventional** (PCA beats identity) |
| **refit** the weights layer-by-layer | inverted (fit loses) | **still inverted** (fit loses) |

The basis inversion is a LayerNorm + tied-embedding artifact. The fit inversion is
architecture-general.

§4d already explained why, and this is its cross-architecture confirmation. **A change of
basis restricts the teacher's operator**: `W_s = V_out^T W V_in` is the teacher's weight
seen through a subspace, so the teacher's layers still compose the way they did, and a
better subspace is straightforwardly better. **A refit replaces the operator** with a
regression solution that happens to reproduce the teacher's activations on a calibration
set. The first preserves alignment. The second buys function accuracy by destroying it,
and alignment is what distillation exploits.

That is why permuting a block matrix (preserving every singular value and every entry,
breaking only alignment) costs 161 → 301 PPL (§4d), and why the exact closed-form solve is
the worst arm on GPT-2 (§4c) *and* on Llama.

### 10b. And the *logit-optimal* basis does not beat plain PCA either

§10 shows that on Llama the basis matters and PCA wins. PCA minimizes error in the residual
stream, `E‖(I−P)h‖²`. But a compressed student pays for error *through the unembedding*:
`E‖W_lm(I−P)h‖²`. Those are different objectives, and the second is not an eigenproblem.
`f(P) = tr(M(I−P)S(I−P))` with `M = W_lm^T W_lm` is quadratic in `P`, so its minimizer is a
genuine Grassmann-manifold problem. `grassmann_logit_basis` solves it directly (Adam on `V`
with a QR retraction, started from PCA, so it can only improve on the objective).

It works, as an optimizer. Started from PCA it cuts relative logit error by **32%** while
retaining slightly *less* variance. The two criteria genuinely disagree, which makes this a
clean test of which one distillation cares about (n=3, `runs/llama_grassmann_s012.json`):

| basis | var kept | rel. logit error | init PPL | final PPL |
|---|---|---|---|---|
| **pca** | 0.9773 | 0.0192 | **19,514** | **80.94 ± 0.90** |
| grassmann (logit-optimal) | 0.9756 | **0.0130** | 65,800 | 81.56 ± 0.46 |

Tied at best, and the diagnostic is the init-PPL column: the basis with **32% less logit
error has 3.4× worse initial perplexity**. The surrogate fails to predict even the quantity
it is a surrogate for.

The reason is that `M = W_lm^T W_lm` is the Jacobian of the *final* layer alone, while the
residual basis is shared by all twelve. A direction that barely reaches the logits directly
may be exactly what layer 3 needs to compute what layer 9 writes. Linearizing the network
at its output throws that away. Variance is agnostic about which layer uses what, and that
agnosticism is apparently the right prior.

**This closes the loop on the whole document.** Six criteria have now been tried against
plain reconstruction-optimal projection, across two architectures:

| criterion | result vs plain reconstruction |
|---|---|
| variance-ranked coordinate selection (§2) | worse on GPT-2, tied on Llama |
| logit-weighted coordinate selection (§2) | worse |
| ablation-importance head selection (§9a) | worse |
| coverage / diversity head selection (§9b) | worse |
| layerwise refit of the weights (§4, §4c, §10a) | worse, on **both** architectures |
| Grassmann logit-optimal basis (§10b) | tied, and a worse init |

Not one beats it. The prescription that survives every experiment here is embarrassingly
simple: **restrict the teacher's operator onto its maximum-variance subspace, keep whole
circuits, and change nothing else.**

### What this does *not* license

Tested on Llama: §2's basis inversion (**refuted**, a LayerNorm artifact), §1's init-PPL
correlation across bases (**refuted**, same cause), and §4/§4c's fit inversion
(**confirmed**, it generalizes).

Not tested outside GPT-2: the spectrum-vs-alignment ablation (§4d), though §10a is strong
indirect support, and the head-selection results (§9a, §9b). Of the things we did check,
one flipped (§2's basis inversion) and one held (§4's fit inversion). Do not assume the rest
transfer.

## 11. Remaining open questions

1. **Why is fit quality non-monotone (§4c)?** Early stopping regularizes, but an explicit
   L2 trust region does not reproduce the effect. What *is* the implicit constraint that
   300 Adam steps impose, and can it be written down? A method that names it would be a
   real contribution; we did not find it.
2. **Does finding 2 (importance ranking hurts) hold on an RMSNorm model with untied
   embeddings?** The mechanism (LayerNorm normalizing across a stream of outlier
   coordinates) predicts the effect should weaken on Llama-family models.
3. **Does any of this transfer past GPT-2 / WikiText-2 / 4.15×?** Nothing here has been
   tested at another scale, ratio, or architecture.
4. **Does the ordering hold at convergence?** §8 only rules out a 4×-budget artifact.

## 12. The objective axis, and a second bug of ours

§6 concludes that no initialization criterion beats plain reconstruction, so if a technique
exists that conclusively beats prior distillation work, it must live in the **objective** or
the **data policy**. §5 admitted we had established nothing there. This section closes that
gap as far as we honestly can.

### 12a. Our skew-KL was measuring nothing

`scripts/analysis/bench.py`'s `skew_kl` arm (403.33 ± 5.09 PPL) computed
`KL(p_t ‖ 0.9·p_t + 0.1·p_s)`. DistiLLM (2402.03425) defines

    SKL_a(p ‖ q) = KL( p ‖ a·p + (1−a)·q ),   a = 0.1

That mixture is dominated by the **student**, not the teacher. Ours was bounded above by
log(1/0.9) = 0.105 and carried **1.6%** of forward-KL's gradient signal at a typical
teacher/student gap. That arm was barely training the model. It is fixed in
`scripts/analysis/h2h.py` and reimplemented in `scripts/analysis/objectives.py`, pinned by
`tests/test_objectives.py::test_skew_kl_mixes_toward_the_student_not_the_teacher`.

**The 403.33 (skew-KL) and 314.95 (reverse-KL, no on-policy sampling) numbers reported in
§4 are void.** With the corrected divergence, DistiLLM's objective reaches 222.73 ± 2.79.

### 12b. Objectives, holding the initialization fixed

Same absorbed-init student (320/5 heads, 30.0M), same LR, same data order, n=3, matched on
**wall-clock** (`runs/obj_absorbed_s012.json`). On-policy sampling costs **22.7× per step**
(112 sequential decode steps at batch 4), so on-policy arms get proportionally fewer steps:

| arm | objective | on-policy | steps | final PPL | s |
|---|---|---|---|---|---|
| **kd** (Hinton) | forward KL | 0.0 | 1500 | **166.85 ± 1.91** | 58 |
| distillm | skew KL, a=0.1 | 0.0 | 1500 | 222.73 ± 2.79 | 74 |
| rkl | reverse KL | 0.0 | 1500 | 330.59 ± 7.46 | 60 |
| distillm_op | skew KL | 0.5 | 126 | 1089.62 ± 104.89 | 63 |
| gkd | JSD, β=0.5 | 0.5 | 126 | 1353.15 ± 239.51 | 64 |
| minillm | reverse KL | 1.0 | 66 | 2198.95 ± 70.03 | 61 |

At equal wall-clock, plain forward KL wins. But that comparison is unfair *to the
objectives*: it prices on-policy generation at our naive implementation's cost and gives
MiniLLM 66 steps. An equal-**steps** table at a reduced common budget is reported alongside
it, so both readings exist. Whether on-policy sampling repays its cost at a realistic budget
is a question about compute, not about the objective, and we cannot settle it here.

### 12c. The metric is rigged in forward KL's favour, and that caps what §12 can claim

**Validation perplexity is the exact quantity forward KL optimizes.** Minimizing
`KL(p_t ‖ p_s)` on teacher-forced data is, up to the teacher's own entropy, minimizing the
student's next-token cross-entropy on that data, which is what perplexity measures.

MiniLLM and GKD *argue for* reverse KL and on-policy sampling **on the grounds that they
trade perplexity for generation quality**: reverse KL is mode-seeking, so the student stops
spreading mass over teacher modes it cannot represent, which raises PPL and lowers
exposure-bias error in free generation. Their papers evaluate with ROUGE, GPT-4 judgments
and held-out task accuracy, not with LM perplexity.

So "forward KL wins on perplexity" is close to a tautology, and §12b **cannot** be read as
"our method beats MiniLLM/GKD/DistiLLM". To make that claim one would have to evaluate
free-running generation quality on an instruction-following or summarization benchmark, on
models large enough for those metrics to mean something. We did not, and nothing in this
repository does.

What §12b *does* establish, narrowly: at this scale and budget, on this metric, swapping the
divergence or adding on-policy sampling to a well-initialized student does not help, and the
initialization contributes far more than any objective we tried. That is a statement about
initialization, which is what this document is about.

### 12d. What this is not

These are the *objectives and data policies* those papers contribute, dropped into one
controlled rig. They are **not** the published systems. MiniLLM's sequence-level policy
gradient, its length normalization and teacher-mixed rollouts are absent; DistiLLM's
adaptive off-policy schedule is a fixed ratio; neither is tuned. Where a number here
disagrees with a paper's, believe the paper. We report them because §5's placeholder arms
were worse than nothing (one of them was literally not training the model), and a wrong
number sitting next to a right one is more damaging than no number at all.
