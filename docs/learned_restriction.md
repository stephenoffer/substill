# Learned Restriction Distillation (LRD)

**Date:** 2026-07-10
**Setup:** `JackFram/llama-160m` (768 hidden, 12 layers, 12 heads, RMSNorm, untied
embeddings) → 384-hidden / 6-head / 1536-intermediate student, **3.07×** compression,
WikiText-2, seq 128, batch 4, lr 1e-3, forward KL, A10G. Teacher val PPL 28.66. Every
number is n=3 seeds, mean ± sd. Reproduce with `scripts/analysis/lrb.py`; the module is
`substill/compression/restricted.py`, pinned by `tests/compression/test_restricted.py`.

**Library API.** The method is exposed as the recommended public entry point.
`substill.learned_restriction_distill(teacher, train_loader, config=substill.LRDConfig.for_ratio(teacher, 0.5, steps=2000))`
returns a folded, zero-overhead `LlamaForCausalLM` (see `substill/lrd.py`,
`examples/learned_restriction.py`). `LRDConfig.for_ratio` sizes the student from a
compression ratio. The Stiefel step is a **trust region** — `v_lr` is the RMS angle `V` turns
per step (§9c), so it is a *physical* quantity that transfers between teachers rather than a
fitted `1/d` rule; the default `0.002` rad/step is the best value at both scales measured. It is
**not** tuning-free — see §11f, where my first default (0.005) turned out to over-rotate
catastrophically at 1.3B. The width-dependent rule `min(1e-3, 0.77/d)` that this document used to
prescribe (§5) is retained only behind `v_trust_region=False`, to reproduce the published ambient
numbers.

> ## ⚠ Read §9–§11 before anything below them.
>
> A soundness audit (2026-07-13) found **two silent-wrongness bugs** (a corrupted teacher on
> tied-embedding models; re-paired query/kv heads on every GQA model), **three modelling
> errors**, **two false claims** (one theoretical, one about frontier scaling), and **one piece
> of statistical overreach**. All are fixed or withdrawn there, each pinned by a test.
>
> **The method works at 160M. Most of the published margin was baseline weakness. The scaling
> claim is withdrawn.**
>
> | | strongest frozen basis | `lrd` | LRD's margin |
> |---|---:|---:|---:|
> | as published (n=3) | 80.81 ± 0.85 | 74.20 ± 0.38 | −6.61 (−8.2%) |
> | **corrected (n=6)** | **75.00 ± 1.07** | **71.25 ± 1.12** | **−3.75 (−5.0%), p < 0.001** |
>
> Two bugs in the initialization that **both arms share** were flattering LRD, because a trained
> projection can absorb a bad start and a frozen basis cannot. A mis-scaled RMS gain (§9b) was
> worth ~1.3 PPL to the baseline; a residual covariance dominated by its deepest layers (§11) was
> worth another ~4.5. Fix both and the frozen baseline climbs from 80.8 to **75.0** — landing *on
> top of the originally-published LRD result*.
>
> What survives is **smaller and far better established**: measured at n=6 against the strongest
> corrected frozen basis (AIR), training the projection is worth **−3.75 PPL (−5.0%)**, 95% CI
> [−5.16, −2.34], p < 0.001, with all six seeds beating all six. The best model in the study is
> LRD on the corrected basis (**71.25**).
>
> **But §5's "the win grows with scale" is withdrawn (§11g).** At 1.3B the seed noise (±68 PPL
> between two seeds) dwarfs every effect anyone is measuring; re-measured at n=4 on the corrected
> map, LRD trends −7.9% with **p = 0.35**. The published 2.7B row reports a baseline sd of ±113
> against a claimed effect of 107 — *the noise exceeds the effect*. **160M is settled; scale is
> not.**
>
> **§9–§11 are the headline; the tables below are pre-audit, kept for the record.**

This document reports the one construction in this repository that **beats the strongest
baseline at matched wall-clock**, on the architecture family the library targets. It is
built directly on the single principle that survived `docs/init_findings.md`.

## 0. The opening the audit left

`docs/init_findings.md` demolished every mechanism this project had proposed and ended on
exactly one surviving principle, confirmed on two architectures:

> A change of **basis** *restricts* the teacher's operator (`W_s = Vᵀ W_T V` is still the
> teacher's weight, seen through a subspace, so its layers compose as before) and transfers
> through distillation. A **refit** *replaces* the operator with a regression solution and
> always makes the distilled student worse.

It then tried **six** ways to choose that subspace `V`: variance ranking, logit-weighted
variance, ablation-importance and coverage head selection, layerwise refit, and the
Grassmann-optimal *logit-error* basis. **Not one beats plain PCA** (§10b). The diagnosis it
gave for the best of them, the logit-error basis, is the key:

> `M = W_lm^T W_lm` is the Jacobian of the *final* layer alone, while the residual basis is
> shared by all twelve — a direction that barely reaches the logits directly may be exactly
> what layer 3 needs to compute what layer 9 writes. Linearizing the network at its output
> throws that away.

All six are **surrogates**. Each optimizes a proxy of student quality (retained variance;
logit error under a one-layer linearization) instead of the quantity actually being
minimized: the KD loss of the assembled student, through the whole network.

The un-surrogated arm was never run. That is this document.

## 1. The construction

Parameterize the compressed student as a point on the Grassmannian:

    W_s = Vᵀ W_T V,      V ∈ St(d, k),   V column-orthonormal

and train **`V` itself against the true KD loss, through the whole network**. In this pure
form `V` is the *only* degree of freedom (768×384 ≈ 295k numbers against the folded student's
30M weights), every reachable point is an exact restriction of the teacher, and there is no
way to leave the class that transfers. The model then depends on `V` only through `span(V)` —
it is invariant under `V → V R` for orthogonal `R`, so it is genuinely a function on the
Grassmannian, and its Riemannian gradient is purely *horizontal* (it tilts the subspace and
never merely re-bases it). Both facts are pinned by
`test_pure_restriction_is_a_grassmann_function`.

`RestrictedLlama` (`substill/compression/restricted.py`) materializes, on the fly and
differentiably in `V`,

    embed = W_E V     q,k,v = W_{qkv}[:rows] V      gate,up = W_{g,u}[idx] V
    lm    = W_lm V    o     = Vᵀ W_o[:, :rows]      down    = Vᵀ W_d[:, idx]
    norms_ℓ = √(d/k · ρ_ℓ(V)) · 1     (a per-norm RMS gain, differentiable in V — see §9b)

`V` is optimized by `StiefelAdamV`: Adam on the Riemannian gradient `G̃ = G − V·sym(VᵀG)`,
with a **trust-region step** (the learning rate *is* the RMS angle `V` turns per step, §9c),
transported momentum, and a **polar** retraction (the metric projection; unlike QR it is
gauge-equivariant, §9c). At deployment `fold()` returns a plain `LlamaForCausalLM` with
weights `Vᵀ W_T V`, function-identical to the trained module (pinned by
`test_fold_is_function_identical`, `test_fold_tracks_a_moved_basis`). Inference carries
**zero overhead**.

**Joint variant (the one that wins).** Add a zero-initialized Euclidean residual `D` to every
weight: `W_s = Vᵀ W_T V + D`. This is the *same function class* as an ordinary absorbed-init
student, since `D` alone spans every weight, plus `d·k` parameters for `V` itself (295k, ~1%
of the student; an earlier draft claimed an *identical* parameter count, which was wrong).
The difference is one extra coordinate: moving `V` moves all twelve layers coherently, in the
single direction that keeps the student a restriction of the teacher. Because `D` starts at
zero, training **begins at exactly the PCA-absorbed student the baseline starts from**, so the
comparison isolates the `V` coordinate and nothing else
(`test_zero_residual_is_the_plain_restriction`).

> **The claim this document used to make here is false, and is withdrawn.** Earlier drafts
> said that under the joint variant the student "stays inside the class of restrictions at
> every step" and that "there is no way to leave the class." That is true only of the pure
> variant. `D` is unconstrained: it can reach *any* student the baseline can, so with a free
> core the restriction is **not an invariant of training**. It is two weaker things, both
> real: an **initialization** (`D = 0`) and a **coordinate system** (`V` moves every layer
> coherently; `D` moves each weight alone).
>
> Whether the trained student nonetheless *stays near* the class is an empirical question, so
> we measure it instead of asserting it. `RestrictedLlama.restriction_gap` reports
> `‖D‖_F / ‖Vᵀ W_T V‖_F`, and at the end of the headline run it is **0.46** — the free residual
> has grown to 46% of the restriction's norm. The student that wins is emphatically *not* an
> exact restriction of its teacher. What the restriction buys is the starting point and the
> coordinates, and — per §2 — that is enough. Weight decay on `D` pulls it back toward zero,
> i.e. toward the class, which is a regularizer *toward the restriction* rather than toward
> the origin; that is worth knowing, and it is not a constraint.

## 2. The controlled result: training `V` beats freezing it, in isolation

Three arms. **Identical** student geometry, FFN neuron selection, whole heads, data order,
seed, optimizer; the last two also share a code path and a per-step wall-clock. Between
`pca_reparam` and `lrb_joint`, the only thing that changes is whether the `V` coordinate is
trained. All at 2000 steps, n=3.

| arm | what trains | final PPL | wall-clock |
|---|---|---|---|
| `pca` (best baseline in `init_findings.md`) | weights, frozen PCA basis | 80.94 ± 0.90 | 115s |
| `pca_reparam` (control) | weights `D`, **`V` frozen** | 81.25 ± 1.06 | 148s |
| **`lrb_joint`** | weights `D` **+ `V` trained** | **75.45 ± 0.79** | 151s |

The control is airtight. `pca_reparam` runs `lrb_joint`'s exact code path (same `D` residual,
same optimizer, same cost) with the Stiefel learning rate set to zero, and it **reproduces
the PCA baseline** (81.25 vs 80.94, within noise). The reparameterization adds nothing on its
own. Turning on the `V` coordinate drops PPL by **5.5 points**, and every `lrb_joint` seed
beats every seed of both baselines.

*(The "~6σ" this paragraph used to claim is not supportable from three seeds — see §9e. The
effect is real; the honest statement is a Welch interval plus the exact permutation p, whose
floor at n=3 is 0.05.)*

This is the missing arm from `init_findings.md` §10b: optimizing the restriction against the
true KD loss, through the network, does what all six surrogates could not.

### 2a. What does *not* work, and why the mechanism is joint co-adaptation

| arm | final PPL | lesson |
|---|---|---|
| `lrb_only` (train **only** `V`, 0.5% of params, weights frozen) | ~100 (2000 steps) | the restriction alone is a real but weak model |
| `lrb` (train `V` for 250 steps, then release weights) | 81.4 ± 0.9 | two-phase **loses** — the same "better init, worse final" inversion the audit documents |
| `lrb_amortized` (train weights every step, refresh `V` every 20) | 87.5 ± 1.0 | periodic `V`-refresh **jumps derail the optimizer** more than they help |
| `lrb_joint` started from the **AIR** basis (`gn`) instead of PCA | 77.07 ± 0.31 | a *better frozen start is worse* (75.45 from PCA): the init-fidelity inversion again |

The win requires `V` and the weights **co-adapting every step**. It is not a better
initialization, and not a rare correction.

The last row deserves its own note. Starting LRD from the strongest *frozen* basis (AIR's
activation+influence, the best in §4a at 79.98) gives a **worse** distilled model than
starting from plain PCA: the same "higher init fidelity, worse final" inversion
`init_findings.md` documents. PCA-start is not a tuning artifact. It is the right starting
point, and the win comes from where `V` *travels*, not where it begins. `lrb_only` is worth a
line on its own terms: training 0.5% of the parameters (the projection, weights untouched)
already reaches 100 PPL, a striking parameter-efficiency point. Only joint training beats the
baseline, though.

### 2b. What *does* improve it: dense per-layer gradient for `V` (2026-07-11)

In `lrb_joint` the only signal reaching `V` is the gradient of the single scalar KD loss,
backpropagated from the logits through all twelve layers. But the construction hands us a
target no ordinary KD has: **the student *is* the teacher restricted to `V`, so the student's
residual stream should point the way the teacher's stream does, seen through `V`**. Written
out, `h_s^(ℓ)` should agree in direction with `Vᵀ h_T^(ℓ)` at every layer. Add a
scale-invariant (cosine) **restriction-consistency** term over all layers, annealed linearly
to zero so the *final* basin is still chosen by KD alone, and `V` gets dense per-layer credit
assignment early: "a direction layer 3 needs is paid for by layer 3," now literally. A small
`V`-LR floor (`v_floor=0.1`) keeps `V` travelling through the whole budget rather than
freezing at cosine's tail. Both stack. Same geometry, same data, same seeds, matched steps
(n=3, 2000 steps):

| arm | final PPL | vs `lrb_joint` |
|---|---|---|
| `pca` (strongest frozen baseline) | 80.94 ± 0.90 | — |
| `lrb_joint` (train `V`, plain KD) | 75.45 ± 0.79 | — |
| **`lrb_joint` + consistency aux + `V`-floor** | **74.36 ± 0.06** | **−1.09, ~13× tighter** |

Every seed of the new arm (74.42 / 74.30 / 74.36) beats every seed of `lrb_joint`
(74.98 / 75.01 / 76.37), lifting the win over the baseline from 6.8% to **8.1%**. The second
effect is as notable as the first. The aux term collapses the seed spread from **±0.79 to
±0.06**: training `V` against a denser signal lowers the mean *and* makes the run almost
deterministic, the same stabilization §5 reports for `V`-training at 2.7B. The aux forward
costs ~1.5× per step (a second teacher/student pass for the hidden states), so as in §3 the
matched-*wall-clock* margin is smaller than the matched-step margin above. The variance
collapse is budget-independent. `aux_w`, `v_floor` and `aux_until_frac` are exposed on
`LRDConfig` (defaults on); the arm is `scripts/analysis/lrb.py --aux-w 1.0 --v-floor 0.1`.

**A reliability fix that came with it.** Training `V` is unusually sensitive to a
low-diversity data trajectory. Hand the public entry point a *plain, unshuffled* list (a
natural thing to do) and it walks the corpus in order, landing at **95.0 PPL instead of
75.7**: a silent 19-point regression that never touched the research driver, which shuffles
internally. `substill.lrd._cycle_ids` now buffers and draws a seeded permutation by default
(`LRDConfig.shuffle`, opt out when the loader already shuffles each epoch), so the front door
reproduces the driver regardless of how the caller built the loader.

## 3. The wall-clock win, and that it compounds

`lrb_joint` costs 1.31× per step, from re-projecting the teacher every forward. The overhead
is the full-rank `D` residual, not `V`: a `V`-only restricted forward is +5%. So the honest
test gives the baseline **1.31× the steps** for the same wall-clock. The `V` coordinate
improves per-step quality; the question is whether that outruns the overhead.

| matched wall-clock | `lrb_joint` (steps) | `pca` (steps) | joint advantage |
|---|---|---|---|
| ~150s | 75.45 ± 0.79 (2000) | 75.16 ± 1.06 (2600) | −0.29 (tied) |
| ~226s | **67.01 ± 0.23** (3000) | 68.13 ± 0.54 (3930) | **+1.12** |
| ~302s | **61.89 ± 0.75** (4000) | 62.71 ± 0.41 (5240) | **+0.82** |

At the shortest budget the per-step edge exactly cancels the 1.31× overhead. Past a crossover
around ~150s the edge **overtakes** it. The win then holds: ~1 PPL at matched wall-clock,
stable across the two longer budgets, and outside the seed noise at both (at 302s the bars
are 61.89 ± 0.75 vs 62.71 ± 0.41). Training the projection is not a head-start the baseline
erases with cheaper steps. It reaches a better basin per unit compute, and stays there.

The claim is therefore bounded: **beyond the overhead crossover, LRD beats the strongest
baseline at equal wall-clock**; below it, the two tie. It never loses.

## 4. Why this is novel

The conjunction is unclaimed against the 2024–2026 literature surveyed in
`papers/gap_analysis.md`. Among the frozen-subspace compressors (FWSVD, SliceGPT, MoDeGPT,
ESPACE, Eigen Attention), `V` is picked by a reconstruction surrogate and then frozen; LRD
trains `V` against KD, end to end. Dobi-SVD and LLRC learn *rank / truncation* against a
reconstruction or perplexity objective, not the projection, and not through the network. For
the **KV cache**, MatryoshkaKV does train orthogonal projections with a distillation
objective, but that is not a weight-side full-model restriction. Stiefel-factor methods
("Don't be so Stief!") likewise train KV-cache factors on the manifold.

LRD's delta is precise: **the weight-side residual-stream projection of the entire model,
trained on the Stiefel manifold against the KD loss through the whole network, while the
student remains an exact restriction of the teacher at every step**, folding to a plain model
with zero inference overhead. The isolated control (§2) shows the gain is attributable to that
and nothing else.

## 4a. Measured against the 2026 SVD-compression wave

> **Superseded by §11a.** Every basis in the table below — and every method it stands in for —
> was built from the **pooled** covariance §11 shows to be depth-imbalanced, so all of them are
> depressed by ~5–6 PPL at once, and LRD's 5.7% margin over the best of them is inflated for the
> same reason its margin over PCA was. §11a re-runs the whole contest on the corrected covariance
> at n=6. The qualitative finding *strengthens* (every criterion ties; the covariance is what
> matters) and LRD still wins, but by **4.3%, not 5.7%**.

A survey of the last ~6 months of LLM low-rank compression (as of 2026-07) turns up a dense
cluster of methods: **AIR** (2606.19993, activation+influence-weighted SVD), **LASER**
(2604.17224), **Swift-SVD** (2605), **SVD-LLM v2** (2503.12340), **IO-SVD** (2605.15626),
**SigmaScale** (2606.07098, learned *scaling* on SVD factors), **COMPOT** (2602.15200,
Procrustes-fit orthogonal transform), **SAES-SVD** (ICLR'26, sequential error suppression),
and the rank-allocation methods **UniRank** (2606.21847), **ARA** (2510.19389), and
differentiable rank selection **LLRC / Dobi-SVD** (2512.13733, 2502.02723). Two structural
facts hold across the whole cluster.

First, the basis is chosen by a surrogate and then frozen. Activation covariance, activation
whitening, or Fisher/influence weighting, then SVD-truncate. None trains the projection
against the KD loss through the network. (SigmaScale learns diagonal scalings; COMPOT fits an
orthogonal transform to *reconstruction*. Still not the projection against KD.)

Second, they do not distill. They are post-training, function-preserving compressors, so at
the 3× ratio here their *raw* PPL is far worse than any distilled number. The only fair
contest is their **basis-selection principle** dropped into an identical distillation
pipeline.

That contest is what `scripts/analysis/llama_basis.py` runs. Each recent method's principle is
reproduced as the frozen residual basis `V`, then distilled under `lrb_joint`'s exact recipe:
same budget, same data, same student (llama-160m, 3.07×, n=3).

| basis principle (representative 2026 method) | rel. logit error | final PPL |
|---|---|---|
| activation SVD: LASER / Swift-SVD / ASVD (`pca`) | 0.0192 | 80.94 ± 0.91 |
| activation-whitened SVD: SVD-LLM (`whiten`) | 0.0192 | 81.04 ± 1.35 |
| activation **+ influence**: AIR (`gn`) | 0.0155 | 79.98 ± 0.61 |
| influence-**optimal** surrogate (`grassmann`, upper bound on the above) | **0.0130** | 81.56 ± 0.46 |
| **LRD — `V` trained against KD (`lrb_joint`)** | — | **75.45 ± 0.79** |

LRD beats the **best** frozen basis, AIR's activation+influence principle at 79.98, by
**5.7% (≈5σ)**, and every other by more. One column shows *why*. Down the surrogates,
relative logit error falls monotonically (0.0192 → 0.0130) while final PPL does not track it:
the influence-**optimal** basis has the lowest error and a near-worst PPL. Every 2026 method
is climbing a proxy that does not predict distilled quality. LRD optimizes the quantity itself.

The **rank-allocation** axis (UniRank, ARA, LLRC) is orthogonal. LRD keeps whole heads at a
uniform rank, and the repo's `use_diff_rank` DDR composes on top of it, so those methods are
complements, not competitors, to the basis-training claim. The one recent method that *does*
train a Stiefel projection, **MAPL** (2606.05484), targets pipeline-parallel activation
communication, not weight-side model compression; **MatryoshkaKV** (2410.14731) trains
projections with a distillation objective, but for the KV cache. Neither is a weight-side
full-model restriction.

## 5. It is not a 160M artifact: the win grows from 1.3B to 2.7B

> ## ⚠ WITHDRAWN. See §11g.
>
> The claim in this section's title — *"the win grows from 1.3B to 2.7B"* — **is not supported by
> its own data**, and is withdrawn.
>
> Both rows are `n=2` / `n=3`, and the seed noise at this scale is enormous: re-running 1.3B finds
> the frozen baseline swinging **±68 PPL between two seeds**, and the published 2.7B row's own
> baseline sd (**±113**) *exceeds its entire claimed effect* (107 PPL). At those `n`, these tables
> describe which seeds were drawn as much as how the method behaves.
>
> Both rows are also measured on the **pooled** covariance, which §11 shows is depth-imbalanced
> just as badly at 1.3B (the pooled basis keeps 54.5% of the embedding's energy against the
> balanced basis's 91.8%), so both baselines are handicapped on top of being noisy.
>
> Re-measured properly at 1.3B (corrected covariance, shipped `v_lr`, n=4): LRD trends **−7.9%**
> but with **p = 0.35** and a CI spanning zero. The mechanism transfers; **the margin at scale is
> unmeasured**, in either direction. The 160M result (§11b) is the only one settled.

Repeated on two larger Sheared-LLaMA teachers (RMSNorm, untied), WikiText-2, n=2:

| teacher | d | compression | `pca` | `lrb_joint` | LRD margin | n |
|---|---|---|---|---|---|---|
| Sheared-LLaMA-**1.3B** | 2048 | 3.64× | 366.70 ± 0.75 | **341.99 ± 7.15** (v-lr 1e-3) | **−6.7%** | 2 |
| Sheared-LLaMA-**2.7B** | 2560 | 9.8× | 635.57 ± 112.87 | **528.66 ± 36.06** (v-lr 3e-4) | **−16.8%** | 3 |

(1.3B: 1000 steps; 2.7B: 600 steps. Both matched-steps; the 2.7B run fits a 22 GB A10G with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.) The mechanism transfers intact from 160M
through 2.7B, 17× the smallest teacher, and the win *grows* with scale. Note the standard
deviations. At 2.7B the frozen-basis baseline is **erratic** (±113, seeds 514/654/738) while
LRD is **tight** (±36, seeds 493/528/565): training `V` lowers the mean and stabilizes the
run. These are lightly-tuned short-budget runs, so treat the exact magnitudes as noisy. What
is solid is that the effect holds, and holds larger, across an order of magnitude of scale.

### The knob that used to have to scale with the teacher: the Stiefel learning rate

> **Superseded by §9c.** The premise of this subsection — that `v-lr` is "roughly the rotation
> per step" — was **false**, and everything below is the symptom. Under the ambient step the
> rotation is proportional to the norm of an unnormalized Adam direction, so it scales like
> `√d` *and* swings 25× within the first 32 steps of a single run. That is why the same `v-lr`
> "over-rotates and destabilizes" on a bigger teacher, and why a hand-fitted constant was
> needed at all. The trust region makes the premise *true* — `v_lr` now **is** the RMS rotation
> per step — and the rule below then has nothing to correct. Keep reading only for the
> historical ambient numbers.
>
> Note also that the fitted exponent is wrong: the geometry gives `1/√d`, not `1/d`. The three
> teachers used to fit it varied width and compression ratio together, so a `k/d` effect was
> absorbed into a `d` exponent — which is exactly the failure mode of extrapolating a constant
> nobody derived.

`StiefelAdamV` normalizes the `V`-gradient (Adam-style), so `v-lr` is roughly the *rotation
per step*. A bigger, deeper, more-compressed student accumulates that rotation over more layers
and more aggressive truncation, so the same `v-lr` over-rotates and destabilizes. Measured at
2.7B / 9.8× (600 steps, n=2):

| v-lr | final PPL |
|---|---|
| 1e-3 (the 160M optimum) | 758.8 ± 393.6, **unstable**: one seed diverges to 1037 |
| **3e-4** | **510.5 ± 25.0: stable, −27% vs pca** |
| 1e-4 | 677.4 ± 157.5, too slow, high variance |

The practical rule that fits every scale tested: **lower `v-lr` for larger teachers.** `1e-3`
is right up to d≈2048 (160M–1.3B); `3e-4` is right at d=2560 / 9.8×. Roughly `v-lr ∝ 1/d`. The
weight learning rate and everything else are unchanged.

## 5a. How far compression pushes: LRD wins at every ratio tested

Same `llama-160m` teacher, student width swept down (matched steps, n=2). The embedding
dominates the parameter count at this scale, so shrinking hidden size raises the ratio fast:

| student hidden | compression | `pca` | `lrb_joint` | LRD margin |
|---|---|---|---|---|
| 384 | 3.07× | 80.94 ± 0.90 | **75.45 ± 0.79** | **6.8%** (n=3) |
| 256 | ~5.6× | 92.98 ± 0.34 | **89.02 ± 0.42** | **4.3%** |
| 192 | ~8.4× | 101.10 ± 1.22 | **97.03 ± 1.40** | **4.0%** |

LRD beats the strongest baseline at all three, at matched steps. The margin is *largest at
moderate compression* and narrows as the student gets tighter, the opposite of the naive
guess. A plausible reading: at extreme compression the retained subspace is so small that
where exactly it points matters less than that there is too little of it, so optimizing `V`
has less to work with. The honest headline is that the win holds across ratios, not that it
grows with them. (At every ratio `lrb_joint` costs ~1.3× per step, so as in §3 the
matched-*wall-clock* margin at the tightest ratios is smaller than the matched-*step* margin
shown here.)

## 6. Why the design scales further

> **Corrected by §9g. Every sentence in this section is true of `free_core=False` — the arm
> that reaches ~100 PPL and *loses*.** The arm that wins carries `D`, a free residual on every
> student weight *including* `D_emb` and `D_lm`, which are `(vocab, k)` each: at the 3.07×
> benchmark that is 53.2M trainable parameters, **46% of them vocabulary-sized**. So the
> winning arm is *not* vocabulary-independent and its per-step cost is not either. What
> survives is the narrower claim: no optimizer state on the *teacher*, and `V` itself costs
> only `d·k`. The student is otherwise trained like any other distilled student, at the same
> cost. See §9g.

The trainable projection is one `(d, k)` matrix regardless of teacher depth or vocabulary, and
the teacher's weights are read-only: **no optimizer state on the frozen teacher**. The
embedding and unembedding are never materialized (`W_E V` is indexed per batch;
`h (W_lm V)ᵀ = (h Vᵀ) W_lmᵀ` lifts through the teacher's own head), so per-step cost is
independent of vocabulary size. These are the properties a frontier-scale student needs. The
same recipe that runs on this 30M student is the one where re-projecting every weight each step
is affordable but carrying `|W_T|` optimizer states is not.

## 7. Vision (ResNet50): the principle transfers, and it locates the win precisely

A ReLU-CNN has no rotation-equivariant residual stream. A BN+ReLU sits between every pair of
convs, and ReLU does not commute with a dense basis rotation, so LRD's residual-stream
rotation is unavailable. Only *selection* survives ReLU, so the vision analog of "optimize the
restriction against KD, not a surrogate" is **KD-driven channel selection**: put a soft gate on
every inner channel, train the gates and weights against KD under a budget, harden to the same
width the variance baseline uses (`substill/vision/gated.py`). Hardening is
function-preserving, the CNN counterpart of `fold()`.

ResNet50 → 0.5-width bottlenecks (~2.3×, 10.35M params), CIFAR-10, teacher top1 0.7908, n=2:

| arm | top1 |
|---|---|
| random init | 0.7346 ± 0.0426 |
| variance selection (substill.vision baseline) | **0.8201 ± 0.0170** |
| KD-driven selection (ours) | 0.8125 ± 0.0218 |

Two things, and together they are the cleanest evidence in this document for *where the LRD win
lives*:

1. **The restriction principle transfers.** Both selection methods beat random init by ~8
   points. Absorbing the teacher into a kept-channel subspace and distilling is a large, real
   win in vision too.
2. **KD-driven selection ties variance selection** (0.8125 vs 0.8201, bars overlapping). This
   is *not* a failure; it is exactly what `init_findings.md` §2/§9 reports on transformers:
   **selection criteria are interchangeable; importance ranking never beats an arbitrary or
   variance choice.** The LRD win on transformers was never the *selection*. It was the
   **rotation** (a dense trainable `V`), which no channel-selection can express, KD-driven or
   not, and which ReLU makes illegal in a CNN.

So vision confirms the half of the story that survives without a rotation-equivariant stream
(restriction ≫ random) and, by tying, confirms the other half by elimination: the 6σ gain in §2
comes from rotating the residual subspace, not from choosing which coordinates to keep. Run with
`scripts/vision/resnet_kd_select.py`; invariants pinned by `tests/vision/test_vision_gated.py`.

## 8. Reproducing

```bash
# the controlled result (§2): pca vs pca_reparam vs lrb_joint, n=3
PYTHONPATH=. python -m scripts.analysis.lrb --steps 2000 --v-lr 1e-3 --seeds 0 1 2 \
    --arms pca pca_reparam lrb_joint --no-compute-match

# the consistency-aux + V-floor improvement (§2b): 74.36 ± 0.06, n=3
PYTHONPATH=. python -m scripts.analysis.lrb --steps 2000 --v-lr 1e-3 --seeds 0 1 2 \
    --arms lrb_joint --aux-w 1.0 --v-floor 0.1 --no-compute-match

# the wall-clock win and its compounding (§3): matched-wall-clock pairs
PYTHONPATH=. python -m scripts.analysis.lrb --steps 3000 --v-lr 1e-3 --seeds 0 1 2 --arms lrb_joint
PYTHONPATH=. python -m scripts.analysis.lrb --steps 3930 --seeds 0 1 2 --arms pca

# the negative arms (§2a)
PYTHONPATH=. python -m scripts.analysis.lrb --steps 2000 --v-lr 1e-3 --seeds 0 1 2 \
    --arms lrb_only lrb lrb_amortized

# scale-up to 1.3B and 2.7B (§5). `--v-lr 0` requests the auto rule (v-lr ~ 1/d).
PYTHONPATH=. python -m scripts.analysis.lrb --teacher princeton-nlp/Sheared-LLaMA-1.3B \
    --hidden 1024 --interm 2752 --n-head 8 --n-kv 8 --batch-size 2 --steps 1000 \
    --v-lr 1e-3 --seeds 0 1 --arms pca lrb_joint --no-compute-match
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python -m scripts.analysis.lrb \
    --teacher princeton-nlp/Sheared-LLaMA-2.7B --hidden 768 --interm 2048 --n-head 6 --n-kv 6 \
    --batch-size 1 --steps 600 --v-lr 3e-4 --seeds 0 1 --arms pca lrb_joint --no-compute-match

# vs the 2026 SVD-compression wave (§4a): each method's basis principle, distilled equally
PYTHONPATH=. python -m scripts.analysis.llama_basis --steps 2000 --seeds 0 1 2 \
    --hidden 384 --interm 1536 --n-head 6 --n-kv 6 --bases pca whiten gn grassmann

# vision: KD-driven vs variance channel selection (§7)
PYTHONPATH=. python -m scripts.vision.resnet_kd_select --width-ratio 0.5 --steps 2000

python -m pytest tests/compression/test_restricted.py tests/vision/test_vision_gated.py

# --- the soundness audit (§9) ---------------------------------------------------------
# every defect in §9, pinned; each of these fails on the pre-audit source
python -m pytest tests/compression/test_lrd_soundness.py tests/compression/test_stiefel_geometry.py

# reproduce the published §2 numbers on the pre-audit code paths (the credibility check)
PYTHONPATH=. python -m scripts.analysis.lrd_validate --stage train --arms pca lrd --legacy \
    --seeds 0 1 2 --out A_legacy.json

# the same arms with every fix on (per-norm gain, write-aware FFN, trust region, polar)
PYTHONPATH=. python -m scripts.analysis.lrd_validate --stage train --arms pca lrd \
    --v-rule trust --v-lr 0.005 --seeds 0 1 2 --out D_fixed.json

# what each init-side fix is worth, with no training at all
PYTHONPATH=. python -m scripts.analysis.lrd_validate --stage init

# statistics that survive n=3: Welch CI + the exact permutation floor
PYTHONPATH=. python -m scripts.analysis.lrd_stats A_legacy.json
```

---

# 9. Soundness audit (2026-07-13)

Everything above was re-derived from the construction rather than taken on trust, and the
construction was then attacked on the assumptions it never states. Eight findings came out. Two
are **silent-wrongness bugs** that make the method compute something other than a restriction
on architectures this library advertises; three are **modelling errors** in the restriction map
and its optimizer; two are **false claims** (one theoretical, one about frontier scaling); one
is **statistical overreach**. All are fixed or withdrawn below, each pinned by a test.

**First, the numbers above reproduce.** Before changing anything, the audit re-ran §2's arms
end to end on a fresh harness (`scripts/analysis/lrd_validate.py --legacy`, same teacher, data,
geometry, budget, seeds 0/1/2, A10G):

| arm | published | audit re-run |
|---|---|---|
| `pca` | 80.94 ± 0.90 | **80.81 ± 0.85** (81.19 / 79.84 / 81.40) |
| `lrd` (joint + aux + floor) | 74.36 ± 0.06 | **74.20 ± 0.38** (74.63 / 74.05 / 73.93) |

Both land within noise of the published values, so the harness is faithful and everything below
is measured against a baseline that is really the baseline. (The published `± 0.06` looks
optimistic against a re-measured `± 0.38`, which is what one expects from an sd estimated on 2
degrees of freedom — see §9e.) The win is real and it survives every fix.

Why they survived until now is itself the lesson: **every teacher benchmarked in this document
is MHA with untied embeddings** (llama-160m and both Sheared-LLaMAs), and the test suite used
one tiny fixture of the same shape. The two fatal bugs are invisible in exactly that
configuration.

| # | finding | severity | where it bites | measured effect |
|---|---|---|---|---|
| 9a | tied `lm_head` corrupts the gamma fold | **critical, silent** | Llama-3.2-1B/3B, most small Llamas | teacher's own logits off by 20–28% |
| 9a | GQA query/kv heads get re-paired | **critical, silent** | every GQA teacher (Llama-3, Mistral, Qwen) | student is a different operator |
| 9b | one RMS gain shared by all 2L+1 norms | high | every teacher | 39% error at layer 0; **init PPL −48%** |
| 9b | FFN neurons ranked by what they *hold*, not what they *write* | medium | a ranking error, unbounded in principle | **no effect on this teacher** |
| 9c | Stiefel step in ambient units → a fitted `0.77/d` constant | high | every teacher not in the fit | rotation ∝ √d; 25× swing within a run |
| 9c | QR retraction not gauge-equivariant; momentum never transported | medium | the manifold story | polar is equivariant to 5e-7; QR to 0.195 |
| 9d | "the student stays inside the restriction class" | **false** | the headline framing | `restriction_gap` ends at **0.46** |
| 9e | "~6σ" from n=3 | overreach | the headline number | exact-permutation floor is p=0.05 |
| 9f | aux term's last-layer state was never *chosen* | low | the 2026-07-11 improvement | the tidier option costs **1.3 PPL** |
| 9g | the frontier-scaling story describes the arm that **loses** | high | the forward-looking claim | winner is 46% vocab-sized params |

Note the fourth and second-to-last rows. Two of these findings are *mathematically* clear-cut
and buy **nothing** — and one of them, followed on the strength of its argument alone, would
have made the method **worse**. They are reported at the same weight as the wins. An audit that
only reports the corrections that helped is not an audit.

## 9a. The two silent-wrongness bugs

**Tied embeddings corrupt the teacher.** `gamma_fold_llama` folds the final RMSNorm's gain
into `lm_head`. When `tie_word_embeddings=True` — the default for Llama-3.2-1B/3B and most
small Llamas — HuggingFace makes `lm_head.weight` *the same `nn.Parameter` object* as
`embed_tokens.weight`, not a copy. So folding the head also rescales the **input embedding**,
and the "function-preserving" fold silently changes the teacher's function. Measured against
the pristine pre-audit source on a tied toy Llama: **20–28% relative logit error**, depending on
how far the final norm's gain sits from 1. Everything downstream — the profiled second moment,
the PCA basis, the KD targets — is then computed against a model that is not the teacher.
Nothing raises. The module docstring asserted "`lm_head` is untied" as a fact about
llama-160m, and then relied on it for every teacher.

*Fix:* `untie_lm_head` runs first, on the copy. Untying is itself exactly function-preserving.
Pinned by `test_gamma_fold_preserves_the_teacher_function[tied]`.

**GQA re-pairs the heads.** Under grouped-query attention, query head `i` was trained to attend
against key/value head `i // G`, for the teacher's group size `G = n_head / n_kv`. The
restriction copies teacher q head `i` and teacher kv head `j` into the student *verbatim*, and
the student's attention then re-derives the pairing from **its own** `G' = n_head' / n_kv'`.
`plan_restricted_geometry` chose `n_head` and `n_kv` independently (each rounded from the width
ratio, then `n_kv` snapped down to a divisor), so `G' ≠ G` in general:

| teacher | ratio | old plan | teacher `G` | student `G'` |
|---|---|---|---|---|
| any GQA, `G=4` | 0.5 | `n_head=16, n_kv=4` | 4 | 4 ✓ |
| any GQA, `G=4` | **0.6** | `n_head=19, n_kv=1` | 4 | **19** ✗ |
| any GQA, `G=4` | **0.4** | `n_head=13, n_kv=1` | 4 | **13** ✗ |

When `G' ≠ G`, student query head `i` attends against a kv head whose keys its query weights
have **never seen**. The student is then not the teacher seen through a subspace; it is a
different operator, and the premise of the entire method is void — while the KD loss descends
happily and the student folds without complaint.

*Fix:* the group size is an invariant, not a free parameter. Pick `n_kv`, then *derive*
`n_head = G · n_kv`. The cost is coarser achievable widths on a GQA teacher — the honest price
of keeping whole groups. `check_head_geometry` enforces it from both `RestrictedLlama` and
`absorb_llama`, so hand-written geometries (which the research drivers accept) fail loudly
rather than silently. On MHA teachers `G = 1` and the constraint is vacuous, so **every number
published above is unaffected**.

This one had already bitten: the repo's own `test_joint_training_moves_V_and_reduces_kd_stably`
was building a `G'=1` student from a `G=2` teacher.

## 9b. Two modelling errors in the restriction map

**One RMS gain cannot serve 2L+1 norms.** The student's norms must restore the scale the
truncation costs: `√(d/k · ρ(V))`, where `ρ` is the fraction of energy `V` retains. But `ρ` is
a property of *the distribution entering that norm*, and the code used a single `ρ` from a
second moment pooled over every layer at once. That pooled moment is dominated by the
high-norm deep layers and describes the raw embedding not at all. Measured on llama-160m at
3.07×, against what each norm actually needs:

| norm | needed gain | applied | error |
|---|---:|---:|---:|
| layer 0 `input_layernorm` | 1.0074 | 1.3987 | **+38.8%** |
| layer 0 `post_attention_layernorm` | 1.0191 | 1.3987 | **+37.3%** |
| layers 1–11 (mean) | ~1.36 | 1.3987 | +1 to +5% |

The first block's input is scaled ~39% too large. RMSNorm weights are already per-layer, so
per-norm gains are *free*, and `tr(Vᵀ S_ℓ V) = ⟨S_ℓ, VVᵀ⟩` means all 2L+1 come from one shared
projector — one `(d,k)` product, still differentiable in `V`.

*Measured effect, absorbed-init PPL at 3.07× (no training):* **17,743 → 9,296, a 48%
reduction.** (`LRDConfig.per_norm_gain`, on by default; costs a `(2L+1, d, d)` buffer — 59 MB
on llama-160m — and can be switched off when that does not fit.)

**FFN neurons were ranked by what they hold, not what they write.** Neuron `i` contributes
`a_i · W_down[:, i]` to the residual stream, so dropping it perturbs the stream by that vector
and its expected squared cost is `E[a_i²] · ‖Vᵀ W_down[:, i]‖²`. `ffn_energy_indices` ranked on
`E[a_i²]` alone — so a loud neuron whose output column is ≈0 (it writes nothing) outranked a
quiet one whose output column is large (it writes a lot). It is a *ranking* error, so no
rescaling of the calibration set can hide it, and it is easy to construct a teacher where it is
exactly backwards (`test_ffn_selection_ranks_by_the_residual_write_not_the_activation`).

*Measured effect on llama-160m: none.* Init PPL 9,296 → 9,332 with the fix — a wash. On a real
trained LLM, activation energy and output-column norm are evidently correlated enough that the
old criterion was accidentally fine. **We keep the corrected criterion because it is the right
one and costs nothing, and we report that on this teacher it bought nothing.** It is a
guard against a failure mode, not a source of the win.

(The criterion remains *greedy*: it ignores correlation between neurons, so near-duplicates are
double-counted. Fixing that is a subset-selection problem, not a ranking, and is out of scope.)

## 9c. The Stiefel step was in the wrong units

The repository carried `v_lr = min(1e-3, 0.77/d)` — a constant fitted to three teachers, with
no derivation, guarding the one knob the docs called "the single knob a frontier-scale run must
set." That is a fragile place to keep a magic number, and the reason it was needed is a units
error.

An Adam direction has entries of order 1, so its Frobenius norm grows like `√(dk)` while `‖V‖_F
= √k`. The rotation the step produces therefore scales like **`√d`**, and the same `lr` turns a
wide teacher's subspace much further than a narrow one's. Measured, one step at fixed `k/d`:

| `d` | rotation, ambient `lr=1e-3` |
|---|---|
| 256 | 0.0145 rad |
| 1024 | 0.0289 rad |
| 4096 | 0.0578 rad |

Exactly `√d`: a 4× width doubles the rotation. **Note the exponent — the fitted rule is `1/d`,
but the geometry says `1/√d`.** The `1/d` fit came from three teachers whose width and
compression ratio moved together, so it absorbed a `k/d` effect into a `d` exponent. That is a
rule with no reason to extrapolate, protecting a step size that was already known to diverge at
2.7B.

Worse, the ambient rotation is not even stable *within* a run. Instrumenting the real benchmark:

| step | ‖Adam direction‖_F | realized rotation |
|---|---:|---:|
| 0 | 27,934 | 0.0228 rad |
| 8 | 182 | 0.0064 rad |
| 32 | 53 | 0.0009 rad |

The step-0 direction is *larger than `V` itself* (`‖V‖_F = 19.6`); only the LR warmup keeps the
first step from throwing the basis away. The subspace turns 25× further on one step than on
another. This is the fragility behind "`1e-3` is unstable at 2.7B, one seed diverges to 1037."

*Fix — put the step in units of rotation.* Normalize the update to `‖D‖_F = √k`, and `lr`
becomes **the RMS angle, in radians, that `V` turns per step**: dimensionless, and meaning the
same thing on a 768-wide teacher and an 8192-wide one. The `1/d` rule then has nothing left to
correct, and the trust region subsumes gradient clipping for `V` (the step length is bounded by
construction). Measured: the realized rotation is now **independent of the gradient scale over
six orders of magnitude, and constant across a 64× change in `d`**
(`tests/compression/test_stiefel_geometry.py`).

Two further geometry defects came out with it:

- **Momentum was never transported.** `m` is a tangent vector at `V`; after the retraction moves
  `V`, an untransported `m` lives in the wrong tangent space and its normal component leaks into
  the next update. It is now projected onto the new tangent space each step.
- **The QR retraction is not gauge-equivariant.** A subspace has no preferred basis, but
  `qr(A R) ≠ qr(A) R` (measured: 0.195 max abs) — QR's triangular factor pins a canonical,
  column-order-dependent basis, so the trajectory depends on an arbitrary representative. The
  **polar** retraction `A(AᵀA)^{-1/2}` — the metric projection — is exactly equivariant, and it
  has no sign ambiguity, which is why it needs none of the `diag(R)` sign-patching QR required.
  *The sign patch was a symptom of the wrong retraction.* Polar is now the default.

An honest limitation remains: Adam's **elementwise** second moment is itself not gauge-equivariant
(it is a coordinatewise statistic on a quantity with no preferred coordinates). Under
`free_core=True` the loss is not gauge-invariant either — `D` is pinned to `V`'s coordinates —
so there is no symmetry to break and this is merely a choice of coordinates. Under
`free_core=False` the loss *is* a Grassmann function and the preconditioner is then a formally
gauge-dependent approximation; `StiefelAdamV(gauge_invariant=True)` swaps in the scalar
`EMA(‖g‖_F²)`, which with the polar retraction is exactly equivariant (verified to 7e-8 in
float64). Both the fixed path and the limitation of the default are pinned by test, rather than
papered over.

## 9d. The theoretical claim was false, and is withdrawn

See the box in §1. In short: with `free_core=True` — the default, and the arm that wins — `D`
is unconstrained and can reach any student the baseline can, so the student does **not** "stay
inside the class of restrictions at every step." The restriction is the *initialization* and
the *coordinate system*, not an invariant.

We now measure the thing we used to assert. `restriction_gap = ‖D‖_F / ‖Vᵀ W_T V‖_F` on the
headline run ends at **0.46**: the free residual grows to 46% of the restriction's norm. The
winning student is decisively *not* an exact restriction of its teacher — and, per §2's
controls, it wins anyway. That is a *more* interesting result than the false one, and it
survives contact with the code.

## 9e. The statistics

"5.8 points (~6σ)" and "74.36 ± 0.06" were computed from **three seeds**. A σ from n=3 has 2
degrees of freedom and ~50% relative standard error of its own, so a "6σ" built on it is a ratio
whose denominator is barely determined. And "every seed of A beats every seed of B" — a real
result — is the extreme case of the exact permutation test, which with 3 vs 3 has only
`C(6,3) = 20` labellings: **the smallest one-sided p the design can ever produce is 1/20 = 0.05**,
no matter how large the effect. That is a fact about `n`, not about the method.

Restated defensibly, on the audit's own re-run (`scripts/analysis/lrd_stats.py`, no SciPy
dependency):

> `pca` 80.81 ± 0.85 (81.19 / 79.84 / 81.40) vs `lrd` 74.20 ± 0.38 (74.63 / 74.05 / 73.93), n=3.
> **Difference −6.61 PPL (−8.2%), 95% CI [−8.40, −4.81] (Welch, dof=2.8), t = −12.3, p = 0.002.**
> Complete seed separation; exact one-sided permutation p = 0.05 — the floor of a 3-vs-3 design.

The effect is real, large, and its interval excludes zero by a wide margin. It is simply not 6σ.
The honest way to claim more is to raise `n`, not to re-describe the same three runs — and note
that the re-run's sd (±0.38) is six times the published ±0.06, which is exactly what an sd
estimated on two degrees of freedom does.

## 9f. The aux term's last layer, and the audit's own worst mistake

This one is here because the audit **got it wrong**, twice, and the way it went wrong is the
most useful thing in §9.

The restriction-consistency term (§2b) asks the student's residual stream to point the way the
teacher's does *seen through `V`*, at every layer. It read both streams off HuggingFace's
`output_hidden_states` — where **the last entry is the state after `model.norm`**, not the last
layer's output. On the teacher that is harmless (after the gamma fold its final norm weight is a
*scalar*, and cosine ignores positive scaling). On the **student** it is not: the student's final
norm weight is `g·1 + D`, and `D` trains it into a **per-channel** vector. So the last of the `L`
terms compared `diag(γ_s) h_s` against `Vᵀ h_T` while every other compared `h_s` against
`Vᵀ h_T`. `gap_fit_llama` **documents this exact trap** in a comment written for the same
codebase.

**Mistake 1: calling it a bug.** It is not obviously one. The state after the final norm is
exactly what `lm_head` reads, and the teacher's corresponding state is exactly what *its*
`lm_head` reads — so matching them is an honest restriction statement *at that point in the
network*, and it supervises the final norm into the bargain. What was actually wrong is narrower:
**the code and its description disagreed, and nobody had chosen between them.**

**Mistake 2: choosing between them on one seed.** The "corrected" raw-residual version measured
**1.3 PPL worse** on seed 0 — same init, same data order, a clean paired comparison. That looked
decisive. It was noise. At n=3:

| aux stream at the last layer | final PPL, n=3 |
|---|---|
| `prelogit` — post-final-norm (the accidental original) | 74.59 ± 1.07 |
| `residual` — raw post-layer (what the description says) | **74.22 ± 0.48** |

Statistically tied, and if anything the *opposite* of what one seed said. The seed spread on this
method is **±0.5 to ±1.1 PPL**, so a single-seed comparison cannot resolve a 1 PPL difference at
all — and the audit came within one decision of shipping a default chosen from pure noise.

Both are now named options (`LRDConfig.aux_stream`), pinned by test. The default is `residual`:
it costs nothing measurable, and it is the one that means what the method says it means.

**The lessons, and the second is the one that matters.**

1. A trap that is documented but not *enforced* will be re-entered. Every finding in §9 is now a
   test, not a comment.
2. **An audit is not exempt from the standard it enforces.** "Mathematically cleaner" is a
   hypothesis, not a result; and a paired single-seed comparison on a chaotic optimization is not
   evidence, however clean the pairing looks. This section spends its length criticizing a "~6σ"
   claim built on three seeds — and then nearly set a default from **one**.


## 9g. The scaling story describes the arm that does not win

§6 argues that LRD is built for frontier scale:

> The trainable projection is one `(d, k)` matrix regardless of teacher depth or vocabulary,
> and the teacher's weights are read-only — **no optimizer state on the frozen teacher**. The
> embedding and unembedding are never materialized […] so per-step cost is independent of
> vocabulary size.

Every clause of that is true — **of `free_core=False`**, the arm that reaches ~100 PPL and
loses. The arm that wins carries `D`, a free residual on *every* student weight, including
`D_emb` and `D_lm`, which are `(vocab, k)` each. At the 3.07× benchmark:

| what trains | free_core=False | free_core=True (the winner) |
|---|---:|---:|
| `V` | 294,912 | 294,912 |
| block weights | — | 28,321,152 |
| `D_emb` + `D_lm` (**vocab-sized**) | — | 24,576,000 |
| **total trainable** | **0.29M** | **53.2M** |
| vocabulary-dependent? | **no** | **yes — 46% of it** |

So the winning arm is *not* vocabulary-independent, and its per-step cost is not either: the
logit correction `h @ D_lmᵀ` is a `(B,T,k) × (k,vocab)` matmul. Extrapolated to a
`d=8192 → k=4096`, `vocab=128k` student, `D_emb + D_lm` alone are **1.05B parameters and 8.4 GB
of fp32 AdamW state** — precisely the cost §6 claims the design avoids.

None of this is a *bug*. A compressed student needs its own embedding and head, and so does
every distillation baseline; LRD is not worse than the alternatives here. What is wrong is the
**claim**: §6 attributes to the method a scaling property that belongs only to the variant the
method does not use. The honest version is:

> LRD adds `d·k` parameters and no optimizer state on the *teacher*. Its student is trained
> like any other distilled student, with the same vocabulary-sized embedding and head, and the
> same costs. The `free_core=False` arm *is* vocabulary-independent and *is* the shape a
> frontier run would want — but it is 25 PPL worse, and closing that gap is open work, not a
> solved problem.

This matters because "runs on a frontier decoder" was the strongest forward-looking claim in
the document, and it was resting on the wrong arm.


---

# 10. What the audit did to the headline

The corrections are not free, and one of them costs the method something. Re-running §2's
comparison end to end — same teacher, data, geometry, budget, and seeds (n=3, 2000 steps):

| | `pca` (baseline) | `lrd` | margin |
|---|---|---|---|
| **legacy** — pooled RMS gain, ambient Stiefel step, QR | 80.81 ± 0.85 | 74.20 ± 0.38 | **−6.61 PPL (−8.2%)** |
| **fixed** — per-norm gain, trust region, polar | **79.46 ± 0.82** | 74.59 ± 1.07 | **−4.87 PPL (−6.1%)** |

Read the two middle columns, not the last one. **The fixes barely move LRD (74.20 → 74.59) and
move the baseline by 1.35 PPL (80.81 → 79.46).** The per-norm RMS gain is a correction to the
*shared* absorbed init, so the baseline gets it too — and the baseline, which has no `V`
coordinate to compensate with, benefits far more from it.

The uncomfortable implication is worth stating plainly: **roughly a quarter of LRD's published
margin was measured against a baseline handicapped by a bug in the initialization both arms
share.** `V`-training was partly buying back a mis-scaled residual stream that should never have
been mis-scaled in the first place. Give the baseline a correctly-scaled init and it closes 1.35
of the 6.61 PPL gap on its own, without touching `V`.

What survives is still a real, clean win:

> `pca` 79.46 ± 0.82 vs **`lrd` 74.59 ± 1.07**, n=3.
> **−4.87 PPL (−6.1%), 95% CI [−7.10, −2.64] (Welch, dof=3.7), t = −6.24, p = 0.004.**
> Complete seed separation (every LRD seed beats every baseline seed); exact one-sided
> permutation p = 0.05 — the floor of a 3-vs-3 design.

Every LRD seed still beats every baseline seed. The interval still excludes zero comfortably. The
mechanism §2 isolates — training the residual-stream projection against the KD loss, through the
whole network — is real, and it is worth about **6%**, not 8%.

That is the number this document should have reported, and now does.

---

# 11. The deepest finding: the baseline was weak for a reason nobody checked

§0 opens this document with the claim that licenses everything after it:

> It then tried **six** ways to choose that subspace `V` — variance ranking, logit-weighted
> variance, ablation-importance and coverage head selection, layerwise refit, and the
> Grassmann-optimal *logit-error* basis. **Not one beats plain PCA.**

Six ways to choose a basis *from a covariance matrix* were tried. **Nobody checked the
covariance matrix.**

## The statistic, not the criterion

`llama_residual_second_moment` builds the profiled covariance by *summing* the raw second moment
of every residual state:

```python
for h in hidden_states:          # embedding, then every layer's output
    acc += h.T @ h               # <-- a raw sum
```

A transformer's residual norm grows steeply with depth — often by an order of magnitude — and a
**sum is dominated by its largest terms**. So the "activation covariance" that this library, and
the entire SVD-compression literature it benchmarks against, takes its basis from is in effect
the covariance of the **last few layers**. The basis it induces barely sees the early ones.

Measured on `llama-160m` at 3.07×: the PCA basis retains **97.8% of the pooled energy** and
**~51% of the embedding's**. The first block's input is half-destroyed by a basis that reports
itself as near-perfect. (This is also the mechanism behind §9b: the *gain* was 39% wrong at layer
0 for exactly the same reason.)

Nothing intends this. It is an artifact of adding together quantities with wildly different
scales. The fix is one line — give every layer an equal vote:

    S_balanced = mean_ℓ  S_ℓ / tr(S_ℓ)

## What it is worth

Absorbed init, no training at all (`--stage init`, llama-160m, 3.07×):

| basis pooling | per-norm gain | init PPL |
|---|---|---:|
| pooled (as published) | no | 17,529 |
| pooled | yes | 8,792 |
| **balanced** | no | 4,421 |
| **balanced** | yes | **3,840** |

**4.6× better initialization**, from normalizing a sum. But this document's own thesis is that
init fidelity *anti-correlates* with final quality (§2a: "a better frozen start is worse"), so
that proves nothing on its own. It has to be trained. n=3, everything else identical:

| basis pooling | `pca` (frozen basis) | `lrd` (trained `V`) | LRD's margin |
|---|---:|---:|---:|
| pooled (as published) | 80.81 ± 0.85 | 74.20 ± 0.38 | −6.61 (−8.2%) |
| pooled + per-norm gain | 79.46 ± 0.82 | 74.59 ± 1.07 | −4.87 (−6.1%) |
| **balanced + per-norm gain** | **74.94 ± 1.29** | **71.68 ± 1.51** | **−3.26 (−4.4%)** |

Three things, in order of how much they should hurt.

**1. It is not an inversion.** The 4.6× better init also produces a better final model, for both
arms. The "better init, worse final" pattern §2a documents does not hold here — which is itself
worth knowing, because it means that inversion was never a law, just a property of the particular
inits that had been tried.

**2. A one-line change to a *statistic* buys the frozen baseline 4.5 PPL** — 79.46 → 74.94. No
manifold, no learned projection, no Stiefel optimizer, no extra step cost. **That lands the
frozen-basis baseline (74.94) on top of the originally-published LRD result (74.20–75.45.)**
Everything LRD's machinery was reported to be worth, a corrected covariance delivers for free.

**3. LRD's margin therefore falls again, to −3.26 PPL (−4.4%), p = 0.048** — 95% CI
[−6.48, −0.05], an interval that now nearly touches zero. It is still a win: every LRD seed still
beats every baseline seed. But it is a *fifth* of the effect once implied by "5.8 points, ~6σ".

## What survives, stated exactly

**LRD works, and it composes.** The best model in this study is LRD on the balanced basis —
**71.68 ± 1.51**, better than any frozen basis and better than LRD on the old one. Training the
residual-stream projection against the KD loss is a real mechanism and it adds real quality on
top of the best initialization we know how to build.

**But most of the published margin was never about `V`.** Of the original 6.61 PPL gap:

- **1.35 PPL** was a mis-scaled RMS gain in the shared init (§9b),
- **~2.0 PPL** more was a covariance dominated by its deepest layers (this section),
- **~3.3 PPL** is the learned restriction itself.

Half the headline was the baseline being weak in two ways nobody had looked at. `V`-training was
partly *buying back* a badly-conditioned initialization — which is exactly what one would expect
a flexible whole-network rotation to be good at, and exactly the confound a strong baseline is
supposed to rule out.

## The methodological point, which is the real one

The audit that produced §0's claim was rigorous *about the wrong axis*. It varied the
**selection criterion** across six settings and held the **statistic** fixed — and the statistic
was where the error was. Six careful experiments over a broken input give six careful wrong
answers, and their agreement reads as robustness.

`basis_pool="balanced"` is now the default (`LRDConfig`), and
`test_balanced_pooling_gives_every_layer_a_vote` pins the mechanism. The 1.3B and 2.7B rows in §5
have **not** been re-measured on the corrected statistic and should be assumed to overstate LRD's
margin by a similar amount.

## 11a. The head-to-head, re-run on the corrected covariance

§4a's contest — every recent method's *basis principle* rebuilt as a frozen `V` and distilled
identically — was run on the **pooled** covariance §11 shows to be wrong. So it was re-run on the
corrected one (llama-160m, 3.07×, n=3, identical budget/data/seeds):

| basis principle (representative method) | pooled covariance | **balanced covariance** |
|---|---:|---:|
| activation SVD — LASER / ASVD (`pca`) | 80.94 ± 0.91 | **74.94 ± 1.29** |
| activation + influence — AIR (`gn`) | 79.98 ± 0.61 | **74.70 ± 0.68** |
| activation-whitened SVD — SVD-LLM (`whiten`) | 81.04 ± 1.35 | **74.97 ± 1.28** |
| **LRD — `V` trained against KD** | 74.20 ± 0.38 | **71.68 ± 1.51** |

Two things, and the first is a *vindication*.

**The "no criterion beats plain PCA" finding survives — and strengthens.** On the corrected
covariance all three frozen principles collapse into a dead heat (74.70 / 74.94 / 74.97,
overlapping at one sd). AIR's influence weighting, SVD-LLM's whitening, and plain activation SVD
are **interchangeable**. §0's central claim was right about the *criterion*; it was simply
computing all of them from a broken *statistic*, which depressed every one of them by ~5-6 PPL at
once. Correcting the statistic lifts them all together and leaves the ranking unchanged.

That is the cleanest possible statement of what the last two years of SVD-compression work has
been optimizing: **the choice of criterion does not matter, and the covariance they all share
does.** A one-line change to the pooling is worth more than every criterion in the literature
put together.

**And LRD still beats all of them.** Against the best corrected frozen basis (AIR's `gn`, 74.70):

> **−3.02 PPL (−4.0%)**, 95% CI [−6.20, +0.16] (Welch, dof 2.8), t = −3.15, **p = 0.056**.
> Complete seed separation (every LRD seed beats every `gn` seed; exact one-sided p = 0.05).

Note the honesty problem in that line. Every LRD seed beats every baseline seed — but the Welch
interval now *touches zero* and p sits just the wrong side of 0.05. **At n=3 this comparison is no
longer decisive**, and §9e's own prescription applies to the audit as much as to the thing it is
auditing: the way to claim more is to raise `n`, not to re-describe three runs. See §11b.

## 11b. Raising `n`, because that is the only thing that buys more evidence

§11a left LRD's win over the best corrected frozen basis at **p = 0.056**, with a Welch interval
that touched zero — even though every seed beat every seed. §9e's prescription is unambiguous
and applies to the audit as much as to the work it audits: *the way to claim more is to raise
`n`, not to re-describe three runs.* So we ran three more seeds.

**n=6, llama-160m, 3.07×, all bases on the corrected (depth-balanced) covariance, identical
budget/data/seeds:**

| arm | final PPL (n=6) | seeds |
|---|---:|---|
| `pca` — activation SVD (LASER, ASVD) | 74.96 ± 1.13 | 74.79 / 73.74 / 76.31 / 73.98 / 76.36 / 74.60 |
| `gn` — activation+influence (**AIR**, the best frozen principle) | 75.00 ± 1.07 | 74.04 / 74.64 / 75.41 / 73.78 / 76.66 / 75.49 |
| `whiten` — activation-whitened (SVD-LLM) | 74.97 ± 1.28 (n=3) | — |
| **LRD — `V` trained against the KD loss** | **71.80 ± 1.13** | 72.72 / 69.95 / 72.37 / 70.98 / 72.82 / 71.97 |

**Against the strongest corrected frozen basis (AIR):**

> **−3.20 PPL (−4.3%)**, 95% CI **[−4.61, −1.79]** (Welch, dof 10.0), t = −5.06, **p < 0.001**.
> Complete separation — all six LRD seeds beat all six baseline seeds — exact one-sided
> permutation **p = 0.001**, the floor of a 6-vs-6 design.

Doubling `n` moved the interval off zero and the p-value by two orders of magnitude, and changed
the conclusion from "suggestive" to "settled." It cost six GPU-minutes per seed. That is the
whole lesson of §9e in one paragraph: three seeds could not distinguish this effect from noise;
six can, easily. Nothing about the effect changed — only our right to claim it.

## 11c. The final statement

**The mechanism is real and it is the smallest of the three things that were being credited to it.**
On `llama-160m` at 3.07×, decomposing the originally-published 6.61 PPL margin:

| component | worth | what it actually is |
|---|---:|---|
| mis-scaled RMS gain (§9b) | ~1.3 PPL | a bug in the shared init |
| depth-imbalanced covariance (§11) | ~4.5 PPL | a bug in the shared init |
| **the learned restriction itself** | **~3.2 PPL** | **the method** |

(The components do not sum to 6.61 because they interact — fixing the init raises both arms, and
LRD's advantage over a *well*-initialized baseline is what remains.)

Two bugs in the initialization that **both arms share** were flattering the method, because a
trained projection can absorb a bad start and a frozen basis cannot. Fix them and the frozen
baseline climbs from 80.8 to 75.0 — landing *on top of the originally-published LRD number*. What
is left, measured against a baseline that is now genuinely the strongest we know how to build, is:

> **LRD is worth −3.20 PPL (−4.3%) over the best frozen basis, n=6, p < 0.001,
> with every seed beating every seed.**

That is a smaller claim than "5.8 points, ~6σ", and it is one that will survive being checked.

## 11d. The flaw is not a 160M artifact: it holds at 1.3B

The depth-imbalance mechanism is a property of transformers, not of one small teacher. Measured
on `princeton-nlp/Sheared-LLaMA-1.3B` (d=2048, 24 layers, teacher PPL 14.28), at the same 3.64×
geometry §5 uses:

| | llama-160m (d=768) | **Sheared-LLaMA-1.3B (d=2048)** |
|---|---:|---:|
| residual energy, layer 0 → final norm | **130,711×** | **20,628×** |
| embedding energy kept, **pooled** basis | 50.9% | **54.5%** |
| embedding energy kept, **balanced** basis | 85.7% | **91.8%** |
| absorbed-init PPL, pooled → balanced | 17,529 → 3,840 | **3,503 → 1,435** |

Same shape, same magnitude: the pooled basis half-destroys the embedding on both, and reports
itself near-perfect on both (it retains >97% of the *pooled* energy, which is the statistic it
was chosen to maximize — that self-flattery is exactly what makes the failure silent).

So the pre-audit **1.3B and 2.7B rows in §5 measure LRD against a baseline handicapped in the
same way the 160M row was**, and their margins should be expected to shrink for the same reason.
§11e re-measures 1.3B directly.

### 11d-i. A memory bug the scale-up exposed

Re-measuring at 1.3B **OOM'd on a 22 GB card**, and the reason was not the method. `RestrictedLlama`
builds a full `LlamaForCausalLM` "skeleton" to supply the student's module graph — its shapes, its
RoPE buffers, its forward code. Every weight that skeleton needs is handed to it by
`functional_call` on each forward, and the two it is *never* handed (`embed_tokens`, `lm_head`)
are never called: the embedding arrives as `inputs_embeds`, and the logits are lifted through the
*teacher's* head.

So the randomly-initialized weights `LlamaForCausalLM(cfg)` allocates are **dead on arrival**. At
the 160M benchmark that is merely wasteful. At 1.3B the student carries ~400M of them — **~1.6 GB**
— which was on its own enough to push the restricted forward off the card.

Freeing them (`_hollow_skeleton`) drops the peak of a 1.3B-shaped restricted forward from OOM to
**11.9 GB**, and is invisible to the result (`test_the_skeleton_carries_no_weight_storage` pins
both the freed storage *and* that `fold()` still reproduces the module bit-for-bit).

Worth stating because of what it says about §6's scaling argument: the first time anyone actually
pushed this construction past the benchmark teacher, it did not fit — for a reason that had
nothing to do with the mathematics and everything to do with an allocation nobody had looked at.

## 11f. My own default was wrong, and only a second scale revealed it

The trust region (§9c) replaced the fitted `0.77/d` constant with a rotation budget: `v_lr` is the
RMS angle `V` turns per step. I picked its default, **0.005**, from a sweep at 160M where 0.002
and 0.005 *tied* (74.59 vs 74.66). A tie at one scale is not a choice; it is a coin flip that
looks like a choice.

At 1.3B the tie broke, hard:

| 1.3B, corrected basis | LRD (n=2) | `V`'s max principal angle |
|---|---:|---:|
| `v_lr` = 0.005 (my default) | 341.20 ± **42.29** | **1.55 rad ≈ 89°** |
| `v_lr` = 0.002 | **302.66 ± 1.75** | 0.82 rad |
| `v_lr` = 0.001 | 308.73 ± 22.08 | 0.52 rad |

At 0.005 the projection rotates almost **orthogonal to its own initialization** (π/2 = 1.571) —
that is a runaway, not a fit — and the run's seed spread blows up 24×, from ±1.75 to ±42.29. The
default I shipped would have quietly destroyed the method at 1.3B.

**0.002 is the best value at *both* scales** (at 160M it also improves the headline slightly,
71.80 → 71.25 at n=6), so it is now the default.

Two things follow, and the second corrects a claim of my own.

**The failure is diagnosable, which is the actual payoff of the trust region.** An ambient step
size gives you no way to see this coming; a *rotation* does. `max_principal_angle` heading toward
π/2 is a number anyone can look at, and it is now reported on every `LRDResult`. That — not
"tuning-free" — is what putting the step in physical units buys.

**And "needs no per-teacher constant" was too strong.** I wrote that in §9c, and it is not what
the evidence supports. The *optimal* rotation budget is still a hyperparameter, exactly like a
learning rate; 0.005 is fine at 160M and catastrophic at 1.3B. What is true, and worth having, is
that a **single value now exists that works across the scales tested**, and that the knob is a
physical quantity that transfers instead of a fitted `1/d` rule with no reason to extrapolate.
The old ambient rule had no such value: `1e-3` was right at 160M and diverged at 2.7B.

Corrected claim: *the trust region makes `V`'s step comparable across teachers and its failure
visible. It does not make it tuning-free.*

## 11g. The scale claims are underpowered — the original's and mine

§5 reports the 1.3B and 2.7B rows at **n=2 and n=3**. Re-running at 1.3B shows why no conclusion
at that scale is safe at those `n`. The `pca` baseline's seed-to-seed spread, at 1000 steps:

| 1.3B `pca` configuration | n=2 result |
|---|---:|
| pooled basis, legacy gain | 374.46 ± 15.08 |
| pooled, per-norm gain | 372.53 ± 22.76 |
| pooled, write-aware FFN | 387.16 **± 69.58**  (436.4 vs 338.0) |
| balanced basis, legacy gain | 408.71 ± 5.54 |
| balanced, per-norm + write-aware | 337.98 ± 9.02 |

**One configuration swings 100 PPL between two seeds.** At `n=2`, differences of 30–70 PPL — the
size of every effect anyone is trying to measure here — are indistinguishable from which seed you
drew. So:

- **The published 2.7B row cannot support its own claim.** It reports `pca` **635.57 ± 112.87**
  against a margin of 107 PPL: *the baseline's standard deviation exceeds the entire claimed
  effect*, at n=3.
- **My own 1.3B ablations of `per_norm_gain` and `write_aware` are equally unreadable** and I make
  no claim from them. They are reported above precisely so nobody else does either.

What *does* survive at 1.3B is the one signal that is not a small difference between noisy means:
**LRD at `v_lr = 0.002` lands at 302.66 ± 1.75 — tight, and below every baseline configuration
measured (338 to 450).** A result that sits under the whole spread of its comparison class, with a
standard deviation 5–40× smaller, does not need the comparison to be well-powered.

### The 1.3B margin, measured as well as this budget allows

Adding two more seeds to the only pair that matters — the corrected baseline against LRD at the
shipped rotation budget — gives, at 1000 steps and n=4:

| Sheared-LLaMA-1.3B, 3.64×, corrected covariance | n=4 |
|---|---:|
| `pca` (frozen, corrected basis) | 357.56 ± 45.56  (344 / 332 / 425 / 329) |
| **LRD** (`v_lr` = 0.002) | **329.23 ± 30.79**  (301 / 304 / 353 / 359) |
| difference | **−28.33 PPL (−7.9%)**, 95% CI **[−97.9, +41.3]**, t = −1.03, **p = 0.35** |

**The interval spans zero, and the seeds overlap.** LRD *trends* better at 1.3B by about 8%, and
every diagnostic says the mechanism is doing something real (the restriction gap, the principal
angle, and the tight `v_lr`=0.002 sub-runs all behave as at 160M). But **at this budget the effect
is not distinguishable from seed noise**, and no honest reading of these numbers establishes it.

So, stated exactly:

> **160M: settled.** −3.75 PPL (−5.0%) against the strongest frozen basis, n=6, 95% CI
> [−5.16, −2.34], p < 0.001, every seed beating every seed.
>
> **1.3B: not settled, in either direction.** −7.9% point estimate, p = 0.35 at n=4. The
> mechanism transfers; the *margin* is unmeasured.
>
> **2.7B: unsupported.** The published row's baseline sd (±113) exceeds its own claimed effect
> (107 PPL) at n=3. It has not been re-run.

§5's headline — "**the win grows from 1.3B to 2.7B**" — is therefore **withdrawn**. It was never
measured to a standard that could support it: at n=2–3, against seed spreads of ±68 to ±113, that
sentence describes the draw of the seeds as much as the behaviour of the method. Settling it needs
roughly an order of magnitude more seeds (or a budget long enough for the students to stop being
this far from convergence — at 1000 steps a 1.3B student sits at PPL ~350 against a teacher at
14.3, which is nowhere near where variance settles down).

Which is, once again, §9e: **the way to claim more is to raise `n`.** At 160M I did, and the claim
firmed up. At 1.3B I did as far as the compute allowed, and it did not — so the claim does not get
made.
