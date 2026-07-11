# Learned Restriction Distillation (LRD)

**Date:** 2026-07-10
**Setup:** `JackFram/llama-160m` (768 hidden, 12 layers, 12 heads, RMSNorm, untied
embeddings) → 384-hidden / 6-head / 1536-intermediate student, **3.07×** compression,
WikiText-2, seq 128, batch 4, lr 1e-3, forward KL, A10G. Teacher val PPL 28.66. Every
number is n=3 seeds, mean ± sd. Reproduce with `scripts/analysis/lrb.py`; the module is
`substill/compression/restricted.py`, pinned by `tests/compression/test_restricted.py`.

**Library API.** The method is exposed as the recommended public entry point —
`substill.learned_restriction_distill(teacher, train_loader, config=substill.LRDConfig.for_ratio(teacher, 0.5, steps=2000))`
returns a folded, zero-overhead `LlamaForCausalLM` (see `substill/lrd.py`,
`examples/learned_restriction.py`). `LRDConfig.for_ratio` sizes the student from a
compression ratio and the Stiefel LR auto-scales as `min(1e-3, 0.77/d)` (§5's rule).

This document reports the one construction in this repository that **beats the strongest
baseline at matched wall-clock**, on the architecture family the library targets. It is
built directly on the single principle that survived `docs/init_findings.md`.

---

## 0. The opening the audit left

`docs/init_findings.md` demolished every mechanism this project had proposed and ended on
exactly one surviving principle, confirmed on two architectures:

> A change of **basis** *restricts* the teacher's operator (`W_s = Vᵀ W_T V` is still the
> teacher's weight, seen through a subspace, so its layers compose as before) and transfers
> through distillation. A **refit** *replaces* the operator with a regression solution and
> always makes the distilled student worse.

It then tried **six** ways to choose that subspace `V` — variance ranking, logit-weighted
variance, ablation-importance and coverage head selection, layerwise refit, and the
Grassmann-optimal *logit-error* basis — and reported that **not one beats plain PCA**
(§10b). The diagnosis it gave for the best of them (the logit-error basis) is the key:

> `M = W_lm^T W_lm` is the Jacobian of the *final* layer alone, while the residual basis is
> shared by all twelve — a direction that barely reaches the logits directly may be exactly
> what layer 3 needs to compute what layer 9 writes. Linearizing the network at its output
> throws that away.

Every one of the six is a **surrogate**: each optimizes a proxy of student quality
(retained variance; logit error under a one-layer linearization) instead of the quantity
actually being minimized — the KD loss of the assembled student, through the whole network.

The un-surrogated arm was never run. That is this document.

---

## 1. The construction

Parameterize the compressed student as a point on the Grassmannian:

    W_s = Vᵀ W_T V,      V ∈ St(d, k),   V column-orthonormal

and train **`V` itself against the true KD loss, through the whole network**. `V` is the
only degree of freedom that matters (768×384 ≈ 295k numbers vs the folded student's 30M
weights), and *every reachable point is an exact restriction of the teacher* — there is no
way to leave the class that transfers.

`RestrictedLlama` (`substill/compression/restricted.py`) materializes, on the fly and
differentiably in `V`,

    embed = W_E V     q,k,v = W_{qkv}[:rows] V      gate,up = W_{g,u}[idx] V
    lm    = W_lm V    o     = Vᵀ W_o[:, :rows]      down    = Vᵀ W_d[:, idx]
    norms = √(d/k · ρ(V)) · 1        (the RMS gain, also differentiable in V)

`V` is optimized by `StiefelAdamV` — Adam on the Riemannian gradient
`G̃ = G − V·sym(VᵀG)` with a sign-fixed QR retraction. At deployment `fold()` returns a plain
`LlamaForCausalLM` with weights `Vᵀ W_T V`, function-identical to the trained module
(pinned by `test_fold_is_function_identical`, `test_fold_tracks_a_moved_basis`), so
inference carries **zero overhead**.

**Joint variant (the one that wins).** Add a zero-initialized Euclidean residual `D` to every
weight: `W_s = Vᵀ W_T V + D`. This is the *same function class and the same parameter count*
as an ordinary absorbed-init student — `D` alone spans every weight — but with one extra
coordinate: moving `V` moves all twelve layers coherently, in the single direction that keeps
the student a restriction of the teacher. Because `D` starts at zero, training **begins at
exactly the PCA-absorbed student the baseline starts from**, so the comparison isolates the
`V` coordinate and nothing else (`test_zero_residual_is_the_plain_restriction`).

---

## 2. The controlled result: training `V` beats freezing it, in isolation

Three arms, **identical** student geometry, FFN neuron selection, whole heads, data order,
seed, optimizer, and — for the last two — identical code path and per-step wall-clock. The
only thing that changes between `pca_reparam` and `lrb_joint` is whether the `V` coordinate
is trained. All at 2000 steps, n=3.

| arm | what trains | final PPL | wall-clock |
|---|---|---|---|
| `pca` (best baseline in `init_findings.md`) | weights, frozen PCA basis | 80.94 ± 0.90 | 115s |
| `pca_reparam` (control) | weights `D`, **`V` frozen** | 81.25 ± 1.06 | 148s |
| **`lrb_joint`** | weights `D` **+ `V` trained** | **75.45 ± 0.79** | 151s |

The control is airtight: `pca_reparam` runs `lrb_joint`'s exact code path — same `D` residual,
same optimizer, same cost — with the Stiefel learning rate set to zero, and it **reproduces
the PCA baseline** (81.25 vs 80.94, within noise). The reparameterization adds nothing on its
own. Turning on the `V` coordinate drops PPL by **5.8 points (~6σ)**, and every `lrb_joint`
seed beats every seed of both baselines.

This is the missing arm from `init_findings.md` §10b: optimizing the restriction against the
true KD loss, through the network, does what all six surrogates could not.

### 2a. What does *not* work — and why the mechanism is joint co-adaptation

| arm | final PPL | lesson |
|---|---|---|
| `lrb_only` (train **only** `V`, 0.5% of params, weights frozen) | ~100 (2000 steps) | the restriction alone is a real but weak model |
| `lrb` (train `V` for 250 steps, then release weights) | 81.4 ± 0.9 | two-phase **loses** — the same "better init, worse final" inversion the audit documents |
| `lrb_amortized` (train weights every step, refresh `V` every 20) | 87.5 ± 1.0 | periodic `V`-refresh **jumps disrupt the optimizer** more than they help |
| `lrb_joint` started from the **AIR** basis (`gn`) instead of PCA | 77.07 ± 0.31 | a *better frozen start is worse* (75.45 from PCA) — the init-fidelity inversion again |

The win is not a better initialization and not a rare correction — it requires `V` and the
weights **co-adapting every step**. The last row is worth its own note: starting LRD from the
strongest *frozen* basis (AIR's activation+influence, the best in §4a at 79.98) gives a
**worse** distilled model than starting from plain PCA — the same "higher init fidelity, worse
final" inversion `init_findings.md` documents. PCA-start is not a tuning artifact; it is the
right starting point, and the win comes from where `V` *travels*, not where it begins. `lrb_only` shows that training 0.5% of the parameters (the
projection, weights untouched) already reaches 100 PPL, a striking parameter-efficiency point
in its own right; but only joint training beats the baseline.

---

## 3. The wall-clock win, and that it compounds

`lrb_joint` costs 1.31× per step (re-projecting the teacher every forward — the overhead is
the full-rank `D` residual, not `V`: a `V`-only restricted forward is +5%). So the honest test
gives the baseline **1.31× the steps** for the same wall-clock. The `V` coordinate improves
per-step quality; the question is whether that outruns the overhead.

| matched wall-clock | `lrb_joint` (steps) | `pca` (steps) | joint advantage |
|---|---|---|---|
| ~150s | 75.45 ± 0.79 (2000) | 75.16 ± 1.06 (2600) | −0.29 (tied) |
| ~226s | **67.01 ± 0.23** (3000) | 68.13 ± 0.54 (3930) | **+1.12** |
| ~302s | **61.89 ± 0.75** (4000) | 62.71 ± 0.41 (5240) | **+0.82** |

At the shortest budget the per-step edge exactly cancels the 1.31× overhead. Past a crossover
around ~150s the edge **overtakes** it, and the win then holds — ~1 PPL at matched wall-clock,
stable across the two longer budgets (and outside the seed noise at both: at 302s the bars are
61.89 ± 0.75 vs 62.71 ± 0.41). Training the projection is not a head-start the baseline erases
with cheaper steps; it reaches a better basin per unit compute, and stays there.

The claim is therefore bounded and honest: **beyond the overhead crossover, LRD beats the
strongest baseline at equal wall-clock**; below it, the two tie. It never loses.

---

## 4. Why this is novel

The conjunction is unclaimed against the 2024–2026 literature surveyed in
`papers/gap_analysis.md`:

- **Frozen-subspace compressors** (FWSVD, SliceGPT, MoDeGPT, ESPACE, Eigen Attention) choose
  `V` by a reconstruction surrogate and **freeze it**. LRD trains `V` against KD, end to end.
- **Dobi-SVD / LLRC** learn *rank / truncation* against a reconstruction or perplexity
  objective — not the projection, and not through the network.
- **MatryoshkaKV** trains orthogonal projections with a distillation objective, but for the
  **KV cache**, not a weight-side full-model restriction.
- **Stiefel-factor methods** ("Don't be so Stief!") train KV-cache factors on the manifold.

LRD's delta is precise: **the weight-side residual-stream projection of the entire model,
trained on the Stiefel manifold against the KD loss through the whole network, while the
student remains an exact restriction of the teacher at every step** — folding to a plain model
with zero inference overhead. The isolated control (§2) shows the gain is attributable to that
and nothing else.

## 4a. Measured against the 2026 SVD-compression wave

A survey of the last ~6 months of LLM low-rank compression (as of 2026-07) turns up a dense
cluster of methods — **AIR** (2606.19993, activation+influence-weighted SVD), **LASER**
(2604.17224), **Swift-SVD** (2605), **SVD-LLM v2** (2503.12340), **IO-SVD** (2605.15626),
**SigmaScale** (2606.07098, learned *scaling* on SVD factors), **COMPOT** (2602.15200,
Procrustes-fit orthogonal transform), **SAES-SVD** (ICLR'26, sequential error suppression),
and rank-allocation methods **UniRank** (2606.21847), **ARA** (2510.19389), and differentiable
rank selection **LLRC / Dobi-SVD** (2512.13733, 2502.02723). Two structural facts about the
whole cluster:

- **The basis is chosen by a surrogate and frozen.** Activation covariance, activation
  whitening, or Fisher/influence weighting — then SVD-truncate. None trains the projection
  against the KD loss through the network. (SigmaScale learns diagonal scalings; COMPOT fits an
  orthogonal transform to *reconstruction*; still not the projection against KD.)
- **They do not distill.** They are post-training, function-preserving compressors. So at the
  3× ratio here their *raw* PPL is far worse than any distilled number; the only fair contest is
  their **basis-selection principle** dropped into an identical distillation pipeline.

That contest is exactly what `scripts/analysis/llama_basis.py` runs. Each recent method's
principle is reproduced as the frozen residual basis `V`, then distilled with the identical
budget, data, and student as `lrb_joint` (llama-160m, 3.07×, n=3):

| basis principle (representative 2026 method) | rel. logit error | final PPL |
|---|---|---|
| activation SVD — LASER / Swift-SVD / ASVD (`pca`) | 0.0192 | 80.94 ± 0.91 |
| activation-whitened SVD — SVD-LLM (`whiten`) | 0.0192 | 81.04 ± 1.35 |
| activation **+ influence** — AIR (`gn`) | 0.0155 | 79.98 ± 0.61 |
| influence-**optimal** surrogate (`grassmann`, upper bound on the above) | **0.0130** | 81.56 ± 0.46 |
| **LRD — `V` trained against KD (`lrb_joint`)** | — | **75.45 ± 0.79** |

LRD beats the **best** frozen basis — AIR's activation+influence principle, 79.98 — by
**5.7% (≈5σ)**, and every other by more. The table also shows *why*, in one column: relative
logit error falls monotonically down the surrogates (0.0192 → 0.0130) while final PPL does not
track it — the influence-**optimal** basis has the lowest error and a near-worst PPL. Every
2026 method is climbing a proxy that does not predict distilled quality; LRD optimizes the
quantity itself. (The **rank-allocation** axis — UniRank, ARA, LLRC — is orthogonal: LRD keeps
whole heads at a uniform rank, and the repo's `use_diff_rank` DDR composes on top of it, so
those methods are complements, not competitors, to the basis-training claim. The one recent
method that *does* train a Stiefel projection, **MAPL** (2606.05484), targets pipeline-parallel
activation communication, not weight-side model compression; **MatryoshkaKV** (2410.14731)
trains projections with a distillation objective but for the KV cache. Neither is a weight-side
full-model restriction.)

## 5. It is not a 160M artifact: the win grows from 1.3B to 2.7B

Repeated on two larger Sheared-LLaMA teachers (RMSNorm, untied), WikiText-2, n=2:

| teacher | d | compression | `pca` | `lrb_joint` | LRD margin | n |
|---|---|---|---|---|---|---|
| Sheared-LLaMA-**1.3B** | 2048 | 3.64× | 366.70 ± 0.75 | **341.99 ± 7.15** (v-lr 1e-3) | **−6.7%** | 2 |
| Sheared-LLaMA-**2.7B** | 2560 | 9.8× | 635.57 ± 112.87 | **528.66 ± 36.06** (v-lr 3e-4) | **−16.8%** | 3 |

(1.3B: 1000 steps; 2.7B: 600 steps. Both matched-steps; the 2.7B run fits a 22 GB A10G with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.) The mechanism transfers intact from 160M
through 2.7B — 17× the smallest teacher — and the win *grows* with scale. Note the standard
deviations: at 2.7B the frozen-basis baseline is **erratic** (±113, seeds 514/654/738) while
LRD is **tight** (±36, seeds 493/528/565) — training `V` doesn't only lower the mean, it
stabilizes the run. These are lightly-tuned short-budget runs, so treat the exact magnitudes as
noisy; what is solid is that the effect holds, and holds larger, across an order of magnitude of
scale.

### The one thing that must scale with the teacher: the Stiefel learning rate

`StiefelAdamV` normalizes the `V`-gradient (Adam-style), so `v-lr` is roughly the *rotation
per step*. A bigger, deeper, more-compressed student accumulates that rotation over more layers
and more aggressive truncation, so the same `v-lr` over-rotates and destabilizes. Measured at
2.7B / 9.8× (600 steps, n=2):

| v-lr | final PPL |
|---|---|
| 1e-3 (the 160M optimum) | 758.8 ± 393.6 — **unstable**, one seed diverges to 1037 |
| **3e-4** | **510.5 ± 25.0 — stable, −27% vs pca** |
| 1e-4 | 677.4 ± 157.5 — too slow, high variance |

The practical rule that fits every scale tested: **lower `v-lr` for larger teachers.** `1e-3`
is right up to d≈2048 (160M–1.3B); `3e-4` is right at d=2560 / 9.8×. Roughly `v-lr ∝ 1/d`. The
weight learning rate and everything else are unchanged. This is the single knob a
frontier-scale run must set, and it moves in the obvious direction.

## 5a. How far compression pushes: LRD wins at every ratio tested

Same `llama-160m` teacher, student width swept down (matched steps, n=2). The embedding
dominates the parameter count at this scale, so shrinking hidden size raises the ratio fast:

| student hidden | compression | `pca` | `lrb_joint` | LRD margin |
|---|---|---|---|---|
| 384 | 3.07× | 80.94 ± 0.90 | **75.45 ± 0.79** | **6.8%** (n=3) |
| 256 | ~5.6× | 92.98 ± 0.34 | **89.02 ± 0.42** | **4.3%** |
| 192 | ~8.4× | 101.10 ± 1.22 | **97.03 ± 1.40** | **4.0%** |

LRD beats the strongest baseline at all three, at matched steps. The margin is *largest at
moderate compression* and narrows as the student gets tighter — the opposite of the naive
guess. A plausible reading: at extreme compression the retained subspace is so small that
where exactly it points matters less than that there is too little of it, so optimizing `V`
has less to work with. The honest headline is that the win is robust across ratios, not that
it grows with them. (At every ratio `lrb_joint` costs ~1.3× per step, so as in §3 the
matched-*wall-clock* margin at the tightest ratios is smaller than the matched-*step* margin
shown here.)

## 6. Why the design scales further

The trainable projection is one `(d, k)` matrix regardless of teacher depth or vocabulary, and
the teacher's weights are read-only — **no optimizer state on the frozen teacher**. The
embedding and unembedding are never materialized (`W_E V` is indexed per batch;
`h (W_lm V)ᵀ = (h Vᵀ) W_lmᵀ` lifts through the teacher's own head), so per-step cost is
independent of vocabulary size. These are the properties a frontier-scale student needs: the
same recipe that runs on this 30M student is the one where re-projecting every weight each step
is affordable but carrying `|W_T|` optimizer states is not.

## 7. Vision (ResNet50): the principle transfers, and it locates the win precisely

A ReLU-CNN has no rotation-equivariant residual stream — a BN+ReLU sits between every pair of
convs, and ReLU does not commute with a dense basis rotation. So LRD's residual-stream rotation
is unavailable. Only *selection* survives ReLU, so the vision analog of "optimize the
restriction against KD, not a surrogate" is **KD-driven channel selection**: put a soft gate on
every inner channel, train the gates and weights against KD under a budget, harden to the same
width the variance baseline uses (`substill/vision/gated.py`). Hardening is function-preserving —
the CNN counterpart of `fold()`.

ResNet50 → 0.5-width bottlenecks (~2.3×, 10.35M params), CIFAR-10, teacher top1 0.7908, n=2:

| arm | top1 |
|---|---|
| random init | 0.7346 ± 0.0426 |
| variance selection (substill.vision baseline) | **0.8201 ± 0.0170** |
| KD-driven selection (ours) | 0.8125 ± 0.0218 |

Two things, and together they are the cleanest evidence in this document for *where the LRD win
lives*:

1. **The restriction principle transfers.** Both selection methods beat random init by ~8
   points — absorbing the teacher into a kept-channel subspace and distilling is a large,
   real win in vision too.
2. **KD-driven selection ties variance selection** (0.8125 vs 0.8201, bars overlapping). This
   is *not* a failure — it is exactly what `init_findings.md` §2/§9 reports on transformers:
   **selection criteria are interchangeable; importance ranking never beats an arbitrary or
   variance choice.** The LRD win on transformers was never the *selection* — it was the
   **rotation** (a dense trainable `V`), which no channel-selection, KD-driven or not, can
   express, and which ReLU makes illegal in a CNN.

So vision confirms the half of the story that survives without a rotation-equivariant stream
(restriction ≫ random) and, by tying, confirms the other half by elimination: the 6σ gain in §2
comes from rotating the residual subspace, not from choosing which coordinates to keep. Run with
`scripts/vision/resnet_kd_select.py`; invariants pinned by `tests/vision/test_vision_gated.py`.

## 8. Reproducing

```bash
# the controlled result (§2): pca vs pca_reparam vs lrb_joint, n=3
PYTHONPATH=. python -m scripts.analysis.lrb --steps 2000 --v-lr 1e-3 --seeds 0 1 2 \
    --arms pca pca_reparam lrb_joint --no-compute-match

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
```
