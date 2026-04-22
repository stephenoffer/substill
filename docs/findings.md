# Benchmark Matrix Findings

Four findings from the 23-cell Apr-22 experimental matrix, ranked by
evidence strength. Each includes (a) the data motivating it, (b) the
mathematical interpretation, and (c) what the v3 recipe does about it.

For the full results tables behind each number, see [results.md](results.md).
For the code changes that follow from these findings, see
[v3-improvements.md](v3-improvements.md).

---

## Finding 1 — Logit-KD and subspace-MSE destructively interfere

**Strength: 🔴 Very high. Two independent τ thresholds, consistent magnitude.**

### Data

Ablation sweeps on ResNet50 / CIFAR-10, 15 epochs each:

| variant | τ=0.85 acc | τ=0.70 acc |
|---|---|---|
| task_only | 87.87 | 85.96 |
| classical_kd (task + KD) | 88.89 (+1.02 vs task_only) | 86.67 (+0.71) |
| no_logit_kd (task + subspace + sparsity) | **90.94** (+3.07) | **89.07** (+3.11) |
| full (task + subspace + sparsity + KD) | 89.82 (+1.95) | 88.07 (+2.11) |

### What this says

- KD *alone* helps by ~1pp (classical_kd vs task_only).
- Subspace+sparsity alone helps by ~3pp (no_logit_kd vs task_only).
- Adding KD on top of subspace+sparsity *removes* 1pp (full vs no_logit_kd).

The effect is not a dose-response or a weight-tuning artifact — both
thresholds show the same ~1pp destructive interaction.

### Mathematical interpretation

The two objectives pull the student in incompatible directions:

- **`L_logit_kd`** pulls `student_logits` (produced by the student's own
  classifier from its raw feature maps) toward `teacher_logits` (produced
  by the teacher's classifier from its high-rank feature maps).
- **`L_subspace`** pulls student feature maps (through the learned 1×1
  projector `P_s`) into the teacher's low-rank eigenbasis `V`.

At high compression, the student cannot both (i) produce logits whose
softmax at temperature T matches teacher's, and (ii) live entirely in a
very-low-rank subspace of its own channel space. The gradients of the two
objectives are not orthogonal, and with δ=1.0, β=0.5 the net is negative.

### v3 response

Set δ=0. Zero algorithm change — just a config override.

If KD must be kept for paper-comparability reasons, **PCGrad gradient
surgery** (Yu et al. 2020) removes the conflicting component per-batch:
`g_kd ← g_kd − ⟨g_kd, ĝ_sub⟩₊ · ĝ_sub`. Not implemented in v3; flagged as
v4 work in [open-questions.md](open-questions.md).

---

## Finding 2 — Rank selection collapses on heavy-tailed spectra

**Strength: 🔴 Very high. Single-dataset but catastrophic effect.**

### Data

GPT-2 / WikiText-2, variance-threshold effective rank:

| τ | per-layer ranks | max | min | student ppl |
|---|---|---|---|---|
| 0.95 | [372, 277, 5, 13, 26, 52, 93, 156, 224, 309, 386, 497] | 497 | 5 | 486 |
| 0.90 | [262, 163, 1, 1, 1, 1, 3, 19, 56, 122, 199, 338] | 338 | 1 | **779** |

Six consecutive middle layers collapse to rank ≤ 3 at τ=0.90. Student
perplexity jumps from 486 to 779 (teacher is 44).

### Mathematical interpretation

Variance-threshold rank: smallest k with `Σ_{i≤k} λ_i / Σλ_i ≥ τ`.
GPT-2's residual stream has 1–2 dominant "attention-sink" eigenvalues
carrying >90% of raw L2 variance but negligible task-relevant information.
At τ=0.90 those top 1–2 PCs satisfy the threshold and rank collapses to
1–3. The middle-block useful signal lives in directions 10–500.

### v3 response — nuanced

Tested two alternatives:

- **Participation rank** `(Σλ)² / Σλ²` — threshold-free, robust to heavy
  tails. Results: per-block ranks `[22, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 10]`
  → student n_embd=24, 94× compressed, ppl 5430. Fixes rank-1 collapse
  but *overshoots*: produces uniformly small ranks because the tail signal
  is smooth rather than concentrated in a few useful extra PCs.
- **Mahalanobis loss weighting, same variance rank** — Finding 3 below.
  Student architecture unchanged; only loss weighting changes. ppl 486 → 131.

The rank *statistic* was never the real problem for the LLM; the loss
*weighting* was. The v3 recipe for transformers keeps variance rank but
switches sv_weighting to Mahalanobis.

For the deeper fix — task-gradient covariance (supervised subspace) —
see [open-questions.md](open-questions.md).

---

## Finding 3 — Mahalanobis weighting is regime-dependent

**Strength: 🔴 High on LLM (3.7× ppl improvement). 🔴 High negative on CNN
(−1.6pp at τ=0.70/0.85, −2.7pp at τ=0.95). Clean regime split.**

### Data — LLM (GPT-2 / WikiText-2 @ τ=0.95)

Identical student architecture (n_embd=504, 62.5M params, 1.99× compression).
Only the `sv_weighting` in the subspace MSE changed:

| sv_weighting | ppl (lower better) |
|---|---|
| sqrt (old default) | 486.0 |
| **mahalanobis** | **130.7** |

### Data — CNN (ResNet50 / CIFAR-10)

Clean A/B at τ=0.85 — same student architecture, same epochs, both drop
KD + sparsity; only `sv_weighting` differs:

| cell | sv_weighting | acc @ τ=0.85, 18ep |
|---|---|---|
| v3 | **mahalanobis** | 89.31 |
| v3_no_kd_no_spar | sqrt | **91.87** (Δ = **−2.56pp** caused by Mahalanobis) |

Pre-ablation view (v3 vs no_logit_kd from prior matrix at 15 ep):

| τ | v3 acc | no_logit_kd acc (sqrt) | Δ |
|---|---|---|---|
| 0.70 | 87.47 | 89.07 | −1.60 |
| 0.85 | 89.38 | 90.94 | −1.56 |
| 0.95 | 89.72 | ~92 (from v2 + seeds) | −2.4 to −2.7 |

### Mathematical interpretation

`sv_weighting=mahalanobis` weights each PC coordinate by `1/λ_i`. In the
ideal case, this is the natural squared distance on a Gaussian subspace:
```
L = Σ_i (x̃_i − ỹ_i)² / λ_i  =  d²_Mahalanobis(x̃, ỹ)
```

**Why it works on the LLM**: GPT-2 residual streams are heavy-tailed —
1–2 "attention-sink" directions carry >90% of L2 variance but negligible
task signal. With `w_i ∝ √λ_i` (sqrt), the loss over-weights those sinks
and the student learns them hardest. Mahalanobis equalizes contributions
across all directions in std-units, keeping the tail (useful) directions
in the loss.

**Why it regresses on CNN — numerical pathology**: at τ=0.95 the CNN's
4th stage has effective_rank ~696. Mahalanobis weight for each of those
696 directions is `1/λ_i`, with λ_i spanning ~6+ orders of magnitude
(λ_max down to the 1e-6·λ_max noise floor we clamp at). The smallest-λ
directions get weight ~10⁶/λ_max; the top PC gets weight ~1/λ_max. After
normalizing by the mean across all 696 directions, the mean is dominated
by the large weights at the bottom of the spectrum, so the top PC ends up
with weight ≈ 0 and the near-floor directions get weight ≈ 1. Net effect:
the student is trained to match noise instead of signal.

The regression grows with effective_rank: −1.6pp at τ=0.70 (stage widths
[24, 88, 232, 392]), −1.6pp at τ=0.85 ([56, 192, 464, 904]), −2.7pp at
τ=0.95 ([128, 336, 744, 1504]). Proportional to the number of near-floor
directions pulled into the loss by the higher threshold.

### Why the LLM doesn't hit this pathology

On GPT-2 at τ=0.95, per-block ranks are
`[372, 277, 5, 13, 26, 52, 93, 156, 224, 309, 385, 494]`. Eight of the 12
blocks have k ≤ 156 — only the first, last, and two adjacent blocks are
near 500. With fewer kept directions, fewer of them fall near the noise
floor, so Mahalanobis weighting behaves closer to the textbook ideal.

### Subtler CNN attribution — Mahalanobis alone is neutral; the regression
comes from the *interaction* with drop-KD

The v3 component ablation at τ=0.85 (asd_v3_ablation_t85) gives:

| variant | uses KD? | weighting | acc | Δ from previous |
|---|---|---|---:|---:|
| full | yes | sqrt | 89.82 | — |
| v3_mahalanobis_only | yes | mahalanobis | 89.77 | −0.05 (noise) |
| no_logit_kd (prior) | no | sqrt | 90.94 | +1.12 |
| v3 | no | mahalanobis | 89.31 | −1.63 |

Mahalanobis alone (with KD) = essentially full. Drop-KD alone (with sqrt)
= +1.12pp win. But Mahalanobis + drop-KD together = −1.63pp regression.

Diagnosis: when KD is present, it provides a label-level supervisory
signal that compensates for the Mahalanobis subspace loss's failure modes.
When KD is dropped, the subspace loss is the dominant distillation signal,
and the Mahalanobis pathology becomes the bottleneck.

### v3 response — refined

Mahalanobis is **not** a universal swap. The v3 recipe for transformers
uses it cleanly; the v3 recipe for CNN has two paths, both better than
the full v3 combination:

1. **Keep KD, keep sqrt** (i.e., `full`, prior baseline) — 89.82 @ τ=0.85.
2. **Drop KD, keep sqrt** (i.e., `no_logit_kd`) — **90.94 @ τ=0.85** ← best.

For the v4 recipe, either:

1. **Stricter rank cutoff inside the loss**: truncate at
   `k_effective = argmin(λ_i < 1e-3 · λ_max)` instead of 1e-6. Removes
   the noise tail from the weighted sum.
2. **Use `inv_sqrt` instead** (`w_i ∝ 1/√λ_i`): half-whitening. Gentler
   than full Mahalanobis; less prone to over-weighting the tail. Already
   implemented in `_sv_weights`; untested.
3. **Spectrum-adaptive**: blend sqrt and Mahalanobis keyed on the
   participation ratio. Proposed in
   [open-questions.md](open-questions.md) §1.

---

## Finding 4 — Sparsity loss is neutral alone, positive combined with drop-KD

**Strength: 🟡 Medium alone; 🔴 high in combination at same setup.**

### Data

In isolation (Apr-22 ablations, 15 ep):

| variant | τ=0.85 | τ=0.70 |
|---|---|---|
| full | 89.82 | 88.07 |
| no_sparsity | 89.59 (Δ = −0.23pp) | 87.95 (Δ = −0.12pp) |

Multi-seed stddev at τ=0.85: 0.16pp over 3 seeds. Alone, the sparsity
term is statistically indistinguishable from zero contribution.

Combined with drop-KD at 18 ep (v3 validation):

| variant | τ=0.85 acc | epochs |
|---|---:|---:|
| full | 89.82 | 15 |
| no_logit_kd (drop KD only, sparsity ON) | 90.94 | 15 |
| v3_no_kd_no_spar (drop KD AND sparsity) | **91.87** | 18 |

The extra +0.93pp from no_logit_kd → v3_no_kd_no_spar mixes (a) +3 epochs
of training and (b) the sparsity drop. A clean A/B would run
`v3_no_kd_no_spar` at 15 epochs. Given this confound, the most defensible
claim is: **dropping sparsity is net-neutral-to-positive**, and combined
with drop-KD produces the best-ever ASD result on CIFAR-10 τ=0.85.

### v3 response

γ=0. Simplifies the loss to 4 components (task + subspace + RKD opt +
AT opt). At worst neutral; at best contributes to the +2.05pp v3 delta
over `full` at τ=0.85.

---

## Non-finding — gap_cov failure is a sizing issue

Worth recording because it's easy to misread. `gap_cov` ablation:

| variant | stage widths (τ=0.85) | student params | acc |
|---|---|---|---|
| full (per_pixel covariance) | [56, 192, 464, 904] | 2,684,354 | 89.82 |
| gap_cov (GAP covariance) | [16, 24, 168, 400] | 455,218 | 84.23 (Δ = −5.59pp) |

`gap_cov` builds a ~6× smaller student because GAP covariance measures
variance of spatial *means*, and many channels look collapsed after
pooling → rank-threshold selects much smaller k. The 5.59pp drop is
dominated by the smaller student, not a weaker loss.

This *does* confirm that per-pixel covariance is essential for correct
rank estimation, but it doesn't say anything new about the subspace
training loss itself.
