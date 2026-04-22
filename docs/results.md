# Full Results — Apr-22 Matrix + v3 Validation

Comprehensive tables of every cell's headline numbers. For interpretation
see [findings.md](findings.md). For the v3 recipe see
[v3-improvements.md](v3-improvements.md).

Raw JSON under `/mnt/shared_storage/asd/outputs_jobs/<cell>/`.

## Main matrix + add-ons — ResNet50 / CIFAR-10 (teacher 96.08%)

### Pareto sweep (v2 + dense_sweep, 20–25 epochs)

Each row is a single seed at that τ. FLOPs from `flops_report_v2`:

| τ | student params | FLOPs | acc | param×  | FLOPs× |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 182,130 | 10.3M | 86.97 | 129.14 | 126.53 |
| 0.70 | 557,802 | 24.7M | 89.61 | 42.17 | 52.91 |
| 0.75 | 952,650 | 39.1M | 90.29 | 24.69 | 33.38 |
| 0.80 | 1,619,330 | 66.7M | 90.31 | 14.53 | 19.57 |
| 0.85 | 2,684,354 | 108.1M | 91.23 | 8.76 | 12.07 |
| 0.90 | 4,374,386 | 179.1M | 91.63 | 5.38 | 7.28 |
| 0.95 | 7,330,818 | 320.1M | 92.39 | 3.21 | 4.08 |
| 0.98 | 10,249,058 | 470.0M | 92.12 | 2.29 | 2.78 |
| 0.99 | 11,684,706 | 560.7M | 92.73 | 2.01 | 2.33 |

### Multi-seed stability

Three seeds per threshold at 18–20 epochs:

| τ | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| 0.70 | 88.73 | 88.77 | 88.60 | 88.70 ± 0.07 |
| 0.85 | 90.76 | 90.60 | 90.98 | 90.78 ± 0.16 |
| 0.95 | 92.05 | 92.19 | 92.14 | 92.13 ± 0.06 |

### Baselines at equal compression (Hinton KD, AT, RKD)

All on ResNet50 / CIFAR-10 with same student architecture per τ, 20 epochs:

| τ = 0.85 | acc | Δ vs ASD-full |
|---|---|---|
| AT baseline | **91.40** | +1.58 |
| ASD v2 (single) | 91.23 | +1.41 |
| ASD 3-seed avg | 90.78 | +0.96 |
| RKD baseline | 89.91 | +0.09 |
| ASD full (ablation) | 89.82 | 0.00 |

| τ = 0.95 | acc | Δ vs ASD-full |
|---|---|---|
| AT baseline | 92.40 | +2.47 |
| ASD v2 (single) | 92.39 | +2.46 |
| ASD 3-seed avg | 92.13 | +2.20 |
| RKD baseline | 90.85 | +0.92 |
| ASD full (equiv, from v2 best) | ~89.93 | 0.00 |

### Ablation sweep — 7 variants × 2 thresholds, 15 epochs

Full order-of-contribution picture (same ResNet50/CIFAR-10 setup):

| variant | τ=0.85 acc | τ=0.70 acc |
|---|---:|---:|
| full | 89.82 | 88.07 |
| **no_logit_kd** | **90.94** | **89.07** |
| no_sparsity | 89.59 | 87.95 |
| gap_subspace | 88.86 | 85.63 |
| gap_cov | 84.23 | 79.25 |
| classical_kd | 88.89 | 86.67 |
| task_only | 87.87 | 85.96 |

Best at each τ: `no_logit_kd`. Difference `full − no_logit_kd` quantifies
the destructive interference finding.

## Main matrix — cross-architecture / cross-dataset

| model / dataset | teacher | τ | student params | acc | drop (pp) |
|---|---|---|---|---|---|
| resnet101 / cifar10 | 95.79 | 0.85 | 2,590,562 | 90.79 | 5.00 |
| resnet101 / cifar10 | 95.79 | 0.95 | 7,368,954 | 91.56 | 4.23 |
| resnet50 / cifar100 | 82.65 | 0.85 | 2,842,172 | 71.63 | 11.02 |
| resnet50 / cifar100 | 82.65 | 0.95 | 7,601,100 | 74.58 | 8.07 |
| resnet34 / cifar100 | 79.75 | 0.85 | 223,132 | 58.32 | 21.43 |
| resnet34 / cifar100 | 79.75 | 0.95 | 553,284 | 63.59 | 16.16 |

## LLM — GPT-2 small / WikiText-2 (teacher ppl 44.45)

| cell | rank def | sv-weighting | τ | n_embd | student params | compression | ppl |
|---|---|---|---|---:|---:|---:|---:|
| baseline | variance | sqrt | 0.95 | 504 | 62.5M | 1.99× | 486.0 |
| baseline | variance | sqrt | 0.90 | 348 | 35.3M | 3.52× | 779.4 |
| **mahalanobis** | variance | **mahalanobis** | 0.95 | 504 | 62.5M | 1.99× | **130.7** |
| participation | participation | sqrt | 0.90 | 24 | 1.3M | 94.45× | 5431 |
| participation | participation | sqrt | 0.95 | 24 | 1.3M | 94.45× | 6275 |

Headline: Mahalanobis at τ=0.95 gives a 3.7× perplexity improvement over
the sqrt-weighted baseline at identical student architecture.

## ImageNet (ResNet50, 5 epochs, 100-per-class subset)

| τ | student params | acc | note |
|---|---|---|---|
| 0.95 | 10.8M | 12.9% | Under-trained. Treat as sanity-check only. |

## v3 validation — ResNet50 / CIFAR-10, 18 epochs

Live results; will be filled as cells land.

| cell | variant | τ | student params | acc | Δ vs full @ same τ | Δ vs no_logit_kd |
|---|---|---:|---:|---:|---:|---:|
| asd_v3_t70 | v3 (drop KD + drop spar + Mahalanobis) | 0.70 | 557,802 | 87.47 | −0.60 | −1.60 |
| asd_v3_t85 | v3 | 0.85 | 2,684,354 | 89.38 | −0.44 | −1.56 |
| asd_v3_t95 | v3 | 0.95 | 7,330,818 | 89.72 | **−2.67** vs v2, **−2.68** vs AT | — |

**v3 CNN regression grows with τ**: −1.6pp @ τ=0.70/0.85, −2.7pp @ τ=0.95.
Diagnosis points at a numerical pathology in Mahalanobis weighting at CNN
scale: at τ=0.95 stage 4 has effective_rank ~696 out of 2048. Mahalanobis
weights each of those 696 directions by 1/λ_i. Even with a `1e-6 · λ_max`
noise floor, the smallest-λ directions get weight ~10⁶ and the top PC
gets weight ~1, so after mean-normalization across 696 directions the
top PC gets weight ~0. The student is trained to match noise instead of
signal. The effect grows with the number of kept directions, which is
exactly the τ=0.95 vs 0.70 pattern.

The LLM doesn't hit this pathology because max per-block k is ~497 and
most middle blocks have k=5–50. Few directions, less noise in the kept set.

**Diagnosis needs confirmation from the component ablation** (pending).
If `v3_no_kd_no_spar` matches no_logit_kd, the regression is attributable
to Mahalanobis specifically.

### v3 component attribution @ τ=0.85 (asd_v3_ablation_t85)

All 18 epochs, same student architecture (2.68M params, 8.76× compression):

| variant | uses KD? | spar? | sv-weighting | acc | Δ vs full |
|---|---|---|---|---:|---:|
| full (reference) | yes | yes | sqrt | 89.82 | 0.00 |
| v3_mahalanobis_only | yes | yes | mahalanobis | 89.77 | −0.05 (noise) |
| v3 (all three drops) | no | no | mahalanobis | 89.31 | −0.51 |
| **v3_no_kd_no_spar** | **no** | **no** | **sqrt** | **91.87** | **+2.05** |
| no_logit_kd (prior matrix, 15 ep) | no | yes | sqrt | 90.94 | +1.12 |

**Headline**: `v3_no_kd_no_spar` at 91.87% is a new best for ASD on
ResNet50/CIFAR-10 at τ=0.85, beating:

- `full` by +2.05pp
- 3-seed average at 18 ep (90.78 ± 0.16) by +1.09pp
- AT baseline at 20 ep (91.40) by +0.47pp
- ASD v2 at 25 ep (91.23) by +0.64pp

**Reading**: Mahalanobis on CNN is the culprit, not a hidden benefit.
Keeping `sqrt` weighting and dropping just KD+sparsity gives a clean win
that also surpasses the AT reference point. The full path from `full` to
`v3_no_kd_no_spar`:

| step | acc | Δ |
|---|---:|---:|
| full | 89.82 | — |
| drop sparsity (no_sparsity) | 89.59 | −0.23 |
| also drop KD (no_logit_kd + no_spar ≈ v3_no_kd_no_spar at 18 ep) | **91.87** | **+2.28 from prev, +2.05 from full** |
| additionally swap sqrt → Mahalanobis (v3) | 89.31 | −2.56 (Mahalanobis on CNN is the bug) |

Dropping sparsity alone is within noise; dropping KD on top of that
converts what was a −0.23pp drag into a +2.05pp win. The +0.93pp
difference between no_logit_kd (15 ep, sparsity on) and
v3_no_kd_no_spar (18 ep, sparsity off) mixes the 3 epochs of extra
training with the sparsity drop, so the isolated sparsity contribution
is not cleanly separable — but the combined effect is the best result.

**Refined v3 recipe for CNN**: `use_logit_kd=False, delta=0, gamma=0,
sv_weighting=sqrt` (i.e., keep the old sqrt weighting, just drop KD and
sparsity). Mahalanobis is reserved for LLM targets where the spectrum is
heavy-tailed and the weighting pathology doesn't trigger.

## SVHN resubmit — ResNet{18,50}

**ResNet18 / SVHN (teacher 96.35%)**:

| τ | student params | compression | acc | drop (pp) |
|---|---:|---:|---:|---:|
| 0.85 | 147,186 | **75.9×** | 96.16 | 0.19 |
| 0.95 | 452,426 | 24.7× | **96.44** | **−0.08** (beats teacher) |

**ResNet50 / SVHN (teacher 96.56%)**:

| τ | student params | compression | acc | drop (pp) |
|---|---:|---:|---:|---:|
| 0.85 | 1,187,250 | 19.8× | **96.98** | **−0.41** (beats teacher) |
| 0.95 | 5,211,186 | 4.5× | **97.13** | **−0.57** (beats teacher) |

**3/4 SVHN runs exceed the teacher.** Distilled students beat the over-
parameterized teacher on this low-rank-structured task. SVHN has clear
digit-invariant features that don't need 23M params of teacher capacity;
the compressed student with enforced low-rank-subspace regularization
generalizes better. This is a strong generalization signal.

## FLOPs over complete matrix — flops_report_v2

68 rows covering 17 cells (vs original `flops_report`'s 22 rows). Master
JSON at `outputs_jobs/flops_report_v2/flops_report.json`. Key numbers in
the Pareto table above.

## Failed bench cells — not re-submitted

Deeper architecture-generalization issues, not algorithmic:

| cell | error |
|---|---|
| bench_mobilenet_v2_cifar10 | `RuntimeError: tensor size (32) must match (16) at dim 3` — student/teacher spatial misalignment |
| bench_vgg16_bn_cifar10 | `AssertionError: Need exactly 4 stage widths` — VGG has 5 stages |

See [open-questions.md](open-questions.md) §4 for the fix sketch.

## How to re-aggregate after more cells land

```bash
cd /mnt/shared_storage/asd
python scripts/10_aggregate.py --root outputs_jobs \
  --output-dir outputs_jobs/_aggregate
```

Outputs `results_master.json`, `results_table.md`, `compression_vs_acc.png`.
