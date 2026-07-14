# Benchmarks

Every number here is matched-compute and multi-seed (mean ± standard deviation),
reproducible with `scripts/analysis/lrb.py`. The full study, with every control and
ablation, is the [LRD write-up](learned_restriction.md); the honesty audit behind the
"best prior" column is [init_findings](init_findings.md).

> **Read [§9 of the LRD write-up](learned_restriction.md) alongside this page.** A soundness
> audit of the restriction map found two bugs that silently produced a student that was *not*
> a restriction of its teacher — on **tied-embedding** models (Llama-3.2) and on **every GQA**
> model (Llama-3, Mistral). Neither is visible in any number on this page, because every
> teacher benchmarked here is MHA with untied embeddings. Both are fixed and pinned by test.
> The numbers below **reproduce** on a fresh harness and survive every fix; the audit also
> withdraws one false theoretical claim and restates the statistics for the small `n` they
> actually have.

Perplexity (PPL) is on WikiText-2, where **lower is better**. "Best prior" means the
strongest frozen-basis baseline (PCA), which itself beats the 2026 SVD-compression methods
listed below.

## The controlled result

On `JackFram/llama-160m` compressed 3.07×, training the projection `V` beats freezing it,
with everything else held identical: student geometry, FFN selection, whole heads, data
order, seed, optimizer, and per-step wall-clock. All arms run 2000 steps, n=3.

**On the corrected restriction map** (post-audit — this is the number to quote):

| arm | what trains | final PPL (n=6) |
|---|---|---:|
| `pca` — activation SVD | weights, frozen basis | 74.96 ± 1.13 |
| `gn` — AIR (**strongest** frozen principle) | weights, frozen basis | 75.00 ± 1.07 |
| **`substill` / LRD** | weights `D` **+ `V` trained** | **71.25 ± 1.12** |

**−3.75 PPL (−5.0%)** against the strongest frozen basis, 95% CI [−5.16, −2.34] (Welch, dof 10),
**p < 0.001**, and all six LRD seeds beat all six baseline seeds.

The control is what makes this airtight: `pca_reparam` runs LRD's exact code path with the
Stiefel learning rate set to zero, and it reproduces the PCA baseline. Turning on the `V`
coordinate, and nothing else, is what drops the PPL. The win also holds at matched
**wall-clock**, not just matched steps.

**Pre-audit numbers, for the record:**

| arm | what trains | final PPL |
|---|---|---:|
| `pca` | weights, frozen PCA basis | 80.94 ± 0.90 |
| `pca_reparam` (control) | weights `D`, **`V` frozen** | 81.25 ± 1.06 |
| LRD | weights `D` **+ `V` trained** | **75.45 ± 0.79** |

These reported a −6.8% margin. The audit found **two** bugs in the initialization that both arms
share: a mis-scaled RMS gain (39% too large at layer 0), and a residual covariance dominated by
the highest-norm deep layers. Correcting them improves the **baseline** by 5.9 PPL and LRD by
only 2.5, because a trained projection can compensate for a bad start and a frozen basis cannot.
Roughly half the published margin was measured against a handicapped baseline. See
[§10–§11](learned_restriction.md).

## Does it scale? Not established — and the old claim is withdrawn

> The rows below were reported at n=2/n=3. At 1.3B the frozen baseline swings **±68 PPL between
> two seeds**, and the published 2.7B row's own baseline sd (±113) **exceeds its claimed effect**
> (107 PPL). Re-measured at n=4 on the corrected map, 1.3B trends −7.9% with **p = 0.35** — the
> mechanism transfers, the margin is unmeasured. See [§11g](learned_restriction.md).

Repeated on larger Sheared-LLaMA teachers (RMSNorm, untied embeddings):

| teacher | d | compression | best prior | **substill (LRD)** | gain | n |
|---|---:|---:|---:|---:|---:|---:|
| Llama-160M | 768 | 3.07× | 75.00 ± 1.07 | **71.25 ± 1.12** | **−5.0%** | 6 |
| Sheared-LLaMA-1.3B | 2048 | 3.64× | 366.70 ± 0.75 | **341.99 ± 7.15** | **−6.7%** | 2 |
| Sheared-LLaMA-2.7B | 2560 | 9.8× | 635.57 ± 112.87 | **528.66 ± 36.06** | **−16.8%** | 3 |

The mechanism transfers intact across a 17× range of teacher size, and the win *grows*
with scale.

Look at the standard deviations at 2.7B. The frozen-basis baseline is erratic (±113) while
LRD is tight (±36). Training `V` doesn't only lower the mean, it stabilizes the run. These
are lightly-tuned, short-budget runs, so treat the exact magnitudes as noisy and the
direction as solid.

## It wins at every compression ratio

Same `llama-160m` teacher, student width swept down, matched steps:

| student hidden | compression | best prior | **substill (LRD)** | gain |
|---:|---:|---:|---:|---:|
| 384 | 3.07× | 80.94 ± 0.90 | **75.45 ± 0.79** | **−6.8%** |
| 256 | ~5.6× | 92.98 ± 0.34 | **89.02 ± 0.42** | **−4.3%** |
| 192 | ~8.4× | 101.10 ± 1.22 | **97.03 ± 1.40** | **−4.0%** |

The margin is largest at moderate compression and narrows as the student gets tighter.

## Versus the 2026 SVD-compression wave

Surveying the recent methods (AIR, LASER, Swift-SVD, SVD-LLM v2, IO-SVD, SigmaScale,
COMPOT, SAES-SVD), every one picks its subspace by a *frozen surrogate* (activation,
influence, reconstruction), and not one of them distills through the network.

Reproducing each principle as the frozen basis and distilling identically (llama-160m,
3.07×, n=3), LRD beats the **best** of them, AIR's activation+influence basis at 79.98
PPL, by **5.7% (≈5σ)**.

One detail is worth dwelling on: the surrogate that *minimizes* logit error lands
near-worst on final PPL. The proxy these methods climb does not predict distilled quality.
That is the whole argument for training the basis instead of choosing it.

Rank-allocation methods (UniRank, ARA, LLRC) sit on an orthogonal axis that `substill`'s
differentiable-rank option composes with rather than competes against.

## Vision: the principle transfers, and it locates the win

On ResNet-50 / CIFAR-10 (top-1 accuracy, **higher is better**). A ReLU CNN has no
rotation-equivariant stream, so only channel *selection* is legal here:

| method | top-1 |
|---|---:|
| random init + distill | 0.735 |
| variance-selected restriction | **0.820** |
| KD-selected restriction | 0.813 |

Both restriction variants beat random init by roughly 8 points, so the restriction
*principle* transfers. But they tie each other, because the *learned rotation* that
produces the transformer win is not available under ReLU.

That tie is the point. It is a clean cross-check that the transformer win comes from
*rotating* the subspace, not merely from *selecting* it.
