# Benchmarks

All numbers are **matched-compute, multi-seed** (mean ± standard deviation), reproducible
with `scripts/analysis/lrb.py`. The full study, with every control and ablation, is in
[the LRD write-up](learned_restriction.md); the honesty audit behind the "best prior"
column is [init_findings](init_findings.md).

Perplexity (PPL) is on WikiText-2; **lower is better**. "Best prior" is the strongest
frozen-basis baseline (PCA), which itself beats the 2026 SVD-compression methods (below).

## The controlled result

On `JackFram/llama-160m` compressed 3.07×, training the projection `V` beats freezing it —
with everything else held identical (student geometry, FFN selection, whole heads, data
order, seed, optimizer, per-step wall-clock). All arms at 2000 steps, n=3.

| arm | what trains | final PPL |
|---|---|---:|
| `pca` (best frozen baseline) | weights, frozen PCA basis | 80.94 ± 0.90 |
| `pca_reparam` (control) | weights `D`, **`V` frozen** | 81.25 ± 1.06 |
| **`substill` / LRD** | weights `D` **+ `V` trained** | **75.45 ± 0.79** |

The control is airtight: `pca_reparam` runs LRD's exact code path with the Stiefel learning
rate set to zero and reproduces the PCA baseline. Turning on the `V` coordinate — and nothing
else — drops PPL by **5.8 points (~6σ)**, and every LRD seed beats every baseline seed. The
win holds at matched **wall-clock**, not just steps.

## It scales — and the margin grows

Repeated on larger Sheared-LLaMA teachers (RMSNorm, untied embeddings):

| teacher | d | compression | best prior | **substill (LRD)** | gain | n |
|---|---:|---:|---:|---:|---:|---:|
| Llama-160M | 768 | 3.07× | 80.94 ± 0.90 | **75.45 ± 0.79** | **−6.8%** (~6σ) | 3 |
| Sheared-LLaMA-1.3B | 2048 | 3.64× | 366.70 ± 0.75 | **341.99 ± 7.15** | **−6.7%** | 2 |
| Sheared-LLaMA-2.7B | 2560 | 9.8× | 635.57 ± 112.87 | **528.66 ± 36.06** | **−16.8%** | 3 |

The mechanism transfers intact across a 17× range of teacher size, and the win *grows* with
scale. Note the standard deviations at 2.7B: the frozen-basis baseline is erratic (±113)
while LRD is tight (±36) — training `V` doesn't only lower the mean, it stabilizes the run.
(These are lightly-tuned short-budget runs; treat the exact magnitudes as noisy, the
direction as solid.)

## It wins at every compression ratio

Same `llama-160m` teacher, student width swept down (matched steps):

| student hidden | compression | best prior | **substill (LRD)** | gain |
|---:|---:|---:|---:|---:|
| 384 | 3.07× | 80.94 ± 0.90 | **75.45 ± 0.79** | **−6.8%** |
| 256 | ~5.6× | 92.98 ± 0.34 | **89.02 ± 0.42** | **−4.3%** |
| 192 | ~8.4× | 101.10 ± 1.22 | **97.03 ± 1.40** | **−4.0%** |

The margin is largest at moderate compression and narrows as the student gets tighter.

## Versus the 2026 SVD-compression wave

A survey of recent methods (AIR, LASER, Swift-SVD, SVD-LLM v2, IO-SVD, SigmaScale, COMPOT,
SAES-SVD) finds every one picks its subspace by a *frozen surrogate* — activation, influence,
or reconstruction — and none distills through the network. Reproducing each principle as the
frozen basis and distilling identically (llama-160m, 3.07×, n=3), LRD beats the **best** of
them — AIR's activation+influence basis at 79.98 PPL — by **5.7% (≈5σ)**. Tellingly, the
surrogate that *minimizes* logit error lands near-worst on final PPL: the proxy these methods
climb does not predict distilled quality. Rank-allocation methods (UniRank, ARA, LLRC) are an
orthogonal axis that `substill`'s differentiable-rank option composes with, not against.

## Vision: the principle transfers, and it locates the win

On ResNet-50 / CIFAR-10 (top-1 accuracy, **higher is better**), a ReLU CNN has no
rotation-equivariant stream, so only channel *selection* is legal:

| method | top-1 |
|---|---:|
| random init + distill | 0.735 |
| variance-selected restriction | **0.820** |
| KD-selected restriction | 0.813 |

Both restriction variants beat random init by ~8 points — the restriction *principle*
transfers — but they tie each other, because the *learned rotation* (the source of the
transformer win) is not available under ReLU. This is a clean cross-check that the ~6σ
transformer win comes from *rotating* the subspace, not merely *selecting* it.
