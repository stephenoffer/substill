# Open Questions & Deferred Work

Questions raised by the Apr-22 matrix that would benefit from further
experiments or code work. Ordered by expected research-impact / cost.

## 1. Spectrum-adaptive sv_weighting

**Problem**: The v3 matrix showed Mahalanobis weighting is a huge win on
heavy-tailed transformer spectra (GPT-2 ppl 486 → 131 at τ=0.95) but
regresses on sharp CNN spectra at aggressive compression (CIFAR-10 τ=0.70,
87.47% vs no_logit_kd's 89.07%). The cause is that Mahalanobis equalizes
all PCs in standard-deviation units; when the top PCs are the actual task
signal (sharp spectra), this de-emphasizes the directions that matter most.

**Proposed fix**: blend sqrt and Mahalanobis adaptively, keyed on the
spectrum's participation ratio:

```python
pr = (Σλ)² / Σλ²                # heavy-tail-robust spread measure
alpha = clamp(pr / k, 0, 1)     # 1 for diffuse spectra, → 0 for heavy-tailed
w = alpha · w_sqrt + (1 − alpha) · w_mahalanobis
```

**Cost**: ~20 LoC in `_sv_weights`. Plus one new variant in
`08_ablation.py` and one new cell.

**Expected outcome**: the best result of both regimes — close to Mahalanobis
on LLM, close to sqrt (or v3_no_kd_no_spar) on CNN τ=0.70.

## 2. Supervised subspace (task-gradient covariance)

**Problem**: Both variance-rank and participation-rank are unsupervised —
they use the channel covariance `E[xxᵀ]` which measures where activations
*vary*, not where the loss *moves*. For transformer residual streams, the
variance-covariance top PCs are the "attention-sink" directions with low
task value.

**Proposed fix**: eigendecompose the task-gradient covariance instead:
```
C_task = E[g · gᵀ]     where g = ∂L_task / ∂x_i
```
The top eigenvectors are the directions in `x_i` that move the loss. Use
these as the subspace targets.

Implementation sketch:

1. Add a `backward_hook` alongside the existing forward hook in
   [activation_capture.py](../asd/profiling/activation_capture.py). Run one
   backward pass per calibration batch with `L_task.backward()`.
2. Accumulate `g · gᵀ` in the same double-precision pattern used for the
   activation covariance.
3. Eigendecompose, project, same rest of pipeline.

**Cost**: ~100 LoC. Requires re-running all profiling-dependent cells (which
is cheap relative to training — calibration is ~40 batches).

**Expected outcome**: on GPT-2, should dramatically beat both variance rank
(which collapses) and participation rank (which underestimates). On CNNs,
probably a small improvement because channel-covariance and task-gradient-
covariance agree more on clean spectra.

## 3. Gradient surgery (PCGrad) between KD and subspace

**Problem**: Finding 1 showed `L_kd` and `L_subspace` destructively
interfere. v3 solves this by dropping KD (δ=0). But KD on its own is worth
+1pp over task-only, so the signal is there; it's just mis-combined.

**Proposed fix**: PCGrad (Yu et al. 2020) — before the optimizer step,
remove the component of `g_kd` that conflicts with `g_sub`:
```
if ⟨g_kd, g_sub⟩ < 0:
    g_kd ← g_kd − (⟨g_kd, g_sub⟩ / ‖g_sub‖²) · g_sub
```
Applied pairwise between all loss components.

**Cost**: ~30 LoC, wrapping `loss.backward()`. Slightly more expensive per
step (extra backward-pass bookkeeping) but small.

**Expected outcome**: recovers part of the lost +1pp from KD. If it matches
or exceeds v3 (no KD), we have a principled alternative to dropping KD
outright.

## 4. Architecture-generalization — VGG / MobileNetV2

**Problem**: Two matrix cells failed not because of the algorithm but because
`SlimNet` assumes exactly 4 stages and a specific stride schedule:

- `bench_vgg16_bn_cifar10`: `AssertionError: Need exactly 4 stage widths`
  — VGG16-BN has 5 conv stages.
- `bench_mobilenet_v2_cifar10`: `RuntimeError: tensor size (32) must match
  (16) at dim 3` at the first batch — MobileNetV2's student spatial
  schedule doesn't match the teacher's.

**Proposed fix**: generalize `SlimNet` to N stages, and per-stage stride
derived from the teacher's actual forward graph rather than hardcoded.

**Cost**: ~200 LoC. Also needs extending
[teacher_ext.py](../asd/models/teacher_ext.py) to expose per-stage feature
outputs with consistent channel counts.

**Expected outcome**: closes the "cross-architecture" claim. Both teachers
have fine-tuned weights cached already, so re-run is cheap.

## 5. Real ImageNet result

**Problem**: The 2026-04-22 matrix's ImageNet cell reached 12.9% student
accuracy (teacher 91.2%) — it ran 5 epochs with 100 samples per class. Almost
certainly under-training, but with the current budget it's not a publishable
result.

**Proposed fix**: full ImageNet-1k, ≥30 epochs, proper augmentation, on a
cluster of L4s or equivalent. Or start with a smaller ImageNet-100 subset
(Deng et al. 2009) and work up.

**Cost**: expensive — ≥24 GPU-hours on L4 for a full run. Usually deferred
to v2 of the paper.

## 6. Seed variance on v3 recipe

**Problem**: v3 cells (`asd_v3_t70/85/95`) each run a single seed. If v3's
effect is ±0.5pp, we need 3 seeds to claim significance.

**Proposed fix**: after v3 results land, pick the threshold with the most
interesting delta and re-run 3 seeds. ~1 hour GPU each.

**Cost**: 3 extra cells. Reuses existing infrastructure.

## 7. Paper-ready Pareto figure

**Problem**: `scripts/10_aggregate.py` produces a scatter plot, but the
figure it renders (`compression_vs_acc.png`) mixes ablations, baselines,
multiple datasets, and different epoch budgets in one panel. Not paper-ready.

**Proposed fix**: one clean ResNet50/CIFAR-10 Pareto panel (FLOPs on x,
test accuracy on y), with three curves:
- `task_only` points (no teacher) — lower bound
- `AT baseline` — competitor
- `ASD v3` — our method

**Cost**: ~80 LoC in a new `make_paper_figure.py`.
