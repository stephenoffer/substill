# ASD Algorithm

Detailed reference for the Activation Subspace Distillation pipeline.
Covers all three phases and the mathematical form of every loss component.
For *why* individual components do or don't work empirically, see
[findings.md](findings.md).

## Pipeline at a glance

```
Phase 0: Fine-tune teacher on the target dataset
         → outputs/teacher_finetuned.pt
Phase 1: Capture per-block activation covariance
         → eigendecompose → per-layer effective rank + principal components
Phase 2: Build SlimNet student with per-stage widths equal to effective rank
         (rounded to multiple of 8, floored at min_width=16)
Phase 3: Train student with combined loss
         L = α·L_task + β(t)·L_sub + γ(t)·L_spar + δ·L_kd + ε·L_rkd + ζ·L_at
```

Entry points in `scripts/`:

| script | role |
|---|---|
| `00_finetune_teacher.py` | Phase 0 |
| `01_profile_teacher.py` | Phase 1 — saves profiles to disk |
| `02_build_student.py` | Phase 2 — reports student architecture |
| `03_train_asd.py` | Phase 3 — single-config training |
| `07_bench.py` | 0→3 end-to-end for a (model, dataset) pair |
| `08_ablation.py` | 3 only — ablates one component at a time on cached teacher |
| `09_llm_distill.py` | Self-contained 0→3 for GPT-2/WikiText-2 |

## Phase 1 — activation profiling

Forward hooks on every residual block accumulate channel-wise covariance in
O(C²) memory:

```
Cov(layer) = E[x · xᵀ] − E[x] · E[x]ᵀ
```

`x` is the channel vector at each spatial position
([activation_capture.py](../asd/profiling/activation_capture.py#L35-L67)).
Two covariance modes:

- **`per_pixel` (default)** — treat each (H×W) position as an independent
  sample in C-dim channel space. Optionally sub-sample spatial positions
  (`spatial_subsample=k`) to mitigate adjacent-pixel correlation inflating
  top eigenvalues. Dense signal, better rank estimate.
- **`gap`** — globally pool to (B, C) first, then accumulate. Legacy
  behavior; much weaker rank estimate because it measures the covariance
  of spatial means, not the channel distribution per-pixel.

Covariance is symmetrized defensively before `torch.linalg.eigh`. Any
negative eigenvalue larger than `1e-4 · λ_max` raises — a hard failure
instead of silent `clamp(min=0)` masking.

### Effective rank — four definitions

From [svd_analysis.py:109-178](../asd/profiling/svd_analysis.py#L109-L178):

| name | formula | best for |
|---|---|---|
| `variance` (default) | smallest k with `Σ_{i≤k}λ_i / Σλ_i ≥ τ` | "classical" low-rank, sensitive to tail |
| `stable` | `⌈Σλ_i / λ_max⌉` | dimensionless, no threshold |
| `participation` | `⌈(Σλ_i)² / Σλ_i²⌉` | robust to heavy tails, threshold-free |
| `entropy` | `⌈exp(H(p))⌉` with `p_i = λ_i/Σλ` | between stable and variance |

An `eps_relative = 1e-6` noise floor zeroes eigenvalues below `eps·λ_max`
before the rank statistic is computed, so float32 noise doesn't inflate
participation / entropy ranks.

### Stage aggregation

Multiple blocks per stage share a channel count. `aggregate_stage_profile`
reduces per-block profiles to one per-stage profile:

- `last` (default) — use the last block's profile. Matches stage output.
- `max_rank` — block with the highest effective rank (information bottleneck).
- `average` — sum per-block covariances as `Σ V_b Λ_b V_bᵀ`, re-eigendecompose.

## Phase 2 — student construction

`SlimNet` mirrors the teacher's 4-stage layout. Per-stage channel width =
effective rank, rounded up to `width_multiple=8`, floored at `min_width=16`.

Example (ResNet50 / CIFAR-10):

| τ | stage widths | student params | vs teacher (23.5M) |
|---|---|---|---|
| 0.70 | [24, 88, 232, 392] | 557,802 | 42.2× |
| 0.85 | [56, 192, 464, 904] | 2,684,354 | 8.8× |
| 0.95 | [128, 336, 744, 1504] | 7,330,818 | 3.2× |
| 0.99 | [184, 440, 952, 1904] | 11,684,706 | 2.0× |

A `SubspaceProjectorBank` of 4 1×1 convs maps student feature maps to
the teacher's subspace dim per stage:
`(B, student_width_i, H, W) → (B, teacher_rank_i, H, W)`. Orthogonal init
by default (preserves activation variance for a linear layer); no BN by
default (the `normalize_features` option on the loss covers scale
invariance more cleanly).

## Phase 3 — combined loss

```
L = α·L_task
  + β(t)·L_subspace       (subspace-MSE)
  + γ(t)·L_sparsity       (activation histogram match)
  + δ·L_logit_kd          (Hinton KD)
  + ε·L_relation          (RKD — opt-in)
  + ζ·L_attention         (AT — opt-in)
```

Source: [combined_loss.py](../asd/losses/combined_loss.py).

Defaults from `config/default.yaml`: α=1.0, β=0.5, γ=0.3, δ=1.0, ε=0, ζ=0.

### L_task — cross-entropy

Standard: `CE(student_logits, labels)`.

### L_subspace — the core of ASD

Three modes in [subspace_loss.py](../asd/losses/subspace_loss.py):

**`spatial` (default)** — per-pixel match in the teacher's subspace:
```
L_sub = Σ_stages mean_BHW [Σ_k w_k · (s_feat[b,k,h,w] − (Vᵀ x_t[b,:,h,w])_k)²]
```
where `V` is the stage's top-k principal components matrix (C, k) and
`s_feat` is the student's output of the subspace projector at that stage.
`w_k` is the per-PC weight (below).

**`gap`** — pool spatial dims first, then MSE in k dims. Loses spatial
structure; only use for ablations.

**`cosine_spatial`** — scale-invariant per-position cosine distance:
`L = 1 − cos(s_feat, Vᵀ x_t)` averaged over (B, H, W). Insensitive to
feature-magnitude mismatch between student and teacher. SV weighting is
ignored (direction is already scale-free).

### SV weighting — how per-PC contributions are weighted

Five modes in `_sv_weights` ([subspace_loss.py:13](../asd/losses/subspace_loss.py#L13)):

| mode | formula | interpretation |
|---|---|---|
| `uniform` | `w_i = 1` | Euclidean MSE in subspace |
| `linear` | `w_i ∝ λ_i` | dominated by top 1-2 PCs |
| `sqrt` (old default) | `w_i ∝ √λ_i` | mild top-PC bias |
| **`mahalanobis`** (v3) | `w_i ∝ 1/λ_i` | Mahalanobis distance — all directions equally weighted in std-units |
| `inv_sqrt` | `w_i ∝ 1/√λ_i` | half-whitening |

Mahalanobis is equivalent to MSE on `Λ^(-1/2) Vᵀ x_t` (whitened teacher
projections). A numerical floor at `1e-6 · λ_max` prevents noise-floor
directions from blowing up the inverted weight.

Empirical finding: Mahalanobis substantially outperforms sqrt on GPT-2/
WikiText-2 (ppl 131 vs 486 at τ=0.95, same student arch). See
[findings.md](findings.md).

### L_sparsity — histogram matching

Soft differentiable histograms of student vs teacher per-layer activations;
KL divergence between them plus a BCE/MSE term on sparsity ratios. Ablation
shows contribution within noise; the v3 recipe sets γ=0.

### L_logit_kd — Hinton distillation

```
L_kd = T² · KL(softmax(t_logits/T) ‖ softmax(s_logits/T))
```
Default T=4. Note: the 2026-04-22 ablation matrix found KD to be net-
*negative* when combined with L_subspace. See Finding 1 in
[findings.md](findings.md). v3 recipe sets δ=0.

### L_relation and L_attention

Opt-in baselines (`ε > 0` or `ζ > 0`), used primarily for comparison:

- **Relational KD** (Park et al. 2019) — pairwise distances + triplet angles
  on GAP features. `RelationalLoss` in [relation_loss.py](../asd/losses/relation_loss.py).
- **Attention Transfer** (Zagoruyko & Komodakis 2017) — L2-normalized
  channel-aggregated spatial maps. `AttentionTransferLoss` in
  [attention_loss.py](../asd/losses/attention_loss.py).

On the benchmark matrix, AT beats the full ASD recipe at τ=0.85 (91.40 vs
90.78 3-seed avg) and ties at τ=0.95 (92.40 vs 92.13). Motivates Finding 3.

### Combination strategies

Two modes selected by `training.combination`:

- **`fixed` (default)** — weighted sum with the literal α, β, γ, δ. Schedulers
  apply warmup to β and γ (`β(t) = β · min(1, t/β_warmup_epochs)` etc).
- **`uncertainty`** — Kendall & Gal (2018). Each loss carries a learnable
  `s_i = log σ_i²`; total is `Σ 0.5·exp(−s_i)·L_i + 0.5·s_i`. Self-balances
  across loss scales. Historically under-delivered vs well-tuned fixed weights.

### Auto-normalization

`auto_normalize: true` divides each component by its EMA magnitude before
the weighted sum. Lets α, β, γ, δ express *relative* priorities rather
than absolute scales. Useful when comparing across datasets / teachers.
