# How F-ASD works

For usage, see [quickstart.md](quickstart.md). This document explains
the mechanism and how F-ASD differs from the related literature.

## 1. Name and scope

**F-ASD** stands for **Functionally-Aligned Activation Subspace
Distillation**. The "F-" is not cosmetic — the whole method is
reorganized around *function*: each retained direction is selected
because it causally matters for the teacher's output, not because it
accounts for the most variance.

Disambiguation. The acronym "ASD" is overloaded in recent ML
literature — Adversarial Score Distillation and Adversarial
Self-Distillation both appear. F-ASD is unrelated to either, and
shares only the core "low-rank activation subspace" idea with the
sibling [asd/](../../asd/) package in this repo.

The rest of this document assumes decoder-only LLM teachers. CNN
teachers can use F-ASD in residual mode, but the branchwise story
below is LLM-specific.

## 2. Setup

Given a frozen teacher `T` and a calibration corpus `D`, we want a
smaller student `S`. F-ASD's claim: the single object that should
decide the student's rank, architecture, initialization, supervision,
and cache format is a set of **branchwise output-preserving
subspaces** of `T`.

A "branch" of a transformer block is a specific computation we can
isolate at a module boundary:

- `attn.q`, `attn.k`, `attn.v` — the three query/key/value
  projections (for GPT-2, slices of a fused `c_attn`).
- `attn.o` — the attention output projection.
- `ffn.up`, `ffn.gate`, `ffn.down` — the MLP projections
  (gated MLP blocks expose a gate branch; GPT-2 does not).
- `block.residual` — the whole block's residual output, for users
  who want the classic ASD behavior.

## 3. Behavioral rank (novelty #1)

Instead of picking the retained rank by cumulative variance or by the
MP bulk edge, F-ASD picks it by **intervention**.

For a branch with orthonormal basis `V` from the eigendecomposition of
its activation covariance (columns sorted by descending eigenvalue),
define the rank-`k` projection

```
P_k x = V[:, :k] V[:, :k]^T x
```

For each calibration batch, we replace the branch's activation `x` by
`P_k x` inside the teacher and measure how much the teacher's
next-token logits change:

```
R(k) = sum_t w_t * KL(p_T(.|x_t) || p_T^(k)(.|x_t)) / sum_t w_t
```

The behavioral rank is the smallest `k` with `R(k) <= tol` (default
`0.02` nats). `w_t` are per-token weights from
[`token_weighting`](../../fasd/profiling/token_weighting.py):
uniform, entropy, teacher-student disagreement, or completion-only.

Why this matters. Variance-PCA keeps high-variance-but-output-
irrelevant directions and discards low-variance directions that still
drive the logits. Behavioral rank flips that — it keeps whatever is
causally necessary.

The search is either a monotone bisection on `[1, C]` (default) or a
geometric ladder for diagnostics. The full ``kl_curve`` is saved in
the profile.

## 4. Loss schedule

Three objectives, progressively tighter:

1. **Gram** — `||K_s - K_t||_F^2`, computed via the trace identity so
   only `k x k` inner products are materialized. Basis-invariant,
   stable early in training.
2. **CKA** — centered kernel alignment. Adds scale invariance;
   bounded in `[0, 1]`. Good stabilizer on LLMs.
3. **Whitened Procrustes** — `min_R ||Z_s R - Z_t||_F^2` over
   orthogonal `R`, with a closed-form solution via SVD of
   `Z_s^T Z_t`. Rotation-invariant inside the retained subspace but
   tighter than CKA.

   We add a small **covariance / norm calibration** term so scale and
   second-moment structure cannot collapse under the rotation
   invariance:

   ```
   lambda_cov * ||Cov(Z_s) - Cov(Z_t)||_F^2 / k^2
       + lambda_norm * (||Z_s||_F - ||Z_t||_F)^2 / N
   ```

   Defaults: `lambda_cov=0.01`, `lambda_norm=0.001`.

`F_ASDLoss.Schedule` linearly anneals between objectives. The
default transformer schedule: Gram 0-10%, CKA 10-40%, Procrustes
40-100%, feature-weight fade over the last 20% of training.

## 5. Semi-orthogonal projectors + fold-away

The student's branch activation has some width `C_s`; the teacher's
retained subspace has width `k_behavioral`. A **semi-orthogonal**
projector (parameterized via
`torch.nn.utils.parametrizations.orthogonal`) maps `C_s -> k_behavioral`
during warm-up. Being semi-orthogonal, it cannot rescale or skew the
student features arbitrarily, so it cannot disguise student
deficiencies.

At the end of warm-up, `F_ASDLoss.fold_projectors_into_(student)`
absorbs the learned rotation into the teacher basis buffer (pre-
composes `V <- V R^T`) and replaces the projector with identity. The
feature loss then compares student hidden states to the teacher
directly, with no learnable bridge.

When `C_s == k_behavioral` — which
[`fasd.build_student(..., absorbed_init=True)`](../../fasd/builders.py)
produces by construction — this fold is exact.

## 6. Generative KD

On autoregressive teachers, feature matching alone is not enough; the
distillation objective must close the train/inference mismatch. F-ASD
exposes four token-level losses in
[`fasd/losses/generative_kd.py`](../../fasd/losses/generative_kd.py):

- `forward_kl` — classical Hinton distillation target.
- `reverse_kl` — mode-seeking; better for generative LMs (MiniLLM).
- `skew_kl` — DistiLLM's interpolated target
  `alpha * p_t + (1-alpha) * p_s`.
- `contrastive_response_loss` — DistiLLM-2's margin loss that
  encourages higher student log-prob on teacher sequences than on
  student sequences.

## 7. On-policy stage

Starting at `on_policy_start` (default 50%), the driver mixes batches
from the training corpus with samples from a
[`ReplayBuffer`](../../fasd/training/onpolicy.py) of student rollouts
at a tunable ratio. Rollouts are generated by `student.generate`
under `torch.no_grad`. This closes the teacher-forced / generative
gap and is the practical form of GKD inside F-ASD's stack.

## 8. Profile refresh

At the on-policy transition (or on a periodic schedule), the driver
reruns `fasd.profile` on a fresh batch of student rollouts and calls
`F_ASDLoss.refresh_from_profile(new_profile)` to swap basis buffers
and behavioral ranks in place. Projector shapes are preserved when
rank doesn't change; otherwise they are reallocated lazily on the
next forward.

## 9. Absorbed weight init (novelty #2)

Given the same branch profile used for the loss, build the student
by absorbing the teacher's weights through the retained bases:

```
W_s = V_out^T @ W_T @ V_in
b_s = V_out^T @ b_T
```

For GPT-2's `c_attn` this takes a block-diagonal stack
`V_out = diag(V_q, V_k, V_v)` because `c_attn` is fused. For Llama,
each projection is a separate module so the stack is trivial. With
`V_in, V_out = I` (no compression) the absorbed weight recovers the
teacher weight exactly — see
[`test_fasd_absorbed_init.py`](../../tests/test_fasd_absorbed_init.py).

The same profile also drives
[`profile_to_student_config`](../../fasd/compression/width_pruner.py),
which encodes the Minitron findings:

- Width-first: primary reduction comes from `hidden_size` (residual
  branch rank) and `intermediate_size` (FFN up/gate branch ranks).
- Keep attention heads: `num_attention_heads` is preserved unless a
  branch rank forces a drop; `num_key_value_heads` can drop
  independently (grouped-query).
- Contiguous depth drops: layers are removed as a contiguous block,
  never scattered.
- Progressive chain: for aggressive compression,
  `plan_progressive_stages` returns a sequence of intermediate
  configs that the driver can distill in order.

## 10. Scalability — streaming PCA and feature cache

For 13B-70B teachers, full `C x C` covariances are infeasible.
[`StreamingPCA`](../../fasd/profiling/streaming_pca.py) exposes three
backends:

- `exact` — `CovarianceAccumulator` + `torch.linalg.eigh` (baseline).
- `randomized` — Halko-Martinsson-Tropp randomized SVD on a streamed
  sketch. `O(C * (k + p))` memory.
- `oja` — online k-PCA via normalized Oja updates on a rank-k
  projector. `O(C * k)` memory.

The backend is auto-selected when `C >= 1024` or the teacher exceeds
~1B parameters.

When `cache_teacher_features=True`, the driver pre-computes
`z_T = V^T x_T` for every calibration batch and caches it. The
student loop reads those cached features instead of re-running the
teacher forward, cutting teacher FLOPs when the teacher is much
larger than the student.

## 11. Quantization recovery

[`quantize_student`](../../fasd/compression/quantization.py) is an
AWQ-flavored post-training quantization pass. For each linear, it
pulls the matching `BranchProfile` to identify high-magnitude
"salient" input channels (from the stored eigenvalues and principal
components), protects the top `protect_fraction` of them in fp16, and
per-group quantizes the rest to `bits` bits.

`qad_finetune` runs a short skew-KL fine-tune with the quantized
student against the unquantized teacher to recover accuracy. The
driver calls both when `quantize=True`.

## 12. Multi-stage driver

The full pipeline:

1. Teacher correction (optional; `teacher_correction_steps > 0`).
2. Progressive planning (optional; `progressive_stages > 1`).
3. Profile (or reuse an existing one).
4. Warm-up: Gram/CKA feature loss + forward KL.
5. Projector fold-away at the warm-up/middle boundary.
6. Middle: Procrustes feature loss + skew KL, off-policy batches.
7. On-policy: hybrid batching, reverse/skew KL, optional contrastive.
8. Profile refresh on student rollouts.
9. Quantization-aware final stage.

## 13. Why it works

Two intuitions stacked:

1. **Output-preserving subspaces are the right currency.** A branch's
   variance spectrum is often long-tailed but includes important
   low-variance directions. Behavioral rank keeps them; variance rank
   drops them.
2. **Branches match the actual computation.** Supervising the residual
   stream alone is too coarse — error on `attn.v` and `ffn.down` gets
   averaged out in residual supervision. Branchwise supervision puts
   the loss where the compression error is introduced.

## 14. Knobs

| Knob | Effect | Default |
|---|---|---|
| `rank_tol` | Max KL (nats) between unpatched / patched teacher for the behavioral rank search. | 0.02 |
| `token_weighting` | Per-token weight for the behavioral-rank scoring: uniform / entropy / disagreement / completion. | entropy |
| `mode` | `"branch"` (default) or `"residual"` (classic ASD behavior). | branch |
| `pca_backend` | `"auto"` / `"exact"` / `"randomized"` / `"oja"`. | auto |
| `objective` | `"gram"` / `"cka"` / `"procrustes"`, overridden by `schedule`. | procrustes |
| `projector` | `"semiortho"` or `"linear"`. | semiortho |
| `generative_kd` | `"forward_kl"` / `"reverse_kl"` / `"skew_kl"`. | skew_kl |
| `on_policy_start` | Step fraction at which on-policy sampling begins. | 0.5 |
| `on_policy_ratio` | Fraction of batches drawn from replay buffer. | 0.5 |
| `contrastive_weight` | Weight on DistiLLM-2 contrastive response loss. | 0.0 |
| `quantize` | Run the quantization-aware final stage. | False |
| `progressive_stages` | Number of teacher-assistant stages for aggressive compression. | 1 |
| `cache_teacher_features` | Precompute `z_T = V^T x_T` to disk. | False |
| `instability_downweight` | Scale each branch's loss by its bootstrap stability. | False |

## 15. What is novel, and what is not

**Do not cite F-ASD for**:

- Low-rank activation-space compression (SliceGPT, FLAT-LLM).
- Low-rank projection + activation cloning (LRC).
- Task-relevant hidden-dimension feature selection (Flex-KD).
- Attention / FFN internal supervision (MiniLM, MiniLMv2, MaKD).
- On-policy or skew-KL autoregressive KD (GKD, MiniLLM, DistiLLM-2).
- Structured width pruning + distill (Minitron, Sheared LLaMA).

**What F-ASD contributes**:

- Intervention-based (output-preserving) branch-subspace rank
  selection, instead of variance- or importance-score-based.
- Unified object — the same branch profile drives rank, architecture
  (via `absorbed_init`), supervision, and cache format.
- A basis-invariant branchwise loss schedule (Gram → CKA →
  whitened Procrustes) with covariance/norm calibration.
- A fold-away semi-orthogonal projector that closes the "the
  projector is doing the work" critique without requiring
  architecture gymnastics.

## 16. Kill criteria

F-ASD's main claims are falsifiable. The design intentionally exposes
them so downstream experiments can falsify:

- If behavioral rank picks the same rank as variance PCA on real
  teachers, the novelty claim is weak.
- If removing the projector (fold-away) wipes out the quality gains,
  the projector is doing the real work and F-ASD is just an
  expressive bridge.
- If most branches need near-full rank to preserve teacher logits,
  the compression hypothesis is false for that teacher/task.
- If on-policy training helps chat benchmarks but hurts LM retention
  badly, F-ASD should be framed as instruction-tuning, not
  universal compression.
- If width pruning + absorbed init explains almost all gains, the
  feature loss is secondary and should not be the headline.

## Implementation modules

- [`fasd/profiling/activation_capture.py`](../../fasd/profiling/activation_capture.py) — branchwise hook engine.
- [`fasd/profiling/behavioral_rank.py`](../../fasd/profiling/behavioral_rank.py) — intervention-based rank picker.
- [`fasd/profiling/streaming_pca.py`](../../fasd/profiling/streaming_pca.py) — exact / randomized / Oja.
- [`fasd/profiling/token_weighting.py`](../../fasd/profiling/token_weighting.py) — entropy / disagreement / completion.
- [`fasd/profiling/stability.py`](../../fasd/profiling/stability.py) — bootstrap stability cap.
- [`fasd/losses/subspace.py`](../../fasd/losses/subspace.py) — `F_ASDLoss`, `Schedule`.
- [`fasd/losses/procrustes.py`](../../fasd/losses/procrustes.py) — whitened Procrustes distance.
- [`fasd/losses/generative_kd.py`](../../fasd/losses/generative_kd.py) — forward / reverse / skew KL + contrastive.
- [`fasd/compression/absorbed_init.py`](../../fasd/compression/absorbed_init.py) — `V_out^T W_T V_in`.
- [`fasd/compression/width_pruner.py`](../../fasd/compression/width_pruner.py) — profile → config + progressive chain.
- [`fasd/compression/quantization.py`](../../fasd/compression/quantization.py) — AWQ PTQ + QAD fine-tune.
- [`fasd/builders.py`](../../fasd/builders.py) — `build_student` for GPT-2 and Llama.
- [`fasd/training/distill.py`](../../fasd/training/distill.py) — multi-stage driver.
- [`fasd/training/onpolicy.py`](../../fasd/training/onpolicy.py) — rollouts, replay, hybrid collator.
- [`fasd/training/teacher_correction.py`](../../fasd/training/teacher_correction.py) — short adaptation.
- [`fasd/api.py`](../../fasd/api.py) — `profile`, `capture`, `TeacherProfile`, `BranchProfile`.
