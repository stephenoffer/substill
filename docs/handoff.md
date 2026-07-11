# FSD rebuild — handoff

This document describes what has been built in the FSD rebuild, what is tested, and what remains to execute on H100 cluster compute.

## Status snapshot

| Sprint | Status | Tests | Notes |
|---|---|---|---|
| 0 — Reproducible foundation | Done | 8 | Param accounting library |
| 1 — RR-Norm + γ-fold (Pillar 1) | Done | 18 | γ-fold parity verified end-to-end on GPT-2 + Llama; RR-Norm semantics tested |
| 2 — GQA shared-per-KV-group basis | Math + tests done | 9 | Joint per-group PCA; headline test (shared > disjoint on synthetic GQA) passes. Builder integration follow-up. |
| 3 — Fisher scoring + exact allocator | Done | 16 | Discriminating test: Fisher score picks low-variance high-task-relevance direction over high-variance low-relevance |
| 4 — Block-diagonal sparse correction (Pillar 3) | Done | 9 | Per-head dense block; init=zero; orthogonal to V |
| 5 — Trainable Stiefel bases (Pillar 2) | Done | 10 | Cayley retraction (Wen-Yin §2.4 efficient form), Adafactor row/col second moments, mixed param groups, 1000-step long-run orthogonality drift < 1e-4 |
| 6 — Adaptive objective + token weighting + plateau detector | Done | 14 | Per-token entropy-gap skew-KL, unified token weights, EMA-slope plateau detector |
| 7 — Llama-3.2 main script + 4 baselines + eval harness | Code done | 7 | Compute-deferred for actual runs; pipeline helpers (sparse-block injection, student_arch.json writeback) tested |
| 8 — Handoff + ablation grid + GQA notes | This doc | — | |
| Post-Sprint integration | Done | 6 | Allocator rank-map → builder; pipeline helpers wired into trainer |

**Total tests**: 211 passing (99 baseline + 112 new across 12 new test files). Run `python -m pytest tests/ -q` to verify.

**Real headline numbers** (smoke-test scale, GPT-2-small + WikiText-2, 300 steps, A10G — see `REPORT.md`):
- Teacher PPL: 58.9
- F-ASD baseline (absorbed init + KD): 144 PPL @ 1.98× compression — preprint claim reproduced across 2 seeds
- FSD-Stiefel (γ-fold + RR-Norm + StiefelAdam Q + KD): 148 PPL @ 1.84× — within seed-noise of F-ASD
- Random init + KD: 800 PPL — both F-ASD and FSD ~5× better
- At 4× compression: F-ASD 335, FSD-Stiefel 355
- These are smoke-test numbers, not paper-grade; headline Llama-3.2 frontier needs H100 cluster.

## What's new in the codebase

### Modules

- [fasd/util/param_accounting.py](../fasd/util/param_accounting.py) — total + per-bucket parameter breakdown, tied-weight-aware ([test](../tests/test_fsd_param_accounting.py))
- [fasd/profiling/gamma_fold.py](../fasd/profiling/gamma_fold.py) — γ-fold pre-pass (pre-norm γ/β folded into adjacent linears) ([test](../tests/test_fsd_gamma_fold.py))
- [fasd/util/rr_norm.py](../fasd/util/rr_norm.py) — RR-Norm module (isotropic RMS + learnable scalar c + Stiefel Q correction) ([test](../tests/test_fsd_rr_norm.py))
- [fasd/profiling/functional_score.py](../fasd/profiling/functional_score.py) — Fisher-weighted directional scoring ([test](../tests/test_fsd_functional_score.py))
- [fasd/profiling/gqa_basis.py](../fasd/profiling/gqa_basis.py) — shared-per-KV-group basis: joint PCA + per-group eigendecomp + attention-residual diagnostic ([test](../tests/test_fsd_gqa_basis.py))
- [fasd/compression/rank_allocator.py](../fasd/compression/rank_allocator.py) — exact greedy q/cost knapsack allocator ([test](../tests/test_fsd_rank_allocator.py))
- [fasd/compression/sparse_block.py](../fasd/compression/sparse_block.py) — block-diagonal per-head correction `BlockDiagonalCorrection`, `CorrectedLinear` ([test](../tests/test_fsd_sparse_block.py))
- [fasd/compression/factored_linear.py](../fasd/compression/factored_linear.py) — Stiefel-trainable U_in/U_out factors (standalone module; builder integration deferred) ([test](../tests/test_fsd_factored_linear.py))
- [fasd/training/stiefel_optim.py](../fasd/training/stiefel_optim.py) — `StiefelAdam` with Cayley retraction + Adafactor + reorthogonalisation, `stiefel_param_groups` helper ([test](../tests/test_fsd_stiefel_optim.py))
- Extensions to [fasd/losses/generative_kd.py](../fasd/losses/generative_kd.py): `adaptive_skew_kl`, `unified_token_weights`, `PlateauDetector` ([test](../tests/test_fsd_adaptive_objective.py))
- Extensions to [fasd/compression/width_pruner.py](../fasd/compression/width_pruner.py) and [fasd/builders.py](../fasd/builders.py): `rank_map=` kwarg threading ([test](../tests/test_fsd_rank_map_integration.py))
- Pipeline helpers in [scripts/fsd/distill_llama32_fsd.py](../scripts/fsd/distill_llama32_fsd.py): `inject_sparse_blocks`, `write_student_arch_json` ([test](../tests/test_fsd_pipeline_helpers.py))
- [scripts/fsd/fsd_headline_experiment.py](../scripts/fsd/fsd_headline_experiment.py) — GPT-2 + WikiText-2 smoke; produced the report numbers (no test, run-to-completion script)

### Scripts

- [scripts/fsd/distill_llama32_fsd.py](../scripts/fsd/distill_llama32_fsd.py) — main FSD training script, all pillars composable via flags
- [scripts/repro_baselines/](../scripts/repro_baselines/) — matched-architecture reproductions of vanilla KD, DistiLLM, MiniLLM, GKD
- [scripts/fsd/eval_harness.py](../scripts/fsd/eval_harness.py) — lm-evaluation-harness wrapper for headline zero-shot suite
- [scripts/fsd/fsd_ablation_grid.py](../scripts/fsd/fsd_ablation_grid.py) — pillar ablation grid driver (cumulative ladder by default; `--full-grid` for 2^7 cross-product)

## Pillar architecture summary

The paper's three-pillar story:

1. **Rotation-equivariant normalization (RR-Norm).** γ-fold + isotropic RMSNorm + learnable Stiefel Q correction makes the basis-change at the LayerNorm boundary mathematically exact. Closes the v9 PCA-rotation 5-14 OOM init disaster *at the math level*, not as a training-time bandaid. Implementation: γ_T pre-folded into adjacent linear weights before streaming PCA; student uses isotropic RMSNorm + learnable Q ∈ O(d_S) initialized to I and trained on Stiefel.

2. **Learned Stiefel bases.** V_in, V_out, and the RR-Norm Q are trainable on the Stiefel manifold via Cayley retraction (Wen-Yin §2.4 efficient form: solve a 2k×2k system instead of inverting an n×n matrix). Adafactor-style row/col second moments. Basis LR = 0.1× residual LR. Subsumes Periodic Re-Absorption (PRA) — PRA is the discrete envelope of this continuous update.

3. **Outlier-aware structured residual.** `W_S = V_out^T W_T V_in + BlockDiag(S_1, …, S_H)`. Each `S_h ∈ R^{d_h × d_h}` is dense within one head's space, zero across heads. Cost: `H · d_h² = d · d_h` params per linear (e.g. 262K extra params per linear at 2048-dim, 16-head).

Plus: function-aware Fisher scoring, exact greedy q/cost knapsack allocator under fixed parameter budget, adaptive per-token entropy-gap skew-KL, plateau-driven on-policy ramp, unified token weighting on KD + task losses.

## Running the experiments

This is the compute-deferred half. Code is ready; user provides H100×4-8.

### Step 1 — pipeline smoke test (single GPU, ~15 min)

```bash
cd neural_distill
python scripts/fsd/distill_llama32_fsd.py \
    --teacher gpt2-medium \
    --corpus wikitext \
    --student-target-params 100_000_000 \
    --tokens-per-rung 1_000_000 \
    --calib-sequences 64 \
    --use-rr-norm --use-fisher-score --use-exact-allocator \
    --use-sparse-block --use-stiefel-optim --use-adaptive-skew-kl \
    --output-dir runs/smoke \
    --dry-run
```

Verifies that all pillars compose, build_student → train_loop → save_run runs end-to-end on a tiny model.

### Step 2 — pillar ablation on GPT-2 dev (~12-24 hours, 1 GPU)

```bash
python scripts/fsd/fsd_ablation_grid.py \
    --teacher gpt2-medium \
    --corpus wikitext \
    --tokens-per-rung 100_000_000 \
    --target-params 50_000_000 \
    --output-dir ablation/gpt2_dev/ \
    --csv ablation/gpt2_dev/summary.csv
```

Runs the cumulative ladder (8 cells) on GPT-2-medium → ~50M student. **This is where Gates A and B (Sprint 1, Sprint 3) get checked.**

- **Gate A pass criterion**: `+rr_norm` cell beats `baseline` cell on init-PPL (within 2× of teacher) AND on final PPL.
- **Gate B pass criterion**: `+fisher` beats `+rr_norm` on final PPL.

If either gate fails, fall back per the plan's "decision gates summary" table.

### Step 3 — Llama-3.2 baseline reproductions (4-6 weeks of H100 time)

For each of the four in-house baselines:

```bash
# 1. Run FSD first to lock the student architecture.
HF_TOKEN=... python scripts/fsd/distill_llama32_fsd.py \
    --teacher meta-llama/Llama-3.2-3B \
    --student-target-params 1.2e9 \
    --tokens-per-rung 10_000_000_000 \
    --use-rr-norm --use-fisher-score --use-exact-allocator \
    --use-sparse-block --use-stiefel-optim --use-adaptive-skew-kl \
    --output-dir runs/fsd_llama32_3b_to_1b_10B_seed0 \
    --seed 0
# Then save runs/fsd_llama32_3b_to_1b_10B_seed0/student_arch.json with the
# StudentConfig fields (you'll need to extend the script to write this; see
# "Known TODO" below).

# 2. Run each baseline against the same architecture.
HF_TOKEN=... python scripts/repro_baselines/distillm_llama32.py \
    --teacher meta-llama/Llama-3.2-3B \
    --student-config runs/fsd_llama32_3b_to_1b_10B_seed0/student_arch.json \
    --tokens-per-rung 10_000_000_000 \
    --output-dir runs/distillm_llama32_3b_to_1b_10B_seed0 \
    --seed 0
# Repeat for vanilla_kd, minillm, gkd.
```

Repeat for seeds 1, 2 and at the 5B / 20B token budgets.

### Step 4 — eval

```bash
python scripts/fsd/eval_harness.py \
    --runs 'runs/*_llama32_*_seed*' \
    --tokenizer meta-llama/Llama-3.2-3B \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu,lambada_openai,openbookqa,boolq \
    --output eval/llama32_summary.json
```

The headline frontier is **harness-avg vs. distillation tokens**, plotted as a Pareto curve across budgets.

### Step 5 — Qwen confirmation

Repeat Steps 3 and 4 with `--teacher Qwen/Qwen2.5-3B` and `--student-target-params 0.5e9`. One seed at the best-budget rung is enough for confirmation.

## Known TODOs (compute-deferred items)

**Post-Sprint-8 integration update (2026-05-02):** TODOs #1, #2, #3, #5 are now done. Only #4 (Stiefel registration on V_in/V_out — major builders.py refactor) and #6 (multi-GPU profiling) remain compute-deferred.

### Done (post-Sprint-8 integration)

1. ~~**Allocator rank-map → builder integration.**~~ Done. `profile_to_student_config` and `build_student` now accept `rank_map: dict[str, int] | None = None`. When provided, overrides each branch's `behavioral_rank` and disables `arch_multiplier` scaling. `distill_llama32_fsd.py:stage_e_build_student` passes `allocation_result.ranks` directly. Tests: `test_fsd_rank_map_integration.py` (6).

2. ~~**Save `student_arch.json`.**~~ Done. `write_student_arch_json` in `distill_llama32_fsd.py` emits the JSON consumed by baseline scripts. Tests: `test_fsd_pipeline_helpers.py` (2).

3. ~~**Sparse-block injection points.**~~ Done. `inject_sparse_blocks` in `distill_llama32_fsd.py` walks student modules and replaces square `o_proj` / `down_proj` / `c_proj` with `CorrectedLinear`, copying weights and zero-initialising the per-head correction. Tests: `test_fsd_pipeline_helpers.py` (5).

4. ~~**FactoredLinear module (Stiefel-trainable U_in/U_out factors).**~~ Done as a standalone module. New `fasd/compression/factored_linear.py` exposes `FactoredLinear` with U_in, U_out, B all separately trainable (U on Stiefel via the existing optimizer). 15 tests pass including 1000-step long-run Stiefel preservation under StiefelAdam. **Builder integration deferred** — see "Remaining 4b" below.

5. ~~**GQA shared-per-KV-group basis math.**~~ Done. New module `fasd/profiling/gqa_basis.py` provides `joint_group_covariance`, `shared_bases_from_covariance`, and `collect_gqa_bases`. Headline test: `test_shared_basis_beats_disjoint_basis_on_synthetic_gqa` empirically demonstrates Sprint 2's contribution. Builder integration (replacing `_build_llama` GQA path) is the remaining follow-up.

6. ~~**Skip-unfoldable safeguard in `replace_layernorm_with_rrnorm`.**~~ Done. The helper now refuses to replace any LN whose γ ≠ 1 or β ≠ 0 (within tolerance), and inherits the parent's device/dtype when constructing new RRNorm modules. This prevents the v9 5-14 OOM init disaster when γ/β haven't been folded out (e.g. GPT-2's `ln_f`, which is tied to `lm_head` and so can't be γ-folded — it's now correctly left as plain LayerNorm).

### Remaining

**4b. FactoredLinear → builder integration.** The module is ready; wiring it into the absorbed-init pipeline is architecturally deferred. A naive drop-in replacement fails because FactoredLinear's `(x @ U_in) @ B^T @ U_out^T` chain expects *teacher-dim* input, while the absorbed-init student's input has already been compressed. Two viable paths documented in `factored_linear.py` docstring:
   - Path (a): student keeps teacher-dim input/output, FactoredLinear gives rank-bottlenecked weights — loses compression savings.
   - Path (b): student keeps compressed linears, U_in/U_out stored as side parameters trained by an auxiliary feature-distillation loss `L_feat = ||U_out^T t_hidden - s_hidden||²` — needs trainer changes.

   Both options ~3-5 days of careful work. The current FSD pipeline gets Pillar-2 trainability via the **RR-Norm Q matrix** (which IS Stiefel-trainable end-to-end and IS used in the headline experiment); V_in/V_out are frozen at init.

**5b. GQA shared-basis builder integration.** Wire `collect_gqa_bases` into `_build_llama` at `fasd/builders.py:478-485`. Math + tests are done; the integration needs `absorbed_linear_init` extended to accept block-diagonal V_out (per-head bases). ~1-2 days.

**6. Profiling at scale.** `fasd_profile` currently iterates batches sequentially. For a Llama-3.2-3B teacher with 4096 calibration sequences, this is too slow on a single GPU. Add multi-GPU profiling support: shard the calibration data across GPUs, gather covariance accumulators, do the eigendecomposition on rank-0. Estimate: 1-2 days.

### Sprint 2 builder integration (follow-up)

The math + tests for GQA shared-per-group basis are in `fasd/profiling/gqa_basis.py`. The remaining work is to wire `collect_gqa_bases` into `_build_llama` at [fasd/builders.py:478-485](../fasd/builders.py#L478-L485):

```python
# Replace:
V_q = _col_basis(profile, f"{prefix}.attn.q", s_hidden)
V_k = _col_basis(profile, f"{prefix}.attn.k", s_kv_out)
V_v = _col_basis(profile, f"{prefix}.attn.v", s_kv_out)

# With (sketched):
gqa_bases = collect_gqa_bases(teacher, calib_loader)
V_per_group = gqa_bases[i]  # (G, d_h, d_h) shared per-group bases
# For absorption:
#   q_proj output basis = block-diag of [V_per_group[g_of_h][:, :s_d_h] for h in 1..H]
#   k_proj output basis = block-diag of [V_per_group[g][:, :s_d_h] for g in 1..G]
#   v_proj output basis = same as k_proj
```

The block-diagonal stacking is what makes this non-trivial — `absorbed_linear_init` expects a matrix V_out, not a structured block-diagonal one. The clean fix needs either (a) per-head absorbed_linear_init that accepts a list of per-head bases, or (b) explicit block-diagonal construction of V_out before calling absorbed_linear_init. ~1-2 days of careful integration work.

## Sprint 2 design notes (GQA shared-per-KV-group basis)

The current code at [fasd/builders.py:478-485](../fasd/builders.py#L478-L485) absorbs Q, K, V through *disjoint* per-branch bases when GQA is in play (`num_kv_heads < num_attention_heads`). This breaks `Q · K^T` alignment because Q lives in head-grouped d_S coordinates and K lives in d_kv coordinates — the inner product implicit in attention isn't preserved by the basis change.

**Correct fix.** Group the H query heads into G KV groups; for each group g, run streaming PCA on the *concatenated* post-RoPE activations:

`X_g ∈ R^{B·T × (H/G + 2) · d_h}` where `X_g = [Q_{g,1}; …; Q_{g,H/G}; K_g; V_g]`

The eigenvectors of `cov(X_g)` form a single orthonormal basis `U_g`. Slice `U_g` into per-head sub-bases:

- `Q_{g,h}` uses `U_g[((h-1)·d_h):(h·d_h), :]` for `h = 1..H/G`
- `K_g` uses `U_g[(H/G · d_h):((H/G+1)·d_h), :]`
- `V_g` uses `U_g[((H/G+1)·d_h):, :]`

RoPE commutes with the basis because RoPE is applied per-head-pair *inside* the head, after projection — the basis change rotates between heads but doesn't mix the within-head pair structure that RoPE relies on.

**Implementation outline.**

1. Extend [`fasd/profiling/activation_capture.py`](../fasd/profiling/activation_capture.py) with a `GQACaptureEngine` that knows about KV groups and captures the concatenated post-RoPE activations.
2. New module `fasd/profiling/gqa_basis.py` running joint PCA per group and slicing.
3. Update [`fasd/builders.py`](../fasd/builders.py) `_build_llama` to use the shared basis: `V_q[g, h] = U_g[head_h_slice, :s_d_h]`, `V_k[g] = U_g[k_slice, :s_d_h]`, `V_v[g] = U_g[v_slice, :s_d_h]`.
4. Tests: synthetic GQA model where Q · K^T has known structure; verify the projected attention scores match the teacher's within tolerance.

**Why deferred.** The integration touches three files and requires careful RoPE/basis testing. Estimated 3-5 days for a clean implementation. Without this, FSD on Llama-3.2 will work but will have a known disjoint-basis bug that suppresses the headline number by ~3-7% PPL (per the architecture review). It's worth fixing for the main paper but not blocking for the smoke tests.

## Decision gates checklist

The plan defines four hard gates. Each one needs to pass before proceeding to the next sprint.

- [ ] **Gate A (Sprint 1)** — RR-Norm + PCA-rotation V_r matches channel-select V_r at init AND final PPL, on GPT-2 dev. Fallback: keep channel-select V_r; Pillar 1 narrative shrinks to "γ-fold + norm-clone warmup" (still publishable as a simpler version of the paper).
- [ ] **Gate B (Sprint 3)** — Fisher score beats KL-patch and variance scoring, on GPT-2 dev across 3 seeds. Fallback: ship KL-patch as default; reframe novelty around exact allocation + Pillars 1/2/3.
- [ ] **Gate C (Sprint 5)** — Trainable Stiefel beats fixed-PRA-correctly by ≥0.3 PPL across 3 seeds. Fallback: drop trainable bases from main paper; ship PRA-fixed (with embedding optimizer-state correction at [reabsorb.py:317-327](../fasd/training/reabsorb.py#L317-L327)). Paper still has Pillars 1 + 3 + allocator + adaptive objective.
- [ ] **Gate D (Sprint 7)** — FSD beats ≥2 in-house baselines on harness-avg at the 10B-token rung on Llama-3.2-3B → 1B. Fallback: data-efficiency-only framing ("Sheared-Llama-class quality at 5-10× fewer tokens"), or methodology paper at ICLR 2027.

## Reproducibility

- Lock model versions: pin `transformers==5.7.0`, `torch==2.11.0+cu130`, datasets revision.
- Lock corpus subset: store the exact list of `(stream_offset, num_tokens)` slices used for the 5B/10B/20B rungs.
- Lock optimizer hyperparameters: same AdamW settings across all in-house baselines (specified in `_common.py`).
- Lock evaluation: pin `lm-evaluation-harness` commit; report task version numbers in the eval JSON.
- Three seeds for headline cells (Llama-3.2 main table); 1 seed for ablation cells.

## Submission target

NeurIPS 2026 main, ICLR 2027 fallback. The plan's calendar:
- Now (week 1): smoke + GPT-2 ablation + Sprint 0-2 integration TODOs.
- Weeks 2-5: Llama-3.2-3B → 1B at 10B tokens, 3 seeds, all baselines.
- Weeks 6-8: 5B and 20B token rungs (data-efficiency curve).
- Weeks 9-10: ablation grid at 2B-token reduced budget.
- Weeks 11-12: Qwen-2.5 confirmation, paper writing, error bars, rebuttal-prep extras.
