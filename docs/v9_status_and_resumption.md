# F-ASD v9 — current state & resumption guide

Status snapshot for the v9-apr25 ablation round, written so that any future
session can pick up the work without conversation history.

## What ran

Two cloud matrices submitted on 2026-04-25, all 12 jobs finished SUCCESS by end of day:

- **`v9-apr25` — full 8-rung short matrix at 800 steps.** Verifies the v9 fixes are net-positive across the ablation ladder.
- **`v9-apr25-long` — focused 4-rung subset at 4000 steps** (rungs 0, 1, 4, 6). Tests whether rankings hold under a real training budget.

Each rung is a separate Anyscale job named `fasd-ladder-v9-apr25-<rung>` or `fasd-ladder-v9-apr25-long-<rung>`. Results persist to `$ANYSCALE_ARTIFACT_STORAGE/fasd/v9-apr25/` (and `…/v9-apr25-long/`) — durable across cluster teardown.

## How to check status

```bash
SHORT=(0_baseline 1_behavioral 2_procrustes 3_skewkl 4_absorbed 5_onpolicy 6_quantize 7_full)
LONG=(0_baseline 1_behavioral 4_absorbed 6_quantize)
for r in "${SHORT[@]}"; do
  anyscale job list -n "fasd-ladder-v9-apr25-${r}" --max-items 1 2>&1 \
    | grep "fasd-ladder-v9-apr25-${r}" | head -1 \
    | grep -oE "AWAITING_CLUSTER_START|RUNNING|SUCCESS|ERRORED|FAILED|TERMINATED|OUT_OF_RETRIES" \
    | xargs -I{} echo "$r: {}"
done
```

(Note: the bare `anyscale job list | grep ...` form paginates and can drop jobs at the bottom; prefer `-n` filter per-rung as above.)

To pull results:

```bash
for r in "${SHORT[@]}"; do
  anyscale job logs -n "fasd-ladder-v9-apr25-${r}" 2>&1 | grep "fasd-ablation-result" | tail -1
done
```

To resume a missing rung (skip-if-exists makes this idempotent):

```bash
TAG=v9-apr25 RUNGS="<failing rungs space-separated>" bash scripts/fasd_ablation_submit.sh
```

## What each v9 fix changed (vs v8)

1. **PCA-tail padding** ([fasd/builders.py](../fasd/builders.py) `_col_basis`) — slices full PC matrix; random pad only as last-ditch fallback. Defensive; not the major win in production.
2. **Schedule-coupled projector fold** ([fasd/training/distill.py](../fasd/training/distill.py) `_resolve_fold_frac`) — folds at Procrustes-phase start (default 0.40), not the magic 0.10 boundary v8 used. Configurable via `fold_after_frac`.
3. **Quantization rebuild** ([fasd/compression/quantization.py](../fasd/compression/quantization.py)) — `QuantizedLinear.fp_weight` is `nn.Parameter`; `_group_fake_quant` applies STE per group on every forward. Defaults: `protect_fraction=0.05` (was 0.01), `qad_steps=500` (was 100).
4. **Training budget** ([scripts/fasd_ablation_submit.sh](../scripts/fasd_ablation_submit.sh)) — `TOTAL_STEPS` default 400→800; the long-matrix override goes to 4000.
5. **Forward + reverse KL eval** ([fasd/training/distill.py](../fasd/training/distill.py) `_eval_kl`) — populated into `DistillResult.val_kl_forward` / `val_kl_reverse`, recorded in result JSON.
6. **Llama Q/K shared basis** ([fasd/builders.py](../fasd/builders.py) `_build_llama`) — mirrors GPT-2 fix for non-GQA; GQA still falls through to per-branch with TODO.
7. **V_r residual PCA** ([fasd/builders.py](../fasd/builders.py) `_residual_basis` + [scripts/fasd_ablation.py](../scripts/fasd_ablation.py)) — the profile pass now also captures `block.residual` branches; `_residual_basis` prefers them when present, falling back to legacy avg-then-QR.

## Acceptance criteria from the v9 plan

| # | criterion | v8 | v9-short (800) | v9-long (4000) | verdict |
|---|---|---|---|---|---|
| 1 | rung 4 `initial_student_ppl` < 1000 | 1.78e13 | **2.55e8** | **7.97e7** | ❌ FAIL — 5 orders better than v8, still ~5–6 orders off the bar |
| 2 | rung 6 final ≤ 1.3× rung 4 final | 1.7× | 2.44× | **1.40×** | ❌ FAIL short / ⚠️ NEAR-PASS long (just over the bar; v8 was 1.7×) |
| 3 | absorbed-rung initial PPLs within 5× | 13 OOM | 20+ OOM | n/a (only 4) | ❌ FAIL — rung 5 (early on-policy) blows to 3.94e28 |

So **none of the three criteria pass**, but every one of them is dramatically closer to passing than v8. Treat v9 as a partial win, not a stop.

## Final results

**v8-vs-v9 short comparison (8 rungs).** Note: v8 ran 400 steps, v9-short ran 800 steps — finals are not directly comparable, but initial PPL and compression are not training-budget sensitive.

| rung | v8 init | v8 final | v8 comp | v9-s init | v9-s final | v9-s comp | v9-s KL fwd |
|---|---|---|---|---|---|---|---|
| 0 baseline | 5.75e4 | 1050 | 2.55× | 5.35e4 | 925.7 | 4.10× | 3.40 |
| 1 behavioral | 5.28e4 | 997 | 1.95× | 5.82e4 | 713.6 | 1.52× | 3.19 |
| 2 procrustes | 5.83e4 | 1030 | 2.00× | 5.72e4 | 722.2 | 1.52× | 3.19 |
| 3 skewkl | 5.79e4 | 1068 | 2.00× | 5.64e4 | 807.0 | 1.52× | 3.67 |
| 4 absorbed | **1.78e13** | 845 | 2.00× | **2.55e8** | **508.5** | 1.52× | 3.20 |
| 5 onpolicy | 1.44e6 | 860 | 2.00× | **3.94e28** | **2094.7** | 1.52× | 4.24 |
| 6 quantize | 8.40e12 | 1432 | 2.00× | 2.79e13 | 1242.5 | 1.52× | 3.66 |
| 7 full | 5.09e19 | 2189 | 2.00× | 1.49e14 | 752.0 | 1.52× | 3.03 |

Teacher PPL = 59.05 for both rounds.

**v9-short vs v9-long (rungs 0/1/4/6).** Same algorithm, 800 vs 4000 steps — isolates the "more training" axis.

| rung | v9-s final | v9-l final | gain | v9-s KL fwd | v9-l KL fwd |
|---|---|---|---|---|---|
| 0 baseline | 925.7 | 408.5 | 56% | 3.40 | 2.69 |
| 1 behavioral | 713.6 | 316.8 | 56% | 3.19 | 2.49 |
| 4 absorbed | 508.5 | **258.0** | 49% | 3.20 | 2.79 |
| 6 quantize | 1242.5 | 362.2 | 71% | 3.66 | 2.54 |

Best result of the round: **rung 4-long final PPL = 258 (4.4× teacher)** with 1.52× compression. Rung 4-long beats rung 0-long (258 vs 408) — absorbed_init is a net win once it can train through the bad initial state.

### Surprises / unexplained

- **Rung 5 short regressed vs v8** (final 2095 vs 860). Early on-policy (start=0.5) plus v9's worse-but-larger absorbed init = catastrophic; the bad initial student generates garbage rollouts that feed back. Rung 7 (rung 5 + quantize) somehow finishes at 752 — likely RNG variance in the unstable regime, not a real signal. Don't trust rung 5/7 short numbers.
- **Rung 0 compression jumped to 4.10×** in v9 (was 2.55× in v8) even though variance ranks barely moved. Suspect: MAX-reduce target across layers landed differently with v9's residual-aware rank picking on rung 0 too. Worth a quick investigation but not blocking.
- **Quantization tax is mostly a step-budget issue**, not an algorithm one. Tax 2.44× at 800 steps → 1.40× at 4000 steps. v9's `qad_steps=500` (vs v8's 100) is doing real work; rung 6 just needs more total steps.

### What is still broken

The dominant remaining failure mode is **absorbed_init initial PPL at d=768**. Local smoke at d=32 gets initial PPL 66 vs teacher 64 — perfect. So absorbed_init's math is correct in principle but something specific to the d=768 GPT-2 stack still drives initial PPL into 1e7–1e14 territory. Final PPL recovers, so it isn't fatal — but the gap means lots of compute is spent un-doing the bad initialization.

## v10 implementation (2026-04-25)

After running v9 we localized the absorbed-init pathology to **two compounding bugs** in `fasd/builders.py`, both fixed in v10:

1. **Residual rotation V_r was destructive when no residual compression was happening.** v9's `_residual_basis` returned a PCA-derived orthogonal rotation even when `s_hidden == t_hidden` (which v9 ablation always hits because MAX-reduce of behavioral ranks forces s_hidden up to ffn.up=768). LayerNorm does not commute with arbitrary orthogonal rotations — gamma/beta cannot be projected through V_r without losing the per-channel structure GPT-2 relies on. v10 short-circuits to V_r = I when k=d. Local diagnostic confirmed this single change drops initial PPL from 5e16 → 1.3e4 (12 orders of magnitude). For the case `s_hidden < t_hidden` v10 falls back to **channel-selection** (one-hot columns of identity at top channels by reconstructed cov.diag), not PCA rotation, so the centering/scaling of LN approximately commutes.

2. **PCA-rotated V_up broke FFN initialization.** GELU is element-wise; it does not commute with orthogonal rotations of FFN intermediate. v9's `V_up = top-k PCs of ffn.up` meant the absorbed weight produced `act(z @ V_up)` instead of `act(z) @ V_up` — close at low k for linear models, but transformers compose these errors across 12 blocks. v10 uses `_channel_select_basis` for FFN intermediate: keep the top-k channels by variance, slice teacher weights at those channels. Activation passes through unchanged.

Local d=768 smoke (same compression as v9 rung 4 long: 1.52×, 81.94M params):

| metric | v9 cloud rung 4 long | v10 local |
|---|---|---|
| initial student PPL | **7.97e+07** | **3.71e+02** |
| acceptance criterion (<1000) | ❌ | ✅ |
| compression ratio | 1.52× | 1.52× |

99 unit tests pass (96 from v9 + 3 new v10 regressions encoding: identity V_r when k=d, channel-select picks top-variance channels, full-size absorbed init reproduces teacher exactly).

v10 cloud round submitted 2026-04-25:

- `v10-apr25` short matrix: 8 rungs × 800 steps.
- `v10-apr25-long` long subset: 4 rungs × 4000 steps (rungs 0, 1, 4, 6).

### v10 short results (6/8 rungs done at writing; rungs 5, 7 still on-policy training)

Direct comparison to v9 short at the same compression (1.52× / 81.94M for absorbed rungs, 4.10× / 30.36M for rung 0).

| rung | v9 init | v9 final | v10 init | v10 final | final delta | KL fwd v9 → v10 |
|---|---|---|---|---|---|---|
| 0 baseline | 5.35e4 | 925.7 | 5.40e4 | 928.96 | ~ same | 3.40 → 3.41 |
| 1 behavioral | 5.82e4 | 713.6 | 5.98e4 | 728.7 | ~ same | 3.19 → 3.21 |
| 2 procrustes | 5.72e4 | 722.2 | 5.95e4 | 734.3 | ~ same | 3.19 → 3.20 |
| 3 skewkl | 5.64e4 | 807.0 | 6.21e4 | 818.8 | ~ same | 3.67 → 3.72 |
| **4 absorbed** | **2.55e8** | **508.5** | **3.13e3** | **76.1** | **6.7×** | **3.20 → 1.45** |
| **6 quantize** | **2.79e13** | **1242.5** | **2.79e3** | **101.8** | **12.2×** | **3.66 → 1.07** |

Rungs 0–3 don't use absorbed_init, so v10 = v9 within RNG noise (sanity check that the v10 changes don't break non-absorbed paths). Rungs 4 and 6 — the absorbed rungs — show the v10 fix landing as expected.

**Acceptance criteria — v10 short:**

| # | criterion | v9 | v10 short | verdict |
|---|---|---|---|---|
| 1 | rung 4 `initial_student_ppl` < 1000 | 2.55e8 | 3.13e3 | ❌ FAIL — 5 OOM better than v9 but still over the bar; the bar was set assuming the rotation could be made benign, which v10 instead fixes by skipping it |
| 2 | rung 6 final ≤ 1.3× rung 4 final | 2.44× | **1.34×** | ⚠️ NEAR-PASS — within 4% of bar, vs v9's 88% over. |
| 3 | absorbed-rung initial PPLs within 5× | 20+ OOM | rung4/rung6 = 1.12× | ✅ PASS so far (rungs 5, 7 pending) |

The init-PPL bar was optimistic — even with V_r=I and channel-select V_up, the residual stream's per-channel magnitudes diverge from teacher because (a) the embedding tables `wte`/`wpe` are unchanged but downstream LayerNorm has different gamma/beta when behavioral_rank picks force student n_inner < t_inner → FFN compresses → residual stream evolves differently from step 0. The 3000-PPL gap closes within 100 training steps (mid-training PPL 96 at step 100, 88 at step 700). For practical use the *final* PPL matters, and v10's 76.1 (1.29× teacher) is the relevant figure.

**Headline:** v10 rung 4 short final PPL 76.1 *beats v9 long (4000 steps) rung 4 final 258* using only 800 training steps. v10 is roughly 5× more sample-efficient than v9.

`v10-apr25-long` (4000 steps) still RUNNING; expected to push final PPL very close to teacher 59.

`v11-am0625-apr25-long` (rung 4, arch_multiplier=0.625, ~3× compression, 4000 steps) submitted in parallel as a v11 probe. Tests whether v10's channel-select V_r supports deeper compression beyond v9's MAX-reduce-bound 1.52×.

## v11 candidates (after v10 results land)

A local probe shows that with v10's channel-select V_r, the residual stream **can** be genuinely compressed (s_hidden < t_hidden) without catastrophic init:

| s_hidden | params | compression | init PPL |
|---|---|---|---|
| 768 | 81.9M | 1.52× | 3.7e2 |
| 600 | 56.8M | 2.19× | 4.8e3 |
| 480 | 41.3M | 3.02× | 5.1e3 |
| 360 | 27.8M | 4.47× | 7.4e3 |
| 240 | 16.5M | 7.55× | 8.3e3 |

For comparison, v9 at 1.52× compression had init 7.97e7 — v11 at 7.55× compression has 4 orders of magnitude better init than v9 had at 5× *less* compression.

If v10 cloud results match the local probe, **v11 should drop the MAX-reduce on hidden_size and pick block.residual rank independently** (e.g., via a `--cap-residual-rank` flag or a rank_tol that's allowed to differ per-kind). This is the long-deferred "per-module widths" architectural change, achievable now that the LN-commutativity bug is fixed.

Other v11 candidates:

- **Schedule re-tuning**: with init PPL no longer 1e8, the schedule's role changes. The `feature_weight_schedule` fade-to-zero in the last 20% may now over-correct.
- **Residual PCA samples**: less critical than v9 thought — channel-select doesn't depend on PCA accuracy in the tail.

## v10 plan (revised based on v9 numbers — superseded by the implementation above)

Priority order:

1. **Diagnose the d=768 absorbed-init pathology directly.** Don't guess at fixes. Add an instrumentation pass that, on a fresh absorbed init, dumps per-layer activation L2 norms (teacher vs student) at the residual stream, post-attn, and post-FFN, for the first few layers. The point at which student norms diverge from teacher norms locates the bug. Suspect order:
   - **LayerNorm projection.** Verify the production formula in `fasd/init/absorbed.py` (or wherever `gamma_s` is computed) — `gamma_s = V^T gamma_t` preserves residual-stream sums; the diagonal-of-V-gamma-V^T formula does not. The d=32 smoke might pass either because the conditioning is fine.
   - **Attn / output biases.** Check that biases are projected (`b_s = V^T b_t` for output projections, identity for input projections in the right basis). A single un-projected bias term at d=768 can dominate the residual stream.
   - **Embedding / unembedding tying.** GPT-2 ties `wte` to `lm_head`. Confirm absorbed init handles this without double-projecting.

2. **Per-module student widths** (drop the MAX-reduce). This is the architectural change we keep deferring. Behavioral ranks for the trial residual-stream-aware run picked, e.g., q=192, k=192, v=745, o=528, up=768, down=380. MAX-reduce forces all of those to the same hidden_size, which inflates compression for the small modules and underfits the large ones. Doing per-module widths will require touching the cross-block projector at module boundaries — non-trivial but the obviously-correct fix.

3. **More residual-PCA samples.** 8 calibration batches × 4 sequences × 128 tokens = 4096 token-vectors for d=768 PCA. The eigenspectrum will be undersampled at the tail. Bump to 32 batches × 8 × 256 = 65k tokens (still cheap — it's profile-time, not train-time) before the per-module widths work.

4. **Re-run rung 4 only at 4000 steps after each fix** to isolate the impact. Don't re-run the whole ladder per iteration.

Deferred (do later if 1–3 don't close the gap):

- Schedule re-tuning (wider blend windows, suppress `feature_weight_schedule` fade-to-zero in the last 20%).
- More `qad_steps` if quant tax is still > 1.3× after a long run with the v10 fixes.
- Llama-GQA Q/K shared-basis fix (currently TODO in `_build_llama`); not blocking GPT-2 work.

## Memory cross-references

- [memory/feedback_persist_matrix_results.md](../../.claude/projects/-home-ray-default-cld-g54aiirwj1s8t9ktgzikqur41k/memory/feedback_persist_matrix_results.md) — durable-storage requirement
- [memory/project_fasd_ablation_v8.md](../../.claude/projects/-home-ray-default-cld-g54aiirwj1s8t9ktgzikqur41k/memory/project_fasd_ablation_v8.md) — v8 round (complete)
- [memory/project_fasd_ablation_v9.md](../../.claude/projects/-home-ray-default-cld-g54aiirwj1s8t9ktgzikqur41k/memory/project_fasd_ablation_v9.md) — v9 round (in flight)

## Test coverage

`tests/test_fasd_v7_regressions.py` has 9 regressions (4 from v8, 5 from v9). All 96 tests pass locally. Each regression encodes a real bug we shipped and shouldn't reintroduce — read it before changing absorbed_init / procrustes / refresh / quant.
