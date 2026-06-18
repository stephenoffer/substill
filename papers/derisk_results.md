# Phase 0.3 de-risk results

**Date:** 2026-06-17. Prototypes in `runs/derisk/`. All three hard questions for CPSD
were answered decisively before committing build/compute.

## 0.3a — Distillation-driven differentiable rank gate: STABLE + SELECTS CORRECTLY

`runs/derisk/optim_derisk.py` (§0.3a). A soft column gate `g = σ(α)` trained against a KD
(output-matching) loss under a budget penalty.

- **Stability:** gradients finite throughout 400 steps; grad norm decays monotonically
  (4.4e-3 → 5.8e-5). No Taylor stabilization (Dobi-SVD's trick) needed in this setting.
- **Correctness:** `corr(gate, column-importance) = +0.81`; the budget-many open gates
  overlap the truly-important columns **100%**. The KD-driven gate learns to keep the
  high-importance directions and drop the rest under the budget.

**Verdict:** DDR (component 3) is viable. Risk retired for the simple gate; revisit only if
composing with the SVD basis introduces ill-conditioning at scale.

## 0.3b — Frozen-W_T + Stiefel V_in/V_out, route-through-teacher-dim: TRACTABLE BUT COSTLY

`runs/derisk/optim_derisk.py` (§0.3b), representative edge `D=2048, R=1024`, batch 8×512 on
CUDA.

| Form | ms/step | peak MB |
|---|---|---|
| collapsed compressed linear | 1.31 | 161.7 |
| frozen-W_T + Stiefel V (route-through) | 8.46 | 199.5 |

- **Train-time overhead: ~6.4× time, ~1.2× memory.** Inference folds back to the collapsed
  `W_S` → **zero** inference overhead.

**Verdict:** Component 2 (manifold-trained factors) is tractable at GPT-2/TinyLlama scale and
fine for research validation. The 6× train cost is a real consideration for large frontier
runs — mitigations: train bases on a subset of edges only, or use periodic re-absorption for
cheap edges and continuous Stiefel for the circuit-critical (QK/OV) edges. Not a blocker.

## 0.3c — RoPE breaks the shared basis (existing claim is FALSE); fix identified

`runs/derisk/rope_circuit_basis.py`. Relative Frobenius error of attention scores, dense
plane-mixing covariance, `d_h=64`, keep 32:

| scheme | no-RoPE resid | with-RoPE resid |
|---|---|---|
| full-rank shared (sanity) | 9.0e-7 | 8.5e-7 |
| arbitrary-PCA truncation | **9.5e-3** | **6.7e-2** |
| plane-aligned truncation | 4.3e-1 | 4.4e-1 |

- **The `gqa_basis.py:33-37` "commutes with RoPE" claim is empirically false:** RoPE inflates
  the arbitrary-PCA shared-basis score error **~7×** (9.5e-3 → 6.7e-2).
- **Plane-aligned truncation commutes with RoPE** (no inflation) but compresses poorly when
  energy lies in plane-mixing directions (0.44 error) — not usable alone.
- **The "no-RoPE" column is the OV/value circuit** (no RoPE there): arbitrary-PCA shared
  subspace works cleanly (9.5e-3). So the **OV-circuit delta is the easy, RoPE-free win**;
  only the **QK circuit** needs RoPE-aware handling.

**Verdict & fix:** the QK half of CPI requires a **decoupled RoPE/NoPE basis** (Palu RoRoPE /
Eigen-Attention style): apply the shared subspace to the RoPE-commuting part, keep/handle RoPE
planes separately. The acceptance test must score **post-RoPE** (extend
`attention_score_residual`, which currently omits RoPE). This is Phase 2.1 work; the OV
circuit can proceed without it.

## Combined readout

All three risks are retired or scoped. The full CPSD system is viable: DDR is stable and
selective, manifold training is tractable (with a known train-cost tax), and the RoPE
correctness issue has a concrete fix (decoupled basis) with the OV circuit as an immediate
RoPE-free win. No finding forces the differentiable-rank-only fallback.
