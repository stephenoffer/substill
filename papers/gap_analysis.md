# Gap analysis: what is actually novel in FASD/FSD, and what is not

**Date:** 2026-06-17
**Purpose:** Establish, honestly and adversarially, which parts of this method are
genuinely unclaimed against the 2024–2026 literature, so the paper leads with a
contribution that survives review rather than one that gets desk-rejected on novelty.

This document is the output of Phase 0.1 of the rebuild plan
(`~/.claude/plans/twinkly-wishing-glacier.md`). It is deliberately self-critical: we
would rather kill a claim here than have a reviewer kill it.

---

## 1. The original six pillars are not novel

The FASD/FSD pillars, taken individually, recombine published techniques:

| Pillar | Closest prior art | Relationship |
|---|---|---|
| Absorbed init `W_S = Vᵀ W_T V` | FWSVD (2207.00112), LoRD, LASER, ESPACE (2410.05437) | Activation/Fisher-subspace weight projection — standard since 2022 |
| γ-fold (fold LayerNorm into linears) | **SliceGPT** (2401.15024) | SliceGPT folds LN into adjacent linears as its core move |
| RR-Norm (isotropic norm + Stiefel Q) | QuaRot (2404.00456), SpinQuant | Orthogonal/rotation transforms of the residual stream |
| Learned Stiefel bases | Cayley retraction (Wen & Yin 2013); orthogonal-constraint nets | Classical manifold optimization applied to bases |
| Block-diagonal residual | Structured-sparsity / LoRA-style corrections | Natural design, not a new idea |
| Fisher score + greedy allocator | FWSVD Fisher importance + textbook knapsack | Established importance + standard optimization |
| Adaptive skew-KL + on-policy | **DistiLLM** (2402.03425) skew-KL; **GKD** (2306.13649) / **MiniLLM** (2306.08543) on-policy | skew-KL used essentially verbatim |

**Conclusion:** the six-pillar framing cannot be the novelty claim. Each pillar must be
cited to its origin in the paper rewrite (Phase 4.3).

---

## 2. Adversarial novelty check on the "new" ideas — all dead standalone

Before committing to a contribution we web-searched (2026-06) for prior work that already
does each candidate idea. Every component-level claim is already published:

| Candidate novelty | Verdict | Killed by |
|---|---|---|
| Joint bilinear `QKᵀ` best-rank factorization + provable attention-score distortion bound | **DEAD** | **KQ-SVD** (arXiv 2512.05916, Dec 2025) — factorizes the K·Q interaction as a single bilinear operator with optimal low-rank factorization and formal attention-fidelity theorems |
| Shared low-rank basis across GQA KV-groups to preserve attention | **DEAD** | **Eigen Attention** (2408.05646) — already shares `Uᴷ` across GQA heads in a layer |
| Differentiable / learned rank (soft truncation) | **DEAD** | **Dobi-SVD** (2502.02723, ICLR'25) — differentiable truncation-value learning; **LLRC** (2512.13733) — differentiable rank via learned masks |
| Per-expert MoE rank allocation fusing routing-frequency × information-density | **DEAD** | **RFID-MoE** (2602.09316) — name and mechanism match exactly; explicitly warns pure-frequency allocation degenerates |

Each prior method's *scope limits* leave a narrow opening (see §3), but none of these can
be claimed as a standalone contribution.

---

## 3. What actually survives: the integrated system

No single paper does the **conjunction** below, and the scope limits of the killers leave
it open:

> **Circuit-Preserving Subspace Distillation (CPSD):** initialize a compressed student
> with a *circuit-preserving* shared-subspace construction (preserving both the QK score
> circuit and the OV value circuit, weight-side), then train *those same factors* on the
> **Stiefel manifold** against a **knowledge-distillation loss**, with per-edge **rank
> differentiable and learned jointly against the KD loss** — extended to per-expert
> allocation for MoE students.

Why the killers do not cover it:

- **KQ-SVD** is training-free SVD, QK-only, and KV-cache-only. It does not train the
  factors, does not touch the **OV circuit**, and is not weight-side compression. Our
  delta: OV circuit + weight-side + factors used as the *initialization for end-to-end
  training*.
- **Eigen Attention** factors Q/K/V/O independently (no bilinear preservation, no bound)
  and freezes the SVD. Our delta: circuit preservation + end-to-end manifold training.
- **Dobi-SVD / LLRC** learn rank against a **reconstruction / perplexity** objective and
  do not train factors on a manifold. Our delta: rank learned against the **KD loss**,
  factors on the Stiefel manifold, jointly.
- **DistiLLM-2 / Minitron** do KD but with fixed (or separately-pruned) architectures —
  no circuit-preserving, manifold-trained, differentiable-rank compressed factors.
  Minitron is strictly *prune-then-distill* (two-stage). Our delta: the compression
  factors and their ranks are trained *by* the distillation loss in one pipeline.

**Lead with the system, never the components.** A reviewer who reads CPI in isolation sees
KQ-SVD; DDR in isolation sees Dobi-SVD; MoE allocation in isolation sees RFID-MoE.
Ablations must isolate the *gain of the conjunction*, not of each part.

The repository is well-positioned for this: absorbed-init, `StiefelAdam`, and skew-KL are
already implemented. The three missing pieces are exactly the three the repo built but left
**unwired** — `factored_linear.py` (trainable bases), `gqa_basis.py` (circuit basis), and a
differentiable rank in place of the frozen PCA/Fisher allocator. The novelty work and the
integration/cleanup work coincide.

---

## 4. Honest defensible contributions (in priority order)

1. **The CPSD system** — circuit-preserving init → Stiefel-manifold factor training against
   KD → differentiable rank, beating two-stage compress-then-finetune at fixed parameter and
   token budgets, and holding at >50–60% compression where SVD/pruning baselines collapse.
2. **Matched-compression ablation methodology** — controlling for capacity confounds when
   comparing distillation/compression components (already advocated in the repo).
3. **Negative result on Periodic Re-Absorption** — the precise Adam optimizer-state bug
   diagnosis is a reproducible, useful negative result.
4. **MoE per-active-parameter results** — applying CPSD inside MoE experts; cite RFID-MoE as
   the closest allocation prior and position as a mechanism extension, not a new allocation
   principle.

---

## 5. Known correctness issue uncovered during this analysis

`fasd/profiling/gqa_basis.py:33-37` claims the shared per-group basis "commutes with RoPE."
This is **mathematically false** for a general basis `V`: RoPE applies a position-dependent
rotation `R(θ, pos)` within each head, and `V` only commutes with all `Rᵢᵀ Rⱼ` if it is
block-diagonal with respect to RoPE's 2D rotation planes. The diagnostic
`attention_score_residual` (line 266) only tests the **no-RoPE** case, so the claim is both
incorrect and unverified. The QK half of CPI is invalid on any RoPE model (Llama/Qwen/
Mistral) until a RoPE-aware basis is implemented (Phase 0.3c / Phase 2.1). This is the same
subtlety that Palu (RoRoPE), TransMLA, and Eigen Attention handle explicitly.

---

## 5b. Empirical refinement of the CPI claim (2026-06-17)

A real-Llama fidelity probe (`tests/test_fsd_cpi_llama_fidelity.py`) refined the CPI claim:
- **OV / value circuit (no RoPE): shared basis is a clean, robust win** — this is the
  strongest CPI delta and the part KQ-SVD does not cover (weight-side OV).
- **QK / score circuit under RoPE: not a clean fidelity win.** The plane-aligned basis
  provably commutes with RoPE, but sacrifices too much energy to beat disjoint cross-plane
  PCA at typical compression. The QK-CPI benefit is unestablished on real models. Lead with
  OV + the manifold/DDR system; treat QK-under-RoPE as an open limitation, not a claim.

## 6. Citations to confirm before paper submission

- GKD arXiv ID (2306.13649), DistiLLM v1 (2402.03425), Sheared-LLaMA (2310.06694) — confirm.
- KQ-SVD (2512.05916), MFA (2412.19255), RFID-MoE (2602.09316) — confirm code availability.
- "GFWSVD" vs canonical FWSVD (2207.00112) — confirm whether distinct.
- SVD-LLM v2 (2503.12340), Dobi-SVD (2502.02723), LLRC (2512.13733), ESPACE (2410.05437),
  Eigen Attention (2408.05646), Palu (2407.21118), TransMLA (2502.07864) — verified in search.
