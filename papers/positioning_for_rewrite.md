# Preprint rewrite guide: honest positioning of CPSD

Actionable edits for `papers/fasd_preprint.tex` so the claims survive review. Grounded in
`gap_analysis.md` (what's not novel) and `novel_mechanism.md` (what is). **Do not** claim
component-level novelty; lead with the integrated system.

## 1. Reframe the contribution (abstract + intro)

**Old framing (reject-risk):** "we introduce absorbed init / RR-Norm / Fisher allocation /
skew-KL." Each is prior art (FWSVD, SliceGPT, DistiLLM, GKD).

**New framing:** *We introduce CPSD, the first method to train circuit-preserving
low-rank compression factors end-to-end against a distillation objective, with the
per-edge rank learned jointly with that objective.* The contribution is the **conjunction**:
no prior method trains the factors (KQ-SVD/Eigen Attention freeze the SVD), learns rank
against KD rather than reconstruction (Dobi-SVD/LLRC), or preserves both the QK and OV
circuits weight-side while doing so.

## 2. Add explicit "Relationship to prior work" paragraph (cite, don't claim)

| Component | Cite | One-line delta |
|---|---|---|
| Absorbed/SVD init | FWSVD (2207.00112), SVD-LLM v2 (2503.12340), ESPACE (2410.05437) | we train, not freeze, the factors |
| LayerNorm folding | SliceGPT (2401.15024) | reused as a pre-pass, not claimed |
| QK bilinear bound | KQ-SVD (2512.05916) | we add the OV circuit, weight-side, and as trainable init |
| GQA shared basis | Eigen Attention (2408.05646) | we make it RoPE-correct + manifold-trained |
| Differentiable rank | Dobi-SVD (2502.02723), LLRC (2512.13733) | ours is KD-driven, not reconstruction-driven |
| skew-KL / on-policy | DistiLLM-2 (2503.07067), GKD (2306.13649), MiniLLM (2306.08543) | reused as the objective |
| Per-expert MoE alloc | RFID-MoE (2602.09316) | we extend CPSD into experts; cite their allocation |
| Stiefel/Cayley opt | Wen & Yin (2013) | reused optimizer |

## 3. Demote / fix specific claims

- **Drop** "we invent absorbed initialization" and "RR-Norm is novel."
- **Drop** the "trainable bases generalize PRA" framing unless the experiments
  substantiate it; keep PRA as a **negative result** (the Adam optimizer-state bug
  diagnosis is a genuine, citable contribution).
- **Fix the RoPE claim.** The old text/`gqa_basis.py` asserted the shared basis "commutes
  with RoPE." It does **not** (proven: ~7× post-RoPE score-error inflation,
  `runs/derisk/rope_circuit_basis.py`). State the RoPE-aware (plane-aligned / decoupled)
  construction and that the OV circuit is RoPE-free.

## 4. Headline experiment + baseline table (must-beat)

Report harness-avg vs distillation tokens at fixed parameter budget, with ablations
isolating CPI / MT / DDR and the **conjunction**. Required head-to-heads:

- **Dobi-SVD** (isolates KD+Stiefel rank delta), **KQ-SVD** (OV + weight-side + trained),
  **DistiLLM-2** and **Minitron** (vs fixed/separately-pruned arch), **RFID-MoE** (MoE).
- Full comparison set: SliceGPT, SVD-LLM v2, ESPACE, Basis-Sharing, Eigen Attention,
  Palu, TransMLA, Sheared-LLaMA, MiniLLM, GKD.

**Confirm arXiv IDs before submission:** GKD (2306.13649), DistiLLM-v1 (2402.03425),
Sheared-LLaMA (2310.06694); KQ-SVD/MFA/RFID-MoE code availability; GFWSVD vs FWSVD.

## 5. Honest limitations section

- The win is empirical and conjunction-dependent — single components are anticipated.
- Manifold training costs ~6× train-time compute (0 at inference; measured).
- MoE absorbed-build on fused-expert tensor layouts is implemented at the
  enumeration/profiling level; full build is version-specific (state what was run).
- Frontier (Llama-3.2-3B→1B, ≥3 seeds) requires the deferred H100 runs.
