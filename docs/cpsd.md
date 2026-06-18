# CPSD — Circuit-Preserving Subspace Distillation

CPSD is the novel contribution layered onto the FASD/FSD codebase. It is the **first
method to train circuit-preserving low-rank compression factors end-to-end against a
distillation objective, with the per-edge rank learned jointly with that objective.**

The honest novelty position, the formal mechanism, the de-risk evidence, and the
preprint-rewrite guidance live in `papers/`:
`gap_analysis.md`, `novel_mechanism.md`, `derisk_results.md`, `positioning_for_rewrite.md`.

## The three coupled components

1. **CPI — circuit-preserving init** (`fasd/compression/cpi.py`,
   `fasd/profiling/rope.py`, `fasd/profiling/gqa_basis.py`): shared-subspace init that
   preserves the QK score circuit (RoPE-aware, plane-aligned) and the OV value circuit
   (weight-side, RoPE-free). Distinct from KQ-SVD's operator-SVD.
2. **MT — manifold-trained factors** (`fasd/compression/factored_linear.py:TeacherFactoredLinear`):
   frozen teacher weight + trainable Stiefel `V_in/V_out`; route-through forward, folds to
   a zero-overhead inference linear. Trained via `StiefelAdam`.
3. **DDR — distillation-driven differentiable rank** (`fasd/compression/diff_rank.py`):
   soft per-column gate optimized against the KD loss under a global parameter budget;
   per-expert capable for MoE.

## Usage

```python
import fasd
pipe = fasd.FSDPipeline(teacher, config=fasd.FSDConfig(
    arch_multiplier=0.5, use_cpsd_factored=True, generative_kd="skew_kl",
    total_steps=1000, lr=3e-4))
result = pipe.run(calib_loader, train_loader)   # profile -> build -> convert -> distill
```

Architecture support is declarative (`fasd/arch/`, `docs/adding_architectures.md`):
GPT-2, Llama 3.x/Mistral/Qwen2.5/Qwen3-dense, and Mixtral/Qwen3-MoE (branch enumeration).

## Results (GPT-2 + WikiText-2, A10G, 500 steps)

Final validation PPL (mean ± std, **n=5 seeds**), forward-KL for all variants so the only
difference is the novel component. Teacher GPT-2 PPL 58.90. Ratios are inference (folded).
Raw data: `runs/cpsd_v1/` (`summary.csv`); reproduce with `scripts/cpsd_experiment_submit.sh`.

| compression | F-ASD (absorbed) | FSD (RR-Norm-Q) | CPSD-MT [novel] | CPSD-full [novel] |
|---|---|---|---|---|
| 2× | 164.9 ± 2.7 (2.57×) | **159.1 ± 3.7** (2.34×) | 160.4 ± 3.4 (2.34×) | 160.4 ± 3.7 (1.57×) |
| 4× | 282.5 ± 5.3 (4.33×) | 277.7 ± 23.9 (4.00×) | **276.2 ± 21.8** (4.00×) | 278.8 ± 20.0 (2.83×) |

**Honest read (do not overclaim):**
- **CPSD-MT is statistically tied with the FSD baseline** at both ratios (2×: Δ−1.3 within
  ±3.7; 4×: Δ+1.5 within ±22). It does not clearly win at GPT-2 scale.
- **Manifold training does help:** both Stiefel-trained methods (FSD-Q, CPSD) beat the
  frozen-absorbed F-ASD baseline by ~5 PPL. But CPSD's *extra* projection-factor training
  adds nothing over FSD's RR-Norm-Q-only training here.
- **GPT-2 cannot test CPI** (no GQA, no RoPE), so the headline circuit-preserving component
  never engaged. On a real GQA+RoPE Llama (`tests/test_fsd_cpi_llama_fidelity.py`): the
  **OV/value-circuit shared basis is a clean win**, but the **QK-circuit under RoPE is not** —
  the RoPE-aware plane-aligned basis provably commutes with RoPE yet sacrifices too much
  energy to beat disjoint PCA at typical compression. The QK-CPI benefit is unestablished.
- The Stiefel-trained methods are far more seed-sensitive (±22 at 4×) than F-ASD (±5).

## Status

- **Built & tested** (249 tests, `PYTHONPATH=. pytest tests/ -q`): the four CPSD components,
  end-to-end composition (`test_fsd_cpsd_integration.py`), real-GPT-2 + exact-forward CPSD
  conversion (`test_fsd_pipeline_cpsd.py`), real-Llama GQA conversion
  (`test_fsd_llama_cpsd.py`), CPI fidelity on real Llama (`test_fsd_cpi_llama_fidelity.py`),
  and ArchitectureSpec equivalence (`test_fsd_arch_spec.py`).
- **CPSD-factored conversion wired for both GPT-2 and Llama-family** (`convert_*_to_factored`).
- **Correctness fixes:** the prior "shared basis commutes with RoPE" claim was false (~7×
  post-RoPE score-error inflation) — CPI is now RoPE-aware; and a width-pruner floor bug that
  could make a tiny student larger than the teacher (now capped at teacher width).
- **Remaining / deferred:** wiring CPI's GQA+RoPE shared basis into `_build_llama` (fixes the
  disjoint-basis bug at `builders.py:483-485` — the decisive integration for a Llama win);
  fused-tensor MoE absorbed-*build* (enumeration done); and the Llama-3.2-3B→1B ≥3-seed
  frontier vs Dobi-SVD/KQ-SVD/DistiLLM-2/Minitron/RFID-MoE. **Whether CPSD beats SOTA is
  unproven** — GPT-2 shows it is competitive (tied), not winning.
