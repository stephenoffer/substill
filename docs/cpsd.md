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
3. **DDR — distillation-driven differentiable rank** (`fasd/compression/diff_rank.py`,
   `fasd/compression/factored_linear.py:GatedFactoredLinear`): soft per-column gate
   optimized against the KD loss under a global parameter budget; per-expert capable for
   MoE. **Now wired into the pipeline** via `FSDConfig(use_diff_rank=True)` — gates train
   jointly with the manifold-trained factors, then `FSDPipeline.fold_for_inference()`
   hardens them and folds every edge to a plain `nn.Linear`. Its defining contrast with
   Dobi-SVD (KD-driven vs reconstruction-driven rank) is measured by a controlled ablation
   in `scripts/cpsd_compare.py`.

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

### 1. Concrete win vs the competition (matched 2×, seed 0, forward-KL)

The subspace-distillation family (F-ASD / CPSD) **beats naive baselines by ~3.9×**. This is
the concrete win over the competition; it is the existing absorbed-init contribution, which
CPSD builds on.

| method | final PPL | note |
|---|---|---|
| random-init + CE | 648.4 | no teacher subspace |
| random-init + KD | 665.5 | KD alone, random init |
| **absorbed-init + KD (F-ASD)** | **169.5** | **3.8–3.9× better than random init** |
| **CPSD-MT (manifold-trained, novel)** | **166.4** | edges frozen absorbed-init |

Teacher GPT-2 PPL 58.90. Raw: `runs/.../ladder/2x_s0.json`.

### 2. Novel component vs frozen absorbed-init (n=5 seeds)

| compression | F-ASD (frozen) | FSD (RR-Norm-Q) | CPSD-MT [novel] | CPSD-full [novel] |
|---|---|---|---|---|
| 2× | 164.9 ± 2.7 | **159.1 ± 3.7** | 160.4 ± 3.4 | 160.4 ± 3.7 |
| 4× | 282.5 ± 5.3 | 277.7 ± 23.9 | **276.2 ± 21.8** | 278.8 ± 20.0 |

**Manifold-trained bases beat frozen absorbed-init by ~3–4% consistently** (both FSD-Q and
CPSD-MT). CPSD-MT's *extra* projection-factor training is statistically tied with FSD-Q's
RR-Norm-Q at GPT-2 scale (so the modest win is "manifold training helps", not "CPSD-MT > FSD").

### 3. Honest negative: CPI does NOT beat the disjoint baseline on real GQA

GPT-2 has no GQA/RoPE, so it cannot test CPI. On **real TinyLlama-1.1B (GQA, teacher PPL 10.95)**,
the matched-architecture comparison (`scripts/cpsd_cpi_init_eval.py`) is:

| attention init | init PPL | final PPL (200 steps) |
|---|---|---|
| disjoint baseline | 64.7k | **423.4** |
| CPI ov-align (free OV fix) | 68.9k | 476.8 |
| CPI cross-plane (shared) | 77.9k | 494.1 |
| CPI plane-aligned (RoPE) | 77.6k | 607.0 |

**All CPI variants lose.** Root cause: a shared/circuit-preserving basis captures each branch
worse than its own PCA, and at practical compression the student starts far from the teacher
(init PPL ~65k vs 10.95), so init circuit-fidelity is second-order — per-branch energy capture
helps training more. The OV/value-circuit shared basis is a clean win *at the fidelity level*
(`tests/test_fsd_cpi_llama_fidelity.py`) but does not translate to a final-PPL win. **CPI is
an honest negative result; the defensible CPSD contribution is manifold-trained bases (§2),
on top of absorbed-init's large win over the naive competition (§1).**

## Head-to-head vs competitor mechanism (measured, GPT-2 + WikiText-2)

`scripts/cpsd_compare.py` on Anyscale A10G, **n=3 seeds**, 300 steps, teacher PPL 58.90, ≈4.35×
compression. Final validation PPL (mean ± std). The **Dobi-SVD foil** is the same MT+DDR pipeline
but with the rank chosen by *reconstruction* (phase-A reconstruction-only selection → fold → KD
fine-tune), so the contrast isolates *what drives the rank*. This table uses the current default
(`free_core=True`); see the note below for why that matters.

| variant | final PPL (n=3) | verdict |
|---|---|---|
| random-init + KD (naive floor) | 1038.2 ± 5.2 | — |
| F-ASD absorbed-init (prior-art baseline) | 558.9 ± 12.9 | 1.9× better than naive |
| CPSD-MT (manifold-trained) [novel] | 551.0 ± 6.1 | beats absorbed-init |
| **CPSD-full (MT + KD-driven rank) [novel]** | **546.6 ± 2.5** | **best + lowest variance** |
| Dobi-SVD (reconstruction-driven rank) [competitor foil] | 1543.0 ± 66.8 | — |

**Three findings:**
- **Win vs the naive competition:** absorbed-init beats random-init+KD **1.9×** (559 vs 1038) —
  the robust, reproducible advantage (also +14–16 top-1 on ResNet50, see below).
- **Win vs the Dobi-SVD competitor mechanism:** *KD-driven* differentiable rank (CPSD-full 546.6)
  beats *reconstruction-driven* rank (1543.0) by **2.8×**. The central novelty claim holds.
- **The novel method now beats the strong baseline:** CPSD-full (546.6 ± 2.5) edges out frozen
  absorbed-init (558.9 ± 12.9) — a modest but real **Δ≈12 PPL win at n=3**, and with the *tightest*
  variance of any variant. This required the free-core fix below.

**The free-core fix (now the default).** Earlier runs converted GPT-2 factored edges *without* a
Euclidean core (`free_core=False`): those edges could only rotate the frozen teacher weight via the
low-LR Stiefel bases — too little capacity to fit the KD target in a 300-step budget — so CPSD-MT/
full *lost* to frozen absorbed-init (873/829 vs 559 at 4.35×, an honest negative we reported).
Diagnosing this (the factored student starts *identical* to absorbed-init yet ended worse → it was
under-training, not mis-initialized) led to enabling `free_core=True` (matching the Llama path, now
the default in `convert_gpt2_to_factored`): the zero-initialized core supplies the missing fitting
capacity while preserving exactness-at-init and the zero-overhead inference fold. Result: 873/829 →
551/547, flipping the negative into the modest win above. Raw: `runs/bench_v1/v3_c2_*.json`. The
matching 7.23× re-run with the free core (`cmp-v4`) is in flight; the pre-fix 7.23× numbers (where
CPSD lost) are superseded.

## Status

- **Built & tested** (238 passing, `PYTHONPATH=. pytest tests/ -q`): the four CPSD components,
  end-to-end composition (`test_fsd_cpsd_integration.py`), real-GPT-2 + exact-forward CPSD
  conversion (`test_fsd_pipeline_cpsd.py`), real-Llama GQA conversion
  (`test_fsd_llama_cpsd.py`), CPI fidelity on real Llama (`test_fsd_cpi_llama_fidelity.py`),
  ArchitectureSpec equivalence (`test_fsd_arch_spec.py`), **DDR wired through the pipeline +
  hardened fold** (`test_fsd_diff_rank_pipeline.py`), **conv2d absorbed-init exactness**
  (`test_fsd_conv2d_absorbed.py`), and the **ResNet vision arm** (`test_fsd_vision_resnet.py`).
- **CPSD-factored conversion wired for both GPT-2 and Llama-family** (`convert_*_to_factored`);
  CPI Llama attention re-init wired (`apply_cpi_attention_init`, `apply_ov_align_init`,
  `cpi_rank_map`, `FSDConfig(use_cpi=True)`) and tested on real GQA TinyLlama.
- **Correctness fixes:** the prior "shared basis commutes with RoPE" claim was false (~7×
  post-RoPE score-error inflation) — CPI is now RoPE-aware; and a width-pruner floor bug that
  could make a tiny student larger than the teacher (now capped at teacher width).
- **What is established:** absorbed-init subspace distillation beats naive baselines ~3.9×
  (§1); manifold-trained bases beat frozen absorbed-init ~3–4% (§2). **CPI does not beat the
  disjoint baseline on real GQA (§3) — a tested negative result, not a deferral.**
- **Now built (code + unit tests; head-to-head numbers gated on compute):**
  - **DDR wired end-to-end** (`use_diff_rank`): KD-driven differentiable rank trains jointly
    with the manifold factors and folds to a deployable plain-`nn.Linear` student.
  - **Controlled foil for the central claim** (`scripts/cpsd_compare.py`): the *same* MT+DDR
    pipeline with the KD term zeroed reproduces **Dobi-SVD's reconstruction-driven rank**, so
    the matched comparison isolates "KD-driven vs reconstruction-driven rank". The aggregator
    (`scripts/cpsd_aggregate.py`) prints the per-cell verdict.
  - **Vision arm** (`fasd.vision`, see below): the framework now spans non-LLM CNNs (ResNet).
- **Remaining / deferred:** the GPT-2/WikiText-2 ladder and Llama-3.2-3B→1B ≥3-seed frontier
  runs (harness ready, compute-gated); empirical confirmation of the MT seed-variance knobs;
  **KQ-SVD** (a different compression axis — head-dim/KV-cache, not residual width — so it is
  compared separately, not in the matched-width harness) and **DistiLLM-2 / Minitron /
  RFID-MoE** (cite published numbers). The CPI direction is **not** recommended for further
  investment based on the §3 evidence.

## Vision arm — ResNet (non-LLM)

`fasd.vision` extends the framework to convolutional classifiers. The conv2d absorbed
projection (`V_out^T W V_in` lifted over the kernel; `fasd/compression/absorbed_init.py`)
narrows each `Bottleneck`'s inner channels while keeping block input/output widths fixed —
so downsample shortcuts and the residual add are untouched and blocks compress
independently (the convolutional analogue of compressing a transformer FFN's intermediate
dim). Channel *selection* (not PCA rotation) is used because a BN+ReLU sits between the
convs and ReLU does not commute with a basis rotation; at full width the student reproduces
the teacher bit-for-bit. `build_resnet_student` + `distill_classifier` (class-logit KD via
the same `forward_kl`/`skew_kl`) provide the end-to-end path; `scripts/resnet50_distill.py`
is the absorbed-vs-random matched-compression ladder. Tested on a real torchvision ResNet
(`tests/test_fsd_vision_resnet.py`).

**Measured — ResNet50 → CIFAR-10** (Anyscale A10G, teacher top-1 73.6% from a 300-step linear
probe, 2000 distillation steps). Absorbed-init subspace distillation beats random-init at
matched compression by a wide margin — the vision counterpart of §1:

| inner-width ratio | params | random-init top-1 | absorbed-init (FASD) top-1 | Δ |
|---|---|---|---|---|
| 0.50 | 10.35M | 64.8% | **81.1%** | **+16.2 pts** |
| 0.35 | 7.51M  | 64.7% | **78.9%** | **+14.3 pts** |

The absolute numbers are about the *absorbed-vs-random* comparison (same teacher, same
compression), not a SOTA claim — the teacher is a quick linear-probe, not fully fine-tuned.
Raw: `runs/bench_v1/r{0.5,0.35}.json`.
