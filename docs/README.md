# ASD Documentation

Project documentation for Activation Subspace Distillation. Organized around
the two axes most useful to a future reader: (a) how the algorithm works,
and (b) what the experimental matrix revealed about where it succeeds and
fails.

## Map

| File | Audience | Contents |
|---|---|---|
| [algorithm.md](algorithm.md) | Someone learning ASD | Pipeline, math, and the moving parts of the combined loss |
| [findings.md](findings.md) | Someone evaluating the method | Four findings from the 23-cell benchmark matrix, ranked by evidence |
| [v3-improvements.md](v3-improvements.md) | Someone implementing improvements | Proposed algorithmic changes, code diffs, expected impact |
| [results.md](results.md) | Someone compiling the paper | Full tables of every cell's headline numbers + baselines |
| [open-questions.md](open-questions.md) | Someone planning v4 | Deferred work, speculative improvements, known blockers |

## Quick reference — where results and run state live

- Raw results: `/mnt/shared_storage/asd/outputs_jobs/<cell>/` (NFS, not in git).
  See [results.md](results.md) for the cell index.
- Resume-safe submission log: `/mnt/shared_storage/asd/RUN_STATE.md`.
- Consolidated eval write-up: `/mnt/shared_storage/asd/EVALUATION.md`
  (same content as these docs, single-file form).
- Teacher checkpoint: `/mnt/shared_storage/asd/teacher_resnet50_cifar10.pt`
  (ResNet50/CIFAR-10, 96.08% — re-used across every cell).

## One-line summary of the state of the method

ASD works, and the Apr-22 benchmark matrix plus v3 validation produced one
**new best CNN result** (91.87% @ τ=0.85, +0.47pp over AT baseline) and
one **major LLM improvement** (GPT-2 ppl 486 → 131 at τ=0.95, same arch).

Findings:

- **✅ Drop logit-KD + drop sparsity, keep sqrt weighting (CNN).** Gives
  **91.87% @ τ=0.85 on ResNet50/CIFAR-10**, a new best for the method and
  +0.47pp over the AT baseline at the same compression. The ablation data
  (Apr-22) predicted this and the v3 component ablation confirmed it.
- **✅ Use Mahalanobis weighting for transformer distillation.** GPT-2 /
  WikiText-2 perplexity drops 486 → 131 at τ=0.95, same student arch.
  Simple one-line change to `sv_weighting`.
- **❌ Do *not* use Mahalanobis on CNN.** The current implementation
  regresses by 0.5–2.7pp on CIFAR-10 because `w_i ∝ 1/λ_i` over-weights
  near-noise directions when effective_rank is large. Numerical pathology,
  not a theoretical failure — fixable by a stricter in-loss rank cutoff.
  See [findings.md](findings.md) §3 and [open-questions.md](open-questions.md) §1.
- **✅ Mahalanobis alone (with KD still on) is neutral on CNN**. The
  regression comes from combining it with drop-KD. This means the LLM
  finding is real and not an artifact.

The "v3" recipe in [08_ablation.py](../scripts/08_ablation.py) landed
three new variants: `v3`, `v3_mahalanobis_only`, `v3_no_kd_no_spar`. The
third is the actual winner for CNN. On LLM, swap `sv_weighting` to
`mahalanobis` via `09_llm_distill.py --sv-weighting mahalanobis`.
