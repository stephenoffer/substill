# Activation Subspace Distillation (ASD)

A knowledge distillation method that exploits the low-rank structure of neural
network activations to auto-size a compact student and train it to match the
teacher's principal activation subspace.

## Core Idea

Deep networks tend to have low-rank, sparse per-layer activations — a handful
of principal components capture most of the signal. ASD exploits this by:

1. **Profiling** the teacher's per-pixel or GAP activation covariance at every
   residual/backbone block via eigendecomposition.
2. **Auto-sizing** a student whose per-stage channel widths equal the teacher's
   effective activation rank at each stage (not the full channel count).
3. **Training** with a combined loss that matches the student to the teacher's
   principal subspace (in spatial or GAP form), aligns sparsity patterns,
   distills logits (Hinton KD), and optionally adds relational and
   attention-transfer terms.

## What's In This Repo

- **Core ASD pipeline** — profiling, auto-sized student, combined loss, trainer.
- **Baseline-comparison losses** — logit KD, attention transfer (AT), relational
  KD (RKD), wired into `ASDLoss` so they can be ablated or combined.
- **Teacher wrappers** for ResNet-{18,34,50,101}, MobileNetV2, VGG16-BN, and
  DenseNet-121.
- **Datasets**: CIFAR-10, CIFAR-100, SVHN for vision; WikiText-2 for the
  GPT-2 experiment.
- **Benchmark runners** (`scripts/06`–`14`) for sweeps, ablations, dense Pareto
  curves, multi-seed runs, cross-architecture evaluation, LLM distillation,
  and v1-vs-v2 algorithm comparisons.
- **Aggregation / plotting scripts** that turn raw JSON results into a paper-
  ready master table and plots.

## Pipeline

```
Phase 0: Fine-tune teacher on the target dataset (adapts CIFAR stem + classifier)
Phase 1: Capture per-block activation covariance → eigendecompose
         → effective rank per layer → sparsity stats
Phase 2: Build student with per-stage widths = effective rank (rounded to
         multiple of 8)
Phase 3: Train student with ASDLoss: task CE + subspace match + sparsity
         pattern + logit KD (+ optional RKD / attention)
```

## Quick Start

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# End-to-end on ResNet50 / CIFAR-10
python scripts/00_finetune_teacher.py
python scripts/01_profile_teacher.py
python scripts/02_build_student.py
python scripts/03_train_asd.py
python scripts/04_evaluate.py
python scripts/05_visualize.py
```

Most benchmark scripts accept `--model`, `--dataset`, `--thresholds`, and
`--output-dir` flags; see each file's docstring for specifics.

## Reproducing the Full Benchmark Matrix

```bash
# Main matrix: ablations + ResNet-{18,34,50,101} × CIFAR-{10,100} + GPT-2/WikiText-2
bash scripts/run_matrix.sh

# Extension: MobileNetV2, VGG16-BN, SVHN, dense Pareto sweep, multi-seed stability
bash scripts/run_matrix_extended.sh

# Aggregate everything into paper tables / plots
python scripts/10_aggregate.py
```

Tuning knobs: `EPOCHS`, `FT_EPOCHS`, `THRESHOLDS_CNN`, `ABL_EPOCHS`,
`ABL_THRESHOLD` as environment variables.

## Project Structure

```
neural_distill/
├── config/default.yaml             # All hyperparameters
├── asd/
│   ├── profiling/
│   │   ├── activation_capture.py   # Hook-based covariance accumulation
│   │   ├── svd_analysis.py         # Eigendecomposition + effective rank
│   │   │                           # (variance / stable / participation / entropy)
│   │   └── sparsity_analysis.py    # Activation histogram + sparsity statistics
│   ├── models/
│   │   ├── teacher.py              # ResNet wrappers (18/34/50/101)
│   │   ├── teacher_ext.py          # MobileNetV2, VGG16-BN, DenseNet-121 wrappers
│   │   ├── student.py              # SlimNet (auto-sized from activation rank)
│   │   └── projectors.py           # Student → teacher subspace projectors
│   ├── losses/
│   │   ├── subspace_loss.py        # MSE in principal subspace (spatial or GAP)
│   │   ├── sparsity_loss.py        # Soft-histogram KL + sparsity ratio loss
│   │   ├── attention_loss.py       # Attention Transfer (Zagoruyko & Komodakis)
│   │   ├── relation_loss.py        # Relational KD (Park et al. 2019)
│   │   └── combined_loss.py        # Weighted sum + Hinton KD + warmup scheduling
│   ├── training/
│   │   ├── trainer.py              # Training loop (grad clip, LR warmup + cosine)
│   │   └── scheduler.py            # β-warmup and γ-warmup for loss weights
│   ├── data/cifar10.py             # CIFAR-10/100, SVHN loaders + calibration subset
│   └── utils/visualization.py      # Spectrum / compression / curve plots
├── scripts/
│   ├── 00_finetune_teacher.py      # Fine-tune teacher on target dataset
│   ├── 01_profile_teacher.py       # Activation profiling
│   ├── 02_build_student.py         # Build + inspect student architecture
│   ├── 03_train_asd.py             # Single-config ASD training
│   ├── 04_evaluate.py              # Teacher vs student comparison
│   ├── 05_visualize.py             # Plots for a single run
│   ├── 06_sweep.py                 # Variance-threshold sweep (ResNet50/CIFAR-10)
│   ├── 07_bench.py                 # Generic (model, dataset) benchmark runner
│   ├── 08_ablation.py              # Ablation runner (10 variants)
│   ├── 09_llm_distill.py           # ASD on GPT-2 / WikiText-2
│   ├── 10_aggregate.py             # Collect all results → paper tables
│   ├── 11_compare_sweeps.py        # Baseline vs improved sweep plot
│   ├── 12_dense_sweep.py           # Dense threshold + multi-seed sweep
│   ├── 13_bench_ext.py             # Bench runner for non-ResNet teachers
│   ├── 14_v2_compare.py            # v1 vs v2 algorithm A/B comparison
│   ├── run_matrix.sh               # Main paper-matrix orchestration
│   └── run_matrix_extended.sh      # Extended matrix (non-ResNet, SVHN, dense)
└── tests/                          # Unit + integration tests
```

## How It Works

### Phase 1: Activation Profiling

Forward hooks on every teacher block accumulate per-layer channel covariance
in `O(C²)` memory (not raw activations):

```
Cov(layer) = E[x · xᵀ] - E[x] · E[x]ᵀ
```

`x` can be either GAP-pooled features (`covariance_mode: gap`) or raw per-pixel
samples (`covariance_mode: per_pixel`, default) for a spatial-aware covariance.

Effective rank per layer is then derived from the eigenspectrum using one of
four definitions configurable via `profiling.rank_definition`:

- `variance` (default): smallest `k` with cumulative variance ≥ threshold.
- `stable`: `⌈Σ λᵢ / λ_max⌉`. Dimensionless, no threshold.
- `participation`: `⌈(Σ λᵢ)² / Σ λᵢ²⌉`. Robust to long tails.
- `entropy`: `⌈exp(H(p))⌉` with `pᵢ = λᵢ / Σ λ`.

### Phase 2: Student Construction

`SlimNet` mirrors the teacher's 4-stage layout but each stage's channel width
equals the teacher's effective rank at that stage (rounded to a multiple of 8
and floored at `student.min_width`):

```
Teacher ResNet50: [256, 512, 1024, 2048]
Student (τ=0.95): [ 48,  96,  160,  320]   # example — derived from data
```

### Phase 3: ASD Training

```
L = α·L_task + β(t)·L_subspace + γ(t)·L_sparsity + δ·L_logitKD
        + ε·L_relation + ζ·L_attention
```

- **L_task** — cross-entropy on ground-truth labels.
- **L_subspace** — MSE between student's projected features and teacher's
  principal-subspace projections. Modes: `spatial` (default), `gap`,
  `cosine_spatial`. SV weighting options: `uniform`, `linear`, `sqrt`.
- **L_sparsity** — KL divergence between differentiable soft histograms plus
  a BCE or MSE term on sparsity ratios; adaptive bin ranges.
- **L_logitKD** — standard Hinton KD: `T² · KL(softmax(t/T) ‖ softmax(s/T))`.
- **L_relation** — Relational KD (pairwise distances + triplet angles on GAP
  features). Opt-in via `ε > 0`.
- **L_attention** — Attention Transfer (channel-aggregated spatial maps,
  L2-normalized). Opt-in via `ζ > 0`.
- **γ(t), β(t)** — linear warmups (default 10 and 3 epochs) so the student
  doesn't try to match subspace / sparsity before its own activations have
  stabilized.

Two combination modes (`training.combination`):
- `fixed` (default) — manual weights α, β, γ, δ, ε, ζ.
- `uncertainty` — Kendall & Gal learnable-σ weighting across components.

Optional EMA auto-normalization (`auto_normalize: true`) divides each loss by
its running magnitude so the weights express *relative* priorities rather than
absolute scales.

## v1 → v2 Improvements

`scripts/14_v2_compare.py` toggles all v2 knobs on. The differences vs v1:

- Auto-normalized loss components (EMA).
- β-warmup (0.1 → 1.0) in addition to γ-warmup.
- LR warmup (10% → 100% over first 2 epochs) before cosine.
- Best-val checkpoint (`keep_best: true`) instead of final epoch.
- KD temperature 4 → 2 for CIFAR-10.
- Optional L2-normalization of the channel axis in subspace loss.

## Configuration

Defaults live in `config/default.yaml`. Selected keys:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profiling.variance_threshold` | 0.95 | Cumulative variance for effective rank |
| `profiling.covariance_mode` | `per_pixel` | `per_pixel` or `gap` |
| `profiling.rank_definition` | `variance` | Rank formula (see above) |
| `student.width_multiple` | 8 | Round widths to multiples of N |
| `student.min_width` | 16 | Lower bound on stage width |
| `training.loss_alpha` | 1.0 | Task CE weight |
| `training.loss_beta` | 0.5 | Subspace weight |
| `training.loss_gamma` | 0.3 | Sparsity weight |
| `training.loss_delta` | 1.0 | Logit-KD weight |
| `training.gamma_warmup_epochs` | 10 | γ warmup length |
| `training.beta_warmup_epochs` | 3 | β warmup length |
| `training.lr_warmup_epochs` | 2 | LR linear warmup length |
| `training.subspace_mode` | `spatial` | `spatial` \| `gap` \| `cosine_spatial` |
| `training.sv_weighting` | `sqrt` | `uniform` \| `linear` \| `sqrt` |
| `training.use_logit_kd` | `true` | Enable Hinton KD term |
| `training.logit_temperature` | 4.0 | KD temperature |
| `training.combination` | `fixed` | `fixed` \| `uncertainty` |
| `training.auto_normalize` | `false` | EMA-normalize each loss component |
| `training.keep_best` | `true` | Checkpoint best-val, not final |

## Ablations Supported

`scripts/08_ablation.py --variant <name>`:

- `full` — all improvements on (reference point).
- `no_logit_kd` — disable Hinton KD.
- `no_sparsity` — disable sparsity loss.
- `gap_subspace` — legacy GAP subspace loss.
- `gap_cov` — legacy GAP covariance in profiling.
- `linear_sv` — linear (not sqrt) SV weighting.
- `uncertainty` — uncertainty-weighted loss combination.
- `classical_kd` — task CE + Hinton KD only (no ASD components).
- `task_only` — task CE only (pure supervised baseline).
- `with_relation` — full + RKD (ε = 0.5).

## Testing

```bash
python -m pytest tests/ -v
```

Test modules cover activation capture, SVD analysis, loss functions,
student construction, a training smoke test, and a suite targeting the
algorithmic v2 improvements.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0, torchvision ≥ 0.15
- `omegaconf`, `matplotlib`, `tqdm`, `tensorboard`
- For the LLM experiment: `transformers`, `datasets`

## Extending to Other Teachers

1. Add a wrapper in `asd/models/teacher_ext.py` that exposes 4 stage features
   (`forward → (logits, [f1..f4])`) with a consistent per-stage channel count.
2. Register it in `_NON_RESNET_WRAPPERS` and give it a `teacher_hook_names`
   entry, or hook its blocks from the bench script.
3. Fine-tune on the target dataset. The rest of the pipeline (profiling,
   sizing, training) works unchanged.

## License

MIT
