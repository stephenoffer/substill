# neural_distill

### Shrink a big model into a small, fast one — without starting from scratch.

Big neural networks are expensive to run. `neural_distill` builds a **small "student"
model that behaves almost like a big "teacher"** by compressing it inside its
**activation subspace** — the low-dimensional space a trained network actually uses,
rather than guessing channel-by-channel. The student **starts from the teacher's own
behavior** instead of random weights, so it reaches good quality with far less training,
and ships as a plain, fast model with **zero inference overhead**.

**One method, two model families — measured wins over the naive baseline:**

| teacher → student | naive distillation | **FASD (absorbed-init)** | gain |
|---|---|---|---|
| GPT-2 / WikiText-2 (≈4.3× smaller) | 1038 perplexity | **559** | **1.9× better** |
| ResNet-50 / CIFAR-10 (≈2× smaller) | 64.8% accuracy | **81.1%** | **+16 points** |

*New to model distillation or compression?* Start with the plain-language
**[explainer](docs/explainer.md)** — it explains the idea, why it matters, and what's
proven, with no prior background assumed.

FASD is a **general** activation-subspace framework: the LLM path covers GPT-2, Llama,
Mistral, and Qwen decoders, and a **vision path** (`fasd.vision`) covers convolutional
classifiers such as **ResNet50**. The same machinery — covariance/PCA profiling, the
absorbed projection `W_s = Vᵀ W V` (now conv2d-aware), basis-invariant feature losses,
and distillation — applies across both.

The public API is the **`fasd`** package: Functionally-Aligned ASD plus the
novel **CPSD** pipeline. (`asd` is an internal subpackage providing the
low-level profiling utilities `fasd` builds on — covariance accumulation,
spectrum/Marchenko-Pastur analysis, stability diagnostics — and is not a
public API.)

## Install

```bash
pip install -e .            # core library
pip install -e ".[llm]"     # + transformers, datasets (for the LLM path)
pip install -e ".[dev]"     # + pytest, ruff
pip install -e ".[all]"     # everything
```

Requires Python 3.10+, PyTorch 2.0+.

## Quickstart — `FSDPipeline`

The recommended entry point is `fasd.FSDPipeline`: one object that profiles
the teacher, builds a width-narrowed student, optionally converts it to
circuit-preserving trainable factors (CPSD), and runs the distillation.

```python
import fasd

pipe = fasd.FSDPipeline(
    teacher,
    config=fasd.FSDConfig(
        arch_multiplier=0.5,      # student width relative to the teacher
        use_cpsd_factored=True,   # train low-rank factors on the Stiefel manifold (CPSD)
        generative_kd="skew_kl",  # KD objective
        total_steps=2000,
    ),
)
result = pipe.run(calib_loader, train_loader)  # profile -> build -> (convert) -> distill
student = pipe.student                          # the trained, compressed student
```

Key `FSDConfig` fields (all optional, sensible defaults):

| field                | default     | what it controls                                            |
|----------------------|-------------|-------------------------------------------------------------|
| `arch_multiplier`    | `1.0`       | student width as a fraction of the teacher                  |
| `rank_map`           | `None`      | explicit per-edge rank overrides                            |
| `template`           | `"auto"`    | `"gpt2"`, `"llama"`, or `"auto"` (dispatch on teacher class)|
| `absorbed_init`      | `True`      | initialize student linears with absorbed teacher weights    |
| `use_cpsd_factored`  | `False`     | replace absorbed linears with Stiefel-trainable factors (CPSD MT) |
| `use_diff_rank`      | `False`     | KD-driven differentiable rank on the factored edges (CPSD DDR; needs `use_cpsd_factored`) |
| `use_rr_norm`        | `False`     | swap norms for rotation-equivariant `RRNorm`                |
| `generative_kd`      | `"skew_kl"` | KD objective: `forward_kl` / `reverse_kl` / `skew_kl`       |
| `total_steps`, `lr`  | `200`, `5e-5` | distillation budget                                       |
| `profile_kwargs`, `distill_kwargs` | `{}` | passthrough to `profile()` / `distill()`           |

## Public API

`import fasd` exposes (see `fasd.__all__`):

- **Pipeline (start here):** `FSDPipeline`, `FSDConfig`
- **Profiling:** `profile`, `capture`, `TeacherProfile`, `BranchProfile`,
  `BranchSpec`, `StreamingPCA`, `choose_behavioral_rank`, `compute_weights`,
  `autodetect_branches`, `autodetect_layers`, `register_detector`,
  `StabilityStats`, `bootstrap_principal_angles`, `stability_adjusted_rank`
- **Student construction:** `build_student`, `StudentConfig`,
  `profile_to_student_config`, `plan_progressive_stages`,
  `absorbed_linear_init`, `quantize_student`, `qad_finetune`
- **Losses & schedules:** `F_ASDLoss`, `gram_distance`, `cka_distance`,
  `procrustes_distance`, `forward_kl`, `reverse_kl`, `skew_kl`,
  `contrastive_response_loss`, `Schedule`, `ScheduleStage`, `default_schedule`
- **Training:** `distill`, `DistillResult`, `correct_teacher`,
  `generate_rollouts`, `HybridCollator`, `ReplayBuffer`, `RolloutBatch`
- **CPSD conversions (advanced):** `convert_gpt2_to_factored`,
  `convert_llama_to_factored`

## Lower-level / advanced

`FSDPipeline` is a thin orchestration over public stages you can also drive
yourself — profile the teacher, build a student, then either run your own
training loop with `F_ASDLoss` or call the multi-stage `distill` driver:

```python
import fasd

profile = fasd.profile(teacher, calib_loader)
student = fasd.build_student(teacher, profile, absorbed_init=True)

loss_fn = fasd.F_ASDLoss(profile, objective="procrustes")
for batch in train_loader:
    with fasd.capture(teacher, profile) as t_hid:
        teacher(**batch)
    with fasd.capture(student, profile) as s_hid:
        s_logits = student(**batch).logits
    loss = loss_fn(dict(s_hid.items()), dict(t_hid.items()))
    loss.backward()
```

`convert_gpt2_to_factored` / `convert_llama_to_factored` are advanced
post-build CPSD helpers (what `FSDPipeline` calls when
`use_cpsd_factored=True`); use them directly only for custom pipelines.

## CPSD — the novel method

`FSDPipeline(..., use_cpsd_factored=True)` enables **Circuit-Preserving
Subspace Distillation**: the student is initialized with circuit-preserving
factors (including the OV/value circuit), those factors are trained on the
Stiefel manifold against the KD loss, and per-edge rank is learned jointly.

The two trainable components are now both first-class pipeline options:

- **MT (manifold-trained factors):** `use_cpsd_factored=True` swaps absorbed
  linears for `TeacherFactoredLinear` and the pipeline auto-builds a `StiefelAdam`
  optimizer so `V_in/V_out` train on the manifold (warm-started from the absorbed
  bases; `stiefel_lr_ratio`/`stiefel_reorth_every` tune its variance).
- **DDR (distillation-driven differentiable rank):** `use_diff_rank=True` wraps each
  factored edge with a soft column gate trained against the **KD loss** under a global
  parameter budget; `pipe.fold_for_inference()` hardens the gates and folds every edge
  back to a plain `nn.Linear` (zero inference overhead). The central claim — *KD-driven*
  rank beating *reconstruction-driven* rank (Dobi-SVD) — is measured directly by
  `scripts/cpsd_compare.py` as a controlled ablation (same pipeline, KD weight toggled).

**Current status (honest, measured on Anyscale A10G — GPT-2/WikiText-2 n=3, ResNet50/CIFAR-10,
real TinyLlama-1.1B):**

- **Win vs the naive competition (both modalities):** absorbed-init subspace distillation beats
  random-init+KD by **1.4–1.9×** PPL on GPT-2 (e.g. 559 vs 1038 at 4.35×, n=3) and by **+14–16
  top-1 points** on ResNet50/CIFAR-10 (81.1% vs 64.8% at matched compression). The core,
  reproducible advantage — now spanning LLMs *and* vision.
- **Win vs a competitor mechanism (Dobi-SVD):** our *KD-driven* differentiable rank beats
  *reconstruction-driven* rank by **1.45–2.2×** PPL (829 vs 1806 at 4.35×, n=3). The central
  novelty claim holds directionally and consistently.
- **Honest negative #1 (manifold training):** at short (300-step) budgets the manifold-trained
  factors do **not** beat frozen absorbed-init; the MT gain is modest and setting-dependent
  (~3–4% only at gentler 2–4× / 500 steps, n=5).
- **Honest negative #2 (CPI):** the circuit-preserving init does **not** beat the disjoint
  baseline on real GQA — documented as a negative result, not a claim.
- **Not yet run:** the Llama-3.2-3B→1B frontier vs *published* SOTA numbers
  (Dobi-SVD/KQ-SVD/DistiLLM-2/Minitron/RFID-MoE) — the decisive head-to-head remains future work.

See [docs/cpsd.md](docs/cpsd.md) for the full tables, mechanism, and what remains.

## Vision — ResNet (non-LLM)

`fasd.vision` applies the same activation-subspace idea to convolutional classifiers.
A conv is linear in its channels per kernel offset, so the absorbed projection extends to
2D convs unchanged (`absorbed_init` gained a `"conv2d"` layout). `build_resnet_student`
narrows each `Bottleneck`'s inner channels (keeping block input/output widths fixed, so
downsample and the residual add are untouched) and absorbs the teacher's conv weights;
`distill_classifier` distils on class logits with the same `forward_kl`/`skew_kl` losses.

```python
from fasd.vision import channel_variance_scores, build_resnet_student, distill_classifier
scores = channel_variance_scores(teacher, calib_loader)
student, info = build_resnet_student(teacher, scores, width_ratio=0.5)  # absorbed-init
distill_classifier(teacher, student, train_loader, val_loader=val_loader)
```

`scripts/resnet50_distill.py` runs the vision analogue of the GPT-2 ladder (absorbed-init
vs random-init at matched compression); `--smoke` runs on CPU.

## How it works

A trained network does not use all its channels. Profiling eigendecomposes
each branch's activation covariance and keeps the directions that carry
statistically distinguishable signal (a Marchenko-Pastur bulk-edge cutoff
separates signal from the noise bulk). The student is sized to those
*behavioral ranks*, initialized by absorbing the teacher's weights into the
retained subspace, and distilled with a basis-invariant feature loss
(Gram / CKA / Procrustes) alongside a generative-KD objective on the logits.

## Repo layout

```
fasd/            current library (public API in fasd/__init__.py)
  pipeline.py      FSDPipeline / FSDConfig
  api.py           profile / capture / TeacherProfile
  builders.py      student constructors + absorbed init
  compression/     absorbed init, width pruning, factored linears, quantization
  profiling/       activation capture, behavioral rank, RoPE/GQA bases
  losses/          subspace (F_ASDLoss), procrustes, generative KD
  training/        distill driver, Stiefel optimizer, on-policy, re-absorption
  arch/            ArchitectureSpec registry (add a model family declaratively)
  vision/          non-LLM arm: ResNet channel-narrowing + classifier distillation
asd/             internal profiling utilities used by fasd (not a public API)
scripts/         training, ablation, and baseline-reproduction entry points
docs/            technical docs, project status, history  (see docs/README.md)
tests/           pytest suite
```

## Tests & lint

```bash
python -m pytest        # test suite
ruff check .            # lint (config in pyproject.toml)
```

## License

MIT.
