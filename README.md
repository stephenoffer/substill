# neural_distill

Compress a PyTorch **LLM** teacher (GPT-2, Llama, Mistral, Qwen) by distilling
inside its **activation subspace** — the low-dimensional space its features
actually occupy, rather than channel-for-channel.

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
| `use_cpsd_factored`  | `False`     | replace absorbed linears with Stiefel-trainable factors (CPSD) |
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

**Current status (honest):** on GPT-2 + WikiText-2 (n=5 seeds), CPSD is
*competitive with* — but does not yet *beat* — the strongest baseline. Manifold
training helps (both Stiefel-trained variants beat frozen absorbed-init by ~5 PPL),
but CPSD's extra projection-factor training ties the simpler FSD baseline at this
scale. GPT-2 cannot exercise CPSD's headline circuit-preserving component (it has no
GQA/RoPE); on a real GQA+RoPE Llama the OV-circuit basis is a clean win while the
QK-circuit-under-RoPE benefit is unestablished. The decisive Llama-3.2 frontier
result is not yet run. See [docs/cpsd.md](docs/cpsd.md) for the mechanism, the full
results table, and what remains.

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
