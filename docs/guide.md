# User guide

`substill` offers three entry points. Use **LRD** for Llama-family decoders; it is the
verified method and the one to reach for by default. Use **`substill.vision`** for ResNet
classifiers. The older **FSD/CPSD pipeline** is still here for reference, but its headline
numbers did not survive re-measurement, so don't build on it without reading the
[audit](init_findings.md) first.

## Learned Restriction Distillation (LRD)

LRD runs in three phases, exposed both as a one-call function and as an object you can
drive yourself.

### One call

```python
import substill

config = substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000)
result = substill.learned_restriction_distill(teacher, train_loader, config=config)
student = result.student            # folded, plain LlamaForCausalLM
```

`result` is an {class}`~substill.LRDResult`. It carries the trained `student`, the
per-step `history`, the `final_kd` loss, and `max_principal_angle`, which tells you how
far `V` rotated away from its PCA initialization.

### Phase by phase

Drive the phases yourself when you want to inspect or checkpoint between them:

```python
lrd = substill.LearnedRestriction(teacher, config)
lrd.prepare(calib_loader)   # gamma-fold the teacher, profile the residual stream, build V0
lrd.fit(train_loader)       # descend the KD loss in the (V, D) coordinates
student = lrd.fold()        # collapse (V, D) into a plain LlamaForCausalLM
```

`prepare` folds the teacher's RMSNorm gains into the consuming linears, which preserves
the function exactly. It then measures the residual second moment, selects the FFN neurons
and the `V0` basis, and builds a trainable
{class}`~substill.compression.restricted.RestrictedLlama`.

`fit` trains `V` on the Stiefel manifold and, with `free_core=True` (the default and the
winning arm), a zero-initialized Euclidean residual `D` alongside it. Training starts at
exactly the absorbed-init student, so `V` is the only coordinate that differs from a plain
baseline.

`fold` returns a plain `LlamaForCausalLM` whose weights are `Vᵀ W_T V (+ D)`. It is
function-identical to the trained module and carries zero inference overhead.

### Configuration

{class}`~substill.LRDConfig` fields (the API reference has the full list):

| field | default | meaning |
|---|---|---|
| `hidden`, `intermediate`, `n_head`, `n_kv` | — | student geometry (use `for_ratio` to derive) |
| `steps` | `2000` | training steps |
| `lr` | `1e-3` | AdamW LR for the Euclidean residual `D` |
| `v_lr` | `0.0` | Stiefel step for `V`, in **radians of subspace rotation per step**; `0` selects the default. Dimensionless, so it needs no per-teacher constant (see `learned_restriction.md` §9c) |
| `kd` | `"forward_kl"` | KD objective: `forward_kl` / `reverse_kl` / `skew_kl` |
| `basis` | `"pca"` | `V0` initializer: `pca` / `identity` / `gn` |
| `free_core` | `True` | train `V` **and** `D` jointly (the verified winner) |

Rather than hand-specifying the geometry, size the student from a compression ratio:

```python
config = substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000, kd="skew_kl")
geom = substill.plan_restricted_geometry(teacher, width_ratio=0.5)   # inspect the choice
```

**Scope.** LRD supports bias-free RMSNorm decoders of the Llama family (Llama, Mistral).
{func}`~substill.learned_restriction_distill` validates the teacher and raises a clear
`NotImplementedError` on an unsupported architecture, so you will not get a silently wrong
student.

## Vision (ResNet)

`substill.vision` applies the same activation-subspace idea to convolutional classifiers.
Rank a ResNet's channels, narrow each bottleneck, absorb the teacher's conv weights, and
distill on class logits.

```python
from substill.vision import channel_variance_scores, build_resnet_student, distill_classifier

scores = channel_variance_scores(teacher, calib_loader)
student, info = build_resnet_student(teacher, scores, width_ratio=0.5)
distill_classifier(teacher, student, train_loader, val_loader=val_loader)
```

Note that a ReLU CNN has no rotation-equivariant residual stream, so the learned rotation
that drives the LRD win is not available here. Channel *selection* is, and it still beats
random init by about 8 points. See [concepts](concepts.md) for why.

## FSD/CPSD pipeline (earlier method)

`substill.FSDPipeline` is the earlier, more general activation-subspace pipeline. Its
headline numbers were **not** reproduced under compute matching (see the
[audit](init_findings.md)), so it is kept for reference and for the profiling and loss
machinery it exposes.

```python
pipe = substill.FSDPipeline(teacher, config=substill.FSDConfig(arch_multiplier=0.5))
result = pipe.run(calib_loader, train_loader)
student = pipe.student
```

The lower-level stages remain public for custom pipelines: {func}`~substill.profile`,
{func}`~substill.build_student`, {func}`~substill.distill`, and the basis-invariant
{class}`~substill.F_ASDLoss`.
