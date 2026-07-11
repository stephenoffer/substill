# User guide

## Learned Restriction Distillation (LRD)

LRD is the recommended method. It has three phases, exposed both as a one-call function
and as an inspectable object.

### One call

```python
import substill

config = substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000)
result = substill.learned_restriction_distill(teacher, train_loader, config=config)
student = result.student            # folded, plain LlamaForCausalLM
```

`result` is an {class}`~substill.LRDResult` with the trained `student`, the per-step
`history`, the `final_kd` loss, and `max_principal_angle` (how far `V` rotated from its
PCA initialization).

### Phase by phase

Drive the phases yourself to inspect or checkpoint between them:

```python
lrd = substill.LearnedRestriction(teacher, config)
lrd.prepare(calib_loader)   # gamma-fold the teacher, profile the residual stream, build V0
lrd.fit(train_loader)       # descend the KD loss in the (V, D) coordinates
student = lrd.fold()         # collapse (V, D) into a plain LlamaForCausalLM
```

- **`prepare`** folds the teacher's RMSNorm gains into the consuming linears
  (function-preserving), measures the residual second moment, selects the FFN neurons and
  the `V0` basis, and builds a trainable
  {class}`~substill.compression.restricted.RestrictedLlama`.
- **`fit`** trains `V` on the Stiefel manifold and, with `free_core=True` (the default,
  winning arm), a zero-initialized Euclidean residual `D`. Training starts at exactly the
  absorbed-init student, so the `V` coordinate is the only difference from a plain baseline.
- **`fold`** returns a plain `LlamaForCausalLM` whose weights are `Vᵀ W_T V (+ D)`,
  function-identical to the trained module, with zero inference overhead.

### Configuration

{class}`~substill.LRDConfig` fields (see the API reference for the full list):

| field | default | meaning |
|---|---|---|
| `hidden`, `intermediate`, `n_head`, `n_kv` | — | student geometry (use `for_ratio` to derive) |
| `steps` | `2000` | training steps |
| `lr` | `1e-3` | AdamW LR for the Euclidean residual `D` |
| `v_lr` | `0.0` | Stiefel LR for `V`; `0` selects `min(1e-3, 0.77/d)` |
| `kd` | `"forward_kl"` | KD objective: `forward_kl` / `reverse_kl` / `skew_kl` |
| `basis` | `"pca"` | `V0` initializer: `pca` / `identity` / `gn` |
| `free_core` | `True` | train `V` **and** `D` jointly (the verified winner) |

Size the student from a compression ratio instead of hand-specifying geometry:

```python
config = substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000, kd="skew_kl")
geom = substill.plan_restricted_geometry(teacher, width_ratio=0.5)   # inspect the choice
```

**Scope.** LRD supports bias-free RMSNorm decoders of the Llama family (Llama, Mistral).
{func}`~substill.learned_restriction_distill` validates the teacher and raises a clear
`NotImplementedError` on an unsupported architecture.

## FSD/CPSD pipeline (earlier method)

`substill.FSDPipeline` is the earlier, general activation-subspace pipeline. Its headline
numbers were **not** reproduced under compute matching (see [init findings](init_findings.md)), so it is
kept for reference and for the profiling / loss machinery it exposes.

```python
pipe = substill.FSDPipeline(teacher, config=substill.FSDConfig(arch_multiplier=0.5))
result = pipe.run(calib_loader, train_loader)
student = pipe.student
```

The lower-level stages — {func}`~substill.profile`, {func}`~substill.build_student`,
{func}`~substill.distill`, and the basis-invariant {class}`~substill.F_ASDLoss` — remain
public for custom pipelines.

## Vision (ResNet)

`substill.vision` applies the same activation-subspace idea to convolutional classifiers:
rank a ResNet's channels, narrow each bottleneck, absorb the teacher's conv weights, and
distill on class logits.

```python
from substill.vision import channel_variance_scores, build_resnet_student, distill_classifier

scores = channel_variance_scores(teacher, calib_loader)
student, info = build_resnet_student(teacher, scores, width_ratio=0.5)
distill_classifier(teacher, student, train_loader, val_loader=val_loader)
```
