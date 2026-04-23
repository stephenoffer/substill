# Quickstart

Use ASD as a Python library on your own model. Two-minute read.

## Install

```bash
pip install -e .
# For HuggingFace LLMs:
pip install -e ".[llm]"
```

## 1. Profile your teacher

```python
import asd

profile = asd.profile(
    teacher,
    calibration_loader,     # any torch DataLoader; yields (x, ...) batches
    source="delta",         # "output" | "delta" | "branch"
    noise_model="mp",       # "eps" | "mp"; MP gives tighter ranks
)
```

One forward pass over `calibration_loader`. No backprop, no training.
Save to disk if you want to rebuild students later:

```python
profile.save("teacher.profile")
# later...
profile = asd.TeacherProfile.load("teacher.profile")
```

## 2. Build or supply a student

If your teacher is a torchvision ResNet or HuggingFace GPT-2, the
library builds a narrower student from the profile:

```python
student = asd.build_student(teacher, profile, arch_multiplier=1.0)
```

`arch_multiplier > 1` trades compression for accuracy; `1.0` is the
most aggressive compression.

For any other model, build your student yourself. The loss accepts
any module you can hook at the same layer names as the profile:

```python
student = MyCustomStudent(hidden=256)
```

## 3. Attach the subspace loss

```python
loss_fn = asd.SubspaceLoss(
    profile,
    objective="cka",          # "coord_mse" | "gram" | "cka"
    normalize_features=True,  # required for LLM stability
).to(device)
```

`SubspaceLoss` is an `nn.Module`. Its parameters are the per-layer
linear projections from the student's hidden dim to the teacher's
retained rank. Add them to the optimizer alongside the student:

```python
opt = torch.optim.AdamW(
    list(student.parameters()) + list(loss_fn.parameters()),
    lr=5e-5,
)
```

## 4. Train

```python
import torch.nn.functional as F

for x, y in train_loader:
    x, y = x.to(device), y.to(device)

    with asd.capture(teacher, profile) as t_hid:
        with torch.no_grad():
            t_logits = teacher(x)

    with asd.capture(student, profile) as s_hid:
        s_logits = student(x)

    loss = F.cross_entropy(s_logits, y) + 0.5 * loss_fn(
        s_hid.values(), t_hid.values(),
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
```

`capture(model, profile)` hooks the model at the same layers as the
profile; `capture_obj.values()` returns the hidden tensors in profile
order — the order `loss_fn` expects.

## One-call pipeline

For classification-style tasks:

```python
result = asd.distill(
    teacher, student, train_loader,
    profile=profile,       # omit to profile inside distill()
    val_loader=val_loader,
    epochs=20,
    objective="cka",
    alpha=1.0, beta=0.5, delta=1.0,   # weights for task / subspace / KD
)
print(f"best: {result.best_metric*100:.2f}%")
```

## Choosing knobs

For `asd.profile(...)`:

| knob            | options                      | pick                                                                     |
|-----------------|------------------------------|--------------------------------------------------------------------------|
| `source`        | `output` / `delta` / `branch`| `output` general. `delta` strips identity contamination on residual nets. |
| `noise_model`   | `eps` / `mp`                 | `mp` when calibration has ≥ C samples per layer (it almost always does). |
| `shrinkage`     | `none` / `ledoit_wolf`       | `ledoit_wolf` on small/noisy calibration sets (< 1k samples).            |

For `asd.SubspaceLoss(...)`:

| knob                  | options                       | pick                                                             |
|-----------------------|-------------------------------|------------------------------------------------------------------|
| `objective`           | `coord_mse` / `gram` / `cka`  | `cka` for LLMs. `gram` for CNNs. `coord_mse` only if you know you want it. |
| `normalize_features`  | `True` / `False`              | `True`. Turn off only if your features are already bounded.       |

## Common pitfalls

**The loss blows up (NaN / huge values) on an LLM.**
Use `objective="cka"` (or `"gram"` with `normalize_features=True`).
The defaults already do this; you only see the divergence if you've
set `normalize_features=False` or `objective="coord_mse"` on
high-magnitude residual-stream features.

**`asd.capture(student, profile)` gives empty hiddens.**
The profile records layer names from the teacher. If your student has
different module paths, either:

- Rebuild the profile with student-compatible paths:
  ```python
  remapped = asd.TeacherProfile(
      layers=["my_blocks.0", "my_blocks.1", ...],   # student paths
      profiles=profile.profiles,                     # same eigenvectors
      source=profile.source,
  )
  ```
- Or construct the student so its layer names match the teacher's
  (e.g. use `asd.build_student` for known families).

**`asd.autodetect_layers(model)` raises `NotImplementedError`.**
Pass `layers=` explicitly:

```python
profile = asd.profile(
    model, loader,
    layers=[model.encoder.blocks[i] for i in range(4)],
)
```

Or register a detector for your family:

```python
def detect_mynet(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
        return [f"encoder.blocks.{i}" for i in range(len(model.encoder.blocks))]
    return None

asd.register_detector("mynet", detect_mynet)
profile = asd.profile(model, loader)
```
