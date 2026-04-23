# Activation Subspace Distillation

Compress a PyTorch teacher model by distilling inside its activation
subspace. On ResNet-50 / CIFAR-10 a 17x-compressed student reaches
90.74% accuracy; a 25x-compressed student reaches 90.21%. Works on
CNNs, HuggingFace transformers, and any `nn.Module` whose block list
you can point at.

```python
import asd
import torch.nn.functional as F

profile = asd.profile(teacher, calib_loader,
                      source="delta", noise_model="mp")

student = asd.build_student(teacher, profile)
sub_loss = asd.SubspaceLoss(profile, objective="cka")

for x, y in train_loader:
    with asd.capture(teacher, profile) as t_hid:
        teacher(x)
    with asd.capture(student, profile) as s_hid:
        s_logits = student(x)
    loss = F.cross_entropy(s_logits, y) + 0.5 * sub_loss(
        s_hid.values(), t_hid.values(),
    )
    loss.backward()
    opt.step()
    opt.zero_grad()
```

The public API is five callables (`profile`, `build_student`,
`SubspaceLoss`, `capture`, `distill`) plus one container
(`TeacherProfile`). No `Trainer` subclass, no YAML config, no
HuggingFace assumption.

## Method

Most knowledge-distillation methods either match the teacher's final
logits (classic KD) or match intermediate feature maps with a
handcrafted alignment loss (FitNets, AT, RKD). Both treat the teacher
as something to imitate channel for channel.

Trained networks don't use all their channels. At the deepest stage
of a CIFAR-10-trained ResNet-50, under a Marchenko-Pastur bulk-edge
cutoff, roughly 400 of 2048 channels carry statistically
distinguishable signal. The rest are noise or linearly redundant. The
teacher's effective computation has an intrinsic rank well below its
nominal width.

ASD exploits that directly:

1. **Profile.** Eigendecompose the channel covariance at each block.
   Pick the top-`k` principal directions per block.
2. **Size.** Build a student whose per-block width is `k`, not the
   teacher's full width.
3. **Distill.** Train the student to match the teacher inside the
   retained subspace, not on the full feature map. The loss is
   centered kernel alignment or Gram-Frobenius distance on the
   projected features.

Three additional choices improve stability:

- **Marchenko-Pastur rank selection.** Instead of a variance
  threshold, use the Gavish-Donoho bulk-edge cutoff to separate
  signal eigenvalues from the noise bulk. On ResNet-50 / CIFAR-10
  this tightens compression from 4x to 17x at under 1 pp additional
  cost over a variance-threshold baseline.
- **Basis-invariant loss.** When eigenvalues are close, the
  principal basis inside the retained subspace is arbitrary up to
  rotation. Coordinate-MSE forces the student to match that
  arbitrary rotation. Gram and CKA do not.
- **Feature normalization.** L2-normalize features per sample before
  forming kernels. Keeps the loss bounded on LLM-scale residual
  streams where unnormalized Gram reaches 1e6 and training diverges.

## Benchmarks

CIFAR-10 / ResNet-50, 20 epochs, 3 seeds, teacher accuracy 97.0%:

| configuration                         | compression | student acc   |
|---------------------------------------|-------------|---------------|
| baseline (output + gram, tau=0.95)    | 4.15x       | 91.60 +/- 0.15|
| baseline (output + gram, tau=0.85)    | 15.86x      | 90.59 +/- 0.24|
| Marchenko-Pastur cutoff               | 17.08x      | 90.74         |
| MP + Ledoit-Wolf shrinkage            | 25.29x      | 90.21         |
| `arch_multiplier=1.25`                | 2.66x       | 91.65 +/- 0.03|
| `arch_multiplier=2.0`                 | 1.04x       | 92.12 +/- 0.33|

Teacher: 96.98% at 23.5M parameters. All student numbers are at a
fraction of that budget. The 17x / 90.74% row is the headline:
Marchenko-Pastur rank selection compresses 4x further than the
variance-threshold baseline for under a percentage point of accuracy.
All CIFAR numbers are from a 34-job benchmark ladder run as
independent Anyscale prodjobs on A10G GPUs.

GPT-2 / WikiText-2, 1 epoch stability smoke:

| configuration                           | behavior              |
|-----------------------------------------|-----------------------|
| gram without feature normalization      | diverges (loss > 1e6) |
| coord_mse with mahalanobis weighting    | 149 ppl at 2x compression |
| cka with feature normalization          | bounded, stable, converges |

LLM training beyond 1 epoch is future work. The loss numerics that
blocked prior attempts are fixed.

## Choosing knobs

Defaults are sensible for most cases. The two calls you tune are:

`asd.profile(...)`:

| knob          | options                       | pick                                                             |
|---------------|-------------------------------|------------------------------------------------------------------|
| `source`      | `output` / `delta` / `branch` | `output` by default; `delta` strips identity contamination on residual nets |
| `noise_model` | `eps` / `mp`                  | `mp` whenever calibration has at least C samples per layer       |
| `shrinkage`   | `none` / `ledoit_wolf`        | `ledoit_wolf` on small (< 1k-sample) or noisy calibration        |

`asd.SubspaceLoss(...)`:

| knob                 | options                      | pick                                                |
|----------------------|------------------------------|-----------------------------------------------------|
| `objective`          | `coord_mse` / `gram` / `cka` | `cka` for LLMs, `gram` for CNNs                     |
| `normalize_features` | `True` / `False`             | `True`; `False` on transformers lets the loss diverge |

## Install

```bash
pip install -e .              # core library
pip install -e ".[dev]"       # + pytest
pip install -e ".[llm]"       # + transformers, datasets
```

Requires Python 3.10+, PyTorch 2.0+, torchvision 0.15+.

## Example: compress ResNet-50 on CIFAR-10

```python
import asd
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

teacher = resnet50(weights="IMAGENET1K_V2").eval().cuda()
profile = asd.profile(teacher, calibration_loader,
                      source="delta", noise_model="mp")

student = asd.build_student(teacher, profile,
                            arch_multiplier=1.0,
                            num_classes=10,
                            stem_type="cifar").cuda()

result = asd.distill(
    teacher, student, train_loader,
    profile=profile, val_loader=test_loader,
    epochs=20, objective="gram",
    alpha=1.0, beta=0.5, delta=1.0,
)
print(f"best acc: {result.best_metric * 100:.2f}%")
```

## Example: compress GPT-2 on WikiText-2

```python
import asd
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

teacher = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
profile = asd.profile(teacher, calib_loader,
                      source="output", noise_model="mp")
student = asd.build_student(teacher, profile).cuda()
sub_loss = asd.SubspaceLoss(profile, objective="cka").cuda()

for ids in train_loader:
    ids = ids.cuda()
    with asd.capture(teacher, profile) as t_hid, torch.no_grad():
        t_out = teacher(ids, labels=ids)
    with asd.capture(student, profile) as s_hid:
        s_out = student(ids, labels=ids)
    loss = s_out.loss + 0.5 * sub_loss(s_hid.values(), t_hid.values())
    loss.backward()
    opt.step()
    opt.zero_grad()
```

## Example: a custom model

Skip `build_student`. Construct the student yourself and let the
loss build projectors from its hidden widths:

```python
profile = asd.profile(
    my_teacher, loader,
    layers=[my_teacher.encoder[i] for i in range(4)],
)
my_student = MyCustomStudent(hidden=256)
sub_loss = asd.SubspaceLoss(profile, objective="cka")
```

If `asd.autodetect_layers(model)` does not recognize the
architecture, pass `layers=` with a list of modules or dotted names.
Or register a detector:

```python
def detect_mynet(m):
    if hasattr(m, "blocks"):
        return [f"blocks.{i}" for i in range(len(m.blocks))]
    return None

asd.register_detector("mynet", detect_mynet)
```

## Comparison with other libraries

| method                                     | LLM-safe | auto width | arbitrary arch | drop-in API |
|--------------------------------------------|:--------:|:----------:|:--------------:|:-----------:|
| classical Hinton KD                        | yes      | no         | yes            | yes         |
| FitNets / AT (feature-map)                 | no       | no         | partial        | partial     |
| RKD / CRD (relational / contrastive)       | partial  | no         | partial        | yes         |
| task-specific pruning (channel prune + FT) | partial  | yes        | no             | no          |
| ASD (this library)                         | yes      | yes        | yes            | yes         |

ASD fits when:

- you have a working teacher and want to trade accuracy for inference
  cost on a known budget
- the teacher has residual or block structure (CNN or transformer)
- you do not want to rewrite your training loop

It is the wrong tool when the teacher is already small, the student
needs a different architecture, or fine-tuning a smaller model from
scratch is cheaper than running the distillation.

## Runnable scripts

```bash
# Fine-tune a torchvision ResNet-50 for CIFAR-10
python scripts/finetune_teacher.py --output outputs/teacher.pt

# Distill it
python scripts/distill_cifar_resnet.py \
    --teacher outputs/teacher.pt --epochs 20

# GPT-2 end-to-end
python scripts/distill_gpt2_wikitext.py --epochs 3
```

Each script is under 200 lines and uses only the public library API.

## Repo layout

```
asd/                 library
  __init__.py          public API re-exports
  api.py               profile / capture / SubspaceLoss / distill / TeacherProfile
  autodetect.py        layer detection for known model families
  builders.py          student constructors (torchvision ResNets + HF GPT-2)
  models/student.py    SlimNet, the 4-stage ResNet student
  profiling/           activation capture, SVD, MP cutoff, stability diagnostic
  data/                CIFAR-10 / ImageNet loader helpers
scripts/             three runnable examples
docs/                quickstart, algorithm, mechanism
tests/               36 tests (`python -m pytest tests/`)
```

## Further reading

- [docs/quickstart.md](docs/quickstart.md): two-minute library walkthrough.
- [docs/algorithm.md](docs/algorithm.md): mechanism, math, design rationale.
- [asd/api.py](asd/api.py): the full public API in a single file.

## License

MIT.
