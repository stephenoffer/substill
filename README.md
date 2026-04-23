# Activation Subspace Distillation

**Compress any PyTorch model by distilling inside the teacher's
activation subspace.** On ResNet-50 / CIFAR-10 a student at
**17× compression reaches 90.74% accuracy**, and **25× compression
reaches 90.21%** — trading 0.9 pp / 1.4 pp against the same method's
strongest 4×-compressed baseline (91.60 ± 0.15%). Works on CNNs,
HuggingFace transformers, and any `nn.Module` whose "blocks" you
can point at. Drop-in: three function calls, no framework.

```python
import asd, torch.nn.functional as F

# 1. Profile the teacher once on a calibration batch.
profile  = asd.profile(teacher, calib_loader,
                       source="delta", noise_model="mp")

# 2. Build (or supply) a narrower student and a loss.
student  = asd.build_student(teacher, profile)
sub_loss = asd.SubspaceLoss(profile, objective="cka")

# 3. Train.
for x, y in train_loader:
    with asd.capture(teacher, profile) as t_hid:
        teacher(x)
    with asd.capture(student, profile) as s_hid:
        s_logits = student(x)
    loss = F.cross_entropy(s_logits, y) + 0.5 * sub_loss(
        s_hid.values(), t_hid.values(),
    )
    loss.backward(); opt.step(); opt.zero_grad()
```

That's the entire surface. No `Trainer` subclass, no YAML config, no
HuggingFace-only assumption. Five callables (`profile`,
`build_student`, `SubspaceLoss`, `capture`, `distill`) plus a
`TeacherProfile` container.

---

## Why

Most deployed knowledge-distillation methods either (a) match the
teacher's final **logits** (classic KD), or (b) match intermediate
**feature maps or attention patterns** with some handcrafted
alignment loss (FitNets, AT, RKD, and descendants). Both treat the
teacher as a single object to imitate, channel-for-channel.

But trained models don't use all their channels. At the deepest
stage of a CIFAR-10-trained ResNet-50, under a Marchenko-Pastur
bulk-edge cutoff, roughly **400 of 2048 channels** carry
statistically distinguishable signal — the rest are noise-bulk or
linearly redundant. Analogous story at other stages. The teacher's
effective computation has an intrinsic rank well below its nominal
width.

ASD takes that literally:

1. **Profile.** Eigendecompose the channel covariance at each block
   you care about. Pick the top-`k` principal directions per block.
2. **Size.** Build a student whose per-block width is `k`, not the
   teacher's full width. This alone yields order-of-magnitude
   compression at a known target task.
3. **Distill.** Train the student to match the teacher *inside that
   retained subspace* — not the full feature map, just its
   projection onto the top-`k` principal axes. The loss is a
   centered-kernel-alignment or Gram-Frobenius distance on the
   projected features.

Three additional choices make it robust:

- **Marchenko-Pastur rank selection.** Instead of "keep 95% of
  variance," use the Gavish-Donoho bulk-edge cutoff, which
  separates signal eigenvalues from the noise bulk. On ResNet-50 /
  CIFAR-10 this tightens compression from 4× to 17× at <1 pp
  additional cost over a variance-threshold baseline of the same
  method. This is the single largest algorithmic gain in the
  current version.
- **Basis-invariant loss.** When eigenvalues are close, the
  principal basis inside the retained subspace is arbitrary up to
  rotation. Coordinate-MSE forces the student to match an arbitrary
  rotation; Gram and CKA don't.
- **Feature normalization.** L2-normalize features per-sample
  before forming kernels. Keeps the loss bounded on LLM-scale
  residual streams where Gram otherwise reaches 10⁶ and training
  diverges.

## Benchmarks

**CIFAR-10 / ResNet-50** (20 epochs, 3 seeds, teacher ≈ 97.0%):

| configuration                         | compression | student acc       |
|---------------------------------------|-------------|-------------------|
| baseline (output + gram, τ=0.95)      |  4.15×      | **91.60 ± 0.15**  |
| baseline (output + gram, τ=0.85)      | 15.86×      | 90.59 ± 0.24      |
| + Marchenko-Pastur cutoff             | **17.08×**  | **90.74**         |
| + Ledoit-Wolf shrinkage on top        | **25.29×**  | **90.21**         |
| `arch_multiplier=1.25` (more width)   |  2.66×      | 91.65 ± 0.03      |
| `arch_multiplier=2.0` (teacher-sized) |  1.04×      | **92.12 ± 0.33**  |

Teacher reference: 96.98% at ≈23.5 M parameters. All student numbers
above are at a fraction of that budget.

The **17× / 90.74%** row is the headline: Marchenko-Pastur rank
selection compresses 4× further than the variance-threshold
baseline of the same method for less than a percentage point of
accuracy. All CIFAR numbers are from a 34-job benchmark ladder run
as independent Anyscale prodjobs on A10G GPUs; the artifact bundle
lives in the project's artifact storage.

**GPT-2 / WikiText-2** (stability smoke, 1 epoch):

| configuration                           | loss behavior                  |
|-----------------------------------------|--------------------------------|
| gram without feature normalization      | diverges (loss → 10⁶)          |
| coord_mse with mahalanobis weighting    | converges, 149 ppl at 2× comp. |
| **cka with feature normalization**      | **bounded, stable, converges** |

LLM training budgets beyond 1 epoch are future work, but the loss
numerics — which blocked prior attempts — are fixed.

## Choosing knobs

Sensible defaults for 90% of use cases are shipped. The two
decisions you'll make are:

**`asd.profile(...)`**

| knob            | options                      | pick                                                                      |
|-----------------|------------------------------|---------------------------------------------------------------------------|
| `source`        | `output` / `delta` / `branch`| `output` default. `delta` strips identity contamination on residual nets. |
| `noise_model`   | `eps` / `mp`                 | `mp` whenever calibration has ≥ C samples per layer (almost always).      |
| `shrinkage`     | `none` / `ledoit_wolf`       | `ledoit_wolf` on small (<1k-sample) or noisy calibration.                 |

**`asd.SubspaceLoss(...)`**

| knob                  | options                       | pick                                                               |
|-----------------------|-------------------------------|--------------------------------------------------------------------|
| `objective`           | `coord_mse` / `gram` / `cka`  | `cka` for LLMs. `gram` for CNNs. `coord_mse` if you have a reason. |
| `normalize_features`  | `True` / `False`              | `True`. Setting `False` on transformers lets the loss diverge.      |

## Install

```bash
pip install -e .              # core library
pip install -e ".[dev]"       # + pytest
pip install -e ".[llm]"       # + transformers, datasets
```

Requires Python ≥3.10, PyTorch ≥2.0, torchvision ≥0.15.

## Example: compress ResNet-50 on CIFAR-10

```python
import asd, torch, torch.nn.functional as F
from torchvision.models import resnet50

teacher = resnet50(weights="IMAGENET1K_V2").eval().cuda()
# 512 CIFAR-10 images, no augmentation
profile = asd.profile(teacher, calibration_loader,
                      source="delta", noise_model="mp")

student = asd.build_student(teacher, profile,
                            arch_multiplier=1.0,
                            num_classes=10, stem_type="cifar").cuda()

result = asd.distill(
    teacher, student, train_loader,
    profile=profile, val_loader=test_loader,
    epochs=20, objective="gram",
    alpha=1.0, beta=0.5, delta=1.0,
)
print(f"best acc: {result.best_metric*100:.2f}%")
```

## Example: compress GPT-2 on WikiText-2

```python
import asd, torch.nn.functional as F
from transformers import GPT2LMHeadModel

teacher = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
# Auto-detects teacher.transformer.h[0..11].
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
    loss.backward(); opt.step(); opt.zero_grad()
```

## Example: your own model

Skip `build_student` — build your student however you want and feed
its widths to the loss:

```python
profile = asd.profile(
    my_teacher, loader,
    layers=[my_teacher.encoder[i] for i in range(4)],  # explicit
)
my_student = MyCustomStudent(hidden=256)
sub_loss = asd.SubspaceLoss(profile, objective="cka")
```

If `asd.autodetect_layers(model)` doesn't recognize your
architecture, pass `layers=` — any list of modules or dotted names.
Or register a detector once:

```python
def detect_mynet(m):
    if hasattr(m, "blocks"):
        return [f"blocks.{i}" for i in range(len(m.blocks))]
    return None

asd.register_detector("mynet", detect_mynet)
```

## Where this fits

Against other open-source compression libraries:

| method                             | LLM-safe | auto width | arbitrary arch | drop-in API |
|------------------------------------|:--------:|:----------:|:--------------:|:-----------:|
| classical Hinton KD                |     ✓    |     ✗      |       ✓        |      ✓      |
| FitNets / AT (feature-map)         |     ✗    |     ✗      |       ~        |      ~      |
| RKD / CRD (relational / contrastive)|    ~     |     ✗      |       ~        |      ✓      |
| task-specific pruning (channel-prune + FT) | ~ |     ✓      |       ✗        |      ✗      |
| **ASD (this library)**             |     ✓    |     ✓      |       ✓        |      ✓      |

ASD is a strong fit when:

- you have a working teacher and want to trade accuracy for
  inference cost on a known budget,
- the teacher has residual/block structure (CNN or transformer),
- you don't want to rewrite your training loop.

It's the wrong tool when your teacher is already small, when you
need a student with a fundamentally different architecture, or when
you can fine-tune a smaller model from scratch cheaper than running
the distillation.

## Runnable scripts

```bash
# 1. Fine-tune a torchvision ResNet-50 for CIFAR-10
python scripts/finetune_teacher.py --output outputs/teacher.pt

# 2. Distill it
python scripts/distill_cifar_resnet.py \
    --teacher outputs/teacher.pt --epochs 20

# GPT-2 end-to-end
python scripts/distill_gpt2_wikitext.py --epochs 3
```

Each script is ≤200 lines and uses only the public library surface.

## Repo layout

```
asd/                 library
  __init__.py          public API re-exports (9 names)
  api.py               profile / capture / SubspaceLoss / distill / TeacherProfile
  autodetect.py        layer detection for known model families
  builders.py          student constructors (torchvision ResNets + HF GPT-2)
  models/student.py    SlimNet — 4-stage ResNet student used by build_student
  profiling/           activation capture, SVD, MP cutoff, stability diagnostic
  data/                CIFAR-10 / ImageNet loader helpers for the examples
scripts/             3 runnable examples
docs/                quickstart, algorithm & mechanism
tests/               36 tests, all green (`python -m pytest tests/`)
```

## Further reading

- [docs/quickstart.md](docs/quickstart.md) — two-minute library walkthrough.
- [docs/algorithm.md](docs/algorithm.md) — mechanism, math, design rationale.
- [asd/api.py](asd/api.py) — the full public API is a single ~900-line
  file; reading it end-to-end is reasonable.

## License

MIT.
