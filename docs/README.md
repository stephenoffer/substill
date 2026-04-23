# ASD documentation

The [top-level README](../README.md) has the pitch and the API
overview. This directory has the technical docs.

| file                           | what it is                                |
|--------------------------------|-------------------------------------------|
| [quickstart.md](quickstart.md) | Two-minute library walkthrough            |
| [algorithm.md](algorithm.md)   | Mechanism, math, design rationale         |

## One-paragraph description

`asd.profile(teacher, calib_loader)` runs a calibration batch through
the teacher with forward hooks, accumulates channel covariance at
each block, eigendecomposes, and keeps the top-`k` principal
directions per block (with a Marchenko-Pastur bulk-edge cutoff to
reject noise-bulk eigenvalues). `asd.SubspaceLoss(profile)` is an
`nn.Module` loss that projects student hidden states into the
teacher's retained subspace and matches them with centered kernel
alignment (scale-invariant, best for LLMs), Gram-Frobenius
(basis-invariant, good for CNNs), or coordinate MSE. The student
plus the loss's per-layer linear projections train jointly in
whatever training loop you already have.

## Read this file if you want to:

- Use the library on your own model: [quickstart.md](quickstart.md).
- Understand the method: [algorithm.md](algorithm.md).
- Skim the code: [`asd/api.py`](../asd/api.py). Single file, heavily
  commented.
- Reproduce the benchmarks in the top-level README:
  [`scripts/finetune_teacher.py`](../scripts/finetune_teacher.py) and
  [`scripts/distill_cifar_resnet.py`](../scripts/distill_cifar_resnet.py).
