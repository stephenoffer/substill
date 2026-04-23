"""Activation Subspace Distillation (ASD).

Compress a PyTorch teacher model by distilling inside its activation
subspace. Works on torchvision ResNets, HuggingFace GPT-2 / Llama /
Mistral-style transformers, and any nn.Module where you can point at
a list of "stage" or "block" modules.

## Minimal usage

    import asd

    # 1. Profile the teacher once.
    profile = asd.profile(teacher, calibration_loader,
                          source="delta", noise_model="mp")

    # 2. Plug the subspace loss into your training loop.
    loss_fn = asd.SubspaceLoss(profile, objective="cka")
    for x, y in loader:
        with asd.capture(teacher, profile) as t_hid:
            teacher(x)
        with asd.capture(student, profile) as s_hid:
            s_logits = student(x)
        loss = F.cross_entropy(s_logits, y) + 0.5 * loss_fn(
            s_hid.values(), t_hid.values(),
        )
        loss.backward()

    # 3. Or go one call:
    result = asd.distill(teacher, student, train_loader, epochs=20)

## Public API

- `profile(...)` — run the teacher over a calibration loader and return
  a `TeacherProfile` with per-layer principal components, eigenvalues,
  effective ranks.
- `TeacherProfile` — save/load-able snapshot of what a teacher computed.
- `capture(model, profile)` — context manager that hooks `model` at the
  same layer names as the profile and exposes hidden states.
- `SubspaceLoss(profile, objective=...)` — `nn.Module` feature-
  distillation loss you add to your training step.
- `build_student(template, profile)` — narrow-student constructor for
  known families (torchvision ResNets, HF GPT-2).
- `distill(teacher, student, loader, ...)` — batteries-included one-
  call pipeline (profile → loss → train) for classification tasks.
- `autodetect_layers(model)` — propose layer names for known model
  families.
- `register_detector(family, fn)` — add your own layer detector.

Everything else in `asd.*` is implementation detail.
"""

from .api import (
    DistillResult,
    SubspaceLoss,
    TeacherProfile,
    build_student,
    capture,
    distill,
    profile,
)
from .autodetect import autodetect_layers, register as register_detector

__all__ = [
    "DistillResult",
    "SubspaceLoss",
    "TeacherProfile",
    "autodetect_layers",
    "build_student",
    "capture",
    "distill",
    "profile",
    "register_detector",
]

__version__ = "1.0.0"
