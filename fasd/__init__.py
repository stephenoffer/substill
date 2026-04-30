"""F-ASD: Functionally-Aligned Activation Subspace Distillation.

Name disambiguation: F-ASD is unrelated to Adversarial Score
Distillation or Adversarial Self-Distillation, which share the "ASD"
acronym in recent literature.

Core novelty: use output-preserving branch subspaces as the single
object that decides rank (behavioral rank selection by intervention),
architecture (width pruning + absorbed init), supervision (branchwise
Gram/CKA/Procrustes), and cache format.

Minimal usage::

    import fasd

    profile = fasd.profile(teacher, calib_loader)
    student = fasd.build_student(teacher, profile, absorbed_init=True)

    loss_fn = fasd.F_ASDLoss(profile, objective="procrustes")
    for batch in train_loader:
        with fasd.capture(teacher, profile) as t_hid:
            teacher(**batch)
        with fasd.capture(student, profile) as s_hid:
            s_logits = student(**batch).logits
        loss = loss_fn(
            dict(s_hid.items()), dict(t_hid.items())
        )
        loss.backward()

    # Or the full multi-stage driver:
    result = fasd.distill(
        teacher, student, train_loader,
        profile=profile,
        generative_kd="skew_kl",
        on_policy_start=0.5,
        quantize=True,
    )
"""

from __future__ import annotations

from .api import (
    BranchProfile,
    DistillResult,
    TeacherProfile,
    capture,
    profile,
)
from .autodetect import (
    BranchSpec,
    autodetect_branches,
    autodetect_layers,
    register as register_detector,
)
from .builders import build_student
from .compression.absorbed_init import absorbed_linear_init
from .compression.quantization import quantize_student, qad_finetune
from .compression.width_pruner import (
    StudentConfig,
    plan_progressive_stages,
    profile_to_student_config,
)
from .losses.generative_kd import (
    contrastive_response_loss,
    forward_kl,
    reverse_kl,
    skew_kl,
)
from .losses.procrustes import procrustes_distance
from .losses.subspace import (
    F_ASDLoss,
    Schedule,
    ScheduleStage,
    cka_distance,
    default_schedule,
    gram_distance,
)
from .profiling.behavioral_rank import choose_behavioral_rank
from .profiling.stability import (
    StabilityStats,
    bootstrap_principal_angles,
    stability_adjusted_rank,
)
from .profiling.streaming_pca import StreamingPCA
from .profiling.token_weighting import compute_weights
from .training.distill import distill
from .training.onpolicy import (
    HybridCollator,
    ReplayBuffer,
    RolloutBatch,
    generate_rollouts,
)
from .training.teacher_correction import correct_teacher


__all__ = [
    # profiles
    "BranchProfile",
    "BranchSpec",
    "DistillResult",
    "StabilityStats",
    "StreamingPCA",
    "Schedule",
    "ScheduleStage",
    "StudentConfig",
    "TeacherProfile",
    # classes
    "F_ASDLoss",
    "HybridCollator",
    "ReplayBuffer",
    "RolloutBatch",
    # functions
    "absorbed_linear_init",
    "autodetect_branches",
    "autodetect_layers",
    "bootstrap_principal_angles",
    "build_student",
    "capture",
    "choose_behavioral_rank",
    "cka_distance",
    "compute_weights",
    "contrastive_response_loss",
    "correct_teacher",
    "default_schedule",
    "distill",
    "forward_kl",
    "generate_rollouts",
    "gram_distance",
    "plan_progressive_stages",
    "procrustes_distance",
    "profile",
    "profile_to_student_config",
    "qad_finetune",
    "quantize_student",
    "register_detector",
    "reverse_kl",
    "skew_kl",
    "stability_adjusted_rank",
]

__version__ = "0.1.0"
