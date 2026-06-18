"""F-ASD: Functionally-Aligned Activation Subspace Distillation.

Name disambiguation: F-ASD is unrelated to Adversarial Score
Distillation or Adversarial Self-Distillation, which share the "ASD"
acronym in recent literature.

Core idea: use output-preserving branch subspaces as the single object
that decides rank (behavioral rank selection by intervention),
architecture (width pruning + absorbed init), supervision (branchwise
Gram/CKA/Procrustes), and cache format. On top of that, :class:`FSDPipeline`
adds the novel CPSD system (Circuit-Preserving Subspace Distillation):
circuit-preserving init + factors trained on the Stiefel manifold against a
distillation loss.

Recommended entry point — the one-call pipeline::

    import fasd

    pipe = fasd.FSDPipeline(
        teacher,
        config=fasd.FSDConfig(
            arch_multiplier=0.5,
            use_cpsd_factored=True,
            generative_kd="skew_kl",
        ),
    )
    result = pipe.run(calib_loader, train_loader)  # profile -> build -> (convert) -> distill
    student = pipe.student  # the trained, compressed student (result is a DistillResult)

Lower-level / advanced — drive the stages yourself::

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

    # ...or the multi-stage driver behind FSDPipeline:
    result = fasd.distill(
        teacher, student, train_loader,
        profile=profile, generative_kd="skew_kl",
        on_policy_start=0.5, quantize=True,
    )

``convert_gpt2_to_factored`` / ``convert_llama_to_factored`` are advanced
post-build CPSD helpers (used internally by ``FSDPipeline`` when
``use_cpsd_factored=True``); call them directly only for custom pipelines.
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
)
from .autodetect import (
    register as register_detector,
)
from .builders import build_student
from .compression.absorbed_init import absorbed_linear_init
from .compression.quantization import qad_finetune, quantize_student
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
from .pipeline import (
    FSDConfig,
    FSDPipeline,
    convert_gpt2_to_factored,
    convert_llama_to_factored,
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
    # --- Pipeline: the recommended one-call entry point ---
    "FSDPipeline",
    "FSDConfig",
    # --- Profiling: measure the teacher's branch subspaces ---
    "profile",
    "capture",
    "BranchProfile",
    "TeacherProfile",
    "BranchSpec",
    "StreamingPCA",
    "choose_behavioral_rank",
    "compute_weights",
    "autodetect_branches",
    "autodetect_layers",
    "register_detector",
    "StabilityStats",
    "bootstrap_principal_angles",
    "stability_adjusted_rank",
    # --- Student construction & compression ---
    "build_student",
    "StudentConfig",
    "profile_to_student_config",
    "plan_progressive_stages",
    "absorbed_linear_init",
    "quantize_student",
    "qad_finetune",
    # --- Losses & schedules ---
    "F_ASDLoss",
    "gram_distance",
    "cka_distance",
    "procrustes_distance",
    "forward_kl",
    "reverse_kl",
    "skew_kl",
    "contrastive_response_loss",
    "Schedule",
    "ScheduleStage",
    "default_schedule",
    # --- Training ---
    "distill",
    "DistillResult",
    "correct_teacher",
    "generate_rollouts",
    "HybridCollator",
    "ReplayBuffer",
    "RolloutBatch",
    # --- CPSD post-build conversions (advanced) ---
    "convert_gpt2_to_factored",
    "convert_llama_to_factored",
]

__version__ = "0.2.0"
