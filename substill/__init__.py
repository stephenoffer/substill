"""substill: activation-subspace distillation for compressing models.

The one verified, recommended method is **Learned Restriction Distillation (LRD)**:
parameterize the compressed student as an exact restriction ``W_s = V^T W_T V`` of the
teacher and train the residual-stream projection ``V`` on the Stiefel manifold against
the KD loss, through the whole network. It beats the strongest frozen-basis baseline by
6.8% PPL at ~6 sigma on ``JackFram/llama-160m`` (n=3), a win that holds at matched
wall-clock and grows with teacher scale. Full study and honest scope:
``docs/learned_restriction.md``.

Recommended entry point — the one-call LRD API (Llama-family decoders)::

    import substill

    result = substill.learned_restriction_distill(
        teacher, train_loader,
        config=substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000),
    )
    student = result.student   # a plain LlamaForCausalLM, zero inference overhead

Drive the phases yourself for inspection / checkpointing::

    lrd = substill.LearnedRestriction(teacher, config)
    lrd.prepare(calib_loader).fit(train_loader)
    student = lrd.fold()

Earlier method (kept for reference; its headline numbers were **not** reproduced under
compute matching — see ``docs/init_findings.md``): the FSD/CPSD pipeline. It profiles
output-preserving branch subspaces, builds a width-narrowed absorbed-init student, and
optionally trains circuit-preserving factors on the Stiefel manifold::

    pipe = substill.FSDPipeline(teacher, config=substill.FSDConfig(arch_multiplier=0.5))
    result = pipe.run(calib_loader, train_loader)
    student = pipe.student

The lower-level stages — :func:`profile`, :func:`build_student`, :func:`distill`, and
the basis-invariant :class:`F_ASDLoss` — remain public for custom pipelines.

Name disambiguation: this F-ASD is unrelated to Adversarial Score Distillation or
Adversarial Self-Distillation, which share the "ASD" acronym in recent literature.
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
from .lrd import (
    LearnedRestriction,
    LRDConfig,
    LRDResult,
    learned_restriction_distill,
    plan_restricted_geometry,
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
    # --- LRD: the verified, recommended method (docs/learned_restriction.md) ---
    "learned_restriction_distill",
    "LearnedRestriction",
    "LRDConfig",
    "LRDResult",
    "plan_restricted_geometry",
    # --- FSD/CPSD pipeline (earlier method; numbers superseded, see docs/init_findings.md) ---
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

__version__ = "0.4.0"
