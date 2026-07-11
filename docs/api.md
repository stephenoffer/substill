# API reference

Everything below is available directly on the top-level `substill` namespace
(`import substill`).

## Learned Restriction Distillation (recommended)

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   substill.learned_restriction_distill
   substill.LearnedRestriction
   substill.LRDConfig
   substill.LRDResult
   substill.plan_restricted_geometry
```

## FSD/CPSD pipeline (earlier method)

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   substill.FSDPipeline
   substill.FSDConfig
```

## Profiling

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   substill.profile
   substill.capture
   substill.TeacherProfile
   substill.BranchProfile
   substill.BranchSpec
   substill.StreamingPCA
   substill.choose_behavioral_rank
   substill.compute_weights
   substill.autodetect_branches
   substill.autodetect_layers
   substill.register_detector
   substill.StabilityStats
   substill.bootstrap_principal_angles
   substill.stability_adjusted_rank
```

## Student construction & compression

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   substill.build_student
   substill.StudentConfig
   substill.profile_to_student_config
   substill.plan_progressive_stages
   substill.absorbed_linear_init
   substill.quantize_student
   substill.qad_finetune
```

## Losses & schedules

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   substill.F_ASDLoss
   substill.gram_distance
   substill.cka_distance
   substill.procrustes_distance
   substill.forward_kl
   substill.reverse_kl
   substill.skew_kl
   substill.contrastive_response_loss
   substill.Schedule
   substill.ScheduleStage
   substill.default_schedule
```

## Training

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   substill.distill
   substill.DistillResult
   substill.correct_teacher
   substill.generate_rollouts
   substill.HybridCollator
   substill.ReplayBuffer
   substill.RolloutBatch
```
