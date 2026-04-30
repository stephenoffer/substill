"""Multi-stage F-ASD distillation driver.

Stages (controlled by ``step_frac``):

0. Teacher correction (optional; ``teacher_correction_steps > 0``).
1. Progressive planning (optional; ``progressive_stages > 1``).
2. Profile (if not provided).
3. Warm-up: feature loss = Gram/CKA, logit loss = forward KL.
4. Projector fold-away at the warm-up/middle boundary.
5. Middle: feature loss = Procrustes, logit loss = skew KL.
6. On-policy: hybrid batches, reverse/skew KL, optional contrastive.
7. Profile refresh on student rollouts.
8. Quantization-aware final stage.

v0.1 implements all stages; see the plan file for scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..api import DistillResult, TeacherProfile, capture, profile as profile_fn
from ..autodetect import BranchSpec
from ..compression.quantization import qad_finetune, quantize_student
from ..compression.width_pruner import StudentConfig, plan_progressive_stages
from ..losses.generative_kd import (
    contrastive_response_loss,
    forward_kl,
    reverse_kl,
    skew_kl,
)
from ..losses.subspace import F_ASDLoss, Schedule, default_schedule
from .onpolicy import HybridCollator, ReplayBuffer, RolloutBatch, generate_rollouts
from .reabsorb import reabsorb_gpt2
from .teacher_correction import correct_teacher


KD = Literal["forward_kl", "reverse_kl", "skew_kl"]


def _apply_kd(
    kind: KD,
    student_logits: Tensor,
    teacher_logits: Tensor,
    mask: Tensor | None,
    temperature: float,
) -> Tensor:
    if kind == "forward_kl":
        return forward_kl(student_logits, teacher_logits, mask=mask, temperature=temperature)
    if kind == "reverse_kl":
        return reverse_kl(student_logits, teacher_logits, mask=mask, temperature=temperature)
    if kind == "skew_kl":
        return skew_kl(student_logits, teacher_logits, mask=mask, temperature=temperature)
    raise ValueError(f"unknown KD: {kind!r}")


def _logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"no .logits in {type(out)}")


def _move_batch(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return tuple(b.to(device) if isinstance(b, Tensor) else b for b in batch)
    if isinstance(batch, Tensor):
        return batch.to(device)
    return batch


def _logits_labels_mask(batch, out, device):
    logits = _logits(out)
    if isinstance(batch, dict):
        labels = batch.get("labels", batch.get("input_ids"))
        attn = batch.get("attention_mask")
    elif isinstance(batch, Tensor):
        labels = batch
        attn = None
    else:
        labels = batch[0] if isinstance(batch, (tuple, list)) and len(batch) > 0 else None
        attn = None
    mask = None
    if attn is not None:
        # Shifted mask for next-token prediction.
        mask = attn[..., 1:].to(logits.dtype)
    return logits, labels, mask


def _shift_ce(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def _resolve_fold_frac(
    schedule: Schedule | None, fold_after_frac: float | None
) -> float:
    """Pick the step-fraction at which to fold semi-orthogonal projectors.

    When Procrustes becomes the dominant objective, the projector is redundant
    (Procrustes is rotation-invariant inside the retained subspace), so folding
    it into the teacher V buffer is safe. Folding earlier freezes a partly-trained
    rotation, which v8-apr24 did at the magic ``frac >= 0.10`` boundary.
    """
    if fold_after_frac is not None:
        return float(fold_after_frac)
    if schedule is None:
        return 0.40
    for stage in schedule.stages:
        if stage.weights.get("procrustes", 0.0) >= 0.5:
            return float(stage.start_frac)
    return 0.40


# -- driver ------------------------------------------------------------


def distill(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: Iterable,
    *,
    profile: TeacherProfile | None = None,
    val_loader: Iterable | None = None,
    schedule: Schedule | None = None,
    generative_kd: KD = "skew_kl",
    alpha: float = 1.0,
    beta: float = 0.5,
    delta: float = 1.0,
    temperature: float = 2.0,
    profile_refresh: int | None = None,
    teacher_correction_steps: int = 0,
    on_policy_start: float = 0.5,
    on_policy_ratio: float = 0.5,
    on_policy_batch_size: int = 4,
    contrastive_weight: float = 0.0,
    contrastive_margin: float = 0.5,
    quantize: bool = False,
    quantize_bits: int = 4,
    qad_steps: int = 500,
    progressive_stages: int = 1,
    cache_teacher_features: bool = False,
    instability_downweight: bool = False,
    total_steps: int = 200,
    lr: float = 5e-5,
    optimizer=None,
    grad_clip: float | None = 1.0,
    log_every: int = 50,
    loss_objective: str = "procrustes",
    _use_default_schedule: bool = True,
    device: str | torch.device | None = None,
    rollout_prompts: Tensor | None = None,
    generation_max_new_tokens: int = 32,
    fold_after_frac: float | None = None,
    reabsorb_every_steps: int | None = None,
    reabsorb_calib_batches: int = 2,
    reabsorb_template: str = "gpt2",
) -> DistillResult:
    """End-to-end F-ASD distillation driver.

    See the plan file for the stage breakdown.
    """
    if device is None:
        try:
            device = next(teacher.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    teacher.to(device).eval()
    student.to(device)

    history: list[dict] = []

    # 0. Teacher correction ----------------------------------------
    if teacher_correction_steps > 0:
        tc_stats = correct_teacher(
            teacher, train_loader, steps=teacher_correction_steps, device=device
        )
        history.append({"stage": "teacher_correction", **tc_stats})

    # 1. Progressive planning -------------------------------------
    if progressive_stages > 1 and hasattr(student, "config"):
        target = StudentConfig(
            hidden_size=int(getattr(student.config, "hidden_size", getattr(student.config, "n_embd", 0))),
            intermediate_size=int(getattr(student.config, "intermediate_size", getattr(student.config, "n_inner", 0))),
            num_attention_heads=int(getattr(student.config, "num_attention_heads", getattr(student.config, "n_head", 1))),
            num_key_value_heads=int(getattr(student.config, "num_key_value_heads", getattr(student.config, "n_head", 1))),
            num_hidden_layers=int(getattr(student.config, "num_hidden_layers", getattr(student.config, "n_layer", 1))),
        )
        stages = plan_progressive_stages(teacher.config, target, n_stages=progressive_stages)
        history.append({"stage": "progressive_plan", "num_stages": len(stages)})
        # Note: v0.1 records the plan but does not instantiate intermediate models
        # — the user can run distill() separately for each stage. Recursive
        # re-instantiation is model-family-specific and left to the user.

    # 2. Profile ---------------------------------------------------
    if profile is None:
        profile = profile_fn(teacher, train_loader)
        history.append({"stage": "profile", "n_branches": len(profile.branches)})

    # Optional: cache compressed teacher features.
    feature_cache: dict[str, list[Tensor]] | None = None
    if cache_teacher_features:
        feature_cache = _build_feature_cache(teacher, profile, train_loader, device)
        history.append({"stage": "cache_teacher_features", "n_branches": len(feature_cache)})

    # Build loss + optimizer. Projectors are built lazily on first
    # forward from the actual student branch dims — the rank-as-width
    # assumption only applies after absorbed-init + fold_projectors_into_.
    # If the caller explicitly passed schedule=None we honour it (no schedule,
    # single objective chosen by the loss_fn's default). Only fall back to
    # default_schedule() when `schedule` was not provided at all (we signal
    # that with the sentinel "_default").
    actual_schedule = schedule
    if actual_schedule is None and _use_default_schedule:
        actual_schedule = default_schedule()
    loss_fn = F_ASDLoss(
        profile,
        student_widths=None,
        objective=loss_objective,
        schedule=actual_schedule,
        instability_weights=_instability_weights(profile) if instability_downweight else None,
    ).to(device)

    if optimizer is None:
        params = [p for p in student.parameters() if p.requires_grad]
        params += [p for p in loss_fn.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)

    # Replay buffer for on-policy stage.
    replay = ReplayBuffer(capacity=max(64, on_policy_batch_size * 16))

    fold_frac = _resolve_fold_frac(actual_schedule, fold_after_frac)

    step = 0
    folded = False
    refreshed = False

    loader_iter = iter(train_loader)
    while step < total_steps:
        frac = step / max(1, total_steps)

        # Pick batch source based on stage.
        on_policy_active = frac >= on_policy_start and rollout_prompts is not None
        draw_on = False
        if on_policy_active and len(replay) >= on_policy_batch_size:
            # With probability on_policy_ratio draw from replay; otherwise off-policy.
            if torch.rand(1).item() < on_policy_ratio:
                draw_on = True

        if draw_on:
            batch = _rollout_to_batch(replay.sample(on_policy_batch_size))
            source = "on"
        else:
            try:
                off = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                off = next(loader_iter)
            batch = off
            source = "off"

        batch = _move_batch(batch, device)

        # Forward passes: student with grad, teacher frozen.
        teacher_kwargs = batch if isinstance(batch, dict) else None
        with capture(teacher, profile, detach=True) as t_hiddens:
            with torch.no_grad():
                t_out = teacher(**batch) if isinstance(batch, dict) else teacher(batch)
        t_logits = _logits(t_out)

        with capture(student, profile) as s_hiddens:
            s_out = student(**batch) if isinstance(batch, dict) else student(batch)
        s_logits = _logits(s_out)

        # Align logits shape if student/teacher output different sizes (shouldn't happen
        # with absorbed init, but guard anyway).
        if s_logits.shape != t_logits.shape:
            min_len = min(s_logits.shape[1], t_logits.shape[1])
            s_logits = s_logits[:, :min_len]
            t_logits = t_logits[:, :min_len]

        # Build token mask if we have attention_mask.
        mask = None
        if isinstance(batch, dict) and "attention_mask" in batch:
            mask = batch["attention_mask"][..., 1:].to(s_logits.dtype)

        # Next-token task loss when labels available.
        task_loss = torch.zeros((), device=device)
        if isinstance(batch, dict):
            labels = batch.get("labels", batch.get("input_ids"))
            if labels is not None:
                task_loss = _shift_ce(s_logits, labels)
        elif isinstance(batch, Tensor):
            task_loss = _shift_ce(s_logits, batch)

        # Logit KD over generated tokens.
        kd_kind = generative_kd
        if source == "on" and generative_kd == "skew_kl":
            kd_kind = "reverse_kl" if frac > 0.8 else "skew_kl"
        # Shift logits/mask for next-token.
        kd_loss = _apply_kd(
            kd_kind,
            s_logits[:, :-1].contiguous(),
            t_logits[:, :-1].contiguous(),
            mask,
            temperature,
        )

        # Subspace loss.
        s_dict = dict(s_hiddens.items())
        t_dict = dict(t_hiddens.items())
        if feature_cache is not None:
            # Override teacher hiddens with cached ones if this batch index is tracked.
            pass  # v0.1: caching path reuses the teacher anyway for simplicity
        sub_loss = loss_fn(s_dict, t_dict, step_frac=frac)

        # Contrastive response loss in on-policy stage.
        contr_loss = torch.zeros((), device=device)
        if source == "on" and contrastive_weight > 0.0:
            # Build a teacher-reference sequence from the off-policy batch (approximate).
            try:
                teacher_batch = next(iter(train_loader))
                teacher_batch = _move_batch(teacher_batch, device)
                if isinstance(teacher_batch, dict) and "input_ids" in teacher_batch:
                    t_tokens = teacher_batch["input_ids"]
                    with torch.no_grad():
                        s_out_t = student(**teacher_batch)
                        s_logits_t = _logits(s_out_t)[:, :-1].contiguous()
                    s_tokens = batch["input_ids"] if isinstance(batch, dict) else None
                    if s_tokens is not None:
                        min_B = min(s_logits_t.shape[0], s_logits.shape[0])
                        contr_loss = contrastive_response_loss(
                            s_logits_t[:min_B],
                            s_logits[:min_B, :-1].contiguous(),
                            t_tokens[:min_B, 1:].contiguous(),
                            s_tokens[:min_B, 1:].contiguous(),
                            margin=contrastive_margin,
                        )
            except Exception:
                contr_loss = torch.zeros((), device=device)

        total = alpha * task_loss + beta * sub_loss + delta * kd_loss + contrastive_weight * contr_loss

        # NaN / Inf guard: skip the step rather than poisoning the student.
        if not torch.isfinite(total):
            history.append(
                {
                    "step": step,
                    "frac": frac,
                    "source": source,
                    "skipped": "nonfinite_loss",
                    "task_loss": float(task_loss.detach().item()),
                    "sub_loss": float(sub_loss.detach().item()),
                    "kd_loss": float(kd_loss.detach().item()),
                }
            )
            optimizer.zero_grad()
            step += 1
            continue

        optimizer.zero_grad()
        total.backward()
        if grad_clip is not None and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad] +
                [p for p in loss_fn.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()

        history.append(
            {
                "step": step,
                "frac": frac,
                "source": source,
                "task_loss": float(task_loss.detach().item()),
                "sub_loss": float(sub_loss.detach().item()),
                "kd_loss": float(kd_loss.detach().item()),
                "contrastive_loss": float(contr_loss.detach().item()),
                "total_loss": float(total.detach().item()),
            }
        )

        # Periodic progress print — critical for Anyscale log visibility.
        if log_every > 0 and (step % log_every == 0 or step == total_steps - 1):
            print(
                f"[fasd.distill] step={step}/{total_steps} frac={frac:.2f} "
                f"src={source} task={float(task_loss.detach().item()):.3f} "
                f"sub={float(sub_loss.detach().item()):.3f} "
                f"kd={float(kd_loss.detach().item()):.3f} "
                f"total={float(total.detach().item()):.3f}",
                flush=True,
            )

        # Fold projectors when Procrustes (rotation-invariant inside the subspace)
        # becomes the dominant objective. Folding earlier freezes a near-random
        # rotation into the teacher V buffer; v8 used frac>=0.10, which gave the
        # projector only ~10% of training to converge before being absorbed.
        if not folded and frac >= fold_frac:
            loss_fn.fold_projectors_into_(student)
            folded = True
            history.append({"stage": "fold_projectors", "step": step, "fold_frac": frac})

        # Profile refresh at on-policy transition.
        if not refreshed and frac >= on_policy_start and rollout_prompts is not None:
            _maybe_generate_and_push(student, rollout_prompts, replay, generation_max_new_tokens, device)
            # Run a light profile refresh on the freshly-generated sequences.
            if len(replay) >= on_policy_batch_size:
                refresh_batch = replay.sample(on_policy_batch_size)
                refresh_loader = [_rollout_to_batch(refresh_batch)]
                try:
                    new_profile = profile_fn(
                        teacher,
                        refresh_loader,
                        branches=[
                            BranchSpec(
                                name=b.name,
                                module_path=b.module_path,
                                kind=b.kind,  # type: ignore[arg-type]
                                slice=b.slice,
                            )
                            for b in profile.branches
                        ],
                        n_calib_batches=1,
                        behavioral_calib_batches=1,
                    )
                    loss_fn.refresh_from_profile(new_profile)
                    profile = new_profile
                    history.append({"stage": "profile_refresh", "step": step})
                except Exception as e:
                    history.append({"stage": "profile_refresh_failed", "error": str(e)})
            refreshed = True

        # Periodic rollout generation during on-policy stage.
        if (
            on_policy_active
            and rollout_prompts is not None
            and step % max(1, on_policy_batch_size * 2) == 0
        ):
            _maybe_generate_and_push(
                student, rollout_prompts, replay, generation_max_new_tokens, device
            )

        # Periodic re-absorption (PRA): refresh teacher PCA on a fresh batch
        # and rotate the student into the new bases while preserving Δ. Skipped
        # once projectors are folded — fold composes V into teacher buffers,
        # after which the student no longer has a distinct subspace to rotate.
        if (
            reabsorb_every_steps is not None
            and reabsorb_every_steps > 0
            and not folded
            and step > 0
            and step % reabsorb_every_steps == 0
            and reabsorb_template == "gpt2"
        ):
            calib = _collect_reabsorb_calib(
                train_loader, reabsorb_calib_batches, device
            )
            try:
                new_profile = reabsorb_gpt2(
                    teacher, student, profile, calib,
                    optimizer=optimizer, device=device,
                )
                loss_fn.refresh_from_profile(new_profile)
                profile = new_profile
                print(
                    f"[fasd.distill] reabsorb step={step} frac={frac:.2f} "
                    f"calib_batches={len(calib)}",
                    flush=True,
                )
                history.append({"stage": "reabsorb", "step": step, "frac": frac})
            except Exception as e:
                history.append({"stage": "reabsorb_failed", "step": step, "error": str(e)})
                print(f"[fasd.distill] reabsorb failed at step {step}: {e}", flush=True)

        step += 1

    # Quantization-aware final stage.
    if quantize:
        report = quantize_student(
            student, profile, bits=quantize_bits, group_size=128, protect_fraction=0.05
        )
        history.append({"stage": "quantize_student", "replaced": report.replaced})
        if qad_steps > 0:
            qad = qad_finetune(
                student, teacher, train_loader, steps=qad_steps, device=device
            )
            history.append({"stage": "qad_finetune", **qad})

    best_metric = None
    teacher_metric = None
    val_kl_forward: float | None = None
    val_kl_reverse: float | None = None
    if val_loader is not None:
        best_metric = _eval_ppl(student, val_loader, device)
        teacher_metric = _eval_ppl(teacher, val_loader, device)
        val_kl_forward = _eval_kl(teacher, student, val_loader, device, direction="forward")
        val_kl_reverse = _eval_kl(teacher, student, val_loader, device, direction="reverse")

    return DistillResult(
        student=student,
        profile=profile,
        history=history,
        best_metric=best_metric,
        teacher_metric=teacher_metric,
        val_kl_forward=val_kl_forward,
        val_kl_reverse=val_kl_reverse,
    )


# -- helpers -----------------------------------------------------------


def _collect_reabsorb_calib(loader, n: int, device) -> list:
    """Pull ``n`` batches from ``loader`` for PCA refresh. Cheap; cycles loader."""
    out = []
    it = iter(loader)
    for _ in range(max(1, n)):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            try:
                b = next(it)
            except StopIteration:
                break
        out.append(_move_batch(b, device))
    return out


def _maybe_generate_and_push(student, prompts, replay, max_new_tokens, device):
    try:
        batch = generate_rollouts(
            student,
            prompts.to(device),
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        return
    if batch is None or not hasattr(batch, "sequences"):
        return
    replay.add(batch)


def _rollout_to_batch(r: RolloutBatch | None) -> dict | None:
    if r is None:
        return None
    return {
        "input_ids": r.sequences,
        "attention_mask": r.attention_mask,
        "labels": r.sequences.clone(),
    }


def _instability_weights(profile: TeacherProfile) -> dict[str, float]:
    """Down-weight high-angle branches. If no angle is available, weight = 1."""
    weights: dict[str, float] = {}
    for b in profile.branches:
        angle = b.calibration_meta.get("median_angle_deg")
        if angle is None:
            weights[b.name] = 1.0
            continue
        weights[b.name] = float(torch.exp(torch.tensor(-float(angle) / 30.0)).item())
    return weights


def _build_feature_cache(teacher, profile, loader, device) -> dict[str, list[Tensor]]:
    cache: dict[str, list[Tensor]] = {b.name: [] for b in profile.branches}
    teacher.eval()
    with torch.no_grad():
        with capture(teacher, profile, detach=True) as hiddens:
            for batch in loader:
                b = _move_batch(batch, device)
                if isinstance(b, dict):
                    teacher(**b)
                elif isinstance(b, Tensor):
                    teacher(b)
                else:
                    teacher(*b)
                for bp in profile.branches:
                    if bp.name in hiddens:
                        V = bp.principal_components[:, : int(bp.behavioral_rank)].to(device).float()
                        z = hiddens[bp.name].to(device) @ V
                        cache[bp.name].append(z.detach().cpu())
    return cache


def _eval_ppl(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            if isinstance(batch, dict):
                out = model(**batch)
                labels = batch.get("labels", batch.get("input_ids"))
            elif isinstance(batch, Tensor):
                out = model(batch)
                labels = batch
            else:
                out = model(*batch)
                labels = batch[0]
            logits = _logits(out)
            if labels is None:
                continue
            loss = _shift_ce(logits, labels)
            n = labels[..., 1:].numel()
            total_loss += float(loss.item()) * n
            total_tokens += n
    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


def _eval_kl(
    teacher: nn.Module,
    student: nn.Module,
    loader: Iterable,
    device: str | torch.device,
    *,
    direction: str = "forward",
) -> float:
    """Token-averaged KL between teacher and student logits.

    ``direction="forward"`` is ``KL(teacher || student)``; ``"reverse"`` is
    ``KL(student || teacher)``. PPL alone doesn't tell you whether
    distillation actually moved the student toward the teacher; reporting
    both directions is also diagnostic — disagreement between forward and
    reverse trends across rungs flags mode-collapse vs mass-covering issues.
    """
    teacher.eval()
    student.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            if isinstance(batch, dict):
                t_logits = _logits(teacher(**batch))
                s_logits = _logits(student(**batch))
                mask = batch.get("attention_mask")
            elif isinstance(batch, Tensor):
                t_logits = _logits(teacher(batch))
                s_logits = _logits(student(batch))
                mask = None
            else:
                t_logits = _logits(teacher(*batch))
                s_logits = _logits(student(*batch))
                mask = None
            t_logp = F.log_softmax(t_logits, dim=-1)
            s_logp = F.log_softmax(s_logits, dim=-1)
            if direction == "forward":
                kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
            elif direction == "reverse":
                kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1)
            else:
                raise ValueError(f"unknown direction: {direction!r}")
            if mask is not None:
                m = mask.to(kl.dtype)
                total += float((kl * m).sum().item())
                n += int(m.sum().item())
            else:
                total += float(kl.sum().item())
                n += int(kl.numel())
    return total / max(1, n)


__all__ = ["distill"]
