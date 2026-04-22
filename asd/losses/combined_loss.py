"""Combined ASD Loss — task + subspace matching + sparsity pattern + logit KD."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..profiling.svd_analysis import LayerProfile
from .attention_loss import AttentionTransferLoss
from .relation_loss import RelationalLoss
from .subspace_loss import SubspaceMatchingLoss
from .sparsity_loss import SparsityPatternLoss


_LOSS_NAMES = ("task", "subspace", "sparsity", "logit", "relation", "attention")


class ASDLoss(nn.Module):
    """Combined Activation Subspace Distillation loss.

    Components:
      - L_task     : CE on ground truth labels
      - L_subspace : match student to teacher's principal activation subspace
      - L_sparsity : match activation sparsity / histogram patterns
      - L_logit    : (optional) Hinton KD — KL divergence between softened
                     student/teacher logits. Free, strong distillation signal.

    Two combination strategies:

      - "fixed" (default): total = α·L_task + β·L_sub + γ·L_spar + δ·L_logit
        (where γ is subject to warmup scheduling).

      - "uncertainty": Kendall & Gal (2018) uncertainty weighting. Each loss
        carries a learnable parameter `s_i = log σ_i²`; the total is
            total = Σ_i 0.5·exp(−s_i)·L_i + 0.5·s_i
        which is the canonical form for homoscedastic regression (a factor of
        2 different from a previous incorrect parametrization that used
        `exp(−s)·L + s` directly).
    """

    def __init__(
        self,
        profiles: list[LayerProfile],
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        delta: float = 1.0,
        epsilon: float = 0.0,          # relational loss weight (opt-in)
        zeta: float = 0.0,             # attention-transfer loss weight (opt-in)
        sv_weighted: bool = True,
        num_bins: int = 64,
        subspace_mode: str = "spatial",
        sv_weighting: str = "sqrt",
        subspace_normalize_features: bool = False,
        subspace_stage_aggregation: str = "last",
        sparsity_ratio_loss: str = "bce",
        sparsity_adaptive_tau: bool = True,
        use_logit_kd: bool = True,
        logit_temperature: float = 4.0,
        use_relation: bool = False,
        use_attention: bool = False,
        combination: str = "fixed",
        auto_normalize: bool = False,
        auto_norm_momentum: float = 0.95,
    ):
        super().__init__()
        if combination not in ("fixed", "uncertainty"):
            raise ValueError(f"combination must be 'fixed' or 'uncertainty', got {combination!r}")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.use_logit_kd = use_logit_kd
        # Enable flags are now *only* controlled by the explicit flags — the
        # old `use_relation = use_relation or epsilon > 0` implicit coupling
        # silently enabled RKD whenever a caller bumped ε and forgot to also
        # flip the boolean, which made ablations unreproducible.
        self.use_relation = use_relation
        self.use_attention = use_attention
        self.logit_temperature = logit_temperature
        self.combination = combination
        self.auto_normalize = auto_normalize
        self.auto_norm_momentum = auto_norm_momentum

        self.subspace_loss = SubspaceMatchingLoss(
            profiles,
            sv_weighted=sv_weighted,
            mode=subspace_mode,
            sv_weighting=sv_weighting,
            normalize_features=subspace_normalize_features,
            stage_aggregation=subspace_stage_aggregation,
        )
        self.sparsity_loss = SparsityPatternLoss(
            profiles,
            num_bins=num_bins,
            ratio_loss=sparsity_ratio_loss,
            adaptive_tau=sparsity_adaptive_tau,
        )
        self.relation_loss = RelationalLoss() if self.use_relation else None
        self.attention_loss = AttentionTransferLoss() if self.use_attention else None

        if combination == "uncertainty":
            # One `s_i = log σ_i²` per active loss term, in the canonical order
            # (task, subspace, sparsity, logit?, relation?, attention?).
            n = 3 + int(use_logit_kd) + int(self.use_relation) + int(self.use_attention)
            self.log_sigmas = nn.Parameter(torch.zeros(n))
        else:
            self.log_sigmas = None

        # EMA magnitudes of each loss component, stored as persistent buffers so
        # they survive checkpoint save/load. `ema_count` tracks how many updates
        # we've applied so the first few batches use a true running mean rather
        # than the raw EMA (which would bias toward the first batch's magnitude
        # and underweight every subsequent one).
        for name in _LOSS_NAMES:
            self.register_buffer(f"ema_{name}", torch.tensor(1.0))
        self.register_buffer("ema_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        student_logits: Tensor,
        student_projected: list[Tensor],
        student_features: list[Tensor],
        teacher_features: list[Tensor],
        labels: Tensor,
        gamma_scale: float = 1.0,
        beta_scale: float = 1.0,
        teacher_logits: Tensor | None = None,
    ) -> dict[str, Tensor]:
        device = student_logits.device
        loss_task = F.cross_entropy(student_logits, labels)
        loss_subspace = self.subspace_loss(student_projected, teacher_features)
        loss_sparsity = self.sparsity_loss(student_features)

        if self.use_logit_kd and teacher_logits is not None:
            T = self.logit_temperature
            # Standard KD: T² · KL(softmax(t/T) || softmax(s/T))
            s_log = F.log_softmax(student_logits / T, dim=1)
            t_prob = F.softmax(teacher_logits / T, dim=1)
            loss_logit = F.kl_div(s_log, t_prob, reduction="batchmean") * (T * T)
        else:
            loss_logit = torch.zeros((), device=device)

        # Optional RKD on last-stage GAP features of student (projected) + teacher
        if self.use_relation and len(student_projected) == len(teacher_features):
            s_last = student_projected[-1].mean(dim=(2, 3))   # (B, k_last)
            t_last_pooled = teacher_features[-1].mean(dim=(2, 3))  # (B, C_last)
            comp = getattr(self.subspace_loss, "components_3")
            t_last = t_last_pooled @ comp  # (B, k_last)
            loss_relation = self.relation_loss(s_last, t_last)
        else:
            loss_relation = torch.zeros((), device=device)

        # Optional attention transfer (L2-normalized spatial attention maps)
        if self.use_attention and len(student_features) == len(teacher_features):
            loss_attention = self.attention_loss(student_features, teacher_features)
        else:
            loss_attention = torch.zeros((), device=device)

        # Update EMA magnitudes once per forward, in a single place — whether or
        # not the caller enabled auto_normalize. Makes the diagnostic magnitude
        # available to callers (e.g., for logging) even when α/β/γ weights are
        # applied raw.
        self._update_ema(
            task=loss_task, subspace=loss_subspace, sparsity=loss_sparsity,
            logit=loss_logit, relation=loss_relation, attention=loss_attention,
        )

        if self.combination == "uncertainty":
            total = self._uncertainty_combine(
                loss_task, loss_subspace, loss_sparsity, loss_logit, loss_relation,
                loss_attention, gamma_scale,
            )
        else:
            if self.auto_normalize:
                norm_task = loss_task / self.ema_task.clamp(min=1e-6)
                norm_sub = loss_subspace / self.ema_subspace.clamp(min=1e-6)
                norm_spar = loss_sparsity / self.ema_sparsity.clamp(min=1e-6)
                norm_logit = (
                    loss_logit / self.ema_logit.clamp(min=1e-6)
                    if (self.use_logit_kd and teacher_logits is not None)
                    else loss_logit
                )
                norm_rel = (
                    loss_relation / self.ema_relation.clamp(min=1e-6)
                    if self.use_relation else loss_relation
                )
                norm_att = (
                    loss_attention / self.ema_attention.clamp(min=1e-6)
                    if self.use_attention else loss_attention
                )
            else:
                norm_task, norm_sub, norm_spar = loss_task, loss_subspace, loss_sparsity
                norm_logit, norm_rel, norm_att = loss_logit, loss_relation, loss_attention

            eff_gamma = self.gamma * gamma_scale
            eff_beta = self.beta * beta_scale
            total = (
                self.alpha * norm_task
                + eff_beta * norm_sub
                + eff_gamma * norm_spar
            )
            if self.use_logit_kd and teacher_logits is not None:
                total = total + self.delta * norm_logit
            if self.use_relation:
                total = total + self.epsilon * norm_rel
            if self.use_attention:
                total = total + self.zeta * norm_att

        return {
            "total": total,
            "task": loss_task.detach(),
            "subspace": loss_subspace.detach(),
            "sparsity": loss_sparsity.detach(),
            "logit": loss_logit.detach(),
            "relation": loss_relation.detach(),
            "attention": loss_attention.detach(),
        }

    def _update_ema(self, **losses: Tensor) -> None:
        """Blended-EMA update: unbiased mean for the first few batches, then
        exponential. Stored as buffers so checkpoint save/restore preserves
        the normalization calibration — the previous `dict` attribute silently
        dropped state on reload.
        """
        count = int(self.ema_count.item())
        momentum = self.auto_norm_momentum
        # Dynamic bias correction: during the first `1/(1-m)` batches, use a
        # simple running mean; afterwards, use the target EMA.
        warmup_len = max(1, int(round(1.0 / max(1 - momentum, 1e-6))))
        for name, val in losses.items():
            buf: Tensor = getattr(self, f"ema_{name}")
            detached = val.detach().to(buf.device).clamp(min=1e-6)
            if count < warmup_len:
                # (n·old + new) / (n+1) — unbiased for n+1 observations.
                new = (buf * count + detached) / (count + 1)
            else:
                new = momentum * buf + (1 - momentum) * detached
            buf.copy_(new)
        self.ema_count.add_(1)

    # Back-compat alias — callers that reached into the old private attribute
    # can keep doing so. Returns the same values the buffers now hold.
    @property
    def _loss_ema(self) -> dict[str, Tensor]:
        return {name: getattr(self, f"ema_{name}") for name in _LOSS_NAMES}

    def _uncertainty_combine(
        self,
        loss_task: Tensor,
        loss_subspace: Tensor,
        loss_sparsity: Tensor,
        loss_logit: Tensor,
        loss_relation: Tensor,
        loss_attention: Tensor,
        gamma_scale: float,
    ) -> Tensor:
        # Kendall & Gal canonical form: with s = log σ², the homoscedastic
        # Gaussian NLL reduces to 0.5·exp(−s)·L + 0.5·s (plus a constant).
        # Previously we used exp(−s)·L + s which mis-weights by 2× and doubles
        # the regularizer — correct only up to scale, but inconsistent with
        # any published benchmark using this formulation.
        s = self.log_sigmas
        losses = [loss_task, loss_subspace, gamma_scale * loss_sparsity]
        if self.use_logit_kd:
            losses.append(loss_logit)
        if self.use_relation:
            losses.append(loss_relation)
        if self.use_attention:
            losses.append(loss_attention)
        total = torch.zeros((), device=loss_task.device)
        for i, L in enumerate(losses):
            total = total + 0.5 * torch.exp(-s[i]) * L + 0.5 * s[i]
        return total
