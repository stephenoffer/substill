"""ASD training loop — orchestrates teacher, student, projectors, and loss."""

from __future__ import annotations

import time
import warnings
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses.combined_loss import ASDLoss
from ..models.projectors import SubspaceProjectorBank
from ..models.student import SlimNet
from ..models.teacher import TeacherWrapper
from .scheduler import BetaWarmupScheduler, LossWeightScheduler


class ASDTrainer:
    """Orchestrates ASD training: teacher inference → student forward → projection → loss → update.

    LR scheduling convention
    ------------------------
    The trainer does its own linear LR warmup over `lr_warmup_epochs` epochs
    and calls `lr_scheduler.step()` exactly once per post-warmup epoch. If the
    caller hands in `CosineAnnealingLR(T_max=num_epochs)`, the cosine phase
    will be under-stepped by `lr_warmup_epochs` and never reach its nominal
    minimum. Pass `T_max=num_epochs - lr_warmup_epochs` instead — the trainer
    will warn once if it detects the common mismatch.
    """

    def __init__(
        self,
        teacher: TeacherWrapper,
        student: SlimNet,
        projectors: SubspaceProjectorBank,
        loss_fn: ASDLoss,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        loss_scheduler: LossWeightScheduler,
        device: str = "cpu",
        beta_scheduler: BetaWarmupScheduler | None = None,
        lr_warmup_epochs: int = 0,
        base_lr: float | None = None,
        keep_best: bool = True,
        restore_best_on_exit: bool = True,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.projectors = projectors.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_scheduler = loss_scheduler
        self.beta_scheduler = beta_scheduler
        self.lr_warmup_epochs = lr_warmup_epochs
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.keep_best = keep_best
        self.restore_best_on_exit = restore_best_on_exit
        self.device = device
        # Best-model tracking — saves a cheap CPU copy of the student's params
        # when its eval accuracy improves, to avoid keeping the end-of-training
        # state when cosine annealing overshoots.
        self._best_acc = 0.0
        self._best_state: dict | None = None

        # Ensure teacher is frozen
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        self.projectors.train()

        gamma_scale = self.loss_scheduler.get_gamma_scale(epoch)
        beta_scale = self.beta_scheduler.get_beta_scale(epoch) if self.beta_scheduler else 1.0

        # LR warmup — linear ramp from 10% of base LR up to base over
        # `lr_warmup_epochs`, after which the underlying lr_scheduler takes
        # over (typically cosine). Prevents early subspace-loss spikes from
        # pushing the student into a bad minimum with a large LR.
        if self.lr_warmup_epochs and epoch < self.lr_warmup_epochs:
            warmup_frac = (epoch + 1) / self.lr_warmup_epochs
            warmup_lr = self.base_lr * (0.1 + 0.9 * warmup_frac)
            for pg in self.optimizer.param_groups:
                pg["lr"] = warmup_lr

        running: dict[str, float] = {"total": 0, "task": 0, "subspace": 0, "sparsity": 0, "logit": 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Teacher forward (no grad) — keep logits for KD
            with torch.no_grad():
                teacher_logits, teacher_features = self.teacher(images)

            # Student forward
            student_logits, student_features = self.student(images)

            # Project student features to teacher subspace
            student_projected = self.projectors(student_features)

            # Compute loss
            losses = self.loss_fn(
                student_logits=student_logits,
                student_projected=student_projected,
                student_features=student_features,
                teacher_features=teacher_features,
                labels=labels,
                gamma_scale=gamma_scale,
                beta_scale=beta_scale,
                teacher_logits=teacher_logits,
            )

            # Backward + step
            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.projectors.parameters()),
                max_norm=5.0,
            )
            self.optimizer.step()

            # Accumulate — guard "logit" since the loss may not always produce it
            for k in running:
                if k in losses:
                    running[k] += losses[k].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{running['total']/num_batches:.4f}",
                "task": f"{running['task']/num_batches:.4f}",
                "sub": f"{running['subspace']/num_batches:.4f}",
                "kd": f"{running['logit']/num_batches:.4f}",
            })

        # Only advance the cosine schedule after warmup completes, so warmup
        # actually uses its full ramp.
        if epoch >= self.lr_warmup_epochs:
            self.lr_scheduler.step()

        return {k: v / max(num_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate student accuracy on a dataset."""
        self.student.eval()

        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.student(images)
            loss = nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            num_batches += 1

        return {
            "accuracy": correct / max(total, 1),
            "loss": total_loss / max(num_batches, 1),
        }

    def restore_best(self) -> bool:
        """Load the best-val checkpoint into `self.student` and `self.projectors`.

        Returns True if a checkpoint was restored, False if none was saved
        (e.g., `keep_best=False` or training ran zero epochs).
        """
        if self._best_state is None:
            return False
        self.student.load_state_dict({
            k: v.to(self.device) for k, v in self._best_state["student"].items()
        })
        self.projectors.load_state_dict({
            k: v.to(self.device) for k, v in self._best_state["projectors"].items()
        })
        return True

    def _warn_if_cosine_tmax_mismatch(self, num_epochs: int) -> None:
        """Catch the common footgun: CosineAnnealingLR with T_max=num_epochs,
        which under-steps by `lr_warmup_epochs` and never reaches its min."""
        if not self.lr_warmup_epochs:
            return
        t_max = getattr(self.lr_scheduler, "T_max", None)
        if t_max is None:
            return
        expected = num_epochs - self.lr_warmup_epochs
        if t_max == num_epochs and expected > 0:
            warnings.warn(
                f"lr_scheduler.T_max={t_max} but lr_warmup_epochs="
                f"{self.lr_warmup_epochs}; cosine will only be stepped "
                f"{expected} times and won't reach eta_min. Construct the "
                f"scheduler with T_max={expected} to fix.",
                stacklevel=2,
            )

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        log_interval: int = 1,
    ) -> list[dict[str, Any]]:
        """Full training loop."""
        history = []

        print(f"\nStarting ASD training for {num_epochs} epochs")
        print(f"  Student params: {self.student.count_parameters():,}")
        projector_params = sum(p.numel() for p in self.projectors.parameters())
        if projector_params > 0:
            print(f"  Projector params: {projector_params:,} (trained jointly)")
        print(f"  Device: {self.device}")
        print()

        self._warn_if_cosine_tmax_mismatch(num_epochs)

        best_acc = 0.0

        for epoch in range(num_epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluate
            eval_metrics = self.evaluate(test_loader)

            elapsed = time.time() - t0
            gamma_scale = self.loss_scheduler.get_gamma_scale(epoch)
            lr = self.optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "eval": eval_metrics,
                "lr": lr,
                "gamma_scale": gamma_scale,
                "elapsed": elapsed,
            }
            history.append(record)

            if eval_metrics["accuracy"] > best_acc:
                best_acc = eval_metrics["accuracy"]
                self._best_acc = best_acc
                if self.keep_best:
                    # Cheap state snapshot to CPU — lets us restore the best
                    # student at end of training, not the final cosine-annealed
                    # one which can be worse after LR drops to near-zero.
                    self._best_state = {
                        "student": {k: v.detach().cpu().clone()
                                    for k, v in self.student.state_dict().items()},
                        "projectors": {k: v.detach().cpu().clone()
                                       for k, v in self.projectors.state_dict().items()},
                    }

            if epoch % log_interval == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"loss {train_metrics['total']:.4f} "
                    f"(task={train_metrics['task']:.4f} "
                    f"sub={train_metrics['subspace']:.4f} "
                    f"spar={train_metrics['sparsity']:.4f}) | "
                    f"acc {eval_metrics['accuracy']*100:.2f}% | "
                    f"best {best_acc*100:.2f}% | "
                    f"lr {lr:.5f} | "
                    f"γ_scale {gamma_scale:.2f} | "
                    f"{elapsed:.1f}s"
                )

        # Restore best-val weights so the object returned to the caller matches
        # `best_acc` — previously the student kept its final (cosine-annealed)
        # weights even with keep_best=True, silently contradicting the log.
        if self.keep_best and self.restore_best_on_exit and self._best_state is not None:
            self.restore_best()
            print(f"Restored best-val student (acc {best_acc*100:.2f}%).")

        print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")
        return history
