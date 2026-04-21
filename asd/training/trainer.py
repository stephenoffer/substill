"""ASD training loop — orchestrates teacher, student, projectors, and loss."""

from __future__ import annotations

import time
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
from .scheduler import LossWeightScheduler


class ASDTrainer:
    """Orchestrates ASD training: teacher inference → student forward → projection → loss → update."""

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
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.projectors = projectors.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_scheduler = loss_scheduler
        self.device = device

        # Ensure teacher is frozen
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        self.projectors.train()

        gamma_scale = self.loss_scheduler.get_gamma_scale(epoch)

        running: dict[str, float] = {"total": 0, "task": 0, "subspace": 0, "sparsity": 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Teacher forward (no grad)
            with torch.no_grad():
                _, teacher_features = self.teacher(images)

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
            )

            # Backward + step
            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.projectors.parameters()),
                max_norm=5.0,
            )
            self.optimizer.step()

            # Accumulate
            for k in running:
                running[k] += losses[k].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{running['total']/num_batches:.4f}",
                "task": f"{running['task']/num_batches:.4f}",
                "sub": f"{running['subspace']/num_batches:.4f}",
            })

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
        print(f"  Device: {self.device}")
        print()

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

        print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")
        return history
