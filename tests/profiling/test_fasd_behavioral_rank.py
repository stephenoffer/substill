"""Behavioral rank picks output-relevant directions even when variance is low.

Synthetic setup:
- Teacher: a single Linear L that projects a 16-D input onto a 4-D
  output. We carefully rotate the input so that there's a direction
  with huge variance that L zeroes out, and a direction with modest
  variance that L uses.
- Expected: variance PCA picks the high-variance direction first;
  behavioral rank picks the output-relevant (low-variance) one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from substill.autodetect import BranchSpec
from substill.profiling.behavioral_rank import choose_behavioral_rank


class ToyTeacher(nn.Module):
    """A model whose forward is a single linear layer we can hook.

    Structure:
      x (B, T, 16)  -> linear (16 -> 4)  -> logits (B, T, 4)
    We'll hook the pre-linear activation; patching it with subspaces
    tells us whether each direction matters for the linear's output.
    """

    def __init__(self, W: Tensor):
        super().__init__()
        self.embed = nn.Identity()
        self.feature = nn.Linear(16, 16, bias=False)
        # Identity so the branch = module output = the pre-head activation.
        with torch.no_grad():
            self.feature.weight.copy_(torch.eye(16))
        self.head = nn.Linear(16, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(W)

    def forward(self, x: Tensor) -> Tensor:
        h = self.feature(x)
        return self.head(h)


def test_behavioral_rank_prefers_output_relevant_direction():
    torch.manual_seed(0)

    # Choose 16 orthonormal directions.
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    # Head weight uses only the LAST 4 directions (rows of head.weight
    # live in those columns of input basis). This makes those
    # directions the *output-relevant* ones.
    W = Q[:, -4:].T.contiguous()  # shape (4, 16), rows = output-relevant directions

    teacher = ToyTeacher(W)
    teacher.eval()

    # Feed data with huge variance on the FIRST 4 directions (irrelevant)
    # and small variance on the LAST 4 directions (relevant).
    scales = torch.ones(16)
    scales[:4] = 10.0
    scales[-4:] = 0.3
    N = 256
    z = torch.randn(N, 16) * scales
    x = z @ Q.T  # project back to standard basis
    x = x.unsqueeze(0)  # (1, N, 16) — treat as (B, T, C)

    # Compute covariance and its eigendecomposition.
    flat = x.reshape(-1, 16)
    cov = (flat.T @ flat) / flat.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    V = eigvecs[:, order].contiguous()

    # The variance-top-4 basis is the FIRST 4 directions of x (irrelevant).
    # Patching with just 4 variance directions should break teacher logits.
    spec = BranchSpec(name="feature", module_path="feature", kind="attn.o")

    calib = [x]

    # Ask for a rank that preserves teacher logits within a tight tolerance.
    k, curve = choose_behavioral_rank(
        teacher,
        spec,
        V,
        calib,
        tol=1e-3,
        search="linear",
        max_rank=16,
    )
    # The behavioral rank must include the 4 output-relevant directions,
    # which are NOT in the top-4 variance directions. So k > 4.
    assert k > 4, f"behavioral rank should need >4 directions, got {k}"

    # Sanity: with rank 16 (full), KL should be ~0.
    full_kl = [kl for (rank, kl) in curve if rank == 16]
    assert full_kl, "full-rank KL not in curve"
    assert full_kl[0] < 1e-4


def test_behavioral_rank_single_direction_sufficient():
    """When only one direction matters, behavioral rank == 1."""
    torch.manual_seed(1)
    # Head uses only direction e_0.
    W = torch.zeros(4, 16)
    W[0, 0] = 1.0
    W[1, 0] = 0.5
    teacher = ToyTeacher(W)
    teacher.eval()
    # Isotropic input.
    x = torch.randn(1, 64, 16)
    flat = x.reshape(-1, 16)
    cov = (flat.T @ flat) / flat.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    V = eigvecs[:, order].contiguous()
    # Move e_0 to front of V since it's the only one that matters.
    # Actually: behavioral rank search will find the smallest k whose subspace
    # contains e_0. Under random ordering, the first index to contain the
    # bulk of e_0 may not be 1. To make this deterministic, construct V
    # manually with e_0 first.
    V = torch.eye(16)
    spec = BranchSpec(name="feature", module_path="feature", kind="attn.o")
    k, curve = choose_behavioral_rank(teacher, spec, V, [x], tol=1e-4, search="linear", max_rank=16)
    assert k == 1, f"expected rank 1, got {k}"
