"""Tests for fasd.profiling.functional_score.score_directions.

The headline property: a direction with HIGH variance but LOW task-relevance
must score lower than a direction with LOW variance but HIGH task-relevance.
This is the discriminator that distinguishes the Fisher score from variance-only.

We use a tiny synthetic teacher where we can control which directions the
loss actually depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fasd.profiling.functional_score import score_directions


@dataclass
class FakeBranch:
    name: str
    module_path: str
    kind: str
    slice: tuple[int, int] | None
    principal_components: torch.Tensor
    eigenvalues: torch.Tensor
    behavioral_rank: int


@dataclass
class FakeProfile:
    branches: list


class _SyntheticTeacher(nn.Module):
    """A tiny LM where we control which residual direction the output depends on.

    The model is::
        embed -> [residual stream of dim d] -> linear -> logits

    The "linear" layer has weight = w_signal · e_signal^T, where e_signal is a
    designated unit vector. So only the projection of the residual onto e_signal
    affects the output. Variance along e_noise (an orthogonal direction) is
    high but irrelevant.
    """

    def __init__(self, vocab: int, d: int, signal_idx: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.tap = nn.Identity()  # branch capture target
        self.head = nn.Linear(d, vocab, bias=False)

        # Make head depend ONLY on the signal_idx-th coordinate.
        with torch.no_grad():
            W = torch.zeros(vocab, d)
            # Put a non-trivial random map on the signal coordinate, zero elsewhere.
            W[:, signal_idx] = torch.randn(vocab)
            self.head.weight.copy_(W)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        h = self.tap(x)
        logits = self.head(h)
        return _Out(logits=logits)


@dataclass
class _Out:
    logits: torch.Tensor


def test_fisher_score_picks_signal_over_noise_direction():
    """Set up: residual stream has high variance along axis 1 (noise), low along
    axis 0 (signal). The teacher's head only reads from axis 0. Variance scoring
    would pick axis 1 (high λ); Fisher scoring should pick axis 0 (high q).
    """
    torch.manual_seed(0)
    vocab = 50
    d = 4
    signal_idx = 0

    teacher = _SyntheticTeacher(vocab, d, signal_idx).eval()

    # Pre-bake the principal components: identity (the natural axes).
    # Eigenvalues: low along signal, high along noise.
    pcs = torch.eye(d)
    eigenvalues = torch.tensor([0.1, 10.0, 5.0, 1.0])  # low signal variance, high noise

    branch = FakeBranch(
        name="residual",
        module_path="tap",
        kind="block.residual",
        slice=None,
        principal_components=pcs,
        eigenvalues=eigenvalues,
        behavioral_rank=2,
    )
    profile = FakeProfile(branches=[branch])

    # Build a calibration set where embeddings produce activations with the
    # designed variance pattern. The synthetic teacher uses fresh random
    # embeddings; activation magnitudes are determined by them.
    # Override embedding to produce activations with the desired variance:
    #   axis 0 (signal): ~0.1 std
    #   axis 1 (noise): ~10 std
    with torch.no_grad():
        scales = torch.tensor([0.1, 10.0, 5.0, 1.0]).sqrt()
        E = teacher.embed.weight
        E.copy_(torch.randn_like(E) * scales.unsqueeze(0))

    # Calibration loader: a batch of token ids.
    batches = [{"input_ids": torch.randint(0, vocab, (4, 16))}]

    scores = score_directions(teacher, profile, batches, device="cpu")

    s = scores["residual"]
    # The Fisher score q = λ · E[(u^T g)²].
    # - Direction 0 (signal): low λ but the gradient is concentrated here; should be highest q.
    # - Direction 1 (noise): high λ but gradient ≈ 0 along it; should be near zero q.
    assert s.q[signal_idx] > s.q[1], (
        f"Fisher score failed to prefer signal direction over noise: "
        f"q={s.q.tolist()}, λ={s.eigenvalues.tolist()}, "
        f"E[(u^T g)²]={s.grad_inner_sq.tolist()}"
    )
    # Variance-only ranking would have picked direction 1.
    assert s.eigenvalues[1] > s.eigenvalues[signal_idx]


def test_fisher_score_returns_per_branch_scores():
    torch.manual_seed(0)
    teacher = _SyntheticTeacher(vocab=20, d=4, signal_idx=0).eval()
    branches = [
        FakeBranch(
            name="b1",
            module_path="tap",
            kind="block.residual",
            slice=None,
            principal_components=torch.eye(4),
            eigenvalues=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            behavioral_rank=2,
        ),
    ]
    profile = FakeProfile(branches=branches)
    batches = [{"input_ids": torch.randint(0, 20, (2, 8))}]
    scores = score_directions(teacher, profile, batches, device="cpu")
    assert "b1" in scores
    assert scores["b1"].q.shape == (4,)
    assert scores["b1"].eigenvalues.shape == (4,)
    assert scores["b1"].grad_inner_sq.shape == (4,)


def test_fisher_score_max_rank_caps_output():
    torch.manual_seed(0)
    teacher = _SyntheticTeacher(vocab=20, d=8, signal_idx=0).eval()
    branches = [
        FakeBranch(
            name="b",
            module_path="tap",
            kind="block.residual",
            slice=None,
            principal_components=torch.eye(8),
            eigenvalues=torch.ones(8),
            behavioral_rank=4,
        ),
    ]
    profile = FakeProfile(branches=branches)
    batches = [{"input_ids": torch.randint(0, 20, (2, 6))}]
    scores = score_directions(teacher, profile, batches, device="cpu", max_rank=3)
    assert scores["b"].q.shape == (3,)


def test_fisher_score_topk_returns_top_indices():
    """If we synthesise q that has a known order, topk should return correct indices."""
    from fasd.profiling.functional_score import DirectionScores

    ds = DirectionScores(
        branch_name="x",
        eigenvalues=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        grad_inner_sq=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        q=torch.tensor([1.0, 5.0, 2.0, 3.0]),
    )
    top = ds.topk(2).tolist()
    assert top == [1, 3]  # highest q is index 1, then index 3


def test_fisher_score_does_not_leak_grads_into_teacher_params():
    """After scoring, teacher params must have no .grad attached."""
    torch.manual_seed(0)
    teacher = _SyntheticTeacher(vocab=30, d=4, signal_idx=0).eval()
    profile = FakeProfile(
        branches=[
            FakeBranch(
                name="b",
                module_path="tap",
                kind="block.residual",
                slice=None,
                principal_components=torch.eye(4),
                eigenvalues=torch.ones(4),
                behavioral_rank=2,
            )
        ]
    )
    batches = [{"input_ids": torch.randint(0, 30, (2, 8))}]
    score_directions(teacher, profile, batches, device="cpu")
    for name, p in teacher.named_parameters():
        assert p.grad is None, f"teacher param {name} has stale grad attached"
