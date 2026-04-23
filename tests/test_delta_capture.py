"""Tests for T1.1 — residual-delta profiling."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from asd.profiling.activation_capture import (
    ActivationCaptureEngine,
    CovarianceAccumulator,
    _residual_shortcut,
    VALID_SOURCES,
)
from asd.models.student import SlimNet


class _IdentityShortcutBlock(nn.Module):
    """Minimal residual block with an identity shortcut — the delta is
    exactly `output - input`."""

    def __init__(self, C: int = 8):
        super().__init__()
        self.lin = nn.Linear(C, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # f(x) = W x → block output = x + W x
        return x + self.lin(x)


class _DownsampleShortcutBlock(nn.Module):
    """Residual block with a non-identity shortcut — delta must use the
    shortcut path, not the raw input."""

    def __init__(self, C_in: int = 4, C_out: int = 8):
        super().__init__()
        self.f = nn.Linear(C_in, C_out)
        self.downsample = nn.Linear(C_in, C_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x) + self.downsample(x)


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _IdentityShortcutBlock(8)
        self.b2 = _DownsampleShortcutBlock(8, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        return x


def test_valid_sources_constant():
    assert set(VALID_SOURCES) == {"output", "delta", "branch"}


def test_covariance_accumulator_tracks_source():
    acc_out = CovarianceAccumulator(4, source="output")
    acc_delta = CovarianceAccumulator(4, source="delta")
    assert acc_out.source == "output"
    assert acc_delta.source == "delta"
    with pytest.raises(ValueError):
        CovarianceAccumulator(4, source="nope")


def test_residual_shortcut_identity_for_missing_attrs():
    # A plain Linear has neither .downsample nor .shortcut — identity lift.
    m = nn.Linear(4, 4)
    lift = _residual_shortcut(m)
    x = torch.randn(2, 4)
    assert torch.allclose(lift(x), x)


def test_residual_shortcut_downsample_when_present():
    block = _DownsampleShortcutBlock(4, 8)
    lift = _residual_shortcut(block)
    x = torch.randn(2, 4)
    # Lift should apply the downsample, not the identity.
    expected = block.downsample(x)
    assert torch.allclose(lift(x), expected)


def test_residual_shortcut_slimnet_block_shortcut():
    # SlimNet Bottleneck with stride=2 has a non-identity shortcut.
    from asd.models.student import Bottleneck
    block = Bottleneck(in_channels=8, out_channels=16, stride=2)
    lift = _residual_shortcut(block)
    x = torch.randn(1, 8, 8, 8)
    y = lift(x)
    assert y.shape == (1, 16, 4, 4)


def test_engine_output_source_matches_legacy():
    """The default source='output' must produce bit-identical covariance
    to pre-change code on a deterministic input."""
    torch.manual_seed(0)
    model = _ToyNet()
    model.eval()

    class _DS(torch.utils.data.Dataset):
        def __init__(self, n=8):
            torch.manual_seed(1)
            self.x = torch.randn(n, 8)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], torch.zeros(1)

    loader = torch.utils.data.DataLoader(_DS(), batch_size=4)

    engine_out = ActivationCaptureEngine(
        model, ["b1"], covariance_mode="per_pixel", source="output",
    )
    accs_out = engine_out.run(loader, device="cpu")
    cov_out = accs_out["b1"].finalize()

    # Hand-computed covariance of b1 output on the same inputs
    xs = _DS().x
    with torch.no_grad():
        outs = model.b1(xs)
    outs64 = outs.to(torch.float64)
    n = outs64.shape[0]
    mean = outs64.mean(dim=0)
    hand_cov = (outs64.T @ outs64) / n - mean.unsqueeze(1) * mean.unsqueeze(0)
    hand_cov = 0.5 * (hand_cov + hand_cov.T)
    assert torch.allclose(cov_out, hand_cov.float(), atol=1e-5)


def test_engine_delta_identity_block():
    """For a block with identity shortcut, delta = output - input."""
    torch.manual_seed(0)
    model = _ToyNet()
    model.eval()

    class _DS(torch.utils.data.Dataset):
        def __init__(self, n=8):
            torch.manual_seed(2)
            self.x = torch.randn(n, 8)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], torch.zeros(1)

    loader = torch.utils.data.DataLoader(_DS(), batch_size=4)

    engine = ActivationCaptureEngine(
        model, ["b1"], covariance_mode="per_pixel", source="delta",
    )
    accs = engine.run(loader, device="cpu")
    cov_delta = accs["b1"].finalize()

    # Hand-compute delta = b1(x) - x (identity shortcut)
    xs = _DS().x
    with torch.no_grad():
        outs = model.b1(xs)
    deltas = (outs - xs).to(torch.float64)
    n = deltas.shape[0]
    mean = deltas.mean(dim=0)
    hand_cov = (deltas.T @ deltas) / n - mean.unsqueeze(1) * mean.unsqueeze(0)
    hand_cov = 0.5 * (hand_cov + hand_cov.T)
    assert torch.allclose(cov_delta, hand_cov.float(), atol=1e-5)

    # The delta covariance and output covariance must be meaningfully
    # different — otherwise the "source" knob is a no-op.
    engine2 = ActivationCaptureEngine(
        model, ["b1"], covariance_mode="per_pixel", source="output",
    )
    accs2 = engine2.run(loader, device="cpu")
    cov_output = accs2["b1"].finalize()
    assert not torch.allclose(cov_delta, cov_output, atol=1e-3)


def test_engine_delta_downsample_block():
    """For a block with non-identity shortcut, delta uses shortcut(input)."""
    torch.manual_seed(0)
    model = _ToyNet()
    model.eval()

    class _DS(torch.utils.data.Dataset):
        def __init__(self, n=8):
            torch.manual_seed(3)
            self.x = torch.randn(n, 8)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], torch.zeros(1)

    loader = torch.utils.data.DataLoader(_DS(), batch_size=4)

    engine = ActivationCaptureEngine(
        model, ["b2"], covariance_mode="per_pixel", source="delta",
    )
    accs = engine.run(loader, device="cpu")
    cov = accs["b2"].finalize()

    # Hand-compute: run model up through b1, then delta = b2(h) - downsample(h)
    xs = _DS().x
    with torch.no_grad():
        h = model.b1(xs)          # shape (n, 8)
        out_b2 = model.b2(h)      # shape (n, 16)
        shortcut = model.b2.downsample(h)
        deltas = (out_b2 - shortcut).to(torch.float64)

    n = deltas.shape[0]
    mean = deltas.mean(dim=0)
    hand_cov = (deltas.T @ deltas) / n - mean.unsqueeze(1) * mean.unsqueeze(0)
    hand_cov = 0.5 * (hand_cov + hand_cov.T)
    assert torch.allclose(cov, hand_cov.float(), atol=1e-4)


def test_engine_delta_on_slimnet_stages():
    """Smoke test: delta profiling on an actual SlimNet runs without
    shape errors on stage transitions (where shortcut stride != 1)."""
    student = SlimNet(
        stage_widths=[16, 32, 64, 64],
        blocks_per_stage=1,
    )
    student.eval()

    class _DS(torch.utils.data.Dataset):
        def __init__(self, n=4):
            torch.manual_seed(4)
            self.x = torch.randn(n, 3, 32, 32)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], torch.zeros(1)

    loader = torch.utils.data.DataLoader(_DS(), batch_size=2)

    # Hook the first block of stage 2 — that block has a stride-2 shortcut.
    # `SlimNet` stages are `stage1.0`, `stage2.0`, etc.
    engine = ActivationCaptureEngine(
        student, ["stage2.0"], covariance_mode="per_pixel", source="delta",
    )
    accs = engine.run(loader, device="cpu")
    cov = accs["stage2.0"].finalize()
    # Stage 2 output is (B, 32, 16, 16); cov is (32, 32)
    assert cov.shape == (32, 32)
