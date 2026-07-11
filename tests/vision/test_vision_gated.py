"""KD-driven channel selection: the gate must be a faithful, hardenable relaxation.

Two invariants make `substill/vision/gated.py` a valid vision analog of LRD:

1. A fully-open gate reproduces the teacher block, so phase 1 starts at the teacher and the
   budget penalty is the only pressure closing channels.
2. Selection commutes with ReLU, so hardening the learned gates to the top-k channels is
   function-preserving on those channels — the CNN counterpart of `RestrictedLlama.fold()`.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchvision")

from torchvision.models.resnet import Bottleneck, ResNet  # noqa: E402

from substill.vision.gated import (  # noqa: E402
    GatedBottleneck,
    distill_gated_then_harden,
    install_channel_gates,
)


def _tiny():
    torch.manual_seed(0)
    m = ResNet(Bottleneck, [1, 1, 1, 1], num_classes=10).eval()
    # give the BNs non-trivial running stats so the gate/ReLU interaction is exercised
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.3)
            mod.running_var.uniform_(0.5, 1.5)
    return m


def test_open_gate_reproduces_the_block():
    """With gate logits at +inf the GatedBottleneck must equal the wrapped block exactly."""
    m = _tiny()
    blk = next(b for b in m.modules() if isinstance(b, Bottleneck))
    g = GatedBottleneck(blk).eval()
    with torch.no_grad():
        g.g1.fill_(50.0)   # sigmoid(50) == 1
        g.g2.fill_(50.0)
        x = torch.randn(2, blk.conv1.in_channels, 8, 8)
        assert torch.allclose(g(x), blk(x), atol=1e-5), (g(x) - blk(x)).abs().max()


def test_install_gates_preserves_the_forward():
    """Installing all-open gates across the model must not change its output."""
    m = _tiny()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y0 = m(x)
    gated = install_channel_gates(m)
    for g in gated.values():
        with torch.no_grad():
            g.g1.fill_(50.0)
            g.g2.fill_(50.0)
    with torch.no_grad():
        y1 = m(x)
    assert torch.allclose(y0, y1, atol=1e-5), (y0 - y1).abs().max()


def test_gates_receive_gradient_from_kd():
    """The gate logits must get a gradient from the KD loss, or phase 1 learns nothing."""
    import torch.nn.functional as F
    teacher = _tiny()
    student = _tiny()
    gated = install_channel_gates(student)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        tl = teacher(x)
    loss = F.kl_div(F.log_softmax(student(x), -1), F.log_softmax(tl, -1),
                    reduction="batchmean", log_target=True)
    loss.backward()
    assert all(g.g1.grad is not None and g.g1.grad.abs().sum() > 0 for g in gated.values())


def test_gates_land_on_the_block_device():
    """Installed gates must share the block's device.

    Regression guard: a CUDA model got CPU gate parameters and only crashed on GPU, which a
    CPU-only smoke test cannot catch. Assert placement matches wherever the block lives.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _tiny().to(dev)
    gated = install_channel_gates(m)
    for g in gated.values():
        assert g.g1.device.type == dev and g.g2.device.type == dev
    # and the forward runs on that device without a mismatch
    x = torch.randn(2, 3, 32, 32, device=dev)
    m(x)


def test_pipeline_returns_matched_width_student():
    """The hardened student must have the width the ratio asks for, and run end to end."""
    teacher = _tiny()
    train = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(4)]
    out = distill_gated_then_harden(teacher, train, width_ratio=0.5, gate_steps=3,
                                    finetune_steps=3, device="cpu", val_loader=train[:1])
    assert out["final_top1"] is not None
    # every compressed bottleneck kept half (rounded) of its inner channels
    for info in out["info"].values():
        if info["s_width"] != info["width"]:
            assert info["s_width"] == max(1, round(info["width"] * 0.5))
