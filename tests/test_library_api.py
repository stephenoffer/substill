"""Library-API integration tests.

These tests exercise the public top-level surface (`asd.profile`,
`asd.SubspaceLoss`, `asd.capture`, `asd.build_student`, `asd.distill`,
`asd.autodetect_layers`) on tiny synthetic models — both a 2-stage
ResNet-like CNN and a toy transformer block stack — to confirm the
API is ergonomic and the math works on both shapes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import asd


# ---------------------------------------------------------------------------
# Toy image model — ResNet-shaped (B, C, H, W)
# ---------------------------------------------------------------------------

class _ToyResBlock(nn.Module):
    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        if stride != 1 or C_in != C_out:
            self.downsample = nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        y = self.bn(self.conv(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(x + y)


class _ToyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.block1 = _ToyResBlock(16, 16)
        self.block2 = _ToyResBlock(16, 32, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _tiny_image_loader(n=16, batch=4):
    torch.manual_seed(0)
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch)


# ---------------------------------------------------------------------------
# Toy transformer — (B, T, C)
# ---------------------------------------------------------------------------

class _ToyTxBlock(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(C, 2, batch_first=True)
        self.ln1 = nn.LayerNorm(C)
        self.ffn = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))
        self.ln2 = nn.LayerNorm(C)

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class _ToyLM(nn.Module):
    def __init__(self, C=16, vocab=64, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, C)
        self.h = nn.ModuleList([_ToyTxBlock(C) for _ in range(n_layers)])
        self.head = nn.Linear(C, vocab)

    def forward(self, ids):
        x = self.emb(ids)
        for blk in self.h:
            x = blk(x)
        return self.head(x)


def _tiny_token_loader(n=8, seq=12, vocab=64, batch=2):
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (n, seq))
    return DataLoader(TensorDataset(ids, ids), batch_size=batch)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_profile_returns_immutable_snapshot_on_cnn():
    model = _ToyResNet()
    loader = _tiny_image_loader()
    p = asd.profile(model, loader, layers=[model.block1, model.block2],
                    source="output")
    assert isinstance(p, asd.TeacherProfile)
    assert p.layers == ["block1", "block2"]
    assert len(p.profiles) == 2
    assert p.source == "output"
    # Eigenvalue / components must be set
    assert p.profiles[0].principal_components.shape[1] >= 1
    assert all(r >= 1 for r in p.effective_ranks())


def test_profile_delta_source_on_cnn():
    model = _ToyResNet()
    loader = _tiny_image_loader()
    p_out = asd.profile(model, loader, layers=[model.block1], source="output")
    p_del = asd.profile(model, loader, layers=[model.block1], source="delta")
    # The eigenvalue magnitudes should differ — delta strips the
    # identity path so its norm is smaller than output's.
    lam_out = p_out.profiles[0].eigenvalues.sum().item()
    lam_del = p_del.profiles[0].eigenvalues.sum().item()
    assert abs(lam_out - lam_del) > 1e-6


def test_profile_save_load_roundtrip(tmp_path):
    model = _ToyResNet()
    loader = _tiny_image_loader()
    p = asd.profile(model, loader, layers=[model.block1, model.block2])
    path = tmp_path / "profile.pt"
    p.save(path)
    loaded = asd.TeacherProfile.load(path)
    assert loaded.layers == p.layers
    assert loaded.source == p.source
    assert len(loaded.profiles) == len(p.profiles)
    # Eigenvalues equal within float precision
    torch.testing.assert_close(
        loaded.profiles[0].eigenvalues, p.profiles[0].eigenvalues, rtol=1e-5, atol=1e-6,
    )


def test_capture_context_manager_populates_hiddens():
    model = _ToyResNet()
    loader = _tiny_image_loader()
    p = asd.profile(model, loader, layers=[model.block1, model.block2])
    batch = next(iter(loader))
    x = batch[0]
    with asd.capture(model, p) as cap:
        with torch.no_grad():
            model(x)
    vals = cap.values()
    assert len(vals) == 2
    # block1: (B, 16, 8, 8); block2: (B, 32, 4, 4)
    assert vals[0].shape[1] == 16
    assert vals[1].shape[1] == 32


def test_subspace_loss_on_image_features():
    model = _ToyResNet()
    loader = _tiny_image_loader()
    p = asd.profile(model, loader, layers=[model.block1, model.block2])
    # Student has the same architecture; widths match profile so the
    # loss can be constructed eagerly with the known widths.
    student = _ToyResNet()
    widths = [16, 32]
    for obj in ("coord_mse", "gram", "cka"):
        loss_fn = asd.SubspaceLoss(p, widths, objective=obj)
        # One step of forward+backward should be clean.
        batch = next(iter(loader))
        x = batch[0]
        with asd.capture(model, p) as t_cap:
            with torch.no_grad():
                model(x)
        with asd.capture(student, p) as s_cap:
            student(x)
        loss = loss_fn(s_cap.values(), t_cap.values())
        assert loss.dim() == 0
        assert torch.isfinite(loss), f"objective {obj} gave non-finite loss: {loss}"
        loss.backward()
        # At least one student param should have a gradient.
        grads = [p.grad for p in student.parameters() if p.grad is not None]
        assert len(grads) > 0


def test_subspace_loss_lazy_projector_init():
    """If `student_widths` isn't provided, projectors are built on
    first forward from the tensors passed in."""
    model = _ToyResNet()
    loader = _tiny_image_loader()
    p = asd.profile(model, loader, layers=[model.block1, model.block2])
    loss_fn = asd.SubspaceLoss(p, objective="gram")  # no widths
    student = _ToyResNet()
    with asd.capture(model, p) as t_cap:
        with torch.no_grad():
            model(next(iter(loader))[0])
    with asd.capture(student, p) as s_cap:
        student(next(iter(loader))[0])
    val = loss_fn(s_cap.values(), t_cap.values())
    assert torch.isfinite(val)


def test_autodetect_torchvision_resnet():
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    names = asd.autodetect_layers(m)
    # resnet18 has 2 blocks per stage × 4 stages = 8 block names
    assert len(names) == 8
    assert all(n.startswith("layer") for n in names)


def test_autodetect_unknown_raises():
    class _Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Linear(1, 1)

        def forward(self, x):
            return self.foo(x)

    with pytest.raises(NotImplementedError, match=r"no detector matched"):
        asd.autodetect_layers(_Weird())


def test_subspace_loss_on_tokens():
    """Validate the transformer path — (B, T, C) features work end-to-end."""
    model = _ToyLM()
    loader = _tiny_token_loader()
    # Profile over a few batches of tokens.
    p = asd.profile(model, loader, layers=list(model.h), source="output")
    assert len(p.profiles) == 2
    for obj in ("coord_mse", "gram", "cka"):
        loss_fn = asd.SubspaceLoss(p, objective=obj)
        # Same-architecture student for simplicity.
        student = _ToyLM()
        ids = next(iter(loader))[0]
        with asd.capture(model, p) as t_cap:
            with torch.no_grad():
                model(ids)
        with asd.capture(student, p) as s_cap:
            student(ids)
        L = loss_fn(s_cap.values(), t_cap.values())
        assert torch.isfinite(L), f"objective {obj} on tokens gave {L}"
        L.backward()


def test_cka_is_scale_invariant_on_tokens():
    """CKA should be unchanged if we scale student features by a
    constant. That's the property that makes it safe on LLMs."""
    model = _ToyLM()
    loader = _tiny_token_loader()
    p = asd.profile(model, loader, layers=list(model.h))

    loss_fn = asd.SubspaceLoss(p, objective="cka", normalize_features=False)
    torch.manual_seed(0)
    s_hid = [torch.randn(2, 12, 16, requires_grad=True) for _ in range(2)]
    t_hid = [torch.randn(2, 12, 16) for _ in range(2)]

    L1 = loss_fn(s_hid, t_hid)
    s_hid_scaled = [10.0 * h for h in s_hid]
    L2 = loss_fn(s_hid_scaled, t_hid)
    assert abs(L1.item() - L2.item()) < 1e-4, \
        f"CKA should be scale-invariant: got {L1.item()} vs {L2.item()}"


def test_gram_loss_bounded_with_normalization():
    """The gram loss with feature normalization should stay small even
    on very high-magnitude features — the whole point of the fix."""
    model = _ToyLM()
    loader = _tiny_token_loader()
    p = asd.profile(model, loader, layers=list(model.h))

    loss_fn = asd.SubspaceLoss(p, objective="gram", normalize_features=True)
    torch.manual_seed(0)
    # 1000-magnitude features — would blow up without normalize_features.
    s_hid = [1000.0 * torch.randn(2, 12, 16, requires_grad=True) for _ in range(2)]
    t_hid = [1000.0 * torch.randn(2, 12, 16) for _ in range(2)]
    L = loss_fn(s_hid, t_hid)
    assert L.item() < 10.0, f"gram loss should stay small; got {L.item()}"


def test_distill_toy_end_to_end():
    """One-call distillation works on an image model with label loader."""
    torch.manual_seed(0)
    teacher = _ToyResNet()
    student = _ToyResNet()  # fresh copy; distill sees it as student.
    loader = _tiny_image_loader()
    # Shallow run — 2 epochs on tiny data.
    result = asd.distill(
        teacher, student, loader,
        epochs=2, lr=1e-2, objective="cka",
        layers=[teacher.block1, teacher.block2],  # forwarded to profile()
        source="output",
    )
    assert len(result.history) == 2
    assert isinstance(result.profile, asd.TeacherProfile)
