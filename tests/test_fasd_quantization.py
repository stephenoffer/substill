"""AWQ-style quantization replaces linears and preserves rough accuracy."""

from __future__ import annotations

import torch
import torch.nn as nn

from fasd.compression.quantization import (
    QuantizedLinear,
    _group_dequantize,
    _group_quantize,
    quantize_student,
)


def test_group_quantize_roundtrip_int4():
    torch.manual_seed(0)
    W = torch.randn(8, 16) * 3.0
    q, scale, _ = _group_quantize(W, bits=4, group_size=8)
    W_hat = _group_dequantize(q, scale, group_size=8)
    err = (W - W_hat).abs().max().item()
    # int4 quantization over narrow groups should be within ~max/8 per element.
    assert err < 2.0


def test_quantize_student_replaces_linears():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(16, 16, bias=False)
            self.l2 = nn.Linear(16, 8, bias=True)

        def forward(self, x):
            return self.l2(self.l1(x))

    m = Tiny()
    x = torch.randn(4, 16)
    y_before = m(x)
    report = quantize_student(m, profile=None, bits=4, group_size=8)
    assert report.replaced == 2
    assert isinstance(m.l1, QuantizedLinear)
    assert isinstance(m.l2, QuantizedLinear)
    y_after = m(x)
    # Quantization error should be small relative to the output norm.
    rel = float((y_before - y_after).norm().item()) / float(y_before.norm().item() + 1e-8)
    assert rel < 0.3


def test_quantized_linear_bias_preserved():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 4, bias=True)

        def forward(self, x):
            return self.l(x)

    m = Tiny()
    b_before = m.l.bias.detach().clone()
    quantize_student(m, profile=None, bits=4, group_size=4)
    assert torch.allclose(m.l.bias.detach(), b_before, atol=1e-6)
