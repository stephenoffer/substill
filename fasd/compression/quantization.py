"""AWQ-flavored post-training quantization + QAD fine-tune.

This is a reference implementation: correctness first, performance
second. No custom kernels, no int-tensor packing — the
``QuantizedLinear`` simply stores int8 (or int4-packed-in-int8)
weights plus per-group scales, and dequantizes on forward.

The quantization pipeline:

1. Walk student linears.
2. For each, pull the matching :class:`BranchProfile` from the profile
   dictionary to recover per-channel activation magnitudes (from the
   stored principal components + eigenvalues). This is the AWQ
   "salient channel" signal — channels with large activation variance
   matter more, so we protect the top ``protect_fraction`` of them in
   full precision.
3. Per-group min-max quantize the remaining weights to ``bits`` bits.
4. Replace the module in place.

The QAD fine-tune keeps weights quantized during forward but runs the
quantization op with a straight-through estimator so gradients flow.
Teacher logits are used as the supervisory signal via skew KL — the
unquantized teacher and quantized student should have close outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


Scheme = Literal["awq", "minmax"]


# -- helpers -----------------------------------------------------------


def _per_channel_magnitude(branch_profile) -> Tensor | None:
    """Approximate per-input-channel activation magnitude from a profile.

    Uses ``sum_j lambda_j * V_{:, j}^2`` — the diagonal of the
    reconstructed covariance under the retained subspace.
    """
    V = getattr(branch_profile, "principal_components", None)
    eig = getattr(branch_profile, "eigenvalues", None)
    if V is None or eig is None:
        return None
    k = getattr(branch_profile, "behavioral_rank", V.shape[1])
    k = min(int(k), V.shape[1], eig.shape[0])
    V_k = V[:, :k]
    eig_k = eig[:k].clamp_min(0)
    return (V_k.pow(2) * eig_k.unsqueeze(0)).sum(dim=1).sqrt()


def _group_quantize(
    W: Tensor,
    bits: int,
    group_size: int,
    *,
    sym: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Per-group symmetric quantization along the last dim.

    Returns ``(q_int, scale, zero_point)``. For symmetric quantization
    ``zero_point`` is ``None``.
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    q_max = (1 << (bits - 1)) - 1  # signed range
    orig_shape = W.shape
    W = W.reshape(-1, orig_shape[-1])
    d_out, d_in = W.shape
    n_groups = (d_in + group_size - 1) // group_size
    q = torch.zeros_like(W, dtype=torch.int8)
    scale = torch.zeros(d_out, n_groups, dtype=W.dtype)
    for g in range(n_groups):
        a = g * group_size
        b = min(a + group_size, d_in)
        block = W[:, a:b]
        max_abs = block.abs().amax(dim=-1).clamp_min(1e-8)
        s = max_abs / q_max
        q_block = torch.round(block / s.unsqueeze(-1)).clamp(-q_max, q_max)
        q[:, a:b] = q_block.to(torch.int8)
        scale[:, g] = s
    q = q.reshape(orig_shape)
    return q, scale, None


def _group_dequantize(
    q: Tensor,
    scale: Tensor,
    group_size: int,
) -> Tensor:
    orig_shape = q.shape
    q2 = q.to(scale.dtype).reshape(-1, orig_shape[-1])
    out = torch.zeros_like(q2)
    d_out, d_in = q2.shape
    n_groups = scale.shape[1]
    for g in range(n_groups):
        a = g * group_size
        b = min(a + group_size, d_in)
        out[:, a:b] = q2[:, a:b] * scale[:, g : g + 1]
    return out.reshape(orig_shape)


def _group_fake_quant(W: Tensor, bits: int, group_size: int) -> Tensor:
    """Per-group symmetric fake-quantize with straight-through estimator.

    Forward returns the dequantized rounded weight; backward passes the
    gradient through unchanged (STE). The per-group scale is recomputed
    from the current ``W`` and detached from the graph, so gradients flow
    only through ``W`` itself — what QAD needs to actually update the
    weights. v8 had no STE: the quantized weights were a frozen buffer
    and ``qad_finetune`` only updated bias / protected / LN, leaving the
    99% quantized bulk inert. That's why rung 6 was a strict PPL tax.
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    qmax = (1 << (bits - 1)) - 1
    orig_shape = W.shape
    W2 = W.reshape(-1, orig_shape[-1])
    d_in = W2.shape[1]
    n_groups = (d_in + group_size - 1) // group_size
    out = torch.empty_like(W2)
    for g in range(n_groups):
        a = g * group_size
        b = min(a + group_size, d_in)
        block = W2[:, a:b]
        with torch.no_grad():
            scale = block.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q = block / scale
        q_round = q.round().clamp(-qmax, qmax)
        # Standard STE: forward gets q_round, backward gets identity through q.
        q_ste = q + (q_round - q).detach()
        out[:, a:b] = q_ste * scale
    return out.reshape(orig_shape)


# -- quantized linear --------------------------------------------------


class QuantizedLinear(nn.Module):
    """AWQ-style linear with an fp32 master weight + per-forward fake-quant (STE).

    Stores ``fp_weight`` as a trainable ``nn.Parameter``. On every forward, the
    weight is fake-quantized per-group with straight-through round; the
    quantized tensor is what flows into ``F.linear``, but the gradient lands on
    ``fp_weight`` so QAD can actually update it. Salient input channels are
    held in full precision via ``protected_weight`` and patched in after
    fake-quant.

    v8 stored the int8 ``q_weight`` as a frozen buffer with no STE — QAD only
    moved bias / protected / LN parameters, which is why rung 6 (4-bit + 100
    QAD steps) was a strict 40–60% PPL tax over rung 4 (no quant).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int,
        group_size: int,
        bias: bool,
        protected_idx: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.fp_weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        if protected_idx is not None and protected_idx.numel() > 0:
            self.register_buffer("protected_idx", protected_idx.long())
            self.protected_weight = nn.Parameter(
                torch.zeros(out_features, protected_idx.numel()), requires_grad=True
            )
        else:
            self.register_buffer("protected_idx", torch.zeros(0, dtype=torch.long))
            self.protected_weight = nn.Parameter(
                torch.zeros(out_features, 0), requires_grad=False
            )

    def quantized_weight(self) -> Tensor:
        """Apply per-group fake-quant (STE) to ``fp_weight``, then patch protected cols."""
        W = _group_fake_quant(self.fp_weight, self.bits, self.group_size)
        if self.protected_idx.numel() > 0:
            W = W.clone()
            W[:, self.protected_idx] = self.protected_weight.to(W.dtype)
        return W

    def forward(self, x: Tensor) -> Tensor:
        W = self.quantized_weight().to(x.dtype)
        return F.linear(x, W, self.bias)


# -- QuantizedConv1D (GPT-2) ------------------------------------------


class QuantizedConv1D(QuantizedLinear):
    """HF GPT-2 ``Conv1D`` stores weight as ``(d_in, d_out)`` and applies
    ``x @ W + b``. This subclass transposes on forward to match that
    convention. Inherits the fp32-master-weight + STE fake-quant scheme.
    """

    def forward(self, x: Tensor) -> Tensor:
        W = self.quantized_weight().to(x.dtype)  # (out, in)
        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


# -- public API --------------------------------------------------------


@dataclass
class QuantizationReport:
    """Summary of a quantize_student pass."""

    replaced: int = 0
    skipped: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.skipped is None:
            self.skipped = []


def _find_matching_branch(profile, module_path: str):
    """Best-effort: pull the BranchProfile whose module_path matches."""
    if profile is None or not hasattr(profile, "branches"):
        return None
    # Exact match first; then branches whose name starts with module_path.
    for b in profile.branches:
        if getattr(b, "module_path", None) == module_path:
            return b
    for b in profile.branches:
        if b.name.startswith(module_path):
            return b
    return None


def quantize_student(
    student: nn.Module,
    profile=None,
    *,
    scheme: Scheme = "awq",
    bits: int = 4,
    group_size: int = 128,
    protect_fraction: float = 0.05,
    skip_names: Iterable[str] = (),
) -> QuantizationReport:
    """Replace linear modules in ``student`` with quantized versions.

    Returns a :class:`QuantizationReport` summarizing what was done.
    """
    skip_set = set(skip_names)
    report = QuantizationReport()

    # Snapshot the (name, module) list up front because we mutate in-place.
    targets: list[tuple[str, nn.Module]] = []
    for name, module in student.named_modules():
        if name in skip_set:
            continue
        # Identify linears worth quantizing. Embeddings and lm_head are
        # intentionally skipped; users can opt in by unlisting them.
        if isinstance(module, nn.Linear):
            targets.append((name, module))
        elif hasattr(module, "nf") and hasattr(module, "weight") and not isinstance(
            module, nn.Linear
        ):
            # GPT-2 Conv1D
            targets.append((name, module))

    for name, module in targets:
        if "lm_head" in name or "embed" in name.lower():
            report.skipped.append(name)
            continue
        try:
            _replace_linear(student, name, module, profile, scheme, bits, group_size, protect_fraction)
            report.replaced += 1
        except Exception as e:  # pragma: no cover
            report.skipped.append(f"{name} ({type(e).__name__}: {e})")
    return report


def _replace_linear(
    root: nn.Module,
    name: str,
    module: nn.Module,
    profile,
    scheme: Scheme,
    bits: int,
    group_size: int,
    protect_fraction: float,
) -> None:
    is_conv1d = hasattr(module, "nf") and not isinstance(module, nn.Linear)
    if is_conv1d:
        W = module.weight.detach().T.contiguous()  # (d_out, d_in)
        in_features = W.shape[1]
        out_features = W.shape[0]
    else:
        W = module.weight.detach()
        in_features = module.in_features
        out_features = module.out_features
    bias = getattr(module, "bias", None)
    b_t = bias.detach().clone() if bias is not None else None

    protected_idx = None
    if scheme == "awq" and protect_fraction > 0.0 and profile is not None:
        bp = _find_matching_branch(profile, name)
        mag = _per_channel_magnitude(bp) if bp is not None else None
        if mag is not None and mag.numel() == in_features:
            n_protect = max(1, int(round(protect_fraction * in_features)))
            protected_idx = torch.argsort(mag, descending=True)[:n_protect]
        else:
            protected_idx = None

    mask = torch.ones(in_features, dtype=torch.bool)
    if protected_idx is not None:
        mask[protected_idx] = False

    W_init = W.clone()
    if protected_idx is not None:
        # Zero out protected columns in the fake-quant path; the values are
        # restored from protected_weight at forward time.
        W_init[:, protected_idx] = 0.0

    new_cls = QuantizedConv1D if is_conv1d else QuantizedLinear
    new_module = new_cls(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        bias=b_t is not None,
        protected_idx=protected_idx,
    )
    new_module.fp_weight.data.copy_(W_init)
    if b_t is not None:
        new_module.bias.data.copy_(b_t)
    if protected_idx is not None and protected_idx.numel() > 0:
        new_module.protected_weight.data.copy_(W[:, protected_idx])

    _set_submodule(root, name, new_module)


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


# -- QAD fine-tune -----------------------------------------------------


def qad_finetune(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    *,
    steps: int = 500,
    lr: float = 5e-6,
    skew_alpha: float = 0.1,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Quantization-aware fine-tune with STE through the quant op.

    With ``QuantizedLinear`` keeping an fp32 ``fp_weight`` Parameter and
    fake-quantizing on every forward, gradients land on ``fp_weight``
    directly — so this loop transparently updates the bulk of the quantized
    weights along with bias / protected_weight / embeddings / LN.
    """
    from .quantization import QuantizedLinear as _QL  # avoid circular name shadowing

    trainable = [
        p for p in student.parameters() if p.requires_grad
    ]
    if not trainable:
        return {"final_loss": float("nan")}
    opt = torch.optim.AdamW(trainable, lr=lr)
    teacher.eval()
    student.train()
    teacher.to(device)
    student.to(device)
    last: float = float("nan")
    seen = 0
    for batch in loader:
        if seen >= steps:
            break
        seen += 1
        batch = _move(batch, device)
        with torch.no_grad():
            t_logits = _logits(teacher(**batch) if isinstance(batch, dict) else teacher(batch))
        s_out = student(**batch) if isinstance(batch, dict) else student(batch)
        s_logits = _logits(s_out)
        # skew KL student→mix
        t_p = F.softmax(t_logits, dim=-1)
        s_logp = F.log_softmax(s_logits, dim=-1)
        s_p = s_logp.exp()
        mix = skew_alpha * t_p + (1.0 - skew_alpha) * s_p
        log_mix = mix.clamp_min(1e-12).log()
        loss = (s_p * (s_logp - log_mix)).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
    return {"final_loss": last, "steps": seen}


def _move(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return tuple(b.to(device) if isinstance(b, Tensor) else b for b in batch)
    if isinstance(batch, Tensor):
        return batch.to(device)
    return batch


def _logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"no logits in {type(out)}")


__all__ = [
    "QuantizationReport",
    "QuantizedConv1D",
    "QuantizedLinear",
    "qad_finetune",
    "quantize_student",
]
