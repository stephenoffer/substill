"""Top-level Python API for Activation Subspace Distillation.

The library surface is three core pieces::

    profile(model, dataloader, layers=...)   -> TeacherProfile
    SubspaceLoss(profile, objective=...)     -> nn.Module for your training loop
    distill(teacher, student, dataloader)    -> all-in-one trainer

For known architecture families (torchvision ResNets, HuggingFace
GPT-2), layers can be auto-detected: pass only the model. For
anything else, pass ``layers=[module_a, module_b, ...]`` or a list of
dotted names.

Example::

    import asd
    from torchvision.models import resnet50

    teacher = resnet50(pretrained=True).eval()

    profile = asd.profile(
        teacher,
        calib_loader,
        source="delta",
        noise_model="mp",
    )

    loss_fn = asd.SubspaceLoss(profile, objective="gram")

    for x, y in train_loader:
        with asd.capture(teacher, profile) as t_hid:
            t_logits = teacher(x)
        s_logits, s_hid = student(x, return_hiddens=True)
        L = F.cross_entropy(s_logits, y) + loss_fn(s_hid, t_hid)
        L.backward()

See ``docs/quickstart.md`` for a full walkthrough.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .profiling.activation_capture import (
    ActivationCaptureEngine,
    VALID_SOURCES,
    _residual_shortcut,
)
from .profiling.svd_analysis import (
    LayerProfile,
    SVDAnalyzer,
    load_profiles,
    save_profiles,
)


@dataclass
class TeacherProfile:
    """Snapshot of a teacher's activation subspace across a set of layers.

    Created by :func:`profile`. Users should treat it as immutable:
    stash it, save it, pass it to :class:`SubspaceLoss` or
    :func:`build_student`.

    Attributes:
        layers: Fully-qualified layer names that were hooked.
        profiles: Per-layer :class:`LayerProfile` (principal
            components, eigenvalues, effective rank, source tag,
            total channels).
        source: One of ``"output"``, ``"delta"``, ``"branch"``.
        meta: Free-form metadata (for example,
            ``{"noise_model": "mp", ...}``).
    """

    layers: list[str]
    profiles: list[LayerProfile]
    source: str = "output"
    meta: dict[str, Any] = field(default_factory=dict)

    def effective_ranks(self) -> list[int]:
        """Return the retained rank per layer after noise cutoff."""
        return [p.effective_rank for p in self.profiles]

    def compression_ratios(self) -> list[float]:
        """Return ``effective_rank / total_channels`` per layer."""
        return [p.effective_rank / p.total_channels for p in self.profiles]

    def groups_by_channels(self) -> dict[int, list[LayerProfile]]:
        """Group profiles by their channel count.

        Useful for ResNet-style stage aggregation where multiple
        blocks share a width.
        """
        out: dict[int, list[LayerProfile]] = {}
        for p in self.profiles:
            out.setdefault(p.total_channels, []).append(p)
        return out

    def save(self, path: str | Path) -> None:
        """Save to disk. Round-trips through :meth:`load`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_profiles(self.profiles, str(path))
        sidecar = path.with_suffix(path.suffix + ".meta.pt")
        torch.save(
            {"layers": self.layers, "source": self.source, "meta": self.meta},
            sidecar,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeacherProfile":
        """Load a profile saved by :meth:`save`."""
        path = Path(path)
        profiles = load_profiles(str(path))
        sidecar = path.with_suffix(path.suffix + ".meta.pt")
        if sidecar.exists():
            meta = torch.load(sidecar, weights_only=False)
        else:
            meta = {
                "layers": [p.name for p in profiles],
                "source": profiles[0].source if profiles else "output",
                "meta": {},
            }
        return cls(
            layers=meta["layers"],
            profiles=profiles,
            source=meta["source"],
            meta=meta.get("meta", {}),
        )


def _resolve_layers(
    model: nn.Module,
    layers: Sequence[nn.Module] | Sequence[str] | None,
) -> list[str]:
    """Turn a user-supplied layer spec into a list of dotted module names.

    ``None``: auto-detect for known architecture families. Raises if
    the model is not recognized.

    ``list[nn.Module]``: look each module up in
    ``model.named_modules()``.

    ``list[str]``: validate that each name exists.
    """
    if layers is None:
        from .autodetect import autodetect_layers
        return autodetect_layers(model)
    if len(layers) == 0:
        raise ValueError(
            "layers=[] is empty. Pass None to auto-detect, or a "
            "non-empty list of modules or names."
        )
    first = layers[0]
    if isinstance(first, str):
        name_to_mod = dict(model.named_modules())
        out: list[str] = []
        for n in layers:  # type: ignore[arg-type]
            if n not in name_to_mod:
                raise KeyError(f"layer name {n!r} not found in model")
            out.append(n)
        return out
    if isinstance(first, nn.Module):
        mod_to_name = {id(m): n for n, m in model.named_modules()}
        out = []
        for m in layers:  # type: ignore[arg-type]
            mid = id(m)
            if mid not in mod_to_name:
                raise ValueError(
                    f"module {type(m).__name__} is not a submodule of the given model"
                )
            out.append(mod_to_name[mid])
        return out
    raise TypeError(
        f"layers must be list[nn.Module] or list[str], got {type(layers).__name__}"
    )


def _model_device(model: nn.Module) -> str:
    """Return the device of the first parameter or buffer, or ``"cpu"``."""
    for p in model.parameters():
        return str(p.device)
    for b in model.buffers():
        return str(b.device)
    return "cpu"


def profile(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    layers: Sequence[nn.Module] | Sequence[str] | None = None,
    source: Literal["output", "delta", "branch"] = "output",
    noise_model: Literal["eps", "mp"] = "eps",
    shrinkage: Literal["none", "ledoit_wolf"] = "none",
    variance_threshold: float = 0.95,
    rank_definition: Literal["variance", "stable", "participation", "entropy"] = "variance",
    n_effective: int | None = None,
    eps_relative: float = 1e-6,
    covariance_mode: Literal["per_pixel", "gap"] = "per_pixel",
    spatial_subsample: int = 1,
    max_batches: int | None = None,
    device: str | None = None,
) -> TeacherProfile:
    """Profile a model's activations into a subspace-distillation profile.

    Runs the model over ``dataloader``, hooks the specified
    ``layers``, accumulates channel covariance, and eigendecomposes
    each covariance to derive top-k principal components and an
    effective rank.

    Args:
        model: Any ``nn.Module``. Set to ``eval()`` during profiling
            and restored to its prior mode on return.
        dataloader: Yields ``(x, ...)`` tuples or just ``x``. Only
            the first element is passed to the model.
        layers: Which modules to hook. ``None`` auto-detects for
            known families. Otherwise a list of modules or dotted
            names.
        source:
            ``"output"``: hook module outputs (default).

            ``"delta"``: the residual update
            ``dx = output - shortcut(input)``, stripping the
            identity path. Typically gives a tighter effective rank
            than raw output on residual networks.

            ``"branch"``: hook branch sub-modules directly (for
            example a transformer's ``attn`` or ``mlp``). The caller
            passes the correct branch submodules in ``layers``.
        noise_model: ``"eps"`` (fixed relative floor at
            ``eps_relative * lam_max``) or ``"mp"`` (Marchenko-Pastur
            / Gavish-Donoho bulk-edge cutoff; requires
            ``n_effective >= C``). MP typically produces an
            order-of-magnitude smaller rank estimate than variance
            threshold + eps on noisy-tail spectra.
        shrinkage: ``"none"`` or ``"ledoit_wolf"``. Applies linear
            shrinkage to the covariance before eigendecomposition.
        variance_threshold: For ``rank_definition="variance"``, the
            cumulative-variance threshold defining effective rank.
        n_effective: Override the effective sample count for MP
            cutoff. Required for MP to produce a non-fallback
            threshold.
        max_batches: Optional cap on the number of dataloader
            batches. Useful for fast profiling on large datasets.
        device: Where to run the model. ``None`` keeps the model on
            its current device.

    Returns:
        A :class:`TeacherProfile` snapshot ready to feed into
        :class:`SubspaceLoss` or :func:`build_student`.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}; got {source!r}")

    original_device = _model_device(model)
    if device is None:
        device = original_device

    layer_names = _resolve_layers(model, layers)
    if not layer_names:
        raise ValueError(
            "No layers to hook. Pass `layers=` explicitly or use a "
            "model type that autodetect supports."
        )

    was_training = model.training
    model.eval()
    if str(device) != str(original_device):
        model.to(device)

    engine = ActivationCaptureEngine(
        model,
        layer_names,
        covariance_mode=covariance_mode,
        spatial_subsample=spatial_subsample,
        source=source,
    )

    def _take_input(batch):
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    engine.register_hooks()
    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                x = _take_input(batch).to(device)
                model(x)
    finally:
        engine.cleanup()
        if str(device) != str(original_device):
            model.to(original_device)
        if was_training:
            model.train()

    svd = SVDAnalyzer(
        variance_threshold=variance_threshold,
        definition=rank_definition,
        eps_relative=eps_relative,
        noise_model=noise_model,
        shrinkage=shrinkage,
        n_effective=n_effective,
    )
    layer_profiles: list[LayerProfile] = []
    for name in layer_names:
        acc = engine.accumulator(name)
        if acc is None:
            warnings.warn(f"Layer {name!r} never fired; skipping.")
            continue
        cov = acc.finalize()
        layer_profiles.append(svd.analyze(name, cov, source=source))

    return TeacherProfile(
        layers=[p.name for p in layer_profiles],
        profiles=layer_profiles,
        source=source,
        meta={
            "noise_model": noise_model,
            "shrinkage": shrinkage,
            "variance_threshold": variance_threshold,
            "rank_definition": rank_definition,
            "n_effective": n_effective,
            "covariance_mode": covariance_mode,
        },
    )


class _HiddenCapture:
    """Dict-like container returned by :func:`capture`.

    Keyed by layer name after the wrapped forward pass exits.
    """

    def __init__(self, model: nn.Module, profile: TeacherProfile):
        self.model = model
        self.profile = profile
        self._state: dict[str, Tensor] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "_HiddenCapture":
        name_to_mod = dict(self.model.named_modules())
        src = self.profile.source
        self._state.clear()
        self._handles = []

        for name in self.profile.layers:
            mod = name_to_mod[name]
            if src == "delta":
                shortcut = _residual_shortcut(mod)

                def _make_hook(nm, sc):
                    def hk(m, inputs, output):
                        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
                        act = output[0] if isinstance(output, tuple) else output
                        with torch.no_grad():
                            self._state[nm] = act - sc(x_in)
                    return hk

                self._handles.append(
                    mod.register_forward_hook(_make_hook(name, shortcut))
                )
            else:
                def _make_hook(nm):
                    def hk(m, inputs, output):
                        act = output[0] if isinstance(output, tuple) else output
                        self._state[nm] = act
                    return hk

                self._handles.append(mod.register_forward_hook(_make_hook(name)))
        return self

    def __exit__(self, *args) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._state

    def __getitem__(self, name: str) -> Tensor:
        return self._state[name]

    def __len__(self) -> int:
        return len(self._state)

    def values(self) -> list[Tensor]:
        """Return hidden states in the same order as ``profile.layers``."""
        return [self._state[n] for n in self.profile.layers]


def capture(model: nn.Module, profile: TeacherProfile) -> _HiddenCapture:
    """Context manager that hooks ``model`` at the profile's layers.

    Inside the ``with`` block, after the forward pass, access hidden
    states via ``capture_obj[layer_name]`` or ``capture_obj.values()``.
    """
    return _HiddenCapture(model, profile)


class SubspaceLoss(nn.Module):
    """Feature-distillation loss driven by a :class:`TeacherProfile`.

    Accepts both image tensors ``(B, C, H, W)`` and transformer
    tensors ``(B, T, C)``. Takes a list of student hidden tensors
    and a list of teacher hidden tensors at matching layers, and
    returns a scalar loss.

    Args:
        profile: The :class:`TeacherProfile` from :func:`profile`.
        objective: Loss variant:

            * ``"coord_mse"``: MSE between student and teacher
              projections in the specific eigenbasis. Basis-
              sensitive: fragile when eigenvalues are close and
              the retained-subspace basis is not identifiable.
            * ``"gram"``: Frobenius distance between the kernel
              matrices ``K = Z Z^T``. Invariant under rotations of
              ``V`` within the retained subspace.
            * ``"cka"``: centered kernel alignment,
              ``1 - <K_s, K_t>_F / (||K_s|| * ||K_t||)``. Fully
              scale-invariant. Recommended for LLMs, where feature
              magnitudes are large.
        power_weight_p: Spectral weighting ``w_i ~ lam_i^{-p}``.
            ``None`` (default) disables weighting.
        normalize_features: L2-normalize student and teacher features
            along the channel axis before kernel computation.
            Required for LLM stability; defaults to ``True``.
        student_widths: Per-stage student widths. Optional. If
            ``None``, projectors are built lazily from the first
            forward pass's tensor shapes. Supply explicitly to build
            projectors up-front (for example, to add them to an
            optimizer before training starts).

    Example::

        loss_fn = SubspaceLoss(profile, objective="cka")
        for x, y in loader:
            with asd.capture(teacher, profile) as t_hid:
                teacher(x)
            with asd.capture(student, profile) as s_hid:
                s_logits = student(x)
            sub = loss_fn(s_hid.values(), t_hid.values())
            (F.cross_entropy(s_logits, y) + 0.5 * sub).backward()
    """

    def __init__(
        self,
        profile: TeacherProfile,
        student_widths: Sequence[int] | None = None,
        *,
        objective: Literal["coord_mse", "gram", "cka"] = "gram",
        power_weight_p: float | None = None,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.profile = profile
        self.objective = objective
        self.power_weight_p = power_weight_p
        self.normalize_features = normalize_features

        self._per_layer_components: list[Tensor] = []
        self._per_layer_weights: list[Tensor | None] = []
        self._per_layer_k: list[int] = []
        self._per_layer_c: list[int] = []

        for p in profile.profiles:
            k = p.effective_rank
            comps = p.principal_components[:, :k].clone()
            idx = len(self._per_layer_components)
            self.register_buffer(f"V_{idx}", comps)
            self._per_layer_components.append(comps)
            w = _power_law_weights(p.eigenvalues, k, power_weight_p)
            if w is not None:
                self.register_buffer(f"W_{idx}", w)
            self._per_layer_weights.append(w)
            self._per_layer_k.append(k)
            self._per_layer_c.append(p.total_channels)

        self._projectors: nn.ModuleList | None = None
        if student_widths is not None:
            self._build_projectors(list(student_widths))

    def _build_projectors(self, student_widths: list[int]) -> None:
        if len(student_widths) != len(self._per_layer_k):
            raise ValueError(
                f"student_widths has {len(student_widths)} entries but "
                f"profile has {len(self._per_layer_k)} layers"
            )
        projs = []
        for w_in, k in zip(student_widths, self._per_layer_k):
            proj = nn.Linear(w_in, k, bias=False)
            nn.init.orthogonal_(proj.weight)
            projs.append(proj)
        self._projectors = nn.ModuleList(projs)

    def _project_student(self, idx: int, s_hid: Tensor) -> Tensor:
        proj = self._projectors[idx]
        if s_hid.dim() == 4:
            B, C, H, W = s_hid.shape
            flat = s_hid.permute(0, 2, 3, 1).reshape(-1, C)
            out = proj(flat)
            return out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if s_hid.dim() == 3:
            return proj(s_hid)
        raise ValueError(f"unsupported student hidden shape {s_hid.shape}")

    def _project_teacher(self, idx: int, t_hid: Tensor) -> Tensor:
        V = getattr(self, f"V_{idx}")
        if t_hid.dim() == 4:
            return torch.einsum("bchw,ck->bkhw", t_hid, V)
        if t_hid.dim() == 3:
            return t_hid @ V
        raise ValueError(f"unsupported teacher hidden shape {t_hid.shape}")

    @staticmethod
    def _flatten(z: Tensor) -> Tensor:
        if z.dim() == 4:
            return z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        if z.dim() == 3:
            return z.reshape(-1, z.shape[-1])
        raise ValueError(f"unexpected shape {z.shape}")

    def _layer_loss(self, idx: int, s_hid: Tensor, t_hid: Tensor) -> Tensor:
        k = self._per_layer_k[idx]
        z_t = self._project_teacher(idx, t_hid)
        z_s = self._project_student(idx, s_hid)

        if self.normalize_features:
            if z_t.dim() == 4:
                z_t = F.normalize(z_t, p=2, dim=1, eps=1e-6)
                z_s = F.normalize(z_s, p=2, dim=1, eps=1e-6)
            else:
                z_t = F.normalize(z_t, p=2, dim=-1, eps=1e-6)
                z_s = F.normalize(z_s, p=2, dim=-1, eps=1e-6)

        w = getattr(self, f"W_{idx}", None)

        if self.objective == "coord_mse":
            diff_sq = (z_s - z_t) ** 2
            if w is not None:
                if z_t.dim() == 4:
                    diff_sq = diff_sq * w.view(1, -1, 1, 1)
                else:
                    diff_sq = diff_sq * w.view(1, 1, -1)
            return diff_sq.mean()

        zt = self._flatten(z_t)
        zs = self._flatten(z_s)
        if w is not None:
            sw = w.sqrt().clamp(min=1e-12)
            zt = zt * sw
            zs = zs * sw

        if self.objective == "gram":
            n = max(zt.shape[0], 1)
            g_s = (zs.T @ zs) / n
            g_t = (zt.T @ zt) / n
            c_st = (zs.T @ zt) / n
            frob_sq = (g_s ** 2).sum() + (g_t ** 2).sum() - 2 * (c_st ** 2).sum()
            return frob_sq.clamp(min=0.0) / (k * k)

        if self.objective == "cka":
            zt = zt - zt.mean(dim=0, keepdim=True)
            zs = zs - zs.mean(dim=0, keepdim=True)
            g_s = zs.T @ zs
            g_t = zt.T @ zt
            cross = zs.T @ zt
            num = (cross ** 2).sum()
            den = ((g_s ** 2).sum() * (g_t ** 2).sum()).clamp(min=1e-12).sqrt()
            return 1.0 - num / den

        raise ValueError(f"unknown objective {self.objective!r}")

    def forward(
        self,
        student_hiddens: Sequence[Tensor],
        teacher_hiddens: Sequence[Tensor],
    ) -> Tensor:
        """Compute the subspace loss.

        Args:
            student_hiddens: Student tensors in the same order as
                ``profile.layers``. Each is ``(B, C, H, W)`` or
                ``(B, T, C)``.
            teacher_hiddens: Same shapes, from the teacher.

        Returns:
            A scalar loss suitable for ``.backward()``.
        """
        if len(student_hiddens) != len(self._per_layer_k):
            raise ValueError(
                f"expected {len(self._per_layer_k)} student hiddens, "
                f"got {len(student_hiddens)}"
            )
        if len(teacher_hiddens) != len(self._per_layer_k):
            raise ValueError(
                f"expected {len(self._per_layer_k)} teacher hiddens, "
                f"got {len(teacher_hiddens)}"
            )
        if self._projectors is None:
            widths = []
            for s in student_hiddens:
                if s.dim() == 4:
                    widths.append(s.shape[1])
                elif s.dim() == 3:
                    widths.append(s.shape[-1])
                else:
                    raise ValueError(f"unsupported hidden shape {s.shape}")
            self._build_projectors(widths)
            self._projectors.to(student_hiddens[0].device)

        total = torch.zeros(
            (),
            device=student_hiddens[0].device,
            dtype=student_hiddens[0].dtype,
        )
        for idx, (s, t) in enumerate(zip(student_hiddens, teacher_hiddens)):
            total = total + self._layer_loss(idx, s, t)
        return total / max(len(student_hiddens), 1)


def _power_law_weights(
    eigenvalues: Tensor, k: int, power_p: float | None,
) -> Tensor | None:
    """Per-component weights ``w_i ~ lam_i^{-p}``, normalized to mean 1.

    Returns ``None`` when ``power_p is None``.
    """
    if power_p is None:
        return None
    sv = eigenvalues[:k].clone()
    lam_max = sv.max().clamp(min=1e-20)
    sv = sv.clamp(min=1e-6 * lam_max)
    w = sv.pow(-power_p)
    return w / w.mean().clamp(min=1e-20)


def build_student(
    template: nn.Module | str,
    profile: TeacherProfile,
    *,
    arch_multiplier: float = 1.0,
    arch_min: int | None = None,
    **kwargs,
) -> nn.Module:
    """Build a student network with widths derived from ``profile``.

    ``template`` selects the student family:

    - ``"slimnet"`` or a :class:`SlimNet` class: 4-stage ResNet-style
      student using ``profile.stage_widths(arch_multiplier)``.
    - An existing ``nn.Module``: infers the family from its type. For
      ``torchvision.models.ResNet`` a ``SlimNet`` is built. For
      HuggingFace ``GPT2LMHeadModel`` a reduced-width GPT-2 is built.

    For custom architectures, skip this helper. Build the student
    directly, then feed its per-layer hidden widths to
    ``SubspaceLoss(..., student_widths=...)``.
    """
    from . import builders
    return builders.build(
        template, profile,
        arch_multiplier=arch_multiplier,
        arch_min=arch_min,
        **kwargs,
    )


@dataclass
class DistillResult:
    student: nn.Module
    profile: TeacherProfile
    history: list[dict[str, Any]]
    best_metric: float
    teacher_metric: float | None


def distill(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    *,
    profile: TeacherProfile | None = None,
    val_loader: DataLoader | None = None,
    task_loss: Callable[[Any, Any], Tensor] | None = None,
    epochs: int = 20,
    lr: float = 0.1,
    optimizer: torch.optim.Optimizer | None = None,
    objective: Literal["coord_mse", "gram", "cka"] = "gram",
    kd: bool = True,
    kd_temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    delta: float = 1.0,
    device: str | None = None,
    **profile_kwargs,
) -> DistillResult:
    """One-call distillation.

    Profiles the teacher if ``profile`` is ``None``, then trains the
    student with task + subspace + (optional) KD loss.

    ``task_loss(logits, target)`` is the per-batch task loss;
    defaults to ``F.cross_entropy`` for classification.

    Returns a :class:`DistillResult` with the trained student, the
    profile, per-epoch history, and best-val metric.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if profile is None:
        profile = globals()["profile"](teacher, train_loader, **profile_kwargs)

    if task_loss is None:
        def task_loss(logits, target):
            return F.cross_entropy(logits, target)

    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = student.to(device)

    widths = _infer_student_widths(student, profile, train_loader, device)
    loss_fn = SubspaceLoss(
        profile, student_widths=widths, objective=objective,
    ).to(device)

    opt = optimizer or torch.optim.SGD(
        list(student.parameters()) + list(loss_fn.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4,
    )

    history: list[dict[str, Any]] = []
    best = float("-inf")
    for epoch in range(epochs):
        student.train()
        running = {"task": 0.0, "sub": 0.0, "kd": 0.0, "total": 0.0, "n": 0}
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                target = batch[1].to(device) if len(batch) > 1 else None
            else:
                x, target = batch.to(device), None

            with capture(teacher, profile) as t_hid:
                with torch.no_grad():
                    t_out = teacher(x)
                t_logits = _unpack_logits(t_out)
            t_hiddens = t_hid.values()

            with capture(student, profile) as s_hid:
                s_out = student(x)
                s_logits = _unpack_logits(s_out)
            s_hiddens = s_hid.values()

            losses = {
                "task": torch.zeros((), device=device),
                "sub": torch.zeros((), device=device),
                "kd": torch.zeros((), device=device),
            }
            if target is not None and s_logits is not None:
                losses["task"] = task_loss(s_logits, target)
            if len(s_hiddens) == len(t_hiddens):
                losses["sub"] = loss_fn(s_hiddens, t_hiddens)
            if kd and s_logits is not None and t_logits is not None:
                T = kd_temperature
                losses["kd"] = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)

            total = (
                alpha * losses["task"]
                + beta * losses["sub"]
                + delta * losses["kd"]
            )
            opt.zero_grad()
            total.backward()
            opt.step()

            for k, v in losses.items():
                running[k] += float(v.detach().item())
            running["total"] += float(total.detach().item())
            running["n"] += 1

        avg = {k: v / max(running["n"], 1) for k, v in running.items() if k != "n"}
        record = {"epoch": epoch, "train": avg}
        if val_loader is not None:
            record["val"] = _evaluate(student, val_loader, device, task_loss)
            if record["val"].get("accuracy", float("-inf")) > best:
                best = record["val"]["accuracy"]
        history.append(record)

    return DistillResult(
        student=student,
        profile=profile,
        history=history,
        best_metric=best,
        teacher_metric=None,
    )


def _unpack_logits(out) -> Tensor | None:
    """Extract a logits tensor from a model output.

    Handles torchvision tensors, HuggingFace dataclasses with
    ``.logits``, and tuples like ``(logits, features)``.
    """
    if out is None:
        return None
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, Tensor):
        return out
    return None


def _infer_student_widths(
    student: nn.Module,
    profile: TeacherProfile,
    loader: DataLoader,
    device: str,
) -> list[int]:
    """Run one forward pass under ``capture`` to record student hidden widths.

    Returns widths in profile order. Returns an empty list when the
    student does not share the profile's layer names.
    """
    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    x = x.to(device)
    with capture(student, profile) as cap:
        with torch.no_grad():
            student(x)
    widths: list[int] = []
    for n in profile.layers:
        if n not in cap:
            return []
        h = cap[n]
        widths.append(h.shape[1] if h.dim() == 4 else h.shape[-1])
    return widths


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    task_loss: Callable,
) -> dict[str, float]:
    model.eval()
    correct = total = 0
    losses = 0.0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            continue
        out = model(x)
        logits = _unpack_logits(out)
        if logits is None:
            continue
        losses += float(task_loss(logits, y).item())
        correct += int(logits.argmax(-1).eq(y).sum().item())
        total += int(y.size(0))
    if total == 0:
        return {"loss": losses}
    return {"accuracy": correct / total, "loss": losses / max(1, len(loader))}
