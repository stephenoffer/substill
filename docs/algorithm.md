# How ASD works

For usage, see [../README.md](../README.md) and
[quickstart.md](quickstart.md). This document explains the mechanism.

## Setup

Given a large teacher `T` and a task dataset, we want a smaller
student `S` that behaves like `T`. ASD drives the student to match
the teacher inside a low-rank subspace of the teacher's activations,
layer by layer.

## Three phases

### 1. Profile

For each layer of interest, hook the teacher's forward pass and
accumulate the channel covariance `C_l = E[x_l x_l^T]` over a
calibration set. What the "layer's activation" means depends on
`source`:

- `source="output"`: the layer's forward output.
- `source="delta"`: the residual update `dx_l = output - shortcut(input)`,
  which strips the identity path on residual architectures. On
  ResNets, the identity typically dominates the output covariance and
  hides the block's actual contribution. Delta makes that
  contribution observable.
- `source="branch"`: a sub-module's output (attention or MLP inside a
  transformer block). Like delta, this isolates a specific
  computation from the residual stream.

Eigendecompose each `C_l = V_l L_l V_l^T` in descending eigenvalue
order. Pick a retained rank `k_l` using one of:

- **Variance threshold**: smallest `k` with `sum(lam_1..lam_k) / sum(lam) >= tau`
  (default tau = 0.95).
- **Marchenko-Pastur / Gavish-Donoho bulk edge**
  (`noise_model="mp"`): for an i.i.d. Gaussian covariance with
  aspect ratio `beta = C / N_eff`, the noise eigenvalues form a
  bulk whose upper edge sits at `omega(beta)^2 * sigma^2`.
  Eigenvalues above that threshold are signal; below are noise. In
  practice this gives a much tighter rank estimate than variance +
  eps on spectra with a long noise tail.
- Optional Ledoit-Wolf shrinkage applied to `C_l` before
  eigendecomposition for small-N calibration sets.

The profile `{V_l, L_l, k_l}` for each layer is the subspace
snapshot. It is pickle-safe.

### 2. Loss

At training time, hook the teacher and student at matching layers.
For each teacher hidden `x_l^T` and student hidden `x_l^S`, project
into the top-`k_l` subspace:

    z_l^T = x_l^T V_l                  in R^{N x k}
    z_l^S = proj_l(x_l^S)              in R^{N x k}

where `proj_l` is a per-layer learnable linear projection from the
student's hidden dim to `k_l`. Then compute one of three objectives:

- **coord_mse**: `||z^T - z^S||^2`. Point-wise match in the teacher's
  eigenbasis. Basis-sensitive: when eigenvalues are close, the basis
  inside the retained subspace is arbitrary up to rotation, and the
  student has to match an arbitrary basis pointlessly precisely.

- **gram**: `||K_s - K_t||_F^2` where `K = Z Z^T` is the token or
  pixel-wise kernel matrix. Basis-invariant under rotations of `V_l`
  within the retained subspace. Computed via the trace identity
  `||K_s||^2 + ||K_t||^2 - 2 ||Z_s^T Z_t||^2`, so only k x k inner
  products are materialized.

- **cka**: `1 - <K_s, K_t>_F / (||K_s|| * ||K_t||)`. Centered kernel
  alignment. Fully scale-invariant: the student can over- or
  under-shoot the teacher's feature magnitude and CKA still drives
  alignment. This is what makes the loss stable on LLMs, where
  residual-stream magnitudes are large.

**Feature normalization** is L2 per sample over the channel axis
before forming the kernels. Without it, Gram entries scale as
`magnitude^4 * k^2` and blow up to 1e6 on GPT-2-class features. With
it, entries stay in `[-1, 1]` and the loss stays bounded regardless
of residual magnitude. Default is on.

**Spectral weighting** is optional: `w_i ~ lam_i^{-p}` for
`p` in `[-1.5, 1.5]`. `p = 0` (uniform) is the default. `p < 0`
emphasizes large-eigenvalue directions (the "loud" variance). `p > 0`
emphasizes small-eigenvalue directions (whitening), often preferred
on transformer spectra with heavy tails.

### 3. Train

Add the subspace loss to whatever training loop you already have:

    L = alpha * L_task + beta * L_subspace + delta * L_logit_kd

The student and the subspace loss's learnable projectors are both
optimized. Everything else about the training loop (optimizer,
schedule, gradient clipping, mixed precision) is unchanged.

## Why it works

Two intuitions stacked:

1. **Teachers do not use all their channels.** Layer covariance
   spectra are heavy-head: the first few dozen principal directions
   carry most of the teacher's computation. A student with channel
   width equal to the effective rank has the capacity to reproduce
   the teacher's behavior at that layer.

2. **Subspace alignment is a cleaner supervision signal than raw
   feature MSE.** Matching the teacher's top-`k` subspace is
   permissive about where exactly the student puts its features
   within that subspace. Matching full-width features forces the
   student to mimic the teacher's axis choices, many of which are
   redundant or arbitrary.

The objective family (coord_mse, gram, cka) trades increasing
invariance for decreasing fine-grained signal. Coord MSE is the
tightest match but most fragile. Gram is invariant to basis rotations
inside the retained subspace. CKA is additionally invariant to
overall scale. Pick the loosest objective that still drives your
student: typically gram for CNNs and cka for LLMs.

## What the knobs control

| knob                 | effect                                                                        |
|----------------------|-------------------------------------------------------------------------------|
| `source`             | what gets profiled: raw activations, residual updates, or branches            |
| `noise_model`        | how retained rank `k` is chosen: fixed threshold or signal/noise MP cutoff    |
| `shrinkage`          | regularize the covariance before eigendecomposition (small-N setups)          |
| `objective`          | how the student matches the teacher: tightest to loosest is mse, gram, cka    |
| `normalize_features` | bound kernel entries; required for LLM stability                              |
| `power_weight_p`     | reweight components across the spectrum                                       |
| `arch_multiplier`    | scale the retained rank for a larger student (more slack for optimization)    |

## Implementation modules

- [`asd/profiling/activation_capture.py`](../asd/profiling/activation_capture.py):
  forward-hook accumulator, `CovarianceAccumulator`.
- [`asd/profiling/svd_analysis.py`](../asd/profiling/svd_analysis.py):
  `SVDAnalyzer` with MP and Ledoit-Wolf options.
- [`asd/profiling/stability.py`](../asd/profiling/stability.py):
  optional diagnostic, bootstrap principal-angle stability across
  calibration splits.
- [`asd/api.py`](../asd/api.py): `profile`, `capture`,
  `SubspaceLoss`, `distill`, `TeacherProfile`. The public surface.
- [`asd/autodetect.py`](../asd/autodetect.py): layer auto-detection
  for known model families.
- [`asd/builders.py`](../asd/builders.py): student constructors for
  torchvision ResNets and HuggingFace GPT-2.
- [`asd/models/student.py`](../asd/models/student.py): `SlimNet`, the
  4-stage ResNet student used by `build_student` for CNN teachers.
