# How it works

This page explains the idea behind `substill` in depth. For a gentler, no-background
introduction, read the [explainer](explainer.md) first; for the code, see the
[user guide](guide.md).

## The problem with cutting channels

A trained network has far more dimensions than it actually uses. The obvious way to shrink
it — drop the "least important" channels or factor each weight matrix with SVD — runs into
a stubborn wall: **the pieces you keep were tuned to work alongside the pieces you threw
away.** Refit them to compensate and you get a model that reproduces the teacher's
activations on your calibration set but has quietly *replaced* the teacher's computation
with a regression fit. That replacement does not survive further training.

`substill` takes the opposite stance.

## Restrict, don't refit

The core object is the teacher's **activation subspace** — the low-dimensional space its
residual stream actually occupies. Given an orthonormal basis `V` for a `k`-dimensional
subspace of the `d`-dimensional stream, `substill` builds the student by *restricting* every
teacher weight onto that subspace:

```
W_s = Vᵀ W_T V
```

This is not a new matrix fit to data — it is the teacher's own weight, viewed through fewer
coordinates. Because every layer is restricted with the *same* `V`, the student's layers
still compose the way the teacher's do. Behavior is preserved as a *structural* property,
not approximated.

> **The principle in one line:** *restrict the teacher's operator; never refit it.* Changing
> the basis restricts — and transfers through distillation. Refitting the weights replaces —
> and does not. This was confirmed on two architectures and every ablation tried; the full
> evidence is in the [re-measurement audit](init_findings.md).

## Learn the subspace

Which subspace should you keep? Every prior method picks `V` up front by some frozen
criterion — top-variance directions (PCA), an influence score, reconstruction error. Each is
a *surrogate*: it optimizes a proxy for the student's quality, not the quality itself.

`substill`'s key move is to stop guessing and **train `V` directly against the distillation
loss, through the whole network.** `V` lives on the Stiefel manifold (the space of
orthonormal frames), so a Riemannian optimizer (`StiefelAdamV`) keeps it a valid basis at
every step while gradient descent rotates it toward the subspace that actually minimizes the
KD loss. A direction that barely reaches the output on its own — but that layer 3 needs to
compute what layer 9 writes — is exactly what a frozen, output-linearized criterion misses
and what training through the network finds.

Trained jointly with `V` is a zero-initialized Euclidean residual `D` (`W_s = Vᵀ W_T V + D`).
Because `D` starts at zero, training *begins* at exactly the absorbed-init student a plain
baseline starts from — so the learned-restriction coordinate is the only thing that differs,
which is what makes the win measurable in isolation.

## The three phases

1. **Profile** the teacher: fold its RMSNorm gains into the consuming linears
   (function-preserving), then measure the second moment of its residual stream and pick the
   starting basis `V₀` (PCA by default).
2. **Fit**: rotate `V` on the Stiefel manifold and move `D` in the ordinary directions, both
   descending the KD loss on the teacher's own logits.
3. **Fold**: collapse `(V, D)` into a plain `LlamaForCausalLM` whose weights are
   `Vᵀ W_T V + D`. It is function-identical to the trained module, so **inference carries
   zero overhead** — there is nothing custom to run at deploy time.

## Why it scales

The only trainable object that grows with the teacher is one `(d, k)` matrix; the teacher's
weights are read-only, so they carry no optimizer state. That is what lets the same recipe
run on a 30M-parameter student and, in principle, on a frontier decoder — where
materializing `Vᵀ W V` per edge is affordable but keeping optimizer state for every teacher
weight is not.

The single hyperparameter that must track teacher size is the Stiefel learning rate (roughly
the rotation per step): larger, deeper, more-compressed students accumulate rotation over
more layers, so `v_lr` scales as `≈ 1/d`. `LRDConfig` sets it automatically.

## Scope and the vision analogue

Restriction relies on the residual stream being **rotation-equivariant** — true of RMSNorm
transformer decoders (Llama, Mistral), which is where the learned-rotation win lives. A
ReLU CNN has no such stream: only channel *selection* commutes with ReLU, not rotation. So
in `substill.vision` the same *restriction principle* transfers (selecting the teacher's
behavioral channels beats random init by ~8 points on ResNet-50/CIFAR-10), but the *learned
rotation* does not apply — which independently pins the transformer win on the rotation, not
the selection. See the [benchmarks](benchmarks.md).
