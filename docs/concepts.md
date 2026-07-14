# How it works

This page explains the idea behind `substill` in depth. For a gentler introduction that
assumes no background, read the [explainer](explainer.md) first. For the code, see the
[user guide](guide.md).

## The problem with cutting channels

A trained network has far more dimensions than it actually uses. The obvious ways to
shrink it, dropping the "least important" channels or factoring each weight matrix with
SVD, run into a stubborn wall: **the pieces you keep were tuned to work alongside the
pieces you threw away.**

Refit them to compensate and you get a model that reproduces the teacher's activations on
your calibration set, but has quietly *replaced* the teacher's computation with a
regression fit. That replacement does not survive further training.

`substill` takes the opposite stance.

## Restrict, don't refit

The core object is the teacher's **activation subspace**, the low-dimensional space its
residual stream actually occupies. Given an orthonormal basis `V` for a `k`-dimensional
subspace of the `d`-dimensional stream, `substill` builds the student by *restricting*
every teacher weight onto it:

```
W_s = Vᵀ W_T V
```

This is not a new matrix fit to data. It is the teacher's own weight, viewed through fewer
coordinates. Because every layer is restricted with the *same* `V`, the student's layers
still compose the way the teacher's do, and behavior is preserved as a structural property
rather than an approximation.

> **The principle in one line:** *restrict the teacher's operator; never refit it.*
> Changing the basis restricts, and transfers through distillation. Refitting the weights
> replaces, and does not. This held on two architectures and across every ablation we
> tried; the evidence is in the [re-measurement audit](init_findings.md).

## Learn the subspace

Which subspace should you keep?

Every prior method picks `V` up front by some frozen criterion: top-variance directions
(PCA), an influence score, reconstruction error. Each is a *surrogate*. It optimizes a
proxy for the student's quality rather than the quality itself.

`substill`'s key move is to stop guessing and **train `V` directly against the
distillation loss, through the whole network.** `V` lives on the Stiefel manifold, the
space of orthonormal frames, so a Riemannian optimizer (`StiefelAdamV`) keeps it a valid
basis at every step while gradient descent rotates it toward the subspace that actually
minimizes the KD loss.

Why does that beat a frozen criterion? Consider a direction that barely reaches the output
on its own, but that layer 3 needs in order to compute what layer 9 writes. A frozen,
output-linearized criterion scores it as unimportant. Training through the network finds
it.

Trained jointly with `V` is a zero-initialized Euclidean residual `D`, giving
`W_s = Vᵀ W_T V + D`. Because `D` starts at zero, training *begins* at exactly the
absorbed-init student that a plain baseline starts from. The learned-restriction
coordinate is therefore the only thing that differs between the two, which is what makes
the win measurable in isolation.

## The three phases

1. **Profile** the teacher. Fold its RMSNorm gains into the consuming linears (which
   preserves the function), measure the second moment of its residual stream, and pick the
   starting basis `V₀` (PCA by default).
2. **Fit.** Rotate `V` on the Stiefel manifold and move `D` in the ordinary directions,
   both descending the KD loss on the teacher's own logits.
3. **Fold.** Collapse `(V, D)` into a plain `LlamaForCausalLM` whose weights are
   `Vᵀ W_T V + D`. It is function-identical to the trained module, so inference carries
   **zero overhead**. Nothing custom runs at deploy time.

## Why it scales

The only trainable object that grows with the teacher is one `(d, k)` matrix. The
teacher's weights are read-only, so they carry no optimizer state. That is what lets the
same recipe run on a 30M-parameter student and, in principle, on a frontier decoder, where
materializing `Vᵀ W V` per edge is affordable but keeping optimizer state for every
teacher weight is not.

One hyperparameter has to track teacher size: the Stiefel learning rate, which is roughly
the rotation per step. Larger, deeper, more-compressed students accumulate rotation over
more layers, so `v_lr` scales as `≈ 1/d`. `LRDConfig` sets it automatically.

## Scope, and the vision analogue

Restriction relies on the residual stream being **rotation-equivariant**. That holds for
RMSNorm transformer decoders (Llama, Mistral), which is where the learned-rotation win
lives. A ReLU CNN has no such stream: only channel *selection* commutes with ReLU, not
rotation.

So in `substill.vision` the restriction *principle* transfers, and selecting the teacher's
behavioral channels beats random init by about 8 points on ResNet-50 / CIFAR-10. The
learned *rotation* does not apply. That independently pins the transformer win on the
rotation rather than the selection. See the [benchmarks](benchmarks.md).
