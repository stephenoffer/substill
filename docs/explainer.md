# substill, explained from scratch

*A plain-language walkthrough for readers new to model compression and knowledge
distillation. It assumes no background. For the method in depth, see
[concepts](concepts.md); for the numbers and their controls, see
[benchmarks](benchmarks.md).*

## The problem

Modern neural networks are big and expensive to run. A 3-billion-parameter language model
needs a lot of memory and energy for every single prediction, and often you want something
smaller that behaves almost the same: cheap enough to serve at high traffic, or small
enough to fit on a phone, while staying nearly as accurate.

The usual framing calls the big model the **teacher** and the small one the **student**.
Building a good student is what *model compression* and *knowledge distillation* are for.
`substill` is a particular way of doing it.

## The insight: a trained network doesn't use all of itself

Everything rests on one observation.

Take a layer in a trained network with, say, 768 "channels". Think of them as 768 dials
the layer can turn. You might assume all 768 carry independent, useful information. They
don't. Once the network is trained, the activations that layer actually produces tend to
live in a much *smaller* space. Perhaps only 200 of those 768 directions carry real
signal; the rest are close to redundant. The layer has 768 dials, but it effectively turns
about 200 knobs.

> **An analogy.** Picture a 768-piece orchestra in which, for the music actually being
> played, only about 200 instruments ever play a distinct part. The other 568 just double
> notes already covered. Send most of them home and the sound barely changes.

That smaller, genuinely-used space is the **activation subspace**. The whole strategy is
to find the subspace the teacher actually uses, and build the student to live in exactly
that space, instead of compressing blindly channel by channel.

Finding it is not exotic. Run a little representative data through the frozen teacher and
watch the activations, then compute the directions that carry statistically real signal.
(The standard tool for this is PCA, plus a noise-floor cutoff so you keep signal and drop
noise.)

## Idea one: restrict the teacher instead of rebuilding it

Most compression methods pick which channels to keep, then *refit* the remaining weights,
either training them from random initialization or regressing them against the teacher's
activations. Both run into the same wall: the pieces you kept were tuned to work alongside
the pieces you threw away.

`substill` does something different. It doesn't fit new weights at all. Given a basis `V`
for the subspace worth keeping, it *restricts* each of the teacher's weight matrices onto
that subspace:

```
W_s = Vᵀ W V
```

This is not a new matrix fitted to data. It is the teacher's own weight, viewed in fewer
coordinates. And because every layer is restricted with the *same* `V`, the student's
layers still compose the way the teacher's did. The teacher's behavior is preserved
structurally rather than approximated.

> **An analogy.** Rather than copying a painting onto a blank canvas, you start from a
> photograph of the original and touch up the details. You begin most of the way there.

The principle in one line: **restrict the teacher's operator; never refit it.**

## Idea two: don't guess the subspace, learn it

That leaves the real question. *Which* subspace should you keep?

Every prior method answers it up front, with a frozen rule: take the highest-variance
directions, or the ones with the best influence score, or the ones that minimize
reconstruction error. Each of those is a *surrogate*. It optimizes a proxy for the
student's quality rather than the quality itself, and then never revisits the choice.

`substill`'s central move is to stop guessing and **train `V` directly against the
distillation loss, through the whole network**. The basis becomes a parameter. It lives on
the Stiefel manifold (the set of valid orthonormal frames), and a Riemannian optimizer
rotates it, keeping it a legal basis at every step while gradient descent steers it toward
the subspace that genuinely minimizes the loss.

This finds things a frozen rule cannot. Consider a direction that contributes almost
nothing to the output on its own, but that layer 3 needs in order to compute what layer 9
eventually writes. A frozen, output-linearized criterion scores it as unimportant and
discards it. Training through the network keeps it.

That method is **Learned Restriction Distillation (LRD)**, and it is what `substill`
recommends you use.

## What the numbers say

On WikiText-2 perplexity, where lower is better, LRD beats the strongest frozen-basis
baseline (PCA) by **6.8%**, at matched compute, over 3 seeds (95% CI [-8.4, -4.8] PPL, Welch p=0.002). That is a large gap,
and every LRD seed beat every baseline seed.

The comparison is unusually tight. The control arm runs LRD's exact code path with the
rotation learning rate set to zero, which reproduces the PCA baseline. So the only thing
that changes between the two arms is whether `V` is allowed to move, and turning that one
coordinate on is worth 5.8 points of perplexity.

The margin also *grows* with the teacher. It reaches 16.8% on a 2.7B-parameter teacher,
and the training gets steadier at scale rather than shakier. Against the recent wave of
SVD-based compression methods, LRD beats the best of them (AIR) by 5.7%, which is the
result you'd predict from the argument above: those methods all climb a frozen proxy, and
that proxy turns out not to predict distilled quality.

The [benchmarks](benchmarks.md) page has the full tables, along with the controls and the
commands to reproduce them.

## Does it work outside language models?

Partly, and the way it fails is informative.

Restriction depends on the network having a *rotation-equivariant* residual stream. That
is true of RMSNorm transformer decoders such as Llama and Mistral. It is not true of a
ReLU convolutional network, where only channel *selection* commutes with the
nonlinearity, not rotation.

So on a ResNet-50, you can still select the teacher's behaviorally important channels and
absorb its weights into them, and that works: it beats random initialization by around 8
points of accuracy on CIFAR-10. But you cannot *rotate* the subspace, and sure enough, the
learned-rotation variant does no better than plain selection there.

That is a useful negative. It means the transformer win comes specifically from
rotating the subspace, not merely from selecting it. The vision arm isolates the mechanism
by being the case where the mechanism is unavailable.

## What we know, and what we don't

This project has retracted its own headline before, and it's worth being direct about
that. An earlier method here (FSD/CPSD) reported large gains that did not survive
re-measurement: once compute was matched and the baseline's learning rate was tuned, the
advantage went away. The audit also turned up a bug in our own pipeline that had been
silently feeding every one of those students a truncated, unprofiled basis.

LRD is the method that came through that process intact, and the numbers above are stated
with the controls that isolate the win. The full account, including the bugs we found in
our own baselines, is in the [re-measurement audit](init_findings.md). Read it before
citing any number from this project.

What remains open: LRD is verified on Llama-family decoders up to 2.7B parameters, on
WikiText-2, at short training budgets. A decisive head-to-head against published
state-of-the-art on a large model, at a full training budget, is the next milestone rather
than a finished claim.

## Where to go next

To run it, the [user guide](guide.md) covers the one-call API, the configuration options,
and the vision path. To understand it properly, [concepts](concepts.md) gives the
restriction principle in depth. To check the evidence, start with the
[benchmarks](benchmarks.md), then the full [LRD study](learned_restriction.md) and the
[audit](init_findings.md).
