# substill, explained from scratch

*A plain-language walkthrough for readers new to model compression and knowledge
distillation. No prior familiarity with these algorithms is assumed. For how the method
works, see [concepts](concepts.md); for the numbers, see [benchmarks](benchmarks.md).*

---

## The problem in one paragraph

Modern neural networks — the large language models behind chat assistants, the vision
models behind image search — are **big and expensive to run**. A 3-billion-parameter model
needs a lot of memory and energy for every single prediction. Often you want a **smaller
model that behaves almost like the big one**: cheaper to serve, fast enough for a phone or a
high-traffic API, but nearly as accurate. The big model is the **teacher**; the small one we
build is the **student**. Producing a good student is the goal of *model compression* and
*knowledge distillation*. FASD is a method for doing this unusually well — and, as of this
work, it does it for **both language models and vision models** with the same core idea.

---

## The key insight: a trained network doesn't use all of itself

Here is the observation everything rests on.

A layer in a trained network has, say, 768 "channels" (think of them as 768 dials it can
turn). You might assume all 768 carry independent, important information. **They don't.**
After training, the activations a layer actually produces tend to live in a much
*smaller* space — maybe only 200 of those 768 directions carry real signal, and the rest are
near-redundant noise. The network has 768 dials, but it effectively turns only ~200 knobs.

> **Analogy.** Imagine a 768-piece orchestra where, for the music actually being played, only
> ~200 instruments ever play distinct parts and the other ~568 just double notes already
> covered. You could send most of those musicians home and barely change the sound.

That smaller, genuinely-used space is the **activation subspace**. FASD's entire strategy is:
**find the subspace each part of the teacher actually uses, and build the student to live in
exactly that space** — instead of compressing blindly, channel by channel.

How do we find it? We run a bit of representative data through the frozen teacher, watch the
activations, and compute the directions that carry real, statistically-distinguishable signal
(a standard tool called PCA, plus a noise-floor cutoff so we keep signal and drop noise). The
number of directions we keep is the layer's **behavioral rank** — how many knobs it really
uses.

---

## Why where you *start* matters: "absorbed initialization"

Most compression methods build the small student with **random** weights and then train it to
imitate the teacher from scratch. That's like asking someone to copy a painting starting from
a blank canvas — possible, but slow and rarely faithful.

FASD instead **starts the student inside the teacher's subspace** and **folds the teacher's
own weights into it**. Concretely, if the teacher's weight matrix is `W`, and `V` is the basis
for the subspace we're keeping, the student starts with `Vᵀ W V` — the teacher's exact
behavior, *projected onto the directions that matter*. This is called **absorbed
initialization**.

> **Analogy.** Instead of a blank canvas, you start from a high-quality *photograph* of the
> original painting and only touch up the details. You begin most of the way there.

**This is FASD's most important, most reproducible win** — and it shows up on both kinds of
models we tested:

- **Language (GPT-2 on WikiText-2):** at matched size, a student that starts random reaches
  ~1038 perplexity; a student with absorbed-init reaches **559** (lower is better) — roughly
  **1.9× better**, just from where it started.
- **Vision (ResNet-50 on CIFAR-10):** a random-init student reaches 64.8% accuracy; the
  absorbed-init student reaches **81.1%** — a **+16 point** jump at the same compression.

Same idea, two very different model families. That breadth is the headline.

---

## What CPSD adds (and an honest account of what's proven)

The novel research layer on top of absorbed-init is **CPSD — Circuit-Preserving Subspace
Distillation**. It has three ideas; we report each honestly, including where it doesn't (yet)
pay off.

1. **MT — manifold-trained factors.** Instead of freezing the subspace we picked, let the
   student *gently rotate* its subspace during training to better fit the teacher, while
   keeping the teacher's weights frozen inside it. (The "manifold" is the mathematical set of
   valid rotations; we train on it with a specialized optimizer.) *Status: a modest but real win.*
   At first it actually **lost** to the simpler frozen approach — the rotating subspace had too
   little freedom to adapt. We diagnosed that, gave each factor a small extra "free core" of
   adjustable weights (which costs nothing at inference), and it flipped to a small win (546.6 vs
   558.9 perplexity, and *steadier* across runs). A good example of how an honest negative, once
   understood, becomes an improvement.

2. **DDR — distillation-driven differentiable rank.** Rather than fixing in advance how many
   directions each layer keeps, **let the model learn the right number** — driven by the
   distillation loss itself, under a global size budget. The contrast with a well-known
   competitor (Dobi-SVD) is sharp: Dobi-SVD picks ranks to minimize *reconstruction error*;
   we pick them to minimize *what the student gets wrong relative to the teacher*. *Status:
   our distillation-driven rank beats the reconstruction-driven approach by 1.4–2.2× in our
   matched comparison — a real win for the idea.*

3. **CPI — circuit-preserving initialization.** An attempt to align the student's attention
   "circuits" with the teacher's at init. *Status: an honest negative result — it did not beat
   the simpler baseline on a real model. We document it as a negative so others don't repeat
   it.*

We keep the negatives visible on purpose: a method you can trust is one whose authors tell you
where it doesn't work.

---

## Why it works on vision too

Nothing above is specific to language. A convolution is just a linear operation on image
channels, so the *same* "project onto the used subspace, absorb the teacher's weights" math
applies — we simply narrow the channels each layer uses. We compress a ResNet's internal
"bottleneck" channels while leaving the connections between blocks intact, so the network
still fits together. The result (the +16 accuracy points above) shows the core idea is a
*general principle*, not an LLM trick. Most compression papers cover one or the other; FASD
spans both.

---

## Why this matters — the implications

- **Cheaper, faster models without starting over.** Because the student begins from the
  teacher's own behavior, it reaches good quality with far less training than random-init
  distillation — saving compute, time, and money.
- **Zero inference overhead.** All the clever factored/trained machinery **folds back into
  ordinary layers** before deployment. The shipped student is a plain, fast model — none of
  the training-time complexity remains at run time.
- **One method, many model families.** The same pipeline compresses LLMs *and* CNNs. As models
  proliferate, a general compression principle is more valuable than a per-architecture hack.
- **Honest, measurable claims.** Every number here was produced by a reproducible script
  (`scripts/cpsd/cpsd_compare.py`, `scripts/vision/resnet50_distill.py`) on real hardware, with seeds and
  baselines — including the comparisons we *lose* or only tie.

---

## Where we stand, in one honest sentence

FASD's absorbed-initialization beats naive distillation baselines clearly and reproducibly on
both language and vision; its distillation-driven rank beats a real competitor mechanism
(Dobi-SVD); and the full novel method now *modestly* beats even our own strong absorbed-init
baseline (after the free-core fix). The decisive head-to-head against *published* state-of-the-art
on a large model is the next milestone, not a finished claim.

---

## Where to go next

- **Run it:** the [user guide](guide.md) — the LRD one-call API, config, and the vision path.
- **Understand it:** the [concepts](concepts.md) page — the restriction principle in depth.
- **The evidence:** the [benchmarks](benchmarks.md), and the full [LRD study](learned_restriction.md).
