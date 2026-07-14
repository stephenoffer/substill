# substill

**Shrink a big model into a small, fast one, without training from scratch.**

`substill` compresses a neural network inside its *activation subspace*: the
low-dimensional space its activations actually occupy. Instead of guessing which channels
to cut, it keeps the directions the network relies on, starts the small "student" from the
big "teacher's" own weights, and trains a lightweight projection so the student matches
the teacher. The trained student folds back into an ordinary model, so inference costs
nothing extra.

The main method, **Learned Restriction Distillation (LRD)**, beats the strongest known
subspace-compression baseline by **5.0% perplexity** on Llama-160M (n=6, 95% CI [−5.2, −2.3] PPL,
p < 0.001, with every seed beating every baseline seed). At larger scales the margin is **not yet
measured to a standard we will defend** — see below.

## Install

```bash
pip install substill            # core library
pip install "substill[llm]"     # + transformers, datasets (the LLM path)
```

Python 3.10+, PyTorch 2.0+.

## Quickstart

Compress a Llama-family model in one call:

```python
import substill
from transformers import AutoModelForCausalLM

teacher = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")

result = substill.learned_restriction_distill(
    teacher, train_loader,                                   # iterable of {"input_ids": ...}
    config=substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000),
)

student = result.student        # a narrower LlamaForCausalLM, ready to deploy
print(f"final KD loss: {result.final_kd:.3f}")
```

`for_ratio` derives the student's geometry from a single compression ratio. The projection's
learning rate is a **trust region**: the number you set is the angle (in radians) the subspace
turns per step, so it is a physical quantity that transfers between teachers rather than a
constant that has to be re-fitted per model. It still needs *sanity-checking*, not just
accepting — if `result.max_principal_angle` approaches π/2, `V` has rotated nearly orthogonal to
its own initialization and the run has run away. To inspect or checkpoint between stages, drive
the three phases yourself:

```python
lrd = substill.LearnedRestriction(teacher, config)
lrd.prepare(calib_loader)   # profile the teacher's activation subspace, build the student
lrd.fit(train_loader)       # train the projection against the KD loss
student = lrd.fold()        # collapse to a plain, zero-overhead model
```

**Scope.** LRD supports Llama-family decoders (Llama, Mistral), including **grouped-query**
attention and **tied** embeddings — both of which used to be silently mis-compressed and are
now handled and pinned by test (see [§9a](docs/learned_restriction.md)). Hand it anything
else and it raises `NotImplementedError` rather than quietly doing the wrong thing. On a GQA
teacher the achievable widths are coarser, because whole query/key-value groups are kept
together; asking for a geometry that would break a group raises rather than re-pairing the
heads. GPT-2 and ResNet have their own paths, described in
[Also in the box](#also-in-the-box).

## Results

LRD against the strongest frozen-basis baseline, WikiText-2 perplexity. Lower is better;
compute is matched and seeds are reported.

| teacher | compression | best prior | **substill (LRD)** | gain | status |
|---|---:|---:|---:|---:|---|
| Llama-160M | 3.1× | 75.0 ± 1.1 | **71.3 ± 1.1** | **−5.0%** (n=6) | **settled**, p < 0.001 |
| Sheared-LLaMA-1.3B | 3.6× | 357.6 ± 45.6 | 329.2 ± 30.8 | −7.9% (n=4) | **not established**, p = 0.35 |
| Sheared-LLaMA-2.7B | 9.8× | — | — | — | **not re-measured** |

Uncertainty is stated properly rather than as a sigma count. On the 160M row, Welch t gives
**p < 0.001** with a 95% CI of **[−5.2, −2.3] PPL**, and the seeds separate completely (all six
LRD seeds beat all six baseline seeds). The "best prior" column is the *strongest* frozen basis we
could build — AIR's activation+influence principle, rebuilt on a corrected covariance — not a
straw man.

**We used to claim the margin *grew* with scale. It doesn't, as far as anyone can tell.** At 1.3B
the frozen baseline swings **±68 perplexity between two seeds**, so the n=2 and n=3 experiments
that claim −6.7% and −16.8% are describing which seeds were drawn as much as how the method
behaves — the old 2.7B row reports a baseline standard deviation of ±113 against a claimed effect
of 107. Re-measured at n=4 on a corrected map, 1.3B trends −7.9% with p = 0.35. The mechanism
transfers; the *margin at scale* is unmeasured, and we say so rather than quoting the number that
flatters us.

**The 160M row is down from a previously reported −6.8%, and that story is worth reading.** A
[soundness audit](docs/learned_restriction.md) found two bugs in the initialization that *both*
arms share — a mis-scaled RMS gain, and a residual covariance dominated by the deepest layers.
Neither hurt LRD much, because a trained projection can compensate for a bad start. Both hurt
the **baseline** badly, because a frozen basis cannot. Fixing them moves the frozen-basis
baseline from 80.8 to **75.0** — which is where LRD itself used to be. Roughly half the published
margin was baseline weakness, not method. What is left is smaller, better established (n=6,
p < 0.001), and composes: the best model in the study is LRD on the corrected basis. † The two
larger rows have not been re-measured on the corrected map and should be assumed to overstate the
margin similarly.

It also beats the 2026 SVD-compression wave — and the head-to-head has been **re-run on the
corrected covariance** (n=6). Rebuilt as frozen bases and distilled identically, every recent
principle lands in a dead heat: activation SVD (LASER/ASVD) 74.96, activation+influence (AIR)
75.00, activation-whitened (SVD-LLM) 74.97. **The choice of criterion does not matter; the
covariance they all share does.** LRD, which trains the basis instead of choosing it, reaches
71.80.

The win holds at every compression ratio tested, from 3× to 8×, and gets *steadier* as the
teacher grows — though those sweeps are also pre-audit.

Deployment is unaffected. The trained student folds down to an ordinary `LlamaForCausalLM`, so
there is nothing custom to run at inference time.

Full tables, every control and ablation, and how to reproduce them:
**[docs/learned_restriction.md](docs/learned_restriction.md)**.

## How it works

A trained network doesn't use all of its dimensions. `substill` exploits that in three
steps.

1. **Profile.** Run calibration data through the frozen teacher and find the subspace its
   activations actually live in — giving every layer an equal vote, rather than letting the
   high-norm deep layers pick the basis for everyone. (That one detail is worth more than
   anything else in the library; see [§11](docs/learned_restriction.md).)
2. **Restrict.** Project the teacher's weights onto that subspace (`W_s = Vᵀ W V`). The
   student is not a new model fitted to data; it is the teacher *seen through fewer
   dimensions*, so its layers still compose the way the teacher's do.
3. **Train.** Learn the projection `V` itself, on the Stiefel manifold so it stays a valid
   basis, against the distillation loss, jointly with the student's own weights. The student
   lands in the *best* subspace to keep, which is the one thing a frozen SVD cannot give you.

The principle in one line: **restrict the teacher's operator; never refit it.** Change the
basis, and the teacher's behavior transfers through distillation. Refit the weights to a
regression target, and it doesn't.

Two things it would be easy to over-claim here, so we don't:

- **The student is not *confined* to the restriction during training.** In the default
  configuration (the one that wins) the weights carry a free residual `D` alongside `V`, and
  `D` can reach any student an ordinary baseline can. The restriction is the *initialization*
  (`D` starts at zero, so training begins exactly where the baseline begins) and a *coordinate
  system* (moving `V` moves every layer coherently). How far the trained student actually
  drifts out of the restriction is measured rather than assumed:
  `LRDResult.restriction_gap`.
- **`V` is trained, but it is not the only thing trained.** It is about 1% of the parameters.
  The win comes from `V` and the weights co-adapting every step; training `V` alone reaches
  only ~100 PPL.

New to model compression? The plain-language [explainer](docs/explainer.md) assumes no
background.

## Also in the box

**Vision** (`substill.vision`) applies the same idea to ConvNets: narrow a ResNet's
channels and distill on class logits.

```python
from substill.vision import build_resnet_student, channel_variance_scores, distill_classifier
scores = channel_variance_scores(teacher, calib_loader)
student, _ = build_resnet_student(teacher, scores, width_ratio=0.5)
distill_classifier(teacher, student, train_loader, val_loader=val_loader)
```

**Lower-level pieces.** `substill.profile`, `substill.build_student`,
`substill.distill`, and the basis-invariant `substill.F_ASDLoss` are public, for when you
want to assemble your own pipeline.

**The FSD/CPSD pipeline** (`substill.FSDPipeline`) is the earlier, more general
activation-subspace pipeline. Its original headline numbers did not survive re-measurement
(see below), so LRD is the path we recommend, though the machinery it exposes is still
useful.

## Examples and docs

```bash
python examples/learned_restriction.py   # LRD on a tiny Llama (CPU, no download)
python examples/vision_resnet.py         # narrow + distill a small ResNet
```

The documentation site (user guide plus full API reference) is built with Sphinx and
published to GitHub Pages on every push to `main`. To build it locally:

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html   # -> docs/_build/html/index.html
```

Everything public lives on the top-level namespace, so `import substill` is all you need
(see `substill.__all__`).

## Tested, and honest about the numbers

```bash
python -m pytest    # 339 tests, CPU-only, no downloads
ruff check .        # lint
```

Every benchmark above comes from a controlled, multi-seed run you can reproduce. Holding
to that bar is the reason LRD exists at all: an independent re-measurement found that this project's
*original* FSD/CPSD headline numbers collapsed once compute was matched and the baseline
learning rate was tuned. LRD is the method that survived, and it is stated here with the
controls that isolate its win. The full audit, including bugs we found in our own
baselines, is in **[docs/init_findings.md](docs/init_findings.md)**.

We then did the same thing to LRD. A soundness audit of the restriction map itself
(**[docs/learned_restriction.md §9](docs/learned_restriction.md)**) found six defects — two of
which silently produced a student that was *not* a restriction of its teacher at all, on
architectures this README advertises:

- **Tied embeddings** (Llama-3.2-1B/3B and most small Llamas) were silently corrupted by the
  norm fold, changing the teacher's own function by ~20% before distillation even began.
- **Grouped-query attention** (Llama-3, Mistral) had its query heads re-paired with key/value
  heads they were never trained against, at most compression ratios.

Both are fixed and pinned by regression tests. Neither was visible in any published number,
because every teacher benchmarked here is MHA with untied embeddings — which is the point: the
benchmark suite and the test suite shared a blind spot. The audit also withdraws one false
theoretical claim, replaces a hand-fitted constant with a scale-free rule, and restates the
statistics for the small `n` they actually have. The headline result reproduces and survives.

## License

MIT.
