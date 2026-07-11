# substill

**Shrink a big model into a small, fast one — without training from scratch.**

`substill` compresses a neural network inside its *activation subspace* — the
low-dimensional space the model actually uses. Instead of guessing which channels to cut,
it keeps the directions the network relies on, starts the small "student" from the big
"teacher's" own behavior, and trains a lightweight projection so the student matches the
teacher. The result folds down to a **plain model with zero inference overhead**.

Its headline method — **Learned Restriction Distillation (LRD)** — beats the strongest
known subspace-compression baseline by **6.8% perplexity (~6σ)**, and the margin *grows*
with model size.

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

student = result.student        # a narrower LlamaForCausalLM — deploy as-is
print(f"final KD loss: {result.final_kd:.3f}")
```

`for_ratio` picks a sensible student geometry from a single compression ratio and
auto-tunes the one tricky hyperparameter (the projection learning rate). Want more
control? Run the three phases yourself:

```python
lrd = substill.LearnedRestriction(teacher, config)
lrd.prepare(calib_loader)   # profile the teacher's activation subspace, build the student
lrd.fit(train_loader)       # train the projection against the KD loss
student = lrd.fold()         # collapse to a plain, zero-overhead model
```

**Supported teachers:** Llama-family decoders (Llama, Mistral). GPT-2 and ResNet have their
own paths — see [Also in the box](#also-in-the-box).

## Benchmarks

LRD vs the strongest frozen-basis baseline (PCA), WikiText-2 perplexity — **lower is
better**, matched compute, seeds reported:

| teacher | compression | best prior | **substill (LRD)** | gain |
|---|---:|---:|---:|---:|
| Llama-160M | 3.1× | 80.9 ± 0.9 | **75.5 ± 0.8** | **−6.8%**  (~6σ, n=3) |
| Sheared-LLaMA-1.3B | 3.6× | 366.7 ± 0.8 | **342.0 ± 7.2** | **−6.7%**  (n=2) |
| Sheared-LLaMA-2.7B | 9.8× | 635.6 ± 113 | **528.7 ± 36** | **−16.8%**  (n=3) |

- 🏆 **Beats the 2026 SVD-compression wave.** Reproduced head-to-head, LRD beats the best
  recent method (AIR) by **5.7%** — because it optimizes the real distillation objective, not
  a frozen proxy like reconstruction or influence.
- 📈 **Wins at every compression ratio** tested (3×→8×), and gets *more stable* — not just
  better — as the teacher grows.
- ⚡ **Zero inference overhead.** The trained student folds to an ordinary
  `LlamaForCausalLM`; there's nothing custom to run at deploy time.

Full tables, controls, ablations, and reproduction:
**[docs/learned_restriction.md](docs/learned_restriction.md)**.

## How it works

A trained network doesn't use all of its dimensions. `substill`:

1. **Profiles** the teacher to find the subspace its activations actually live in.
2. **Restricts** the teacher's weights onto that subspace (`W_s = Vᵀ W V`) — so the student
   is literally the teacher *seen through fewer dimensions*, and its layers still compose the
   way the teacher's do.
3. **Trains** the projection `V` (on the Stiefel manifold) against the distillation loss, so
   the student learns the *best* subspace to keep — the one thing a frozen SVD can't give you.

The principle in one line: **restrict the teacher's operator; never refit it.** Change the
basis and the teacher's behavior transfers through distillation; refit the weights and it
doesn't.

New to model compression? The plain-language **[explainer](docs/explainer.md)** assumes no
background.

## Also in the box

- **Vision** (`substill.vision`) — the same idea for ConvNets: narrow a ResNet's channels and
  distill on class logits.
  ```python
  from substill.vision import build_resnet_student, channel_variance_scores, distill_classifier
  scores = channel_variance_scores(teacher, calib_loader)
  student, _ = build_resnet_student(teacher, scores, width_ratio=0.5)
  distill_classifier(teacher, student, train_loader, val_loader=val_loader)
  ```
- **Lower-level API** — `substill.profile`, `substill.build_student`, `substill.distill`, and
  the basis-invariant `substill.F_ASDLoss`, for building your own pipeline.
- **FSD/CPSD pipeline** (`substill.FSDPipeline`) — the earlier, more general activation-subspace
  pipeline. Its original headline numbers didn't survive re-measurement (see below), so LRD is
  the recommended path; the machinery it exposes is still useful.

## Examples & docs

```bash
python examples/learned_restriction.py   # LRD on a tiny Llama (CPU, no download)
python examples/vision_resnet.py         # narrow + distill a small ResNet
```

- **Documentation site** (user guide + full API reference) is built with Sphinx and published
  to GitHub Pages on every push to `main`. Build it locally:
  ```bash
  pip install -e ".[docs]"
  python -m sphinx -b html docs docs/_build/html   # -> docs/_build/html/index.html
  ```
- **API:** `import substill` — everything public is on the top-level namespace
  (see `substill.__all__`).

## Tested, and honest about it

```bash
python -m pytest    # 313 tests, CPU-only, no downloads
ruff check .        # lint
```

Every benchmark above is a **controlled, multi-seed, reproducible** result — and holding to
that bar is why LRD exists. An independent re-measurement found the project's *original*
FSD/CPSD headline numbers didn't hold up under compute matching and a tuned learning rate;
LRD is the method that survived, stated with the controls that isolate its win. The full
audit — including bugs we found in our own baselines — is in
**[docs/init_findings.md](docs/init_findings.md)**. We think that transparency is a feature.

## License

MIT.
