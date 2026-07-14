# substill

**Shrink a big model into a small, fast one, without training from scratch.**

`substill` compresses a neural network inside its *activation subspace*: the
low-dimensional space its activations actually occupy. Instead of guessing which channels
to cut, it keeps the directions the network relies on, starts the small "student" from the
big "teacher's" own weights, and trains a lightweight projection so the student matches
the teacher. The trained student folds back into an ordinary model, so inference costs
nothing extra.

The main method, **Learned Restriction Distillation (LRD)**, beats the strongest known
subspace-compression baseline by 5.0% perplexity on Llama-160M (n=6; 95% CI [-5.2, -2.3] PPL,
p < 0.001), with every seed beating every baseline seed. At larger scales the margin is not yet
measured to a standard we will defend. See the [benchmarks](benchmarks.md).

```{admonition} Soundness audit
:class: important

The restriction map was audited on 2026-07-13 and eight findings were fixed or withdrawn —
including two bugs that silently produced a student that was **not** a restriction of its
teacher, on **tied-embedding** models (Llama-3.2) and on **every GQA** model (Llama-3,
Mistral). Neither was visible in any benchmark, because every teacher benchmarked was MHA with
untied embeddings. The headline result reproduces and survives every fix — but the **margin
shrinks from 6.8% to 5.0%**, and the claim that it *grows with scale* is withdrawn as unsupported. The audit found two bugs in the initialization that *both* arms
share (a mis-scaled RMS gain, and a residual covariance dominated by the deepest layers). Neither
hurt LRD much — a trained projection can compensate for a bad start — but both badly hurt the
frozen-basis **baseline**, which cannot. Fixing them moves the baseline from 80.8 to **74.9**,
which is where LRD itself used to be. Roughly half the published margin was baseline weakness.
The audit is [§9–§11 of the LRD write-up](learned_restriction.md); it reports the corrections
that bought nothing, and the one that cost the method half its margin, alongside the ones that
worked.
```

## Install

```bash
pip install substill              # core library
pip install "substill[llm]"       # + transformers, datasets (the LLM path)
pip install "substill[docs]"      # + sphinx, furo, myst-parser (to build these docs)
```

Requires Python 3.10+ and PyTorch 2.0+.

## Quickstart

```python
import substill
from transformers import AutoModelForCausalLM

teacher = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")

result = substill.learned_restriction_distill(
    teacher, train_loader,        # any iterable of {"input_ids": ...} batches
    config=substill.LRDConfig.for_ratio(teacher, width_ratio=0.5, steps=2000),
)
student = result.student          # a narrower LlamaForCausalLM, ready to deploy
```

`LRDConfig.for_ratio` sizes the student from a single compression ratio, keeping whole
attention heads, and auto-tunes the projection learning rate.

**Scope.** LRD supports Llama-family decoders (Llama, Mistral). The [user guide](guide.md)
covers the vision path and the earlier pipeline.

## Where to next

New to model compression? The [explainer](explainer.md) assumes no background. To start
using the library, work through the [user guide](guide.md), with the
[API reference](api.md) open alongside it. The [concepts](concepts.md) page explains why
restriction works at all. And to check the evidence for yourself, read the
[benchmarks](benchmarks.md) and the full [LRD study](learned_restriction.md).

```{toctree}
:maxdepth: 2
:caption: Guide
:hidden:

guide
concepts
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

benchmarks
api
```

```{toctree}
:maxdepth: 1
:caption: Deep dives
:hidden:

explainer
learned_restriction
init_findings
```
