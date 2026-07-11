# substill

**Shrink a big model into a small, fast one — without training from scratch.**

`substill` compresses a neural network inside its *activation subspace* — the
low-dimensional space the model actually uses. Instead of guessing which channels to cut,
it keeps the directions the network relies on, starts the small "student" from the big
"teacher's" own behavior, and trains a lightweight projection so the student matches the
teacher. The result folds down to a **plain model with zero inference overhead**.

Its headline method — **Learned Restriction Distillation (LRD)** — beats the strongest known
subspace-compression baseline by **6.8% perplexity (~6σ)**, and the margin grows with model
size. See the [benchmarks](benchmarks.md).

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
student = result.student          # a narrower LlamaForCausalLM — deploy as-is
```

`LRDConfig.for_ratio` sizes the student from a single compression ratio (keeping whole
attention heads) and auto-tunes the projection learning rate. **Scope:** Llama-family
decoders (Llama, Mistral). The [user guide](guide.md) covers everything else.

## Where to next

- **New to model compression?** The [explainer](explainer.md) assumes no background.
- **Want to use it?** The [user guide](guide.md) and [API reference](api.md).
- **Want to understand it?** The [concepts](concepts.md) page.
- **Want the evidence?** The [benchmarks](benchmarks.md) and the full
  [LRD study](learned_restriction.md).

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
