# Examples

Short, self-contained scripts that run on CPU in seconds and need no downloads, because
they build tiny models from scratch. Install the package first (`pip install -e .`; add
`[llm]` for the GPT-2 examples and `torchvision` for the vision one).

| script | what it shows |
|---|---|
| [`learned_restriction.py`](learned_restriction.py) | **LRD (the verified method):** distil a tiny Llama by training its restriction `Vᵀ W_T V` on the Stiefel manifold, then fold to a plain student. |
| [`quickstart_pipeline.py`](quickstart_pipeline.py) | The earlier `FSDPipeline`: profile a GPT-2 teacher, build a half-width student from its activation subspace, distill a few steps. |
| [`absorbed_init.py`](absorbed_init.py) | Why absorbed init matters: it starts far closer to the teacher than random init (lower KD before any training). |
| [`vision_resnet.py`](vision_resnet.py) | The non-LLM arm: rank a ResNet's channels, narrow each Bottleneck, and distill on class logits. |

```bash
python examples/learned_restriction.py
python examples/quickstart_pipeline.py
python examples/absorbed_init.py
python examples/vision_resnet.py
```

For real training runs and the reproductions behind the docs, see [`../scripts/`](../scripts).
