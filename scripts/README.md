# Scripts

Entry points for training runs, benchmarks, and the reproductions behind the
docs. Run them from the repo root with `PYTHONPATH=.` (or after `pip install -e .`).
For a gentle introduction, start with [`../examples/`](../examples) instead.

| directory | contents |
|---|---|
| [`cpsd/`](cpsd) | CPSD head-to-head comparison (`cpsd_compare.py`), result aggregation, and the GQA-Llama CPI init probe. |
| [`fsd/`](fsd) | The FSD Llama-3.2 trainer (`distill_llama32_fsd.py`), the GPT-2 headline experiment, the pillar ablation grid, and the eval harness. |
| [`vision/`](vision) | The ResNet distillation ladder (`resnet50_distill.py`, `--smoke` for CPU). |
| [`analysis/`](analysis) | The `docs/init_findings.md` reproductions and the Learned Restriction study (`lrb.py`); `h2h.py`/`bench.py` are the shared KD loop and benchmark. |
| [`repro_baselines/`](repro_baselines) | Matched-architecture reproductions of published baselines (vanilla KD, GKD, MiniLLM, DistiLLM). |

`.sh` and `.yaml` files next to a script are cluster-submission launchers for that script.
