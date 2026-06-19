# neural_distill documentation

`fasd` is the library; `asd` is an internal subpackage of profiling
utilities it builds on (not a public API). The
[top-level README](../README.md) has the pitch and the public-API
overview. This directory holds the technical docs.

## fasd

| file                                               | what it is                                            |
|----------------------------------------------------|-------------------------------------------------------|
| [explainer.md](explainer.md)                       | **Start here if you're new** — plain-language walkthrough: the idea, why it matters, what's proven (no background assumed) |
| [cpsd.md](cpsd.md)                                 | CPSD (Circuit-Preserving Subspace Distillation) — the novel system behind `FSDPipeline` |
| [adding_architectures.md](adding_architectures.md) | Checklist for supporting a new model family via `ArchitectureSpec` |
| [fasd/quickstart.md](fasd/quickstart.md)           | F-ASD library walkthrough                             |
| [fasd/algorithm.md](fasd/algorithm.md)             | F-ASD mechanism & math                                |

## Project status & history

| file                     | what it is                                          |
|--------------------------|-----------------------------------------------------|
| [report.md](report.md)   | Integration milestone snapshot                      |
| [handoff.md](handoff.md) | Sprint-by-sprint build status + next-phase plan     |
| [archive/](archive/)     | Older status snapshots (historical)                 |

## Skim the code

- Public API: [`fasd/__init__.py`](../fasd/__init__.py) and
  [`fasd/pipeline.py`](../fasd/pipeline.py) (`FSDPipeline`).
- Internal profiling utilities: [`asd/profiling/`](../asd/profiling/).
