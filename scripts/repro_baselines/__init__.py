"""In-house reproductions of distillation baselines for matched-comparison.

Each script trains a student of *the same architecture and size* as the FSD
student, on *the same corpus and token budget*, with *the same optimizer
and LR schedule*. Only the loss differs, so the headline FSD comparison is
attributable to FSD's contributions — not to data, compute, or arch advantages.

Reproductions:
  - vanilla_kd_llama32.py     Hinton-style forward-KL distillation
  - distillm_llama32.py       DistiLLM (skew-KL)
  - minillm_llama32.py        MiniLLM (reverse-KL + on-policy)
  - gkd_llama32.py            GKD (on-policy generalised JSD)

Each script reads the same student config from the FSD run that produced it,
so the architecture is locked.
"""
