"""Faithful-as-we-can KD objectives, and the on-policy data policy they need.

`docs/init_findings.md` §5 says we never established anything about MiniLLM / GKD /
DistiLLM, because the `reverse_kl` and `skew_kl` arms in `scripts/bench.py` were not those
methods: no on-policy student sampling, no replay, and -- as it turns out -- a skew-KL with
its mixture weights the wrong way round.

    DistiLLM defines  SKL_a(p || q) = KL( p || a*p + (1-a)*q ),  a = 0.1.

`bench.kd_loss` computed `KL(p || 0.9*p + 0.1*q)`, whose value is bounded by log(1/0.9) =
0.105 and which carries **1.6%** of forward-KL's gradient signal at a typical
teacher/student gap. That arm's 403.33 PPL measured a loss that was barely training the
model. It should never have been reported next to the others.

This module implements the divergences as their papers define them, plus the on-policy
sampling that GKD and MiniLLM depend on. It is still not a reimplementation of those
systems -- MiniLLM's sequence-level policy gradient with length normalization and its
teacher-mixed rollouts are absent, and DistiLLM's adaptive off-policy schedule is replaced
by a fixed ratio. Read the arms as "the objective and data policy each paper contributes,
dropped into one controlled harness", not as the published methods. Where a number here
disagrees with a paper's, believe the paper.

References: MiniLLM 2306.08543, GKD 2306.13649, DistiLLM 2402.03425.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _log_mix(a: float, lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """log(a*exp(lp) + (1-a)*exp(lq)), numerically stable."""
    return torch.logsumexp(
        torch.stack([lp + math.log(a), lq + math.log1p(-a)]), dim=0)


def divergence(s_logits, t_logits, kind: str, *, skew: float = 0.1,
               beta: float = 0.5) -> torch.Tensor:
    """Token-mean divergence between teacher `t` and student `s` distributions.

    forward_kl   KL(p_t || p_s)                       -- Hinton KD
    reverse_kl   KL(p_s || p_t)                       -- the MiniLLM objective
    skl          KL(p_t || a*p_t + (1-a)*p_s)         -- DistiLLM skew KL
    srkl         KL(p_s || a*p_s + (1-a)*p_t)         -- DistiLLM skew reverse KL
    jsd          b*KL(p_t||m) + (1-b)*KL(p_s||m),  m = b*p_t + (1-b)*p_s   -- GKD
    """
    ls = F.log_softmax(s_logits.reshape(-1, s_logits.size(-1)), -1)
    lt = F.log_softmax(t_logits.reshape(-1, t_logits.size(-1)), -1)
    def kl(a, b):  # KL(a||b)
        return F.kl_div(b, a, reduction="batchmean", log_target=True)

    if kind == "forward_kl":
        return kl(lt, ls)
    if kind == "reverse_kl":
        return kl(ls, lt)
    if kind == "skl":
        return kl(lt, _log_mix(skew, lt, ls))
    if kind == "srkl":
        return kl(ls, _log_mix(skew, ls, lt))
    if kind == "jsd":
        m = _log_mix(beta, lt, ls)
        return beta * kl(lt, m) + (1.0 - beta) * kl(ls, m)
    raise ValueError(f"unknown divergence: {kind!r}")


@torch.no_grad()
def sample_on_policy(student, ids, *, prefix: int = 16, temperature: float = 1.0):
    """Replace each sequence's continuation with one the *student* generates.

    GKD's central claim is that the student should be trained on its own outputs, to fix
    the train/inference distribution mismatch; MiniLLM samples from the student for the
    same reason. Sampling is done under `no_grad` -- gradients flow only through the
    divergence evaluated on the sampled tokens, which is GKD's formulation (it does not
    backprop through the sampling itself).
    """
    was_training = student.training
    student.eval()
    out = student.generate(
        input_ids=ids[:, :prefix],
        max_new_tokens=ids.shape[1] - prefix,
        min_new_tokens=ids.shape[1] - prefix,
        do_sample=True,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        pad_token_id=getattr(student.config, "eos_token_id", None) or 0,
    )
    if was_training:
        student.train()
    return out[:, : ids.shape[1]]


# name -> (divergence kind, on-policy fraction).  The two axes each paper varies.
ARMS = {
    "kd":       ("forward_kl", 0.0),   # Hinton
    "rkl":      ("reverse_kl", 0.0),   # reverse KL, off-policy
    "minillm":  ("reverse_kl", 1.0),   # MiniLLM: reverse KL on student rollouts
    "gkd":      ("jsd", 0.5),          # GKD: generalized JSD, half on-policy
    "distillm": ("skl", 0.0),          # DistiLLM: skew KL (off-policy)
    "distillm_op": ("skl", 0.5),       # DistiLLM + on-policy
}
