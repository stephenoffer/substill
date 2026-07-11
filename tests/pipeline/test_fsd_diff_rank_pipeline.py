"""DDR (distillation-driven differentiable rank) wired through FSDPipeline.

Validates the DDR integration: ``use_diff_rank=True`` (on top of
``use_cpsd_factored``) wraps every TeacherFactoredLinear with a GatedFactoredLinear,
builds a RankBudgetController, trains the gates jointly with the manifold-trained bases
against the KD loss, then hardens + folds to a plain-nn.Linear student for inference.
"""
from __future__ import annotations

import pytest
import torch

from substill.compression.factored_linear import (
    GatedFactoredLinear,
    TeacherFactoredLinear,
)


def _toy_gpt2(n_layer=2, n_embd=32, n_head=4):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=n_embd,
                     n_layer=n_layer, n_head=n_head, n_inner=4 * n_embd)
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def _loader(n=8, B=2, T=8, vocab=64):
    torch.manual_seed(0)
    return [{"input_ids": (tok := torch.randint(5, vocab - 1, (B, T))),
             "labels": tok, "attention_mask": torch.ones(B, T, dtype=torch.long)}
            for _ in range(n)]


def test_gated_factored_linear_fold_matches_soft_forward():
    """fold(harden=False) reproduces the gated forward exactly (gate folds into columns)."""
    torch.manual_seed(0)
    d_in, d_out, k = 12, 10, 6
    W = torch.randn(d_out, d_in)
    V_in = torch.linalg.qr(torch.randn(d_in, k))[0]
    V_out = torch.linalg.qr(torch.randn(d_out, k))[0]
    tfl = TeacherFactoredLinear(W, V_in, V_out, torch.randn(d_out), free_core=True)
    gated = GatedFactoredLinear(tfl)
    # Push the gate off the all-open point so the fold is a non-trivial column scaling.
    with torch.no_grad():
        gated.gate.alpha.copy_(torch.linspace(-2.0, 4.0, k))
    x = torch.randn(5, k)
    lin = gated.fold(harden=False)
    assert torch.allclose(gated(x), lin(x), atol=1e-4)
    # cost is one (d_in+d_out) entry per latent column.
    assert gated.cost().shape == (k,)
    assert float(gated.cost()[0]) == float(d_in + d_out)


def test_pipeline_diff_rank_trains_hardens_and_folds():
    teacher = _toy_gpt2()
    if teacher is None:
        pytest.skip("transformers not installed")
    import substill

    pipe = substill.FSDPipeline(teacher, config=substill.FSDConfig(
        arch_multiplier=0.5, use_cpsd_factored=True, use_diff_rank=True,
        diff_rank_target_ratio=0.7, total_steps=4, lr=5e-4,
        distill_kwargs={"on_policy_start": 2.0, "quantize": False},
    ))
    pipe.run_profile(_loader())
    student = pipe.build()

    # Gates were attached and a budget controller built over the factored edges.
    kin_by_name = {n: m.k_in for n, m in student.named_modules()
                   if isinstance(m, GatedFactoredLinear)}
    assert kin_by_name, "no GatedFactoredLinear edges attached"
    assert pipe.rank_controller is not None
    assert set(pipe.rank_controller.gates) == set(kin_by_name)
    init_expected = float(pipe.rank_controller.expected_params().item())
    assert init_expected > 0 and torch.isfinite(torch.tensor(init_expected))

    with torch.no_grad():
        student.eval()
        assert torch.isfinite(student(**_loader(n=1)[0]).logits).all()

    result = pipe.train(_loader())
    assert torch.isfinite(next(result.student.parameters())).all()

    # Harden + fold to a deployable student: no factored/gated modules remain, the
    # rank-map is integer-valued within [1, k_in], and the forward stays finite.
    rank_map = pipe.fold_for_inference(harden=True)
    assert rank_map, "expected a hardened rank-map"
    assert set(rank_map) == set(kin_by_name)
    assert all(1 <= rank_map[n] <= kin_by_name[n] for n in rank_map)
    assert not any(isinstance(m, (GatedFactoredLinear, TeacherFactoredLinear))
                   for m in pipe.student.modules())
    with torch.no_grad():
        pipe.student.eval()
        assert torch.isfinite(pipe.student(**_loader(n=1)[0]).logits).all()
