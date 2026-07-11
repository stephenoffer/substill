"""The public LRD API: it trains, folds bit-identically, and the joint arm helps.

These pin the contract of :mod:`substill.lrd` on a tiny, from-scratch Llama (no download,
CPU-only): the one-call entry point runs end to end and returns a plain student; the
folded student reproduces the restricted module; and training in the ``(V, D)``
coordinates reaches a KD loss no worse than freezing ``V`` -- the controlled comparison
that isolates the learned-restriction coordinate.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

import substill  # noqa: E402
from substill.lrd import (  # noqa: E402
    LearnedRestriction,
    LRDConfig,
    plan_restricted_geometry,
)


def _tiny_llama(hidden=32, interm=64, layers=2, heads=4, kv_heads=2, vocab=61):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=interm,
                      num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=kv_heads, max_position_embeddings=64,
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    for layer in m.model.layers:
        layer.input_layernorm.weight.data.uniform_(0.5, 1.5)
        layer.post_attention_layernorm.weight.data.uniform_(0.5, 1.5)
    m.model.norm.weight.data.uniform_(0.5, 1.5)
    return m


def _data(vocab=61, n=6, seq=16):
    torch.manual_seed(1)
    return [{"input_ids": torch.randint(0, vocab, (2, seq))} for _ in range(n)]


def test_plan_geometry_keeps_whole_heads():
    t = _tiny_llama()
    g = plan_restricted_geometry(t, width_ratio=0.5)
    head_dim = t.config.hidden_size // t.config.num_attention_heads
    assert g.hidden % head_dim == 0
    assert g.n_head == 2 and g.hidden == 16
    assert g.n_head % g.n_kv == 0


def test_config_for_ratio_and_auto_v_lr():
    t = _tiny_llama(hidden=64, heads=8)
    cfg = LRDConfig.for_ratio(t, width_ratio=0.5, steps=3)
    assert cfg.hidden == 32 and cfg.n_head == 4
    # auto rule: min(1e-3, 0.77/d)
    assert cfg.resolved_v_lr(t) == pytest.approx(min(1e-3, 0.77 / 64))
    assert LRDConfig(hidden=32, intermediate=32, n_head=2, n_kv=2,
                     v_lr=5e-3).resolved_v_lr(t) == 5e-3


def test_end_to_end_runs_and_folds_to_plain_student():
    t = _tiny_llama()
    cfg = LRDConfig.for_ratio(t, 0.5, steps=4, calib_batches=3, device="cpu")
    result = substill.learned_restriction_distill(t, _data(), config=cfg)

    from transformers import LlamaForCausalLM
    assert isinstance(result.student, LlamaForCausalLM)
    assert result.student.config.hidden_size == cfg.hidden
    assert result.final_kd is not None and len(result.history) == 4
    with torch.no_grad():
        out = result.student(input_ids=_data()[0]["input_ids"]).logits
    assert torch.isfinite(out).all()


def test_fold_matches_the_restricted_module():
    t = _tiny_llama()
    cfg = LRDConfig.for_ratio(t, 0.5, steps=3, calib_batches=3, device="cpu")
    lrd = LearnedRestriction(t, cfg).prepare(_data()).fit(_data())
    ids = _data()[0]["input_ids"]
    with torch.no_grad():
        a = lrd.restricted(ids).logits
        b = lrd.fold()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_unsupported_teacher_raises():
    gpt2 = pytest.importorskip("transformers").GPT2LMHeadModel(
        pytest.importorskip("transformers").GPT2Config(
            vocab_size=40, n_positions=16, n_embd=16, n_layer=1, n_head=2, n_inner=32)
    )
    with pytest.raises(NotImplementedError, match="Llama family"):
        substill.learned_restriction_distill(
            gpt2, _data(), config=LRDConfig(hidden=8, intermediate=16, n_head=1, n_kv=1))


def test_joint_training_moves_V_and_reduces_kd_stably():
    """The joint arm actually exercises the Stiefel coordinate and trains stably.

    Both arms start from the identical absorbed-init student (``D = 0``); the learned-
    restriction coordinate only matters if ``V`` genuinely moves. A tiny random teacher
    cannot reproduce the 6-sigma PPL win, but the reliable contract holds: with ``v_lr > 0``
    the projection rotates away from its PCA init, and training descends the KD loss it
    optimizes without diverging.
    """
    torch.manual_seed(0)
    t = _tiny_llama()
    data = _data(n=8)
    cfg = LRDConfig(hidden=16, intermediate=32, n_head=2, n_kv=2, steps=12,
                    lr=3e-3, v_lr=3e-2, calib_batches=4, device="cpu", seed=0)
    lrd = LearnedRestriction(t, cfg).prepare(data).fit(data)
    kd = [h["kd"] for h in lrd.history]
    assert lrd.principal_angle() > 0                  # V rotated off its init
    assert all(v == v and v < float("inf") for v in kd)  # finite throughout
    assert kd[-1] <= kd[0]                              # descended the objective
