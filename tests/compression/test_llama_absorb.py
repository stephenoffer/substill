"""Correctness of the Llama absorb path (substill.compression.llama_absorb).

This path exists to falsify the GPT-2 residual-basis finding on a model with RMSNorm and
untied embeddings, so its own correctness has to be beyond doubt: if the gamma fold
changed the teacher's function, or full-width absorption did not reproduce it, the
comparison it supports would be meaningless.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from substill.compression.llama_absorb import (  # noqa: E402
    absorb_llama,
    build_narrow_llama,
    gamma_fold_llama,
    llama_residual_second_moment,
    rms_gain,
)


@pytest.fixture(scope="module")
def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=32, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).eval()
    # Non-trivial RMSNorm gains, else the fold is vacuous.
    for layer in model.model.layers:
        layer.input_layernorm.weight.data.normal_(1.0, 0.2)
        layer.post_attention_layernorm.weight.data.normal_(1.0, 0.2)
    model.model.norm.weight.data.normal_(1.0, 0.2)
    calib = [{"input_ids": torch.randint(0, 60, (2, 16))} for _ in range(2)]
    return model, calib


def test_gamma_fold_preserves_logits(tiny):
    model, calib = tiny
    folded = gamma_fold_llama(model)
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a, b = model(input_ids=ids).logits, folded(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    for layer in folded.model.layers:
        assert torch.allclose(layer.input_layernorm.weight,
                              torch.ones_like(layer.input_layernorm.weight))
        assert torch.allclose(layer.post_attention_layernorm.weight,
                              torch.ones_like(layer.post_attention_layernorm.weight))
    assert torch.allclose(folded.model.norm.weight, torch.ones_like(folded.model.norm.weight))


def test_gamma_fold_does_not_mutate_the_original(tiny):
    model, calib = tiny
    before = model.model.layers[0].input_layernorm.weight.detach().clone()
    gamma_fold_llama(model)
    assert torch.equal(model.model.layers[0].input_layernorm.weight, before)


def test_full_width_absorb_reproduces_the_teacher(tiny):
    """k == hidden and interm == intermediate: every basis is the identity, so the
    absorbed student must be the folded teacher bit-for-bit."""
    model, calib = tiny
    folded = gamma_fold_llama(model)
    d, di = folded.config.hidden_size, folded.config.intermediate_size
    student = build_narrow_llama(folded, d, di, folded.config.num_attention_heads,
                                 folded.config.num_key_value_heads)
    absorb_llama(folded, student, torch.eye(d), [torch.eye(di)] * folded.config.num_hidden_layers,
                 norm_gain=1.0)
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a, b = folded(input_ids=ids).logits, student.eval()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_rms_gain_is_unity_at_full_width(tiny):
    model, calib = tiny
    folded = gamma_fold_llama(model)
    S = llama_residual_second_moment(folded, calib, device="cpu")
    assert rms_gain(S, torch.eye(folded.config.hidden_size)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("basis", ["identity", "random_sel", "select", "pca"])
def test_narrow_absorb_is_finite_for_every_basis(tiny, basis):
    from substill.compression.seq_absorb import residual_basis

    model, calib = tiny
    folded = gamma_fold_llama(model)
    d, di = folded.config.hidden_size, folded.config.intermediate_size
    S = llama_residual_second_moment(folded, calib, device="cpu")
    V = residual_basis(S, d // 2, method=basis)
    student = build_narrow_llama(folded, d // 2, di // 2, 2, 1)
    absorb_llama(folded, student, V, [torch.eye(di)[:, : di // 2]] * 2,
                 norm_gain=rms_gain(S, V))
    with torch.no_grad():
        out = student.eval()(input_ids=calib[0]["input_ids"]).logits
    assert torch.isfinite(out).all()


def test_student_keeps_the_teachers_head_dim(tiny):
    model, _ = tiny
    folded = gamma_fold_llama(model)
    student = build_narrow_llama(folded, 16, 32, 2, 1)
    t_hd = folded.config.hidden_size // folded.config.num_attention_heads
    assert student.config.head_dim == t_hd
    assert student.config.tie_word_embeddings is False


def test_hidden_states_last_is_post_final_norm(tiny):
    """As for GPT-2: `hidden_states[-1]` is `model.norm(...)` applied to the last layer's
    output. `gap_fit_llama` must take its last target from a layer hook, not from there,
    or block L-1 is asked to emit a normalized state."""
    model, calib = tiny
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    assert torch.allclose(model.lm_head(out.hidden_states[-1]), out.logits, atol=1e-4)
    assert len(out.hidden_states) == model.config.num_hidden_layers + 1


def test_gap_fit_reduces_drift_uniformly_across_depth(tiny):
    """Every block, including the last, must end with small residual drift. A post-norm
    target for the final block shows up here as a drift spike."""
    from substill.compression.llama_absorb import gap_fit_llama
    from substill.compression.seq_absorb import residual_basis

    model, calib = tiny
    folded = gamma_fold_llama(model)
    d, di = folded.config.hidden_size, folded.config.intermediate_size
    S = llama_residual_second_moment(folded, calib, device="cpu")
    V = residual_basis(S, d // 2, method="pca")
    student = build_narrow_llama(folded, d // 2, di // 2, 2, 1)
    absorb_llama(folded, student, V, [torch.eye(di)[:, : di // 2]] * 2,
                 norm_gain=rms_gain(S, V))
    drifts = gap_fit_llama(folded, student, V, calib, device="cpu")
    assert len(drifts) == folded.config.num_hidden_layers
    assert all(0.0 <= x < 1.0 for x in drifts), drifts
    # the last block must not be an outlier relative to the others
    assert drifts[-1] < 3 * max(drifts[:-1]) + 1e-6, drifts
