"""The restricted student must be exactly the model it folds into.

Everything `substill.compression.restricted` claims rests on one identity: training ``V``
inside `RestrictedLlama` and then calling `fold()` yields a plain `LlamaForCausalLM` that
computes the *same function*. If the two ever drift apart, phase 2 starts from a different
model than phase 1 ended on and the whole comparison is meaningless.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from substill.compression.llama_absorb import (  # noqa: E402
    gamma_fold_llama,
    llama_residual_second_moment,
)
from substill.compression.restricted import (  # noqa: E402
    RestrictedLlama,
    ffn_energy_indices,
    qr_retract,
)


def _tiny_llama(hidden=32, interm=64, layers=2, heads=4, vocab=61):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=interm,
                      num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=heads, max_position_embeddings=64,
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    # A freshly-initialised RMSNorm has weight == 1, which makes the gamma fold a no-op
    # and hides any bug in it. Give the norms real gains.
    for layer in m.model.layers:
        layer.input_layernorm.weight.data.uniform_(0.5, 1.5)
        layer.post_attention_layernorm.weight.data.uniform_(0.5, 1.5)
    m.model.norm.weight.data.uniform_(0.5, 1.5)
    return m


def _calib(vocab=61, n=3, seq=16):
    torch.manual_seed(1)
    return [{"input_ids": torch.randint(0, vocab, (2, seq))} for _ in range(n)]


def _restricted(k=16, interm=32, n_head=2):
    t = gamma_fold_llama(_tiny_llama()).eval()
    calib = _calib()
    S = llama_residual_second_moment(t, calib, device="cpu")
    evals, evecs = torch.linalg.eigh(S.double())
    V0 = evecs[:, -k:].flip(-1).float()
    idx = ffn_energy_indices(t, calib, interm, device="cpu")
    return RestrictedLlama(t, S, V0, idx, n_head, n_head), calib


def test_fold_is_function_identical():
    """`fold()` must reproduce the restricted module's logits, not merely approximate them."""
    rm, calib = _restricted()
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a = rm(ids).logits
        b = rm.fold().eval()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_fold_tracks_a_moved_basis():
    """The identity must hold at a *trained* ``V``, not just the PCA starting point.

    A fold that silently re-derives the basis (or the RMS gain) from the teacher would
    pass the previous test and fail this one.
    """
    rm, calib = _restricted()
    with torch.no_grad():
        rm.V.copy_(qr_retract(rm.V + 0.3 * torch.randn_like(rm.V)))
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a = rm(ids).logits
        b = rm.fold().eval()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_full_width_restriction_reproduces_the_teacher():
    """At ``k == d`` with all heads and neurons kept, restriction is the identity map.

    This is the sanity check that ``V^T W V`` is wired the way the docstring says: any
    transposed factor or mismatched head slice would survive the two tests above (both
    compare the module against its own fold) but not this one.
    """
    t = gamma_fold_llama(_tiny_llama()).eval()
    calib = _calib()
    S = llama_residual_second_moment(t, calib, device="cpu")
    d = t.config.hidden_size
    idx = ffn_energy_indices(t, calib, t.config.intermediate_size, device="cpu")
    rm = RestrictedLlama(t, S, torch.eye(d), idx, t.config.num_attention_heads,
                         t.config.num_key_value_heads)
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        assert torch.allclose(rm(ids).logits, t(input_ids=ids).logits, atol=1e-4)


def test_gain_is_differentiable_in_V():
    """The RMS gain is part of the restriction map, so gradient must flow through it.

    Freezing it at ``rms_gain(S, V0)`` would let ``V`` drift to a subspace whose retained
    energy the norms no longer compensate -- a silent mis-scaling that only shows up as a
    worse final PPL.
    """
    rm, _ = _restricted()
    g = rm.gain()
    g.backward()
    assert rm.V.grad is not None and rm.V.grad.abs().sum() > 0


def _restricted_free(k=16, interm=32, n_head=2):
    t = gamma_fold_llama(_tiny_llama()).eval()
    calib = _calib()
    S = llama_residual_second_moment(t, calib, device="cpu")
    _, evecs = torch.linalg.eigh(S.double())
    V0 = evecs[:, -k:].flip(-1).float()
    idx = ffn_energy_indices(t, calib, interm, device="cpu")
    return RestrictedLlama(t, S, V0, idx, n_head, n_head, free=True), calib


def test_zero_residual_is_the_plain_restriction():
    """``free=True`` with ``D = 0`` must be the absorbed-init student, exactly.

    This is what makes `lrb_joint` a controlled comparison against `pca`: both arms start
    training from the identical model, so any difference in the final student is caused by
    the Stiefel coordinate and not by a different starting point.
    """
    rf, calib = _restricted_free()
    rp, _ = _restricted()
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        assert torch.allclose(rf(ids).logits, rp(ids).logits, atol=1e-5)


def test_hidden_and_logits_match_forward_and_expose_the_stream():
    """`hidden_and_logits` must return the same logits as `forward`, plus the per-layer
    ``(L, B, T, k)`` student stream used by the restriction-consistency aux loss."""
    rf, calib = _restricted_free()
    torch.manual_seed(4)
    with torch.no_grad():          # a non-trivial residual so the two paths could diverge
        for p in rf.D.values():
            p.normal_(0, 0.02)
        rf.D_emb.normal_(0, 0.02)
        rf.D_lm.normal_(0, 0.02)
    ids = calib[0]["input_ids"]
    L = rf.teacher[0].config.num_hidden_layers
    with torch.no_grad():
        ref = rf(ids).logits
        out, hs = rf.hidden_and_logits(ids)
    assert torch.allclose(out.logits, ref, atol=1e-5), (out.logits - ref).abs().max()
    assert hs.shape == (L, ids.shape[0], ids.shape[1], rf.k)


def test_restriction_consistency_is_differentiable_in_V():
    """The cosine consistency term must push gradient back into ``V`` on both sides."""
    from substill.lrd import _restriction_consistency

    rf, calib = _restricted_free()
    ids = calib[0]["input_ids"]
    teacher = rf.teacher[0]
    t_hidden = teacher(input_ids=ids, output_hidden_states=True).hidden_states
    _, hs = rf.hidden_and_logits(ids)
    aux = _restriction_consistency(rf.V, hs, t_hidden)
    assert 0.0 <= float(aux.detach()) <= 2.0
    aux.backward()
    assert rf.V.grad is not None and rf.V.grad.abs().sum() > 0


def test_free_fold_is_function_identical_with_nonzero_residual():
    """`fold()` must add ``D`` into every weight it belongs to -- including the embedding
    and the unembedding, which the forward pass handles on the side and never materializes."""
    rf, calib = _restricted_free()
    torch.manual_seed(3)
    with torch.no_grad():
        for p in rf.D.values():
            p.normal_(0, 0.02)
        rf.D_emb.normal_(0, 0.02)
        rf.D_lm.normal_(0, 0.02)
        rf.V.copy_(qr_retract(rf.V + 0.2 * torch.randn_like(rf.V)))
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a = rf(ids).logits
        b = rf.fold().eval()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_free_residual_matches_the_students_own_parameter_count():
    """``D`` is not extra capacity: it has exactly the shape of the folded student's
    weights, so `lrb_joint` and `pca` deploy models of identical size, and train models of
    identical size up to the single ``(d, k)`` basis."""
    rf, _ = _restricted_free()
    folded = rf.fold()
    trainable = sum(p.numel() for p in rf.D.values()) + rf.D_emb.numel() + rf.D_lm.numel()
    deployed = sum(p.numel() for p in folded.parameters())
    assert trainable == deployed, (trainable, deployed)
    assert rf.V.numel() == rf.d * rf.k


def test_param_groups_separate_the_manifold_from_the_rest():
    rf, _ = _restricted_free()
    stiefel, euclid = rf.param_groups()
    assert len(stiefel) == 1 and stiefel[0] is rf.V
    assert not any(p is rf.V for p in euclid)
    assert sum(p.numel() for p in euclid) == sum(p.numel() for p in rf.fold().parameters())


def test_load_student_residual_reproduces_the_student():
    """After `load_student_residual`, the restricted module must equal the student exactly.

    This is the invariant the amortized loop depends on: the V-step starts from the student
    the cheap loop has trained, so it refines the projection around the current weights
    rather than a stale checkpoint. If the round-trip were lossy, every refresh would inject
    an error the Euclidean loop then has to undo.
    """
    rf, calib = _restricted_free()
    st = rf.fold()   # a plain student to stand in for the cheap loop's model
    torch.manual_seed(4)
    with torch.no_grad():
        for p in st.parameters():
            p.add_(0.05 * torch.randn_like(p))
    # a *different* V than the student was folded from, to exercise the residual solve
    Vnew = qr_retract(rf.V + 0.2 * torch.randn_like(rf.V))
    rf.load_student_residual(st, Vnew)
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a = rf(ids).logits
        b = st.eval()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_write_back_preserves_parameter_objects():
    """`write_back` must move ``.data`` in place, so an optimizer's state stays attached."""
    rf, _ = _restricted_free()
    st = rf.fold()
    ids_before = {id(p) for p in st.parameters()}
    with torch.no_grad():
        rf.V.copy_(qr_retract(rf.V + 0.1 * torch.randn_like(rf.V)))
    rf.write_back(st)
    assert {id(p) for p in st.parameters()} == ids_before
    # and the write actually took: st now equals the moved module
    import torch as _t
    with _t.no_grad():
        x = _t.randint(0, st.config.vocab_size, (1, 8))
        assert _t.allclose(rf(x).logits, st.eval()(input_ids=x).logits, atol=1e-4)


def test_bf16_teacher_matches_fp32_restriction():
    """A bf16-loaded teacher must give ~the same restricted logits as the fp32 one.

    This is what lets the same code run a 7B teacher on a 22 GB card: the teacher's bulk is
    stored in half precision, but every restriction matmul up-casts its small weight slice, so
    the result tracks the fp32 path to within bf16 rounding. A regression that dropped the
    up-cast would silently degrade (or crash on a dtype mismatch) only on large real models.
    """
    t32 = gamma_fold_llama(_tiny_llama()).eval()
    calib = _calib()
    S = llama_residual_second_moment(t32, calib, device="cpu")
    _, evecs = torch.linalg.eigh(S.double())
    V0 = evecs[:, -16:].flip(-1).float()
    idx = ffn_energy_indices(t32, calib, 32, device="cpu")
    rm32 = RestrictedLlama(t32, S, V0, idx, 2, 2)

    t16 = _tiny_llama().to(torch.bfloat16)
    t16 = gamma_fold_llama(t16).eval()
    rm16 = RestrictedLlama(t16, S, V0, idx, 2, 2)

    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a = rm32(ids).logits
        b = rm16(ids).logits
    # bf16 has ~3 significant digits; a relaxed tolerance on a tiny net is expected.
    assert (a - b).abs().max() < 0.5, (a - b).abs().max()
    assert rm16(ids).logits.dtype == torch.float32   # V trains in fp32 regardless


def test_qr_retract_returns_to_the_manifold_and_fixes_signs():
    torch.manual_seed(0)
    A = torch.randn(8, 3)
    Q = qr_retract(A)
    assert torch.allclose(Q.T @ Q, torch.eye(3), atol=1e-5)
    # sign convention: R's diagonal is positive, so Q's columns point "with" A's
    assert (torch.linalg.qr(A)[1].diagonal().sign()
            * (Q * torch.linalg.qr(A)[0]).sum(0).sign() > 0).all()
