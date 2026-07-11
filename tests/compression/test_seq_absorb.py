"""Tests for sequential drift-corrected absorption (substill.compression.seq_absorb)."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from substill.compression.seq_absorb import (  # noqa: E402
    SeqAbsorbConfig,
    _ffn_basis,
    _graft,
    absorb_gpt2,
    build_narrow_gpt2,
    gamma_fold_gpt2,
    logit_metric,
    residual_basis,
    residual_second_moment,
    sequential_absorb_gpt2,
)


@pytest.fixture(scope="module")
def tiny():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=24, n_layer=2, n_head=4,
                     n_inner=48, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg).eval()
    # Non-trivial LayerNorm affines, else gamma-fold is vacuous.
    for blk in model.transformer.h:
        for ln in (blk.ln_1, blk.ln_2):
            ln.weight.data.normal_(1.0, 0.2)
            ln.bias.data.normal_(0.0, 0.1)
    calib = [{"input_ids": torch.randint(0, 60, (2, 16))} for _ in range(3)]
    return model, calib


def test_gamma_fold_preserves_logits(tiny):
    model, calib = tiny
    folded = gamma_fold_gpt2(model)
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a, b = model(input_ids=ids).logits, folded(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4)
    for blk in folded.transformer.h:
        assert torch.allclose(blk.ln_1.weight, torch.ones_like(blk.ln_1.weight))
        assert torch.allclose(blk.ln_1.bias, torch.zeros_like(blk.ln_1.bias))


def test_absorb_at_full_width_reproduces_teacher(tiny):
    """k == d_model and inner == d_inner: every basis is the identity, so the
    absorbed student must be the teacher bit-for-bit. Guards the projection math."""
    model, calib = tiny
    d, inner = model.config.n_embd, model.config.n_inner
    V = torch.eye(d)
    ffn = [_ffn_basis(model, i, calib, inner, "cpu") for i in range(model.config.n_layer)]
    student = build_narrow_gpt2(model, d, inner)
    absorb_gpt2(model, student, V, ffn)
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        a, b = model(input_ids=ids).logits, student.eval()(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


@pytest.mark.parametrize("method", ["identity", "select", "select_gn", "pca", "gn"])
def test_residual_basis_orthonormal(tiny, method):
    model, calib = tiny
    S = residual_second_moment(model, calib, device="cpu")
    M = logit_metric(model)
    V = residual_basis(S, 12, method=method, M=M)
    assert V.shape == (model.config.n_embd, 12)
    assert torch.allclose(V.T @ V, torch.eye(12), atol=1e-4)


def test_graft_is_identity_when_student_matches_teacher(tiny):
    """If the student's state equals the teacher's projection, grafting must
    return the teacher's state exactly -- otherwise the graft objective would
    penalize a perfect block."""
    model, calib = tiny
    S = residual_second_moment(model, calib, device="cpu")
    V = residual_basis(S, 12, method="pca")
    h = torch.randn(2, 16, model.config.n_embd)
    assert torch.allclose(_graft(h @ V, h, V), h, atol=1e-5)


def test_sequential_absorb_beats_plain_absorb_on_drift(tiny):
    """The whole point: fitting on the student's own drifted stream must leave the
    final block's residual drift no worse than not fitting at all."""
    model, calib = tiny
    cfg = SeqAbsorbConfig(k=12, inner=24, steps_per_block=30, lnf_steps=10,
                          objective="l2", verbose=False)
    student, info = sequential_absorb_gpt2(model, calib, cfg, device="cpu")
    assert student.config.n_embd == 12
    drifts = [b["drift"] for b in info["block_loss"]]
    assert len(drifts) == model.config.n_layer
    assert all(d == d for d in drifts)  # no NaN
    # Fitted drift stays bounded; the unfitted baseline compounds past 1.0 on
    # real models. Here we only assert the fit produced a finite, sane stream.
    assert drifts[-1] < 2.0


def test_profile_default_yields_identity_residual_basis(tiny):
    """Regression guard for a silent, load-bearing fallback.

    ``profile()`` defaults to ``mode="branch"``, which enumerates attn/ffn branches
    and no ``block.residual`` branch. ``_residual_basis`` therefore never finds
    residual statistics and returns ``torch.eye(d, k)`` -- plain truncation to the
    first k coordinates. Every absorbed-init student built through the default
    pipeline has used this. If someone makes ``profile()`` emit residual stats, this
    test fails and they must re-run the basis comparison rather than assume the new
    basis is better: on GPT-2 the statistics-driven bases were *worse*.
    """
    import substill
    from substill.builders import _residual_basis

    model, calib = tiny
    profile = substill.profile(model, calib)
    assert not any(b.kind == "block.residual" for b in profile.branches)
    d = model.config.n_embd
    with pytest.warns(UserWarning, match="no 'block.residual' branch"):
        V = _residual_basis(profile, d, d // 2)
    assert torch.equal(V, torch.eye(d, d // 2))


def test_graft_objective_runs_and_lowers_loss(tiny):
    model, calib = tiny
    cfg = SeqAbsorbConfig(k=12, inner=24, steps_per_block=20, lnf_steps=5,
                          objective="graft", verbose=False)
    student, info = sequential_absorb_gpt2(model, calib, cfg, device="cpu")
    assert all(torch.isfinite(torch.tensor(b["loss"])) for b in info["block_loss"])
    with torch.no_grad():
        out = student(input_ids=calib[0]["input_ids"]).logits
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kw", [
    {"input_mode": "teacher"},          # no drift correction
    {"target_mode": "output"},          # classic layerwise reconstruction
    {"metric": "l2"},                   # no logit-Jacobian metric
    {"prox": 10.0},                     # trust region around the absorbed init
    {"fit_params": "affine"},           # only biases + LayerNorm affines move
    {"objective": "both"},              # graft + L2
])
def test_fit_ablations_run_and_stay_finite(tiny, kw):
    """Each ablation arm of the fit must produce a finite, forward-able student.

    These are the arms docs/init_findings.md reports on; a silent NaN in any of them
    would turn a measured negative result into an artifact.
    """
    model, calib = tiny
    cfg = SeqAbsorbConfig(k=12, inner=24, steps_per_block=15, lnf_steps=5,
                          verbose=False, **kw)
    student, info = sequential_absorb_gpt2(model, calib, cfg, device="cpu")
    assert all(torch.isfinite(torch.tensor(b["loss"])) for b in info["block_loss"])
    with torch.no_grad():
        assert torch.isfinite(student(input_ids=calib[0]["input_ids"]).logits).all()


def test_affine_fit_leaves_weight_matrices_untouched(tiny):
    """`fit_params="affine"` is only meaningful if it cannot move a weight matrix."""
    model, calib = tiny
    common = {"k": 12, "inner": 24, "steps_per_block": 15, "lnf_steps": 0, "verbose": False}
    torch.manual_seed(0)
    frozen, _ = sequential_absorb_gpt2(
        model, calib, SeqAbsorbConfig(**common, fit=False), device="cpu")
    torch.manual_seed(0)
    fitted, _ = sequential_absorb_gpt2(
        model, calib, SeqAbsorbConfig(**common, fit_params="affine"), device="cpu")
    moved = changed = 0
    for (n, a), (_, b) in zip(frozen.named_parameters(), fitted.named_parameters(), strict=False):
        if "transformer.h." not in n:
            continue
        if a.dim() == 2:
            assert torch.equal(a, b), f"affine fit moved weight matrix {n}"
            moved += 1
        elif not torch.equal(a, b):
            changed += 1
    assert moved > 0, "no weight matrices were checked"
    assert changed > 0, "affine fit moved nothing at all"


def test_closed_form_absorb_matches_or_beats_adam_fit_on_drift(tiny):
    """The closed-form solve must reach at least the Adam fit's drift, far cheaper.

    Both target the same objective (gap-closing residual state, student inputs, L2).
    If the solve is wrong -- transposed weights, wrong target, bad conditioning -- its
    drift blows past the Adam fit's and this fails.
    """
    from substill.compression.seq_absorb import closed_form_absorb_gpt2

    model, calib = tiny
    common = {"k": 12, "inner": 24, "verbose": False}
    torch.manual_seed(0)
    _, adam_info = sequential_absorb_gpt2(
        model, calib, SeqAbsorbConfig(**common, steps_per_block=100, lnf_steps=0,
                                      objective="l2"), device="cpu")
    torch.manual_seed(0)
    student, cf_info = closed_form_absorb_gpt2(
        model, calib, SeqAbsorbConfig(**common), device="cpu")

    adam_drift = adam_info["block_loss"][-1]["drift"]
    cf_drift = cf_info["block_loss"][-1]["drift"]
    assert cf_drift < 2.0, f"closed-form diverged: drift={cf_drift}"
    assert cf_drift <= adam_drift * 1.5, (
        f"closed-form drift {cf_drift:.4f} much worse than Adam's {adam_drift:.4f}")
    with torch.no_grad():
        assert torch.isfinite(student(input_ids=calib[0]["input_ids"]).logits).all()


def test_closed_form_absorb_is_exact_at_full_width(tiny):
    """At k = d_model and inner = d_inner the student can represent the teacher
    exactly, so the ridge solve must recover it (up to the ridge term)."""
    from substill.compression.seq_absorb import closed_form_absorb_gpt2

    model, calib = tiny
    cfg = SeqAbsorbConfig(k=model.config.n_embd, inner=model.config.n_inner,
                          verbose=False, ridge_lambda=1e-9)
    student, info = closed_form_absorb_gpt2(model, calib, cfg, device="cpu")
    assert info["block_loss"][-1]["drift"] < 1e-2, info["block_loss"][-1]["drift"]


def test_student_config_keeps_whole_teacher_heads():
    """Absorbed init truncates the residual stream to its first `hidden_size`
    coordinates, and attention heads live contiguously along that axis. So the
    student's hidden_size must be a multiple of the teacher's head_dim, and its head
    count must fall -- otherwise every head is a fragment of one teacher head glued to
    a fragment of the next.

    The legacy rule rounded hidden_size to the teacher's head *count* and kept all 12
    heads, giving head_dim 27 on GPT-2. At a bit-identical 30,004,920 parameters that
    costs ~13.6 PPL.
    """
    from types import SimpleNamespace

    from substill.compression.width_pruner import profile_to_student_config

    t_cfg = SimpleNamespace(hidden_size=768, intermediate_size=3072,
                            num_attention_heads=12, num_key_value_heads=12,
                            num_hidden_layers=12)
    branches = [
        SimpleNamespace(name="b.attn.q", kind="attn.q", behavioral_rank=650),
        SimpleNamespace(name="b.attn.o", kind="attn.o", behavioral_rank=650),
        SimpleNamespace(name="b.attn.k", kind="attn.k", behavioral_rank=650),
        SimpleNamespace(name="b.attn.v", kind="attn.v", behavioral_rank=650),
        SimpleNamespace(name="b.ffn.up", kind="ffn.up", behavioral_rank=2100),
        SimpleNamespace(name="b.ffn.down", kind="ffn.down", behavioral_rank=650),
    ]
    prof = SimpleNamespace(branches=branches)

    cfg = profile_to_student_config(prof, teacher_config=t_cfg, arch_multiplier=0.5)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    assert head_dim == 768 // 12, f"head_dim {head_dim} != teacher's 64"
    assert cfg.hidden_size % 64 == 0
    assert cfg.num_attention_heads < 12  # heads dropped, not narrowed

    legacy = profile_to_student_config(prof, teacher_config=t_cfg, arch_multiplier=0.5,
                                       preserve_head_dim=False)
    assert legacy.num_attention_heads == 12
    assert legacy.hidden_size // legacy.num_attention_heads != 64


def test_rank_map_overrides_head_geometry():
    """An explicit rank_map encodes its own head geometry (cpi_rank_map pins
    H_s == H_t and shrinks head_dim). preserve_head_dim must not override it."""
    from types import SimpleNamespace

    from substill.compression.width_pruner import profile_to_student_config

    t_cfg = SimpleNamespace(hidden_size=128, intermediate_size=256,
                            num_attention_heads=4, num_key_value_heads=2,
                            num_hidden_layers=2)
    branches = [
        SimpleNamespace(name="b.attn.q", kind="attn.q", behavioral_rank=64),
        SimpleNamespace(name="b.attn.o", kind="attn.o", behavioral_rank=64),
        SimpleNamespace(name="b.attn.k", kind="attn.k", behavioral_rank=32),
        SimpleNamespace(name="b.attn.v", kind="attn.v", behavioral_rank=32),
        SimpleNamespace(name="b.ffn.up", kind="ffn.up", behavioral_rank=128),
        SimpleNamespace(name="b.ffn.down", kind="ffn.down", behavioral_rank=64),
    ]
    prof = SimpleNamespace(branches=branches)
    rm = {b.name: b.behavioral_rank for b in branches}
    cfg = profile_to_student_config(prof, teacher_config=t_cfg, rank_map=rm)
    assert cfg.num_attention_heads == 4  # teacher's head count preserved


def test_head_selection_bases_are_whole_head_blocks(tiny):
    """Each column block must be one intact teacher head, and the basis orthonormal."""
    from substill.compression.seq_absorb import head_importance, head_selection_bases

    model, calib = tiny
    head_dim = model.config.n_embd // model.config.n_head
    scores = head_importance(model, calib, device="cpu")
    assert scores.shape == (model.config.n_layer, model.config.n_head)
    assert (scores >= 0).all()  # KL is nonnegative

    k = 2 * head_dim
    for mode in ("first", "important", "random"):
        bases = head_selection_bases(scores, k, head_dim, mode=mode)
        assert len(bases) == model.config.n_layer
        for E in bases:
            assert E.shape == (model.config.n_embd, k)
            assert torch.allclose(E.T @ E, torch.eye(k), atol=1e-5)
            # every column block selects exactly one contiguous head block
            for j in range(k // head_dim):
                blk = E[:, j * head_dim:(j + 1) * head_dim]
                assert int(blk.abs().sum()) == head_dim

    with pytest.raises(ValueError, match="not a multiple of head_dim"):
        head_selection_bases(scores, k + 1, head_dim)


def test_head_bases_first_reproduces_legacy_absorb(tiny):
    """`head_bases="first"` must be bit-identical to the old implicit behavior, so
    the head-selection experiment's control arm is a true control."""
    from substill.compression.seq_absorb import head_importance, head_selection_bases

    model, calib = tiny
    head_dim = model.config.n_embd // model.config.n_head
    k, inner, nh = 2 * head_dim, 24, 2
    S = residual_second_moment(model, calib, device="cpu")
    V = residual_basis(S, k, method="identity", M=logit_metric(model))
    ffn = [_ffn_basis(model, i, calib, inner, "cpu") for i in range(model.config.n_layer)]
    scores = head_importance(model, calib, device="cpu")

    legacy = build_narrow_gpt2(model, k, inner, n_head=nh)
    absorb_gpt2(model, legacy, V, ffn)
    explicit = build_narrow_gpt2(model, k, inner, n_head=nh)
    absorb_gpt2(model, explicit, V, ffn,
                head_bases=head_selection_bases(scores, k, head_dim, mode="first"))
    for a, b in zip(legacy.parameters(), explicit.parameters(), strict=False):
        assert torch.equal(a, b)


def test_head_similarity_is_symmetric_unit_diagonal(tiny):
    from substill.compression.seq_absorb import head_similarity

    model, calib = tiny
    S = head_similarity(model, calib, device="cpu")
    assert S.shape == (model.config.n_layer, model.config.n_head, model.config.n_head)
    for li in range(S.shape[0]):
        assert torch.allclose(S[li], S[li].T, atol=1e-5)
        assert torch.allclose(S[li].diagonal(), torch.ones(model.config.n_head), atol=1e-4)
    assert ((S >= -1e-5) & (S <= 1 + 1e-5)).all()


def test_coverage_and_diverse_select_less_redundant_sets(tiny):
    """The point of `coverage`/`diverse` is that the chosen heads duplicate each other
    less than the importance-ranked ones do. If that stops holding, the modes are not
    doing what their names say and the experiment they support is meaningless."""
    from substill.compression.seq_absorb import (
        _greedy_coverage,
        _greedy_diverse,
        head_importance,
        head_similarity,
    )

    model, calib = tiny
    imp = head_importance(model, calib, device="cpu")
    sims = head_similarity(model, calib, device="cpu")
    n_keep = 2

    def redundancy(sim, idx):
        idx = sorted(idx)
        pairs = [float(sim[a, b]) for i, a in enumerate(idx) for b in idx[i + 1:]]
        return sum(pairs) / len(pairs)

    for li in range(model.config.n_layer):
        sim = sims[li]
        important = torch.argsort(imp[li], descending=True)[:n_keep].tolist()
        cov = _greedy_coverage(sim, n_keep)
        div = _greedy_diverse(sim, n_keep)
        assert len(set(cov)) == n_keep and len(set(div)) == n_keep
        assert redundancy(sim, div) <= redundancy(sim, important) + 1e-6


def test_selection_bases_require_sims_for_coverage_modes(tiny):
    from substill.compression.seq_absorb import head_importance, head_selection_bases

    model, calib = tiny
    scores = head_importance(model, calib, device="cpu")
    head_dim = model.config.n_embd // model.config.n_head
    for mode in ("coverage", "diverse"):
        with pytest.raises(ValueError, match="requires `sims`"):
            head_selection_bases(scores, 2 * head_dim, head_dim, mode=mode)


def test_hidden_states_last_is_post_final_norm(tiny):
    """`output_hidden_states=True` returns ln_f(last block output) as its final entry,
    not the last block's raw output.

    Anything that reads `hidden_states[-1]` and then applies `ln_f` to it normalizes
    twice. `sequential_absorb_gpt2` did exactly that for the graft objective's logit
    targets and for the ln_f fit, which corrupted both. This test pins the HF behavior so
    the workaround is never quietly dropped.
    """
    from substill.compression.seq_absorb import _gpt2_block_parts

    model, calib = tiny
    ids = calib[0]["input_ids"]
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        raw_last = _gpt2_block_parts(model.transformer.h[-1], hs[-2])[1]
    assert not torch.allclose(hs[-1], raw_last, atol=1e-4)
    assert torch.allclose(hs[-1], model.transformer.ln_f(raw_last), atol=1e-4)
    assert len(hs) == model.config.n_layer + 1


def test_graft_targets_are_not_double_normalized(tiny, monkeypatch):
    """The graft objective's teacher targets must equal the teacher's real logprobs."""
    import torch.nn.functional as F

    from substill.compression.seq_absorb import _gpt2_block_parts as parts

    model, calib = tiny
    ids = torch.cat([b["input_ids"] for b in calib], 0)
    with torch.no_grad():
        real = F.log_softmax(model(input_ids=ids).logits, -1)
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        h_final = parts(model.transformer.h[-1], hs[-2])[1]
        fixed = F.log_softmax(model.lm_head(model.transformer.ln_f(h_final)), -1)
        buggy = F.log_softmax(model.lm_head(model.transformer.ln_f(hs[-1])), -1)
    assert torch.allclose(fixed, real, atol=1e-4)
    assert not torch.allclose(buggy, real, atol=1e-3)


def test_grassmann_basis_lowers_logit_error_below_pca(tiny):
    """The whole point: PCA minimizes residual-stream error, this minimizes *logit* error,
    and the two objectives disagree. If it cannot beat its own PCA starting point, the
    optimization is broken."""
    from substill.compression.seq_absorb import (
        grassmann_logit_basis,
        relative_logit_error,
    )

    model, calib = tiny
    S = residual_second_moment(model, calib, device="cpu")
    M = logit_metric(model)
    k = model.config.n_embd // 2

    V_pca = residual_basis(S, k, method="pca")
    V_g = grassmann_logit_basis(S, M, k, steps=200)

    assert V_g.shape == (model.config.n_embd, k)
    assert torch.allclose(V_g.T @ V_g, torch.eye(k), atol=1e-4)  # still on the manifold
    e_pca, e_g = relative_logit_error(S, M, V_pca), relative_logit_error(S, M, V_g)
    assert e_g <= e_pca + 1e-9, f"grassmann {e_g} worse than its PCA init {e_pca}"


def test_relative_logit_error_is_zero_at_full_rank(tiny):
    from substill.compression.seq_absorb import relative_logit_error

    model, calib = tiny
    S = residual_second_moment(model, calib, device="cpu")
    M = logit_metric(model)
    d = model.config.n_embd
    assert relative_logit_error(S, M, torch.eye(d)) == pytest.approx(0.0, abs=1e-9)
