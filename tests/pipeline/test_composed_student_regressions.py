"""Regressions in the composed absorbed-init student forward.

The per-linear absorbed_init math is correct (existing tests cover it), but
the *composed* student forward was once catastrophically broken — absorbed-init
students started at PPL 10^10 to 10^15, because:

- Q/K used independent per-branch bases, destroying the Q·K^T dot product.
- Principal-component matrices were zero-padded along columns when behavioral
  rank was smaller than student hidden_size, producing rank-deficient
  absorbed weights.

Separately, the Procrustes loss was unnormalized by k, so a schedule transition
from CKA (O(1)) to Procrustes (O(k)) spiked loss magnitude by 100–1000× and
drove training to NaN.
"""

from __future__ import annotations

import pytest
import torch


def _toy_gpt2(n_layer=2, n_embd=32, n_head=4):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,
    )
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def _shift_ce(logits, labels):
    import torch.nn.functional as F

    return F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        labels[..., 1:].reshape(-1),
    )


def test_absorbed_init_composed_student_ppl_is_sane():
    """Student built with absorbed_init should not start at astronomical PPL.

    Without care, the composed forward can start at initial PPL 10^10 to 10^15.
    With the Q/K-shared-basis fix and orthogonal (not zero) column padding, the composed forward is
    well-conditioned and initial PPL stays within a reasonable multiple of the
    teacher's PPL.
    """
    teacher = _toy_gpt2()
    if teacher is None:
        pytest.skip("transformers not installed")
    teacher.eval()

    import substill

    torch.manual_seed(0)
    B, T = 4, 16
    tokens = torch.randint(5, 50, (B, T))
    batch = {
        "input_ids": tokens,
        "labels": tokens,
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }
    loader = [batch for _ in range(4)]

    profile = substill.profile(
        teacher,
        loader,
        mode="branch",
        rank_tol=0.02,
        max_rank=32,
        n_calib_batches=4,
        behavioral_calib_batches=2,
        search="bisect",
    )

    student = substill.build_student(teacher, profile, absorbed_init=True, template="gpt2")
    student.eval()

    with torch.no_grad():
        t_logits = teacher(**batch).logits
        s_logits = student(**batch).logits
        t_ppl = float(torch.exp(_shift_ce(t_logits, tokens)).item())
        s_ppl = float(torch.exp(_shift_ce(s_logits, tokens)).item())

    assert torch.isfinite(s_logits).all(), "absorbed-init student produced non-finite logits"
    # Teacher is random-init, so its PPL on random tokens is ~vocab_size = 64.
    # A well-conditioned absorbed init should be in the same ballpark, not 10^10+.
    assert s_ppl < 100 * t_ppl, (
        f"absorbed-init student PPL {s_ppl:.3e} is >100× teacher PPL {t_ppl:.3e} — "
        "composition is broken (check Q/K basis sharing and orthogonal col padding)"
    )


def test_procrustes_loss_magnitude_comparable_to_cka():
    """Procrustes loss must stay in [0, ~1] so schedule transitions don't spike.

    Training can diverge when the loss schedule hits frac=0.4 (cka→procrustes)
    if the unnormalized procrustes cost is O(k) vs CKA's O(1). After
    normalizing by the total feature energy, procrustes lives in [0, 1] and
    the objective hand-off no longer explodes the gradient scale.
    """
    from substill.losses.procrustes import procrustes_distance
    from substill.losses.subspace import cka_distance

    torch.manual_seed(0)
    N, k = 256, 384  # k large enough that unnormalized procrustes would be O(100s)
    Z_s = torch.randn(N, k)
    Z_t = torch.randn(N, k)

    cka = cka_distance(Z_s, Z_t)
    proc = procrustes_distance(Z_s, Z_t)

    assert 0.0 <= float(cka) <= 1.01
    assert 0.0 <= float(proc) <= 1.01, (
        f"procrustes={float(proc):.3f} should be in [0, 1] after normalization; "
        "unnormalized value scales with k and breaks schedule transitions"
    )
    # Aligned-case sanity: Z_s = Z_t gives near-zero procrustes.
    proc_aligned = procrustes_distance(Z_s, Z_s)
    assert float(proc_aligned) < 0.05


def test_absorbed_init_orthogonal_col_padding():
    """When behavioral_rank < cols_needed, pad with orthonormal (not zero) cols.

    Zero padding produced rank-deficient absorbed weights on branches where the
    behavioral rank came in below the student's hidden/intermediate size.
    """
    from substill.builders import _pad_cols_orthogonal

    torch.manual_seed(0)
    rows, k0, k_target = 48, 10, 32
    V = torch.linalg.qr(torch.randn(rows, k0))[0]  # rows × k0 orthonormal
    V_padded = _pad_cols_orthogonal(V, k_target)

    assert V_padded.shape == (rows, k_target)
    # First k0 columns unchanged.
    assert torch.allclose(V_padded[:, :k0], V, atol=1e-6)
    # All columns orthonormal.
    gram = V_padded.T @ V_padded
    assert torch.allclose(gram, torch.eye(k_target), atol=1e-5)


def test_refresh_from_profile_preserves_folded_shape():
    """refresh_from_profile must not resize a folded branch.

    The on-policy profile refresh can crash: a new PCA
    computed on student rollouts returns a large ``behavioral_rank`` (e.g. 2664) for
    ``ffn.up`` while the student's intermediate dim is 768 and the projector
    has been folded. A blind basis swap would break the next forward. The refresh
    now keeps the folded rank locked and only updates the basis where shape
    allows.
    """
    from substill.losses.subspace import F_ASDLoss

    torch.manual_seed(0)

    class _Branch:
        def __init__(self, name, d, k):
            self.name = name
            self.principal_components = torch.linalg.qr(torch.randn(d, d))[0]
            self.behavioral_rank = k

    profile = [_Branch("a", d=16, k=4)]
    loss_fn = F_ASDLoss(profile, objective="gram")
    # Simulate the fold.
    loss_fn._folded.add("a")

    # Refresh with a LARGER behavioral rank (the crash case).
    big_profile = [_Branch("a", d=16, k=12)]
    loss_fn.refresh_from_profile(big_profile)

    V = loss_fn._get_v("a")
    assert V.shape == (16, 4), (
        f"folded branch's V must stay at (16, 4), got {V.shape} — "
        "refresh would break the folded forward"
    )
    assert loss_fn.branch_ks["a"] == 4  # rank locked

    # Refresh with a SMALLER behavioral rank (don't upsize silently either).
    small_profile = [_Branch("a", d=16, k=2)]
    loss_fn.refresh_from_profile(small_profile)
    V2 = loss_fn._get_v("a")
    assert V2.shape == (16, 4)
    assert loss_fn.branch_ks["a"] == 4


# -- regressions ---------------------------------------------


def test_col_basis_uses_pca_tail_when_under_ranked():
    """``_col_basis`` should slice PCA tail directions, not random-pad.

    Previously when behavioral_rank < cols_needed the code zero-padded
    or randomly padded. With principal_components stored as the full
    (C, C) eigenvector matrix, we can slice the tail directly. PCA-tail is
    deterministic; random padding gives different absorbed weights every run.
    """
    from substill.builders import _col_basis

    torch.manual_seed(0)
    C = 16
    pc_full, _ = torch.linalg.qr(torch.randn(C, C))

    class _B:
        name = "branch"
        principal_components = pc_full
        behavioral_rank = 4

    profile_obj = type("P", (), {"branches": [_B()]})()

    # cols_needed (10) is larger than behavioral_rank (4) but ≤ C (16).
    V = _col_basis(profile_obj, "branch", 10)
    assert V.shape == (C, 10)
    assert torch.allclose(V, pc_full[:, :10], atol=1e-6), \
        "must return deterministic PCA tail slice, not random padding"


def test_resolve_fold_frac_picks_procrustes_start():
    """Default schedule should fold at the Procrustes phase boundary (0.40)."""
    from substill.losses.subspace import default_schedule
    from substill.training.distill import _resolve_fold_frac

    sched = default_schedule()
    assert _resolve_fold_frac(sched, None) == 0.40
    assert _resolve_fold_frac(sched, 0.25) == 0.25
    assert _resolve_fold_frac(None, None) == 0.40


def test_quantized_linear_fp_weight_receives_grad():
    """QAD must update fp_weight (not just bias / protected / LN).

    An earlier approach froze quantized weights as int8 buffers with no STE — qad_finetune was
    powerless against the quantization error. Now fp_weight is a Parameter and
    fake-quant has STE, so backprop flows through.
    """
    from substill.compression.quantization import QuantizedLinear

    torch.manual_seed(0)
    layer = QuantizedLinear(in_features=64, out_features=32, bits=4, group_size=16, bias=True)
    layer.fp_weight.data.copy_(torch.randn(32, 64))

    x = torch.randn(8, 64, requires_grad=False)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert layer.fp_weight.grad is not None, "fp_weight should receive gradient via STE"
    assert torch.isfinite(layer.fp_weight.grad).all()
    assert layer.fp_weight.grad.abs().sum() > 0


def test_residual_basis_prefers_residual_branch():
    """``_residual_basis`` prefers a ``block.residual`` branch over the legacy fallback.

    When a profile includes a true residual-stream PCA branch, the
    absorbed_init should use it directly rather than averaging attn.o + ffn.down.
    """
    from substill.builders import _residual_basis

    torch.manual_seed(0)
    C = 8
    s = 4
    pc_residual, _ = torch.linalg.qr(torch.randn(C, C))
    pc_attno, _ = torch.linalg.qr(torch.randn(C, C))
    pc_ffndown, _ = torch.linalg.qr(torch.randn(C, C))

    class _B:
        def __init__(self, name, kind, pc):
            self.name = name
            self.kind = kind
            self.principal_components = pc

    profile = type("P", (), {"branches": [
        _B("a.attn.o", "attn.o", pc_attno),
        _B("a.ffn.down", "ffn.down", pc_ffndown),
        _B("a.residual", "block.residual", pc_residual),
    ]})()

    V_r = _residual_basis(profile, t_hidden=C, s_hidden=s)
    # When a residual branch exists, V_r must be derived from it, not the average.
    assert torch.allclose(V_r, pc_residual[:, :s], atol=1e-6), \
        "with a block.residual branch present, V_r must come from it"


def test_residual_basis_falls_back_when_no_residual_branch():
    """When no block.residual branch is present, fall back to identity-truncated.

    There is deliberately no attn.o+ffn.down averaged-then-QR fallback (it had no
    theoretical justification and added noise). The fallback is a truncated
    identity: keep the first ``s_hidden`` channels of the teacher residual.
    """
    from substill.builders import _residual_basis

    torch.manual_seed(0)
    C = 8
    s = 4
    pc_attno, _ = torch.linalg.qr(torch.randn(C, C))

    class _B:
        def __init__(self, name, kind, pc):
            self.name = name
            self.kind = kind
            self.principal_components = pc

    profile = type("P", (), {"branches": [_B("a.attn.o", "attn.o", pc_attno)]})()
    V_r = _residual_basis(profile, t_hidden=C, s_hidden=s)
    assert V_r.shape == (C, s)
    # Identity-truncated, so V_r.T @ V_r = I_s.
    assert torch.allclose(V_r.T @ V_r, torch.eye(s), atol=1e-5)
    assert torch.allclose(V_r, torch.eye(C, s), atol=1e-6)


def test_residual_basis_identity_when_no_compression():
    """When ``s_hidden == t_hidden``, ``_residual_basis`` returns the identity.

    Returning a PCA-derived orthogonal rotation in this case broke
    initial PPL by 5–14 orders of magnitude because LayerNorm does
    not commute with arbitrary orthogonal rotations. With no compression
    needed, the rotation buys nothing and is information-destructive.
    """
    from substill.builders import _residual_basis

    torch.manual_seed(0)
    C = 8
    pc_residual, _ = torch.linalg.qr(torch.randn(C, C))

    class _B:
        def __init__(self, name, kind, pc, eigvals=None):
            self.name = name
            self.kind = kind
            self.principal_components = pc
            self.eigenvalues = eigvals

    profile = type("P", (), {"branches": [
        _B("a.residual", "block.residual", pc_residual,
           torch.linspace(1.0, 0.1, C)),
    ]})()
    V_r = _residual_basis(profile, t_hidden=C, s_hidden=C)
    assert torch.allclose(V_r, torch.eye(C), atol=1e-6), \
        "with no compression, V_r must be exactly the identity"


def test_channel_select_basis_picks_top_variance_channels():
    """``_channel_select_basis`` returns one-hot columns at the top channels
    by reconstructed cov.diag (variance per channel).

    PCA-rotated FFN intermediate broke initial PPL because GELU is element-wise
    and does not commute with orthogonal rotations. Channel-selection slices
    teacher channels, so the activation passes through unchanged.
    """
    from substill.builders import _channel_select_basis

    n = 6
    k = 3
    # Construct V (orthogonal) and eigvals so that cov.diag has known top-k.
    # Easy way: use diagonal cov directly, V = identity.
    V = torch.eye(n)
    # Make channels 1, 3, 5 the "top" ones by giving them larger eigenvalues.
    # With V = I, cov.diag[i] = eigvals[i]. So eigvals controls ranking.
    eigvals = torch.tensor([0.1, 1.0, 0.2, 0.9, 0.3, 0.8])

    class _B:
        def __init__(self, name, pc, eigs):
            self.name = name
            self.principal_components = pc
            self.eigenvalues = eigs

    profile = type("P", (), {"branches": [_B("test.ffn.up", V, eigvals)]})()
    E = _channel_select_basis(profile, "test.ffn.up", k)
    assert E.shape == (n, k)
    # Each column is one-hot.
    assert torch.allclose(E.sum(dim=0), torch.ones(k))
    assert torch.all((E == 0) | (E == 1))
    # Selected channels must be {1, 3, 5} (top-3 eigvals).
    selected = set()
    for j in range(k):
        i = int(E[:, j].argmax())
        selected.add(i)
    assert selected == {1, 3, 5}, f"expected top channels {{1,3,5}}, got {selected}"


def test_gpt2_absorb_full_size_reproduces_teacher():
    """With no residual or FFN compression, absorbed init reproduces the
    teacher exactly (modulo floating point).

    If, even with k=d=t_hidden, V_r were a non-trivial
    orthogonal rotation from PCA, it would break LayerNorm and per-head attention.
    Instead the code short-circuits to V_r = I, V_up = I in that case.
    """
    # Use n_embd >= profile_to_student_config's min_hidden=64 default so the
    # student isn't bumped up; if it's bumped, absorbed init has to project
    # into a different shape and reproduction is no longer expected.
    teacher = _toy_gpt2(n_layer=2, n_embd=64, n_head=4)
    if teacher is None:
        pytest.skip("transformers not installed")

    from substill.api import BranchProfile, TeacherProfile
    branches = []
    eye64 = torch.eye(64)
    eye256 = torch.eye(256)  # n_inner = 4 * 64 = 256
    for i in range(2):
        prefix = f"transformer.h.{i}"
        branches.append(BranchProfile(
            name=f"{prefix}.residual", kind="block.residual",
            module_path=prefix, principal_components=eye64,
            eigenvalues=torch.linspace(1.0, 0.5, 64),
            behavioral_rank=64, variance_rank=64, channels=64,
        ))
        branches.append(BranchProfile(
            name=f"{prefix}.ffn.up", kind="ffn.up",
            module_path=f"{prefix}.mlp.c_fc", principal_components=eye256,
            eigenvalues=torch.linspace(1.0, 0.5, 256),
            behavioral_rank=256, variance_rank=256, channels=256,
        ))
    profile = TeacherProfile(branches=branches)

    import substill
    student = substill.build_student(teacher, profile, absorbed_init=True, template="gpt2")
    assert int(student.config.n_embd) == 64, "student must keep teacher's hidden size"
    assert int(student.config.n_inner) == 256

    teacher.eval()
    student.eval()
    torch.manual_seed(0)
    ids = torch.randint(1, 64, (2, 16))
    am = torch.ones_like(ids)
    with torch.no_grad():
        t_out = teacher(input_ids=ids, attention_mask=am)
        s_out = student(input_ids=ids, attention_mask=am)
    diff = (t_out.logits - s_out.logits).abs().max().item()
    assert diff < 1e-4, \
        f"absorbed init at full size must reproduce teacher; max logit diff = {diff}"
