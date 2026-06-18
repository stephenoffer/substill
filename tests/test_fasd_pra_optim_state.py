"""PRA optimiser-state regression: embedding path must match linear-path policy.

Pre-fix the embedding branch in reabsorb_gpt2 rotated `exp_avg`, zeroed
`exp_avg_sq`, and never reset `state["step"]`. With v=0 but step large, Adam's
bias correction produced m/eps updates of ~10^8x effective LR, which is the
failure mode r2_pra200 hit in the v11-pra-apr30 run.
"""

from __future__ import annotations

import pytest
import torch

from fasd.compression.absorbed_init import (
    _infer_layout,
    absorbed_bias,
    absorbed_linear_init,
    absorbed_weight,
)


def _toy_gpt2(n_layer=2, n_embd=16, n_head=2, vocab=40, n_pos=16):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(
        vocab_size=vocab,
        n_positions=n_pos,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,
    )
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def _populate_optim_state(student, optimizer, loader, n_steps=5):
    """Run a few real backward+step iterations so Adam state has nonzero v
    and step >= 1 for every trainable param."""
    student.train()
    it = iter(loader)
    for _ in range(n_steps):
        batch = next(it)
        out = student(**batch, labels=batch["input_ids"])
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()


def _make_loader(vocab=40, T=8, B=2, n=8):
    tokens = torch.randint(5, vocab - 1, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    return [{"input_ids": tokens, "attention_mask": attn} for _ in range(n)]


def test_embedding_pra_resets_step_and_zeros_state():
    teacher = _toy_gpt2()
    if teacher is None:
        pytest.skip("transformers not installed")

    import fasd
    from fasd.training.reabsorb import reabsorb_gpt2

    torch.manual_seed(0)
    loader = _make_loader()

    profile = fasd.profile(teacher, loader, n_calib_batches=4, behavioral_calib_batches=4)
    student = fasd.build_student(teacher, profile, absorbed_init=True, template="gpt2")

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
    _populate_optim_state(student, optimizer, loader, n_steps=5)

    wte = student.transformer.wte.weight
    wpe = student.transformer.wpe.weight

    # Sanity: state populated with finite v and step >= 1.
    pre_state_wte = optimizer.state[wte]
    assert pre_state_wte["exp_avg_sq"].abs().sum() > 0
    pre_step = pre_state_wte["step"]
    pre_step_val = int(pre_step.item()) if torch.is_tensor(pre_step) else int(pre_step)
    assert pre_step_val >= 1

    # Re-absorb: should zero m, zero v, AND reset step for both embeddings.
    new_profile = reabsorb_gpt2(teacher, student, profile, loader[:2], optimizer=optimizer)
    assert new_profile is not None

    for p, name in [(wte, "wte"), (wpe, "wpe")]:
        st = optimizer.state[p]
        assert torch.equal(st["exp_avg"], torch.zeros_like(st["exp_avg"])), (
            f"{name}: exp_avg should be zeroed (matches linear-path policy)"
        )
        assert torch.equal(st["exp_avg_sq"], torch.zeros_like(st["exp_avg_sq"])), (
            f"{name}: exp_avg_sq should be zeroed"
        )
        step = st["step"]
        step_val = int(step.item()) if torch.is_tensor(step) else int(step)
        assert step_val == 0, f"{name}: step should be reset to 0, got {step_val}"


def test_embedding_pra_first_post_reabsorb_step_does_not_explode():
    """End-to-end: a real optimizer.step after reabsorb produces a small Δ on
    the embeddings, not a 10^8x blowup. Pre-fix this would be ~lr/eps≈1e3+."""
    teacher = _toy_gpt2()
    if teacher is None:
        pytest.skip("transformers not installed")

    import fasd
    from fasd.training.reabsorb import reabsorb_gpt2

    torch.manual_seed(0)
    loader = _make_loader()

    profile = fasd.profile(teacher, loader, n_calib_batches=4, behavioral_calib_batches=4)
    student = fasd.build_student(teacher, profile, absorbed_init=True, template="gpt2")

    lr = 5e-4
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    _populate_optim_state(student, optimizer, loader, n_steps=5)

    wte = student.transformer.wte.weight
    wpe = student.transformer.wpe.weight
    wte_before = wte.detach().clone()
    wpe_before = wpe.detach().clone()

    reabsorb_gpt2(teacher, student, profile, loader[:2], optimizer=optimizer)

    # One real backward+step. Under the fixed policy, post-reabsorb embeddings
    # behave like fresh AdamW state — first update is ~lr scale, not lr/eps.
    batch = loader[0]
    out = student(**batch, labels=batch["input_ids"])
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    delta_wte = (wte.detach() - wte_before).norm().item()
    delta_wpe = (wpe.detach() - wpe_before).norm().item()

    # Normalize by sqrt(numel) * lr to make the bound dimension-independent.
    bound_wte = (wte.numel() ** 0.5) * lr * 5.0
    bound_wpe = (wpe.numel() ** 0.5) * lr * 5.0
    assert delta_wte < bound_wte, (
        f"wte delta {delta_wte:.3e} exceeded sane post-PRA bound {bound_wte:.3e} "
        f"— optimiser state likely re-introduced the rotate-m + zero-v + no-step-reset bug"
    )
    assert delta_wpe < bound_wpe, (
        f"wpe delta {delta_wpe:.3e} exceeded sane post-PRA bound {bound_wpe:.3e}"
    )


def test_absorbed_init_bias_branch_collapse():
    """Regression for the dead conditional in absorbed_init.py:134 — both
    branches were identical, so collapsing them must not change behaviour."""
    torch.manual_seed(0)

    teacher = torch.nn.Linear(8, 8)
    student = torch.nn.Linear(4, 4)

    V_in = torch.linalg.qr(torch.randn(8, 4))[0]
    V_out = torch.linalg.qr(torch.randn(8, 4))[0]

    absorbed_linear_init(teacher, student, V_in=V_in, V_out=V_out)

    expected_W = absorbed_weight(
        teacher.weight.detach(),
        V_in.to(teacher.weight),
        V_out.to(teacher.weight),
        layout=_infer_layout(teacher),
    )
    expected_b = absorbed_bias(teacher.bias.detach(), V_out.to(teacher.bias))

    assert torch.allclose(student.weight.data, expected_W, atol=1e-6)
    assert torch.allclose(student.bias.data, expected_b, atol=1e-6)
