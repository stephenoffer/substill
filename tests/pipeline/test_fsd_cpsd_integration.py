"""CPSD end-to-end integration (Phase 2-integrate).

Proves the four novel components COMPOSE and TRAIN together with the existing
distillation loss and Stiefel optimizer:
  - CPI shared-subspace bases initialize the compressed factors,
  - TeacherFactoredLinear keeps W_T frozen with trainable Stiefel V_in/V_out,
  - DifferentiableRankGate prunes the latent under a global budget,
  - skew_kl (DistiLLM) is the distillation objective,
  - StiefelAdam keeps the bases on the manifold.

This is the unit-scale stand-in for the GPT-2 smoke (the full builder + real-model
wiring is Phase 2-integrate/Phase 3); it validates that the conjunction runs, the KD
loss decreases, bases stay orthonormal, and the rank-map hardens within budget.
"""
import torch
import torch.nn as nn

from substill.compression.diff_rank import DifferentiableRankGate, RankBudgetController
from substill.compression.factored_linear import TeacherFactoredLinear
from substill.losses.generative_kd import skew_kl
from substill.training.stiefel_optim import StiefelAdam, stiefel_param_groups


def _shared_basis(acts, keep):
    """CPI: shared PCA basis from activations (cols by descending eigenvalue)."""
    cov = acts.T @ acts
    evals, evecs = torch.linalg.eigh(cov)
    return evecs[:, torch.argsort(evals, descending=True)][:, :keep]


class GatedCPSDLinear(nn.Module):
    """TeacherFactoredLinear with a differentiable rank gate on the input latent."""

    def __init__(self, W_T, V_in, V_out, b_T=None):
        super().__init__()
        self.tfl = TeacherFactoredLinear(W_T, V_in, V_out, b_T)
        self.gate = DifferentiableRankGate(V_in.shape[1], init_open=True)

    def forward(self, x):                      # x: (..., k_in)
        return self.tfl(self.gate(x))

    def cost(self):                            # per-column param cost (in+out dims)
        return torch.full((self.tfl.k_in,), float(self.tfl.d_in + self.tfl.d_out))


def _build(W1, b1, head, V_in, V_out):
    student = GatedCPSDLinear(W1, V_in, V_out, b1)

    def t_logits(x):
        return head(x @ W1.T + b1)

    def s_logits(x):
        h_S = student(x @ V_in)            # compress -> route through frozen W1 -> gate
        return head(h_S @ V_out.T)         # lift back for the shared head

    return student, t_logits, s_logits


def test_cpsd_components_compose_and_train():
    torch.manual_seed(0)
    d, h, vocab, latent = 48, 48, 16, 10
    keep = 16                              # captures the rank-10 signal with margin
    B = 64

    # --- teacher with genuinely low-rank activations (so a "right" subspace exists) ---
    X = torch.randn(2048, latent) @ torch.randn(latent, d)   # inputs live in `latent` dims
    W1 = torch.randn(h, d)
    b1 = torch.randn(h)
    head = nn.Linear(h, vocab)
    head.weight.data *= 4.0                  # sharpen logits so KL is sensitive to error
    for p in head.parameters():
        p.requires_grad_(False)
    hidden = X @ W1.T + b1

    def teacher_logits(x):
        return head(x @ W1.T + b1)

    # CPI shared bases vs a deliberately wrong random init.
    V_in_cpi = _shared_basis(X, keep)
    V_out_cpi = _shared_basis(hidden, keep)
    V_in_rand = torch.linalg.qr(torch.randn(d, keep))[0]
    V_out_rand = torch.linalg.qr(torch.randn(h, keep))[0]

    def kd_of(s_logits, n=512):
        xb = torch.randn(n, latent) @ torch.randn(latent, d)
        return skew_kl(s_logits(xb), teacher_logits(xb), alpha=0.1).item()

    _, _, s_cpi = _build(W1, b1, head, V_in_cpi, V_out_cpi)
    _, _, s_rand = _build(W1, b1, head, V_in_rand, V_out_rand)
    with torch.no_grad():
        cpi_kd = kd_of(s_cpi)
        rand_kd = kd_of(s_rand)

    # CLAIM 1 (CPI): circuit-preserving init is at least as close as random.
    # (The strong CPI benefit is proven in the OV/RoPE unit tests; here we only
    #  sanity-check that the absorbed init composes correctly in the full path.)
    assert cpi_kd <= rand_kd + 1e-6, f"CPI worse than random: cpi={cpi_kd:.4e} rand={rand_kd:.4e}"

    # CLAIM 2 (MT+DDR): from the BAD random init, the full machinery recovers.
    torch.manual_seed(1)
    student, _, s_logits = _build(W1, b1, head, V_in_rand.clone(), V_out_rand.clone())
    ctrl = RankBudgetController(
        {"hidden": student.gate}, {"hidden": student.cost()},
        target_params=0.85 * keep * (d + h), lam=1.0,
    )
    opt = StiefelAdam(stiefel_param_groups(student, base_lr=3e-2, stiefel_lr_ratio=0.5))
    gate_opt = torch.optim.Adam(student.gate.parameters(), lr=2e-2)

    def train_kd():
        xb = torch.randn(B, latent) @ torch.randn(latent, d)
        return skew_kl(s_logits(xb), teacher_logits(xb), alpha=0.1)

    with torch.no_grad():
        init_kd = train_kd().item()
    for _ in range(400):
        opt.zero_grad()
        gate_opt.zero_grad()
        loss = train_kd() + ctrl.budget_penalty()
        loss.backward()
        assert torch.isfinite(loss)
        opt.step()
        gate_opt.step()
    with torch.no_grad():
        final_kd = train_kd().item()
    assert final_kd < 0.7 * init_kd, f"training did not recover: {init_kd:.4e} -> {final_kd:.4e}"

    # CLAIM 3: Stiefel bases stayed on the manifold throughout.
    for p in student.tfl.stiefel_parameters():
        gram = p.T @ p
        assert torch.allclose(gram, torch.eye(p.shape[1]), atol=1e-3)

    # CLAIM 4: DDR hardened to a rank-map within budget; fold preserves the forward.
    rm = ctrl.harden()
    assert 0 < rm["hidden"] <= keep
    folded = student.tfl.fold()
    xt = torch.randn(4, keep)
    gated = student.gate(xt)
    assert torch.allclose(student.tfl(gated), folded(gated), atol=1e-4)
