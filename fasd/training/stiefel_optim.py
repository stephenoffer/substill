"""Stiefel-manifold optimizer for trainable bases (Pillar 2).

Replaces the discrete Periodic Re-Absorption (PRA) mechanism with continuous
basis updates. A Stiefel parameter ``U ∈ St(n, k) = {U : U^T U = I_k}``
evolves by Riemannian momentum SGD with **Cayley retraction**.

Why this is needed for FSD:
    - Each absorbed-init basis ``V_in, V_out`` and each RR-Norm correction ``Q``
      is an orthonormal matrix that we want to keep orthonormal during training.
    - Standard AdamW on these parameters drifts off the Stiefel manifold
      (``U^T U`` slowly diverges from identity) and mixes basis vectors in
      gradient-coupled ways that are hard to reason about.
    - PRA's "discrete jump from old V to new V" is a special case of Stiefel
      gradient descent with a very large step. The principled version is
      continuous Stiefel descent: the basis evolves smoothly with the residual.

References:
    - Wen & Yin, "A feasible method for optimization with orthogonality
      constraints," Math. Programming, 2013.
    - Lezcano-Casado, "Cheap Orthogonal Constraints in Neural Networks," 2019.
    - Li & Arora, "An exponential learning rate schedule for deep learning," 2019
      (note on momentum-only Stiefel).

Algorithm (per Stiefel parameter U ∈ R^{n × k}):

1. Euclidean grad ``G = ∂L/∂U``.
2. Riemannian grad ``G̃ = G − U sym(U^T G)``, where ``sym(A) = ½(A + A^T)``.
3. Momentum ``M ← β₁ M + G̃``; project back to tangent: ``M ← M − U sym(U^T M)``.
4. Adafactor-style scaling: row stats ``R ∈ R^n`` and col stats ``C ∈ R^k``
   tracking EMA of ``M²``; scale ``M̂ = M / sqrt(R · C^T + ε)``.
5. Skew matrix ``W = M̂ U^T − U M̂^T``, shape (n, n).
6. Cayley retraction: ``U_new = (I − ½η W)^{-1} (I + ½η W) U``.

For ``k ≪ n``, we use the efficient form (Wen-Yin §2.4): solve a 2k × 2k
system instead of inverting an n × n matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


def _sym(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def _project_tangent(M: Tensor, U: Tensor) -> Tensor:
    """Project M onto the tangent space of St(n, k) at U.

    T_U St = {Z : U^T Z + Z^T U = 0} = {Z = U Ω + (I-UU^T) K  : Ω skew, K free}.
    The projection of an arbitrary M is::

        M_T = M − U sym(U^T M)

    This makes ``U^T M_T + M_T^T U = 0`` (skew).
    """
    return M - U @ _sym(U.transpose(-1, -2) @ M)


def _cayley_step(U: Tensor, M: Tensor, lr: float) -> Tensor:
    """Compute Cayley-retracted ``U_new`` along descent direction ``-M`` at step ``lr``.

    Wen & Yin (2013) §2.4 efficient form. For ``M`` being a tangent vector
    (the positive Riemannian gradient direction; descent moves along ``-lr · M``):

        W̃ = M U^T - U M^T,   so step is W = -lr · W̃.
        U_new = (I - W/2)^{-1} (I + W/2) U
              = (I + (lr/2) W̃)^{-1} (I - (lr/2) W̃) U

    Decompose W̃ = X Y^T with X = [M, U], Y = [U, -M] (rank ≤ 2k). Then:

        A = I_{2k} + (lr/2) Y^T X
        Z = A^{-1} Y^T U
        U_new = U - lr · X · Z

    Returns U_new with the same shape as U.
    """
    n, k = U.shape
    # Form W̃ = M U^T - U M^T via rank-2k factorisation.
    X = torch.cat([M, U], dim=1)  # (n, 2k)
    Y = torch.cat([U, -M], dim=1)  # (n, 2k)

    YtX = Y.transpose(0, 1) @ X  # (2k, 2k)
    YtU = Y.transpose(0, 1) @ U  # (2k, k)

    A = torch.eye(2 * k, dtype=U.dtype, device=U.device) + 0.5 * float(lr) * YtX
    Z = torch.linalg.solve(A, YtU)  # (2k, k)
    U_new = U - float(lr) * (X @ Z)
    return U_new


def _orthogonalize(U: Tensor) -> Tensor:
    """QR-orthogonalize columns of U (post-step safety projection).

    Cayley retraction is exact in infinite precision but accumulates float64
    error over many steps in float32. This is an O(nk²) safety net.
    """
    Q, R = torch.linalg.qr(U)
    # Sign-disambiguate so the diagonal of R is positive (cosmetic; keeps
    # eigenvalue ordering stable).
    sign = torch.sign(torch.diag(R))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q * sign.unsqueeze(0)


class StiefelAdam(Optimizer):
    """Stiefel-aware optimizer with momentum + Adafactor row/col second moments.

    Stiefel parameters ``U`` are kept on the manifold via Cayley retraction;
    other parameters fall through to a standard AdamW path.

    Parameters
    ----------
    params : Iterable[Tensor] | list[dict]
        Parameter list or param-groups. Each param-group may set::

            stiefel: bool            (default False)
                Treat this group's parameters as Stiefel matrices.
            lr: float                (default 1e-3)
            betas: tuple[float, float]
                For Stiefel: betas[0] is momentum β₁; betas[1] is Adafactor decay β₂.
                For standard: usual AdamW betas.
            eps: float               (default 1e-8)
            weight_decay: float      (default 0.0; not applied to Stiefel params)
            reorth_every: int        (default 50)
                Re-QR Stiefel parameters every N steps to bound float drift.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        stiefel: bool = False,
        reorth_every: int = 50,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "stiefel": stiefel,
            "reorth_every": reorth_every,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            is_stiefel = group.get("stiefel", False)
            reorth_every = group.get("reorth_every", 50)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                if is_stiefel:
                    self._stiefel_step(p, grad, state, lr, beta1, beta2, eps, reorth_every)
                else:
                    self._adamw_step(p, grad, state, lr, beta1, beta2, eps, wd)

                state["step"] += 1

        return loss

    def _stiefel_step(
        self,
        p: Tensor,
        grad: Tensor,
        state: dict,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        reorth_every: int,
    ) -> None:
        if p.dim() != 2:
            raise ValueError(
                f"Stiefel parameter must be 2-D; got shape {tuple(p.shape)}. "
                "Use a non-stiefel param-group for biases / 1-D params."
            )
        n, k = p.shape
        if n < k:
            raise ValueError(
                f"Stiefel param shape ({n}, {k}) requires n ≥ k; transpose if needed."
            )

        # State init.
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p)
            state["row_sq"] = torch.zeros(n, dtype=p.dtype, device=p.device)
            state["col_sq"] = torch.zeros(k, dtype=p.dtype, device=p.device)

        M = state["momentum"]
        R_stat = state["row_sq"]
        C_stat = state["col_sq"]
        step = state["step"] + 1

        # 1. Riemannian gradient.
        G_riem = _project_tangent(grad, p)

        # 2. Momentum, then re-tangent.
        M.mul_(beta1).add_(G_riem)
        M = _project_tangent(M, p)
        state["momentum"] = M  # keep updated reference

        # 3. Adafactor row / col second moments (EMA of mean of squared M).
        m_sq = M.pow(2)
        R_stat.mul_(beta2).add_(m_sq.mean(dim=1), alpha=1.0 - beta2)
        C_stat.mul_(beta2).add_(m_sq.mean(dim=0), alpha=1.0 - beta2)

        # Bias correction (Adafactor-lite): scale by 1 / (1 - β₂^t) once for both.
        bias_correction = 1.0 - beta2 ** step
        R_hat = R_stat / bias_correction
        C_hat = C_stat / bias_correction

        # Reconstruct outer-product approximation to the elementwise variance.
        # Normalisation: divide so that mean(R_hat ⊗ C_hat) ≈ mean(M²).
        # Using Adafactor's exact form: scale = sqrt(R_hat[:, None] / mean(R_hat) * C_hat[None, :])
        R_norm = R_hat / R_hat.mean().clamp_min(eps)
        scale = torch.sqrt(R_norm.unsqueeze(1) * C_hat.unsqueeze(0).clamp_min(eps))

        M_scaled = M / scale.clamp_min(eps).sqrt().clamp_min(eps)
        # Re-tangent after scaling (scale doesn't preserve tangent in general).
        M_scaled = _project_tangent(M_scaled, p)

        # 4. Cayley retraction along -M_scaled.
        U_new = _cayley_step(p.data, M_scaled, lr)

        # 5. Periodic reorthogonalization for float drift.
        if reorth_every > 0 and step % reorth_every == 0:
            U_new = _orthogonalize(U_new)

        p.data.copy_(U_new)

    def _adamw_step(
        self,
        p: Tensor,
        grad: Tensor,
        state: dict,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        wd: float,
    ) -> None:
        # Standard AdamW with decoupled weight decay.
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        m = state["exp_avg"]
        v = state["exp_avg_sq"]
        step = state["step"] + 1

        if wd != 0:
            p.data.mul_(1.0 - lr * wd)

        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def stiefel_param_groups(
    model,
    *,
    base_lr: float,
    stiefel_lr_ratio: float = 0.1,
    weight_decay: float = 0.01,
    reorth_every: int = 50,
) -> list[dict]:
    """Build param-groups for FSD: Stiefel for tagged params, AdamW for the rest.

    Looks for any module whose ``__class__.is_stiefel_q`` (RRNorm) or attribute
    ``stiefel_parameters()`` returns parameters; those go into the Stiefel group.

    Parameters
    ----------
    base_lr : float
        Standard learning rate (used for AdamW group).
    stiefel_lr_ratio : float
        Stiefel learning rate as a fraction of base_lr. Recommended 0.1× per
        the algorithm review.
    """
    stiefel_params: list[Tensor] = []
    standard_params: list[Tensor] = []
    seen: set[int] = set()

    for mod in model.modules():
        if hasattr(mod, "stiefel_parameters"):
            try:
                sp = list(mod.stiefel_parameters())
            except Exception:
                sp = []
            for p in sp:
                if id(p) not in seen and p.requires_grad:
                    stiefel_params.append(p)
                    seen.add(id(p))

    for p in model.parameters():
        if id(p) not in seen and p.requires_grad:
            standard_params.append(p)
            seen.add(id(p))

    groups = []
    if stiefel_params:
        groups.append(
            {
                "params": stiefel_params,
                "lr": base_lr * stiefel_lr_ratio,
                "stiefel": True,
                "weight_decay": 0.0,
                "reorth_every": reorth_every,
                "betas": (0.9, 0.999),
            }
        )
    if standard_params:
        groups.append(
            {
                "params": standard_params,
                "lr": base_lr,
                "stiefel": False,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.999),
            }
        )
    return groups


__all__ = ["StiefelAdam", "stiefel_param_groups"]
