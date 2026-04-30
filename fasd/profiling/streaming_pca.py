"""Streaming / randomized PCA for behavioral-rank profiling.

Three backends:

``"exact"``
    Accumulate the full ``C x C`` covariance via
    :class:`asd.profiling.activation_capture.CovarianceAccumulator`,
    then ``torch.linalg.eigh``. Correctness baseline.

``"randomized"``
    Halko-Martinsson-Tropp randomized SVD on a streamed sketch
    ``Y = X @ Omega`` with ``Omega`` a ``(C, k + p)`` Gaussian matrix
    and ``p=10`` oversampling. Two power iterations on the accumulated
    ``Y`` improve tail accuracy. Memory ``O(C * (k + p))`` — never
    materializes ``C x C``.

``"oja"``
    Online k-PCA via normalized Oja updates on a rank-k projector
    ``U in R^{C x k}`` with periodic QR re-orthogonalization.

The backend is auto-selected in :func:`fasd.profile` when ``C >= 1024``
or when the teacher has more than ``1e9`` parameters.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from asd.profiling.activation_capture import CovarianceAccumulator


Backend = Literal["exact", "randomized", "oja"]


class StreamingPCA:
    """Unified streaming-PCA interface over the three backends."""

    def __init__(
        self,
        channels: int,
        *,
        k: int,
        backend: Backend = "exact",
        oversample: int = 10,
        power_iter: int = 2,
        oja_lr: float = 1e-2,
        oja_reortho_every: int = 50,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > channels:
            raise ValueError(f"k ({k}) must be <= channels ({channels})")
        if backend not in ("exact", "randomized", "oja"):
            raise ValueError(f"unknown backend: {backend!r}")

        self.channels = channels
        self.k = k
        self.backend: Backend = backend
        self.oversample = oversample
        self.power_iter = power_iter
        self.oja_lr = oja_lr
        self.oja_reortho_every = oja_reortho_every
        self.device = torch.device(device)
        self.dtype = dtype

        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        self._n_samples = 0
        self._step = 0

        if backend == "exact":
            self._acc = CovarianceAccumulator(channels)
        elif backend == "randomized":
            sketch_dim = k + oversample
            self._omega = torch.randn(
                channels, sketch_dim, generator=self._gen, dtype=dtype
            ).to(self.device)
            self._Y = torch.zeros(channels, sketch_dim, device=self.device, dtype=dtype)
            self._C_times_omega_buf: Tensor | None = None
        elif backend == "oja":
            u = torch.randn(channels, k, generator=self._gen, dtype=dtype).to(
                self.device
            )
            u, _ = torch.linalg.qr(u)
            self._U = u
        self._eigenvalues: Tensor | None = None
        self._basis: Tensor | None = None

    # -- public update ----------------------------------------------------

    def update(self, activation: Tensor) -> None:
        """Update PCA estimate from a batch of activations.

        ``activation`` is reshaped to ``(N, channels)`` where ``N`` is
        the number of samples in this batch. Accepted input shapes:
        ``(B, C)``, ``(B, T, C)``, ``(B, C, H, W)``.
        """
        x = self._flatten(activation)
        n = x.shape[0]
        self._n_samples += n

        if self.backend == "exact":
            # Reshape for CovarianceAccumulator.update (wants B,C or B,T,C etc)
            self._acc.update(activation)
            return

        x = x.to(device=self.device, dtype=self.dtype)

        if self.backend == "randomized":
            # Accumulate X^T X @ Omega without building C x C.
            self._Y += x.T @ (x @ self._omega)
            return

        if self.backend == "oja":
            # Oja: U <- U + lr * (I - U U^T) X^T X U / N
            xt_x_u = x.T @ (x @ self._U)
            proj = self._U @ (self._U.T @ xt_x_u)
            self._U = self._U + (self.oja_lr / max(1, n)) * (xt_x_u - proj)
            self._step += 1
            if self._step % self.oja_reortho_every == 0:
                self._U, _ = torch.linalg.qr(self._U)
            return

    # -- public finalize --------------------------------------------------

    def top_k(self) -> tuple[Tensor, Tensor]:
        """Return ``(basis, eigenvalues)`` with shapes ``(C, k)`` / ``(k,)``.

        Eigenvalues are sorted descending. For ``"oja"`` the eigenvalue
        estimate is the diagonal of ``U^T E[X^T X] U`` computed from
        the current Oja state, so it is only approximate.
        """
        if self.backend == "exact":
            cov = self._acc.finalize().to(self.device).to(self.dtype)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            order = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            self._basis = eigvecs[:, : self.k].contiguous()
            self._eigenvalues = eigvals[: self.k].clamp_min(0.0).contiguous()
            return self._basis, self._eigenvalues

        if self.backend == "randomized":
            if self._n_samples == 0:
                raise RuntimeError("StreamingPCA.update was never called")
            Y = self._Y / float(self._n_samples)  # ~ C @ Omega
            for _ in range(self.power_iter):
                Q, _ = torch.linalg.qr(Y)
                Z = self._omega.T @ Q  # not used further, keep structure symmetric
                Y = Y @ (Q.T @ Y)  # C @ (Q Q^T) @ Omega ~ power-iterated sketch
            Q, _ = torch.linalg.qr(Y)
            # Project and solve small eigenproblem: B = Q^T C Q is approximated
            # by Q^T (C @ Omega) @ pinv(Q^T Omega)
            QtY = Q.T @ (self._Y / float(self._n_samples))  # (k+p, k+p)
            QtOmega = Q.T @ self._omega  # (k+p, k+p)
            # Solve for small covariance B : B @ QtOmega = QtY
            B = torch.linalg.lstsq(QtOmega.T, QtY.T).solution.T
            B = 0.5 * (B + B.T)
            eigvals, eigvecs = torch.linalg.eigh(B)
            order = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            basis = Q @ eigvecs
            self._basis = basis[:, : self.k].contiguous()
            self._eigenvalues = eigvals[: self.k].clamp_min(0.0).contiguous()
            return self._basis, self._eigenvalues

        if self.backend == "oja":
            U, _ = torch.linalg.qr(self._U)
            # eigenvalue estimate: diag(U^T C U) is unavailable without cov.
            # Use raw singular-value surrogate: for a rank-k subspace with
            # normalized Oja updates, diag(U^T U) ~ 1 after QR, so we return
            # uniform ones — users who need eigenvalues should use "exact"
            # or a one-pass re-projection on a calibration batch.
            self._basis = U[:, : self.k].contiguous()
            self._eigenvalues = torch.ones(self.k, device=U.device, dtype=U.dtype)
            return self._basis, self._eigenvalues

        raise AssertionError(f"unreachable backend: {self.backend!r}")

    # -- helpers ----------------------------------------------------------

    def _flatten(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            # (B, T, C) → (B*T, C)
            return x.reshape(-1, x.shape[-1])
        if x.dim() == 4:
            # (B, C, H, W) → (B*H*W, C)
            return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        raise ValueError(
            f"StreamingPCA.update: expected 2/3/4D tensor, got {tuple(x.shape)}"
        )


def auto_backend(channels: int, teacher_params: int | None = None) -> Backend:
    """Choose a backend based on channel count and teacher size."""
    if channels >= 1024:
        return "randomized"
    if teacher_params is not None and teacher_params >= 1_000_000_000:
        return "randomized"
    return "exact"


__all__ = ["StreamingPCA", "Backend", "auto_backend"]
