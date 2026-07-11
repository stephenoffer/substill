"""On-policy student rollouts and replay for F-ASD.

Provides:

- :func:`generate_rollouts` — sample from the student with
  ``student.generate`` under ``torch.no_grad()``, returning token
  sequences and prompt lengths.
- :class:`ReplayBuffer` — bounded FIFO buffer of rollouts.
- :class:`HybridCollator` — iterates mixed batches from an off-policy
  loader and the replay buffer at a target ratio.
- :func:`contrastive_response_loss` — re-exported from
  :mod:`substill.losses.generative_kd` for convenience.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..losses.generative_kd import contrastive_response_loss  # noqa: F401


@dataclass
class RolloutBatch:
    """One batch of student-generated sequences."""

    sequences: Tensor  # (B, T_full) int64
    prompt_lens: Tensor  # (B,) int64
    attention_mask: Tensor  # (B, T_full) int64

    def to(self, device):
        return RolloutBatch(
            sequences=self.sequences.to(device),
            prompt_lens=self.prompt_lens.to(device),
            attention_mask=self.attention_mask.to(device),
        )


# -- rollout generator ------------------------------------------------


@torch.no_grad()
def generate_rollouts(
    student: nn.Module,
    prompts: Tensor,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_p: float = 0.95,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    prompt_mask: Tensor | None = None,
) -> RolloutBatch | None:
    """Sample from the student on the given prompts.

    ``prompts`` is ``(B, T_p)`` int64. ``prompt_mask`` is an optional
    ``(B, T_p)`` attention mask — if a prompt row is right-padded,
    sampling should still proceed from the last non-pad position, but
    this assumes left-padded or equal-length prompts.

    Returns ``None`` when generation fails (e.g. the student produced
    NaN logits or an out-of-vocab token that would trip the embedding
    lookup on the next step). Callers should handle ``None`` as "skip
    this on-policy batch, use off-policy instead."
    """
    student.eval()
    cfg = getattr(student, "config", None)
    vocab_size = int(getattr(cfg, "vocab_size", 0)) if cfg is not None else 0
    if pad_token_id is None:
        pad_token_id = getattr(cfg, "pad_token_id", None) if cfg is not None else None
        if pad_token_id is None:
            pad_token_id = getattr(cfg, "eos_token_id", 0) if cfg is not None else 0
            pad_token_id = pad_token_id or 0
    if eos_token_id is None:
        eos_token_id = getattr(cfg, "eos_token_id", None) if cfg is not None else None
    B, T_p = prompts.shape
    device = prompts.device

    # Clamp prompts defensively in case a caller passed out-of-vocab ids.
    if vocab_size > 0:
        prompts = prompts.clamp(0, vocab_size - 1)

    try:
        gen = student.generate(
            prompts,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(1e-5, temperature),
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
        )
    except Exception:
        return None
    sequences = gen.sequences.to(device)
    # Post-hoc safety: if generation produced an out-of-vocab token (can happen
    # when logits have NaN/Inf and argmax wraps), clamp to the valid range.
    if vocab_size > 0:
        sequences = sequences.clamp(0, vocab_size - 1)
    full_len = sequences.shape[1]
    prompt_lens = torch.full((B,), T_p, dtype=torch.long, device=device)
    attention_mask = torch.ones((B, full_len), dtype=torch.long, device=device)
    if pad_token_id is not None:
        attention_mask = (sequences != pad_token_id).long()
        attention_mask[:, :T_p] = 1  # prompts are always attended to
    return RolloutBatch(
        sequences=sequences, prompt_lens=prompt_lens, attention_mask=attention_mask
    )


# -- replay buffer ----------------------------------------------------


class ReplayBuffer:
    """Bounded FIFO of rollouts, sampled uniformly for training."""

    def __init__(self, capacity: int = 1024, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._items: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, batch: RolloutBatch) -> None:
        """Split a batch into per-row samples and push each."""
        B = batch.sequences.shape[0]
        for i in range(B):
            self._items.append(
                RolloutBatch(
                    sequences=batch.sequences[i : i + 1].clone(),
                    prompt_lens=batch.prompt_lens[i : i + 1].clone(),
                    attention_mask=batch.attention_mask[i : i + 1].clone(),
                )
            )

    def sample(self, n: int) -> RolloutBatch | None:
        if len(self._items) == 0 or n < 1:
            return None
        n = min(n, len(self._items))
        chosen = self._rng.sample(list(self._items), n)
        # Pad sequences to a common length.
        max_T = max(c.sequences.shape[1] for c in chosen)
        seqs: list[Tensor] = []
        masks: list[Tensor] = []
        plens: list[Tensor] = []
        for c in chosen:
            seq = c.sequences[0]
            m = c.attention_mask[0]
            if seq.shape[0] < max_T:
                pad = torch.zeros(max_T - seq.shape[0], dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, pad], dim=0)
                m = torch.cat([m, torch.zeros_like(pad)], dim=0)
            seqs.append(seq)
            masks.append(m)
            plens.append(c.prompt_lens[0])
        return RolloutBatch(
            sequences=torch.stack(seqs, dim=0),
            prompt_lens=torch.stack(plens, dim=0),
            attention_mask=torch.stack(masks, dim=0),
        )

    def clear(self) -> None:
        self._items.clear()


# -- hybrid collator --------------------------------------------------


class HybridCollator:
    """Iterate mixed batches from an off-policy loader and a replay buffer.

    Each call to :meth:`__iter__` returns a generator that yields
    ``{"source": "off" | "on", "batch": <...>}`` dicts. The ``ratio``
    is the probability of drawing from the replay buffer at each step;
    when the buffer is empty we fall back to off-policy.
    """

    def __init__(
        self,
        off_policy_loader: Iterable,
        replay_buffer: ReplayBuffer,
        ratio: float,
        *,
        on_policy_batch_size: int = 4,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")
        self.off_policy_loader = off_policy_loader
        self.replay_buffer = replay_buffer
        self.ratio = ratio
        self.on_batch = on_policy_batch_size
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[dict]:
        off_iter = iter(self.off_policy_loader)
        while True:
            draw_on = (
                self._rng.random() < self.ratio and len(self.replay_buffer) >= self.on_batch
            )
            if draw_on:
                batch = self.replay_buffer.sample(self.on_batch)
                if batch is None:
                    draw_on = False
                else:
                    yield {"source": "on", "batch": batch}
                    continue
            try:
                off_batch = next(off_iter)
            except StopIteration:
                return
            yield {"source": "off", "batch": off_batch}


__all__ = [
    "RolloutBatch",
    "generate_rollouts",
    "ReplayBuffer",
    "HybridCollator",
    "contrastive_response_loss",
]
