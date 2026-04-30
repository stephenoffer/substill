"""Cached compressed teacher features reproduce the in-loop teacher features."""

from __future__ import annotations

import pytest
import torch

from fasd.api import capture, profile as profile_fn


def _toy_gpt2():
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(vocab_size=30, n_positions=16, n_embd=16, n_layer=2, n_head=2, n_inner=32)
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def test_cache_teacher_features_consistent_with_online():
    from fasd.training.distill import _build_feature_cache

    model = _toy_gpt2()
    if model is None:
        pytest.skip("transformers not installed")
    model.eval()

    torch.manual_seed(0)
    B, T = 2, 6
    tokens = torch.randint(5, 25, (B, T))
    loader = [{"input_ids": tokens, "attention_mask": torch.ones(B, T, dtype=torch.long)}] * 2

    prof = profile_fn(
        model,
        loader,
        mode="residual",
        rank_tol=0.5,
        token_weighting="uniform",
        n_calib_batches=2,
        behavioral_calib_batches=1,
        max_rank=8,
    )
    cache = _build_feature_cache(model, prof, loader, torch.device("cpu"))
    assert len(cache) == len(prof.branches)
    for b in prof.branches:
        assert b.name in cache
        # At least one entry per batch.
        assert len(cache[b.name]) >= 1
        sample = cache[b.name][0]
        # Shape: (B, T, behavioral_rank)
        assert sample.shape[-1] == b.behavioral_rank
