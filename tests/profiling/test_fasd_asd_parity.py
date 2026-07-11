"""substill's branch engine matches asd's on residual-stream hooks (no branch split).

Both engines share the asd ``CovarianceAccumulator``; on a hook point
that doesn't split (block.residual), they should produce identical
covariances for identical inputs.
"""

from __future__ import annotations

import pytest
import torch

from substill._asd.profiling.activation_capture import ActivationCaptureEngine
from substill.autodetect import BranchSpec
from substill.profiling.activation_capture import BranchCaptureEngine


def _toy_gpt2():
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return None
    cfg = GPT2Config(vocab_size=30, n_positions=16, n_embd=16, n_layer=2, n_head=2, n_inner=32)
    cfg.pad_token_id = 0
    return GPT2LMHeadModel(cfg)


def test_fasd_residual_covariance_matches_asd():
    model = _toy_gpt2()
    if model is None:
        pytest.skip("transformers not installed")
    model.eval()

    torch.manual_seed(0)
    B, T = 2, 6
    tokens = torch.randint(5, 25, (B, T))
    batch = {"input_ids": tokens, "attention_mask": torch.ones(B, T, dtype=torch.long)}

    # asd path — register hooks manually and run model once.
    asd_engine = ActivationCaptureEngine(model, ["transformer.h.0"], source="output")
    asd_engine.register_hooks()
    with torch.no_grad():
        for _ in range(3):
            model(**batch)
    asd_engine.cleanup()
    asd_cov = asd_engine.accumulator("transformer.h.0").finalize()

    # substill residual-mode path.
    spec = BranchSpec(
        name="transformer.h.0.residual",
        module_path="transformer.h.0",
        kind="block.residual",
    )
    f_engine = BranchCaptureEngine(model, [spec])
    f_engine.register_hooks()
    with torch.no_grad():
        for _ in range(3):
            model(**batch)
    f_engine.cleanup()
    f_cov = f_engine.accumulator("transformer.h.0.residual").finalize()

    # Both should match exactly up to float precision.
    assert asd_cov.shape == f_cov.shape
    err = (asd_cov - f_cov).abs().max().item()
    assert err < 1e-5, f"substill/asd covariance diverges by {err}"
