"""Declarative architecture support (ArchitectureSpec) for FSD/CPSD.

The strangler-fig replacement for per-architecture detector/fold/builder forks. See
:mod:`fasd.arch.spec` for the data model and ``docs/adding_architectures.md`` for the
"add a new model family" checklist.
"""
from .interpreter import expand_branches
from .registry import (
    GPT2_SPEC,
    LLAMA_SPEC,
    MIXTRAL_SPEC,
    QWEN3MOE_SPEC,
    register_arch,
    resolve_spec,
)
from .spec import ArchitectureSpec, EdgeTemplate, FoldTemplate, MoESpec

__all__ = [
    "ArchitectureSpec", "EdgeTemplate", "FoldTemplate", "MoESpec",
    "GPT2_SPEC", "LLAMA_SPEC", "MIXTRAL_SPEC", "QWEN3MOE_SPEC",
    "register_arch", "resolve_spec", "expand_branches",
]
