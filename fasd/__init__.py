"""Deprecated import alias for :mod:`substill`.

The package was renamed ``fasd`` -> ``substill`` in v0.4. ``import fasd`` still resolves
to :mod:`substill` (including submodules such as ``fasd.lrd``) but emits a
:class:`DeprecationWarning`; update imports to ``import substill``. The alias will be
removed in a future release.
"""
from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "The 'fasd' package was renamed to 'substill'. Import 'substill' instead; "
    "the 'fasd' alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-point 'fasd' (and every 'fasd.<sub>' path) at the real package so existing imports
# keep working unchanged.
sys.modules[__name__] = importlib.import_module("substill")
