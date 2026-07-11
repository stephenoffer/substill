"""The renamed-away ``fasd`` package must still import, warn, and alias ``substill``."""
from __future__ import annotations

import sys
import warnings


def _fresh_import_fasd():
    for m in [k for k in sys.modules if k == "fasd" or k.startswith("fasd.")]:
        del sys.modules[m]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import fasd  # noqa: F401
    return sys.modules["fasd"], caught


def test_fasd_import_emits_deprecation_warning():
    _, caught = _fresh_import_fasd()
    assert any(issubclass(w.category, DeprecationWarning) and "substill" in str(w.message)
               for w in caught)


def test_fasd_aliases_substill_public_api():
    import substill
    fasd, _ = _fresh_import_fasd()
    assert fasd is substill
    assert fasd.learned_restriction_distill is substill.learned_restriction_distill
    assert fasd.LRDConfig is substill.LRDConfig
    assert fasd.__version__ == substill.__version__
