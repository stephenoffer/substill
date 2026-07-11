"""Sphinx configuration for the substill documentation site."""

from __future__ import annotations

import importlib.metadata as _md

project = "substill"
author = "substill contributors"
copyright = "substill contributors"  # noqa: A001 (Sphinx expects this name)
try:
    release = _md.version("substill")
except _md.PackageNotFoundError:  # building from a source checkout without install
    release = "0.4.0"
version = release

# -- General -----------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Research notes are kept in the repo but not rendered into the site build.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",          # the GitHub-facing docs index; the site uses index.md
    "cpsd.md",            # legacy CPSD system (unverified numbers)
    "adding_architectures.md",  # legacy FSD/CPSD extension how-to
    "report.md",
    "handoff.md",
    "legacy/**",          # earlier F-ASD method docs
    "archive/**",
]

# MyST (Markdown) ------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist"]
myst_heading_anchors = 3
# The build is warning-free; nothing is suppressed, so a broken cross-reference or a
# page dropped from the nav surfaces immediately in the build log.
suppress_warnings = []

# Autodoc / autosummary ------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Heavy runtime deps are mocked so the site builds without a GPU stack when needed.
autodoc_mock_imports = ["transformers", "datasets", "omegaconf", "torchvision"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML --------------------------------------------------------------------
html_theme = "furo"
html_title = f"substill {version}"
html_static_path = []
