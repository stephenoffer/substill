"""Internal subspace-profiling utilities for :mod:`fasd`.

``asd`` is no longer a public package. It retains only the low-level
profiling machinery that :mod:`fasd` builds on:

- :class:`asd.profiling.activation_capture.CovarianceAccumulator`
- :mod:`asd.profiling.stability` (principal-angle stability diagnostics)
- :mod:`asd.profiling.svd_analysis` (Marchenko-Pastur / spectrum analysis)

The original ``asd`` CNN/ResNet feature-KD API (``SubspaceLoss``, ``distill``,
``build_student``, ``autodetect``) has been removed; use :mod:`fasd` instead.
Nothing here is part of the supported public surface.
"""
