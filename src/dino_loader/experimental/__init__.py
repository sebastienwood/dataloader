"""dino_loader.experimental
===========================
Experimental features that are not yet production-ready.

Contents
--------
dynamic_pipeline
    DALI v2 dynamic-mode augmentation pipeline.  Replaces the static
    ``@pipeline_def`` graph in ``pipeline.py`` with imperative Python code
    that DALI JIT-compiles per batch, eliminating ``ExternalSource`` hacks
    for resolution and per-dataset normalisation.

These features are disabled by default in production ``DINODataLoader``
and must be explicitly opted into via the ``DALIBackend`` ``use_dynamic``
flag.  See ``scripts/benchmark.py`` for a head-to-head comparison.

.. warning::
    All APIs in this package are **experimental** and may change without
    notice.  They depend on ``nvidia.dali.experimental.dynamic`` which is
    itself marked experimental by NVIDIA.
"""
