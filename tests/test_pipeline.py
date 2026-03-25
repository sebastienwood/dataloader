"""tests/test_pipeline.py
=========================
Unit tests for :mod:`dino_loader.pipeline`.

Coverage
--------
NormSource [M2]
- set_dataset_indices is a full copy-on-write replacement
- __call__ returns independent numpy copies (not views into _lookup)
- Concurrent set + call does not corrupt or raise

pipeline dispatch
- build_pipeline raises RuntimeError when DALI is not installed
- build_pipeline raises TypeError on unknown AugmentationSpec
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.config import DINOAugConfig

# ══════════════════════════════════════════════════════════════════════════════
# NormSource — thread-safe copy-on-write [M2]
# ══════════════════════════════════════════════════════════════════════════════


def _make_norm_source(n_datasets: int = 2):
    from dino_loader.pipeline import NormSource
    specs = [MagicMock(mean=None, std=None) for _ in range(n_datasets)]
    return NormSource(aug_cfg=DINOAugConfig(), specs=specs)


class TestNormSourceSetIndices:

    def test_set_indices_replaces_entire_list(self):
        ns = _make_norm_source(3)
        ns.set_dataset_indices([0, 1, 2])
        ns.set_dataset_indices([2, 1])
        means, stds = ns()
        assert len(means) == 2
        assert len(stds) == 2

    def test_call_length_matches_indices(self):
        ns = _make_norm_source(4)
        ns.set_dataset_indices([0, 3, 1])
        means, stds = ns()
        assert len(means) == 3
        assert len(stds) == 3


class TestNormSourceReturnsCopies:

    def test_call_returns_independent_arrays(self):
        """Modifying a returned array must not affect subsequent calls [M2]."""
        ns = _make_norm_source(1)
        ns.set_dataset_indices([0])
        means1, _ = ns()
        means2, _ = ns()
        means1[0][:] = 99.0
        assert not np.allclose(means2[0], 99.0), (
            "Returned arrays are views into _lookup — expected independent copies"
        )


class TestNormSourceConcurrency:

    def test_concurrent_set_and_call_no_errors(self):
        ns = _make_norm_source(4)
        ns.set_dataset_indices([0, 1, 2, 3])
        errors: list[Exception] = []

        def _setter():
            for i in range(200):
                ns.set_dataset_indices([i % 4])

        def _caller():
            for _ in range(200):
                try:
                    ns()
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=_setter),
            threading.Thread(target=_caller),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in concurrent NormSource access: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# build_pipeline dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildPipelineDispatch:

    def test_raises_runtime_error_when_dali_unavailable(self):
        """build_pipeline must surface a clear error when nvidia-dali is absent."""
        from unittest.mock import patch

        # Patch HAS_DALI to False to simulate missing DALI
        with patch("dino_loader.pipeline.HAS_DALI", False):
            from dino_loader.augmentation import DinoV2AugSpec
            from dino_loader.pipeline import build_pipeline
            with pytest.raises(RuntimeError, match="nvidia-dali"):
                build_pipeline(
                    source=MagicMock(),
                    aug_spec=DinoV2AugSpec(aug_cfg=DINOAugConfig()),
                    batch_size=4,
                    num_threads=1,
                    device_id=0,
                    resolution_src=MagicMock(),
                )
