"""tests/test_memory.py
=======================
Unit tests for :mod:`dino_loader.memory`.

Coverage
--------
Batch
- Construction with all fields
- __iter__ unpacks as (global_crops, local_crops)
- metadata and masks default correctly

allocate_buffers [M5]
- Returns pinned CPU tensors sized to max crop dimensions
- Shape matches (batch_size, 3, max_size, max_size)
- Tensors are pinned (is_pinned())
- topo parameter accepted for API compatibility

FP8Formatter (NullFP8Formatter used in CPU path)
- quantise returns the same tensor unchanged when TE absent
- quantise does not modify values
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.config import DINOAugConfig
from dino_loader.memory import Batch, allocate_buffers

# ══════════════════════════════════════════════════════════════════════════════
# Batch dataclass
# ══════════════════════════════════════════════════════════════════════════════


class TestBatch:

    def test_construction_with_all_fields(self):
        g = [torch.zeros(4, 3, 32, 32)]
        l = [torch.zeros(4, 3, 16, 16)]
        m = [{"quality_score": 0.9}] * 4
        batch = Batch(global_crops=g, local_crops=l, metadata=m)
        assert batch.global_crops is g
        assert batch.local_crops is l
        assert batch.metadata is m

    def test_iter_unpacks_global_and_local(self):
        g = [torch.zeros(2, 3, 32, 32)]
        l = [torch.zeros(2, 3, 16, 16)]
        batch = Batch(global_crops=g, local_crops=l)
        unpacked_g, unpacked_l = batch
        assert unpacked_g is g
        assert unpacked_l is l

    def test_masks_defaults_to_none(self):
        batch = Batch(global_crops=[], local_crops=[])
        assert batch.masks is None

    def test_metadata_defaults_to_empty_list(self):
        batch = Batch(global_crops=[], local_crops=[])
        assert batch.metadata == []

    def test_masks_can_be_set(self):
        masks = torch.ones(4, 196, dtype=torch.bool)
        batch = Batch(global_crops=[], local_crops=[], masks=masks)
        assert batch.masks is masks


# ══════════════════════════════════════════════════════════════════════════════
# allocate_buffers [M5]
# ══════════════════════════════════════════════════════════════════════════════


class TestAllocateBuffers:

    def _make_topo(self):
        from unittest.mock import MagicMock
        return MagicMock()

    def test_returns_global_and_local_keys(self):
        aug_cfg = DINOAugConfig(global_crop_size=32, local_crop_size=16)
        bufs = allocate_buffers(
            batch_size=4,
            aug_cfg=aug_cfg,
            topo=self._make_topo(),
            device=torch.device("cpu"),
        )
        assert "global" in bufs
        assert "local" in bufs

    def test_global_tensor_shape(self):
        aug_cfg = DINOAugConfig(global_crop_size=32, max_global_crop_size=64)
        bufs = allocate_buffers(
            batch_size=4,
            aug_cfg=aug_cfg,
            topo=self._make_topo(),
            device=torch.device("cpu"),
        )
        for t in bufs["global"]:
            assert t.shape == (4, 3, 64, 64)

    def test_local_tensor_shape(self):
        aug_cfg = DINOAugConfig(local_crop_size=16, max_local_crop_size=32)
        bufs = allocate_buffers(
            batch_size=4,
            aug_cfg=aug_cfg,
            topo=self._make_topo(),
            device=torch.device("cpu"),
        )
        for t in bufs["local"]:
            assert t.shape == (4, 3, 32, 32)

    def test_tensors_are_pinned_on_cpu(self):
        """PCIe path must produce pinned tensors for efficient H2D DMA [M5]."""
        aug_cfg = DINOAugConfig(global_crop_size=32, local_crop_size=16)
        device = torch.device("cpu")
        bufs = allocate_buffers(
            batch_size=4,
            aug_cfg=aug_cfg,
            topo=self._make_topo(),
            device=device,
        )
        for t in bufs["global"] + bufs["local"]:
            assert t.is_pinned(), "PCIe path must produce pinned tensors"
            assert t.device.type == "cpu"

    def test_topo_parameter_accepted(self):
        """Topo is accepted for API compatibility even though all paths are PCIe."""
        from unittest.mock import MagicMock
        aug_cfg = DINOAugConfig(global_crop_size=32, local_crop_size=16)
        topo = MagicMock()
        allocate_buffers(
            batch_size=2,
            aug_cfg=aug_cfg,
            topo=topo,
            device=torch.device("cpu"),
        )  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# FP8Formatter (CPU / no-TE path via NullFP8Formatter)
# ══════════════════════════════════════════════════════════════════════════════


class TestNullFP8Formatter:

    def test_quantise_returns_same_tensor(self):
        from dino_loader.backends.cpu import NullFP8Formatter
        fmt = NullFP8Formatter()
        t = torch.randn(4, 3, 32, 32)
        out = fmt.quantise(t)
        assert out is t

    def test_quantise_does_not_modify_values(self):
        from dino_loader.backends.cpu import NullFP8Formatter
        fmt = NullFP8Formatter()
        t = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        out = fmt.quantise(t)
        assert torch.equal(out, t)
