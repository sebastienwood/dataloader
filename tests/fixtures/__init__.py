"""
tests.fixtures
==============
Helpers for building synthetic WebDataset shards in memory and on disk.

These are used by all unit tests that require shard data.  They deliberately
do NOT import any dino_loader module so they can be used before the package
is on sys.path (e.g. in conftest.py before installation).
"""

from __future__ import annotations

import io
import json
import os
import struct
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic image generation
# ══════════════════════════════════════════════════════════════════════════════

def make_jpeg_bytes(
    width:   int   = 64,
    height:  int   = 64,
    color:   Tuple[int, int, int] = (128, 64, 32),
    quality: int   = 85,
) -> bytes:
    """Return JPEG-encoded bytes for a solid-colour RGB image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_shard_tar(
    n_samples:      int   = 10,
    with_metadata:  bool  = True,
    quality_scores: Optional[List[float]] = None,
    img_width:      int   = 64,
    img_height:     int   = 64,
) -> bytes:
    """
    Build a WebDataset-compatible tar archive in memory.

    Each sample consists of:
    - ``<key>.jpg``  — a JPEG-encoded solid-colour image
    - ``<key>.json`` — (when with_metadata=True) a JSON sidecar with
                        ``{"quality_score": float, "caption": str}``

    Parameters
    ----------
    n_samples      : Number of (jpg, json) pairs in the archive.
    with_metadata  : Whether to include .json sidecars.
    quality_scores : Per-sample quality scores.  Defaults to 1.0 for all.
    img_width / img_height : Synthetic image dimensions.

    Returns
    -------
    bytes — raw tar archive content.
    """
    if quality_scores is None:
        quality_scores = [1.0] * n_samples
    assert len(quality_scores) == n_samples

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_samples):
            key = f"sample_{i:06d}"

            # JPEG member
            jpeg = make_jpeg_bytes(
                width  = img_width,
                height = img_height,
                color  = (i * 25 % 256, (i * 37 + 80) % 256, (i * 53 + 160) % 256),
            )
            _add_bytes(tf, f"{key}.jpg", jpeg)

            # JSON sidecar
            if with_metadata:
                meta = {
                    "quality_score": quality_scores[i],
                    "caption":       f"A test image number {i}",
                    "dedup_hash":    f"hash_{i:08x}",
                }
                _add_bytes(tf, f"{key}.json", json.dumps(meta).encode())

    return buf.getvalue()


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info          = tarfile.TarInfo(name=name)
    info.size     = len(data)
    tf.addfile(info, io.BytesIO(data))


# ══════════════════════════════════════════════════════════════════════════════
# On-disk shard scaffolding
# ══════════════════════════════════════════════════════════════════════════════

def write_shard(
    directory:     Path,
    shard_idx:     int  = 0,
    n_samples:     int  = 10,
    with_metadata: bool = True,
    quality_scores: Optional[List[float]] = None,
) -> Tuple[str, str]:
    """
    Write a synthetic .tar shard and its companion .idx file to *directory*.

    Returns (tar_path, idx_path) as strings.

    The .idx file is a sequence of int64 byte offsets matching the wds2idx
    binary format (8 bytes per entry, little-endian int64).  For testing
    purposes we write monotonically-increasing synthetic offsets.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    tar_data = make_shard_tar(
        n_samples      = n_samples,
        with_metadata  = with_metadata,
        quality_scores = quality_scores,
    )

    tar_path = directory / f"shard-{shard_idx:06d}.tar"
    idx_path = directory / f"shard-{shard_idx:06d}.idx"

    tar_path.write_bytes(tar_data)

    # Synthetic idx: n_samples * 8-byte int64 offsets
    idx_data = struct.pack(f"<{n_samples}q", *range(0, n_samples * 512, 512))
    idx_path.write_bytes(idx_data)

    return str(tar_path), str(idx_path)


def scaffold_dataset_dir(
    root:     Path,
    conf:     str   = "public",
    modality: str   = "rgb",
    name:     str   = "test_dataset",
    split:    str   = "train",
    n_shards: int   = 2,
    n_samples_per_shard: int = 8,
    with_metadata: bool = True,
) -> List[str]:
    """
    Create a full dataset directory hierarchy and populate it with synthetic
    shards.  Returns the list of tar paths.

    Layout::

        root/
          <conf>/
            <modality>/
              <name>/
                <split>/
                  shard-000000.tar
                  shard-000000.idx
                  ...
    """
    split_dir = root / conf / modality / name / split
    tar_paths = []
    for i in range(n_shards):
        tar_path, _ = write_shard(
            directory      = split_dir,
            shard_idx      = i,
            n_samples      = n_samples_per_shard,
            with_metadata  = with_metadata,
        )
        tar_paths.append(tar_path)
    return tar_paths
