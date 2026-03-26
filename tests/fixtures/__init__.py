"""tests.fixtures
==============
Helpers for building synthetic WebDataset shards in memory and on disk.

These are used by all unit tests that require shard data.  Deliberately kept
free of any ``dino_loader`` import so they can be used before the package is
on sys.path (e.g. in ``conftest.py`` before installation).

Public API
----------
make_jpeg_bytes          — encode a solid-colour JPEG in memory
make_shard_tar           — build a WebDataset tar archive in memory
make_minimal_tar_bytes   — bare-bones tar (no valid JPEG, no PIL dep) for low-level tests
write_shard              — write a .tar + .idx pair to disk
write_shm_file           — write a /dev/shm-style header + payload file (for mmap pool tests)
scaffold_dataset_dir     — create the full dataset directory hierarchy
make_spec                — convenience DatasetSpec factory (used across test modules)

Filesystem layout produced by ``scaffold_dataset_dir``
------------------------------------------------------
::

    root/
      <conf>/
        <modality>/
          <name>/
            raw/
            pivot/
            metadata/          ← matches dino_datasets._walk.DATASET_SKELETON_DIRS
            subset_selection/
            outputs/
              <strategy>/
                <split>/
                  shard-000000.tar
                  shard-000000.idx
                  ...
"""

from __future__ import annotations

import io
import json
import struct
import sys
import tarfile
from pathlib import Path

from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
# Synthetic image helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_jpeg_bytes(
    width:   int = 64,
    height:  int = 64,
    color:   tuple[int, int, int] = (128, 64, 32),
    quality: int = 85,
) -> bytes:
    """Return JPEG-encoded bytes for a solid-colour RGB image (requires Pillow)."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_minimal_jpeg_bytes(size: int = 256) -> bytes:
    """Return a minimal JPEG-like byte sequence (SOI marker + padding + EOI marker).

    Does NOT require Pillow.  Useful for low-level tests that only care about
    byte presence in a tar archive, not decodability.
    """
    return b"\xff\xd8" + b"\x00" * size + b"\xff\xd9"


# ══════════════════════════════════════════════════════════════════════════════
# In-memory tar builders
# ══════════════════════════════════════════════════════════════════════════════

def make_shard_tar(
    n_samples:      int = 10,
    with_metadata:  bool = True,
    quality_scores: list[float] | None = None,
    img_width:      int = 64,
    img_height:     int = 64,
) -> bytes:
    """Build a WebDataset-compatible tar archive in memory (requires Pillow).

    Each sample consists of:

    - ``<key>.jpg``  — a JPEG-encoded solid-colour image.
    - ``<key>.json`` — (when *with_metadata* is True) a JSON sidecar with
                       ``{"quality_score": float, "caption": str}``.

    Parameters
    ----------
    n_samples:
        Number of (jpg, [json]) pairs in the archive.
    with_metadata:
        Whether to include ``.json`` sidecars.
    quality_scores:
        Per-sample quality scores.  Defaults to ``1.0`` for all samples.
    img_width / img_height:
        Synthetic image dimensions.

    Returns
    -------
    bytes
        Raw tar archive content.

    """
    if quality_scores is None:
        quality_scores = [1.0] * n_samples
    assert len(quality_scores) == n_samples

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_samples):
            key  = f"sample_{i:06d}"
            jpeg = make_jpeg_bytes(
                width  = img_width,
                height = img_height,
                color  = (
                    i * 25 % 256,
                    (i * 37 + 80) % 256,
                    (i * 53 + 160) % 256,
                ),
            )
            _add_bytes(tf, f"{key}.jpg", jpeg)

            if with_metadata:
                meta = {
                    "quality_score": quality_scores[i],
                    "caption":       f"A test image number {i}",
                    "dedup_hash":    f"hash_{i:08x}",
                }
                _add_bytes(tf, f"{key}.json", json.dumps(meta).encode())

    return buf.getvalue()


def make_minimal_tar_bytes(n_samples: int = 4) -> bytes:
    """Create an in-memory WebDataset-style tar with minimal JPEG-like entries.

    Does NOT require Pillow — entries contain ``make_minimal_jpeg_bytes()``
    payloads.  Intended for low-level tests (mmap pool, async prefetch, …)
    that do not decode images.

    Parameters
    ----------
    n_samples:
        Number of JPEG entries to embed.

    Returns
    -------
    bytes
        Raw tar archive content.

    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_samples):
            data = make_minimal_jpeg_bytes()
            info = tarfile.TarInfo(name=f"sample-{i:06d}.jpg")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add a raw bytes payload as a named entry in an open TarFile."""
    info      = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


# ══════════════════════════════════════════════════════════════════════════════
# On-disk shard scaffolding
# ══════════════════════════════════════════════════════════════════════════════

def write_shard(
    directory:      Path,
    shard_idx:      int = 0,
    n_samples:      int = 10,
    with_metadata:  bool = True,
    quality_scores: list[float] | None = None,
) -> tuple[str, str]:
    """Write a synthetic ``.tar`` shard and its companion ``.idx`` file to
    *directory*.

    The ``.idx`` file uses the ``wds2idx`` binary format: a flat sequence of
    little-endian int64 byte offsets (8 bytes per entry).  Synthetic offsets
    are monotonically increasing (512-byte stride) for test purposes.

    Returns
    -------
    (tar_path, idx_path)
        Both as absolute string paths.

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

    # Binary idx: n_samples × little-endian int64 offsets.
    idx_data = struct.pack(f"<{n_samples}q", *range(0, n_samples * 512, 512))
    idx_path.write_bytes(idx_data)

    return str(tar_path), str(idx_path)


def write_shm_file(path: Path, data: bytes) -> None:
    """Write a /dev/shm-style shard file: a 16-byte header followed by *data*.

    Header layout (two little-endian uint64)::

        [data_len : u64][ready_magic : u64]

    This matches the format that ``NodeSharedShardCache._write()`` produces and
    ``_MmapPool.acquire()`` consumes.  Useful for unit tests that exercise the
    mmap pool without a running ``NodeSharedShardCache``.

    Parameters
    ----------
    path:
        Destination file path (parent directory must exist).
    data:
        Raw shard payload to embed after the header.

    """
    _HDR_FMT     = "<QQ"
    _READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D

    with open(path, "wb") as f:
        f.write(struct.pack(_HDR_FMT, len(data), _READY_MAGIC))
        f.write(data)


def scaffold_dataset_dir(
    root:                Path,
    conf:                str = "public",
    modality:            str = "rgb",
    name:                str = "test_dataset",
    split:               str = "train",
    strategy:            str = "default",
    n_shards:            int = 2,
    n_samples_per_shard: int = 8,
    with_metadata:       bool = True,
) -> list[str]:
    """Create the full dataset directory hierarchy and populate it with synthetic
    shards.  Returns the list of absolute ``.tar`` paths.

    The subdirectory names match ``dino_datasets._walk.DATASET_SKELETON_DIRS``
    exactly: ``raw``, ``pivot``, ``metadata``, ``subset_selection``, ``outputs``.
    Note: the previous ``metadonnees`` name was incorrect and has been removed.

    Layout produced::

        root/
          <conf>/
            <modality>/
              <name>/
                raw/
                pivot/
                metadata/
                subset_selection/
                outputs/
                  <strategy>/
                    <split>/
                      shard-000000.tar
                      shard-000000.idx
                      ...

    Parameters
    ----------
    root:
        Filesystem root (typically ``tmp_path`` in pytest).
    conf:
        Confidentiality label (e.g. ``"public"``, ``"private"``).
    modality:
        Modality label (e.g. ``"rgb"``, ``"multispectral"``).
    name:
        Dataset name.
    split:
        Split name (e.g. ``"train"``, ``"val"``).
    strategy:
        Strategy folder name (default: ``"default"``).
    n_shards:
        Number of synthetic shards to write.
    n_samples_per_shard:
        Samples per shard.
    with_metadata:
        Whether to include ``.json`` sidecars in each shard.

    Returns
    -------
    list of str
        Absolute paths to the generated ``.tar`` files.

    """
    root         = Path(root)
    dataset_root = root / conf / modality / name

    # Skeleton dirs must match dino_datasets._walk.DATASET_SKELETON_DIRS.
    # "outputs" is created implicitly via split_dir below.
    for subdir in ("raw", "pivot", "metadata", "subset_selection"):
        (dataset_root / subdir).mkdir(parents=True, exist_ok=True)

    split_dir = dataset_root / "outputs" / strategy / split
    split_dir.mkdir(parents=True, exist_ok=True)

    tar_paths: list[str] = []
    for i in range(n_shards):
        tar_path, _ = write_shard(
            directory     = split_dir,
            shard_idx     = i,
            n_samples     = n_samples_per_shard,
            with_metadata = with_metadata,
        )
        tar_paths.append(tar_path)

    return tar_paths


# ══════════════════════════════════════════════════════════════════════════════
# DatasetSpec factory
# ══════════════════════════════════════════════════════════════════════════════

def make_spec(name: str, tar_paths: list, weight: float = 1.0, **kwargs):
    """Convenience factory for ``DatasetSpec``.

    Imported by ``conftest.py`` and used directly in test modules.  Defined
    here (in ``fixtures``) rather than in ``conftest.py`` so that test modules
    which do not need pytest fixtures can import it without pulling in the
    entire conftest machinery.

    Parameters
    ----------
    name:
        Dataset name (passed to ``DatasetSpec.name``).
    tar_paths:
        List of shard paths (passed to ``DatasetSpec.shards``).
    weight:
        Mixing weight (default: ``1.0``).
    **kwargs:
        Any additional keyword arguments forwarded to ``DatasetSpec``.

    Returns
    -------
    DatasetSpec

    """
    _src = str(Path(__file__).parent.parent / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from dino_datasets import DatasetSpec
    return DatasetSpec(name=name, shards=tar_paths, weight=weight, **kwargs)
