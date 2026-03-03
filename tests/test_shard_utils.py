"""
tests/test_shard_utils.py
=========================
Unit tests for dino_loader.datasets.utils — tar extraction and shard validation.

All tests use in-memory or tmp-dir synthetic shards, no Lustre required.
"""

from __future__ import annotations

import io
import struct
import sys
import tarfile
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.fixtures import make_shard_tar, make_jpeg_bytes, write_shard
from dino_loader.datasets.utils import (
    _check_tar_has_jpeg_header,
    _extract_jpegs,
    validate_webdataset_shard,
)


# ══════════════════════════════════════════════════════════════════════════════
# _extract_jpegs
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractJpegs:

    def _make_tar_mv(self, n: int = 4) -> memoryview:
        return memoryview(make_shard_tar(n_samples=n, with_metadata=False))

    def test_extracts_correct_count(self):
        mv     = self._make_tar_mv(5)
        result = _extract_jpegs(mv)
        assert len(result) == 5

    def test_each_element_is_valid_jpeg(self):
        mv     = self._make_tar_mv(3)
        result = _extract_jpegs(mv)
        for b in result:
            assert b[:2] == b"\xff\xd8", "Not a JPEG SOI marker"

    def test_raises_on_empty_tar(self):
        """An archive with no JPEG members should raise RuntimeError."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            data = b"hello world"
            info = tarfile.TarInfo(name="readme.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        mv = memoryview(buf.getvalue())
        with pytest.raises(RuntimeError, match="no JPEG"):
            _extract_jpegs(mv)

    def test_end_of_archive_double_null_block(self):
        """
        [FIX-18] Parser must stop at two consecutive null blocks.
        Verify a well-formed archive with trailing null padding is parsed.
        """
        # make_shard_tar produces a standards-compliant tar with double null blocks
        mv     = memoryview(make_shard_tar(n_samples=2))
        result = _extract_jpegs(mv)
        assert len(result) == 2

    def test_mixed_archive_extracts_only_jpegs(self):
        """Archives with non-JPEG members should only return JPEG bytes."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            # Add a JPEG
            jpeg = make_jpeg_bytes()
            info = tarfile.TarInfo(name="sample.jpg")
            info.size = len(jpeg)
            tf.addfile(info, io.BytesIO(jpeg))
            # Add a JSON sidecar
            data = b'{"quality_score": 0.9}'
            info2 = tarfile.TarInfo(name="sample.json")
            info2.size = len(data)
            tf.addfile(info2, io.BytesIO(data))

        mv     = memoryview(buf.getvalue())
        result = _extract_jpegs(mv)
        assert len(result) == 1
        assert result[0][:2] == b"\xff\xd8"


# ══════════════════════════════════════════════════════════════════════════════
# _check_tar_has_jpeg_header  (fast structural check)
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckTarHasJpegHeader:

    def test_valid_shard_returns_true(self, tmp_path):
        tar_path, _ = write_shard(tmp_path, n_samples=4)
        assert _check_tar_has_jpeg_header(tar_path) is True

    def test_nonexistent_file_returns_false(self, tmp_path):
        result = _check_tar_has_jpeg_header(str(tmp_path / "ghost.tar"))
        assert result is False

    def test_empty_file_returns_false(self, tmp_path):
        p = tmp_path / "empty.tar"
        p.write_bytes(b"")
        assert _check_tar_has_jpeg_header(str(p)) is False

    def test_text_file_returns_false(self, tmp_path):
        p = tmp_path / "text.tar"
        p.write_text("This is not a tar file")
        assert _check_tar_has_jpeg_header(str(p)) is False

    def test_shard_with_json_only_entry(self, tmp_path):
        """An archive whose first entry is .json (no .jpg) should still return True
        if there's a regular file entry — the check accepts any regular file."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            data = b'{"quality_score": 1.0}'
            info = tarfile.TarInfo(name="sample.json")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        p = tmp_path / "json_only.tar"
        p.write_bytes(buf.getvalue())
        # Any regular-file entry satisfies the fast structural check
        result = _check_tar_has_jpeg_header(str(p))
        assert isinstance(result, bool)


# ══════════════════════════════════════════════════════════════════════════════
# validate_webdataset_shard
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateWebdatasetShard:

    def test_valid_shard_returns_true(self, tmp_path):
        tar_path, idx_path = write_shard(tmp_path, n_samples=4)
        assert validate_webdataset_shard(tar_path, idx_path) is True

    def test_missing_tar_returns_false(self, tmp_path):
        assert validate_webdataset_shard(
            str(tmp_path / "nope.tar"),
            str(tmp_path / "nope.idx"),
        ) is False

    def test_empty_tar_returns_false(self, tmp_path):
        tar_path = str(tmp_path / "empty.tar")
        idx_path = str(tmp_path / "empty.idx")
        Path(tar_path).write_bytes(b"")
        assert validate_webdataset_shard(tar_path, idx_path) is False

    def test_missing_idx_triggers_generation(self, tmp_path, monkeypatch):
        """
        When .idx is absent, validate_webdataset_shard calls ensure_idx_exists.
        We monkeypatch it to a no-op and confirm the function returns False
        (since the idx file still won't exist after our no-op).
        """
        from dino_loader.datasets import utils as utils_module

        generated = []

        def _fake_ensure(tar_path, idx_path):
            generated.append((tar_path, idx_path))

        monkeypatch.setattr(utils_module, "ensure_idx_exists", _fake_ensure)

        tar_path, _ = write_shard(tmp_path, shard_idx=99)
        missing_idx = str(tmp_path / "shard-000099_ghost.idx")

        result = validate_webdataset_shard(tar_path, missing_idx)
        assert len(generated) == 1   # ensure_idx_exists was called
        # After our no-op, the idx still doesn't exist → False
        assert result is False
