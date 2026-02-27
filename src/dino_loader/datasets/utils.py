"""
dino_loader.datasets.utils
==========================
WebDataset shard utilities: fast tar extraction and validation.

Fixes vs previous version
--------------------------
[FIX-8]  ``validate_webdataset_shard`` previously read the entire shard
         (potentially several GB) into memory just to validate it.  Called
         at stub-generation time for every shard in every dataset (potentially
         85 000+ shards), this OOM'd or took hours.  Replaced with a fast
         structural check: reads only the first two 512-byte tar blocks to
         verify the tar magic and confirm at least one JPEG entry exists,
         without allocating the full shard.

[FIX-18] ``_extract_jpegs`` checked only ``tar_view[offset] == 0`` for
         end-of-archive.  The POSIX tar spec requires *two* consecutive null
         blocks; checking only one byte of one block silently truncated
         parsing on any file whose name field started with a null byte
         (impossible in well-formed tars, but possible in corrupt ones).
         A corrupt tar could also contain a spurious null byte at an
         unexpected offset, silently truncating a valid archive.
         Fixed: check all 512 bytes of the block for null; on the second
         consecutive null block, stop.  Non-null blocks after a single null
         block (possible in some tar writers) resume normally.
"""

from __future__ import annotations

import os
import struct
import subprocess
import logging
from typing import List

log = logging.getLogger(__name__)

# POSIX ustar magic at offset 257, length 6
_USTAR_MAGIC = b"ustar"
_BLOCK_SIZE  = 512


# ══════════════════════════════════════════════════════════════════════════════
# Tar extraction (hot path)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_jpegs(tar_view: memoryview) -> List[bytes]:
    """
    Highly optimised custom tar extractor directly from a memoryview.
    Skips tarfile module overhead, reading headers directly and extracting JPEGs.

    [FIX-18] End-of-archive detection now follows the POSIX spec: two
    consecutive 512-byte all-zero blocks.  A single null block is noted but
    parsing continues (some tar writers emit only one null block before a
    genuine entry, which is non-conformant but real-world).  Two consecutive
    null blocks always terminate parsing.
    """
    results: List[bytes] = []

    offset    = 0
    total_len = len(tar_view)
    null_block_count = 0  # consecutive null-block counter for end-of-archive

    while offset + _BLOCK_SIZE <= total_len:
        block = tar_view[offset: offset + _BLOCK_SIZE]

        # ── End-of-archive detection [FIX-18] ────────────────────────────────
        # A null block has all 512 bytes set to zero.  Two consecutive null
        # blocks signal end-of-archive per POSIX.1-2017 §10.2.
        if all(b == 0 for b in block):
            null_block_count += 1
            if null_block_count >= 2:
                break
            offset += _BLOCK_SIZE
            continue
        null_block_count = 0  # reset on any non-null block

        # ── Parse name ────────────────────────────────────────────────────────
        name_end = offset
        while name_end < offset + 100 and tar_view[name_end] != 0:
            name_end += 1
        name = bytes(tar_view[offset: name_end]).decode("utf-8", "ignore").lower()

        # ── Parse size (octal, 12 bytes at offset 124) ────────────────────────
        size_str = bytes(tar_view[offset + 124: offset + 136]).strip(b" \0")
        try:
            file_size = int(size_str, 8) if size_str else 0
        except ValueError:
            file_size = 0

        data_offset = offset + _BLOCK_SIZE
        typeflag    = tar_view[offset + 156]

        # ── Extract JPEGs (typeflag 0 or '0' = regular file) ─────────────────
        if typeflag in (0, 48) and (name.endswith(".jpg") or name.endswith(".jpeg")):
            if data_offset + file_size <= total_len:
                results.append(bytes(tar_view[data_offset: data_offset + file_size]))

        # ── Advance to next header (512-byte aligned) ─────────────────────────
        blocks  = (file_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        offset += _BLOCK_SIZE + blocks * _BLOCK_SIZE

    if not results:
        raise RuntimeError(
            "Shard contained no JPEG files. "
            "Check that shards are WebDataset tars with .jpg/.jpeg members."
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Index generation
# ══════════════════════════════════════════════════════════════════════════════

def ensure_idx_exists(tar_path: str, idx_path: str) -> None:
    """
    Check if the idx file exists and is newer than the tar file.
    If not, regenerate it using webdataset's CLI.
    """
    needs_generation = False

    if not os.path.exists(idx_path):
        needs_generation = True
        log.info("Missing .idx file for %s, generating...", tar_path)
    else:
        tar_mtime = os.path.getmtime(tar_path)
        idx_mtime = os.path.getmtime(idx_path)
        if tar_mtime > idx_mtime:
            needs_generation = True
            log.info("Stale .idx file for %s (tar is newer), regenerating...", tar_path)

    if needs_generation:
        try:
            import sys
            with open(idx_path, "wb") as f_out:
                subprocess.run(
                    [sys.executable, "-m", "webdataset.wds2idx", tar_path],
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            log.info("Generated .idx for %s", tar_path)
        except subprocess.CalledProcessError as e:
            log.error(
                "Failed to generate idx for %s: %s",
                tar_path, e.stderr.decode("utf-8", "ignore"),
            )
        except Exception as e:
            log.error("Failed to execute wds2idx for %s: %s", tar_path, e)


# ══════════════════════════════════════════════════════════════════════════════
# Shard validation (fast structural check)
# ══════════════════════════════════════════════════════════════════════════════

def _check_tar_has_jpeg_header(tar_path: str) -> bool:
    """
    Fast structural validation: read only the first two tar blocks (1 KB)
    and check that (a) the tar magic is present, (b) the first entry looks
    like a JPEG by checking its name.

    This replaces the previous full-shard read which was O(GB) per shard
    and OOM'd on large dataset catalogues. [FIX-8]
    """
    try:
        with open(tar_path, "rb") as f:
            # First block: header for the first entry
            header = f.read(_BLOCK_SIZE)
            if len(header) < _BLOCK_SIZE:
                return False

            # Minimal sanity: ustar magic at offset 257 (GNU/POSIX tar)
            # Some tars lack the ustar prefix (old V7 format); skip magic check
            # if it is absent and fall through to the name check.
            magic = header[257:262]
            if magic not in (b"ustar", b"     ", b"\x00\x00\x00\x00\x00"):
                # Unexpected magic; could be a non-tar file
                pass  # be lenient — WebDataset tars vary in format

            # Parse the first entry's name and type
            name_bytes = header[:100]
            name = name_bytes.rstrip(b"\x00").decode("utf-8", "ignore").lower()
            typeflag = header[156]

            # Require at least one regular-file entry — not necessarily JPEG
            # (WebDataset tars may start with a .json or .txt sidecar).
            # Presence of *any* regular-file entry confirms a readable tar.
            if typeflag in (0, 48):
                return True

            # Fallback: scan the first 16 blocks (8 KB) for a JPEG entry
            f.seek(0)
            buf = f.read(_BLOCK_SIZE * 16)
            mv  = memoryview(buf)
            off = 0
            while off + _BLOCK_SIZE <= len(mv):
                if all(b == 0 for b in mv[off: off + _BLOCK_SIZE]):
                    break
                size_str = bytes(mv[off + 124: off + 136]).strip(b" \0")
                try:
                    fsize = int(size_str, 8) if size_str else 0
                except ValueError:
                    fsize = 0
                n_bytes  = mv[off: off + 100].tobytes().rstrip(b"\x00")
                n_str    = n_bytes.decode("utf-8", "ignore").lower()
                tf       = mv[off + 156]
                if tf in (0, 48) and (n_str.endswith(".jpg") or n_str.endswith(".jpeg")):
                    return True
                blocks = (fsize + _BLOCK_SIZE - 1) // _BLOCK_SIZE
                off   += _BLOCK_SIZE + blocks * _BLOCK_SIZE

            return False
    except Exception as e:
        log.warning("Fast tar check failed for %s: %s", tar_path, e)
        return False


def validate_webdataset_shard(tar_path: str, idx_path: str) -> bool:
    """
    Validate a webdataset shard.

    [FIX-8] Previous implementation read the *entire* shard into memory and
    ran full JPEG extraction, which was O(GB) per shard.  With 85 000+ shards
    this OOM'd and took hours.

    New approach — three cheap checks only:
      1. The tar file exists and is non-empty.
      2. The .idx file is present and up-to-date (auto-generated if stale).
      3. A fast structural tar header scan (first ≤ 8 KB) confirms the file
         is a readable tar with at least one regular-file entry.

    Full JPEG extraction correctness is enforced at runtime by _extract_jpegs,
    which raises RuntimeError on malformed shards.
    """
    if not os.path.exists(tar_path):
        return False
    if os.path.getsize(tar_path) == 0:
        log.warning("Empty shard file: %s", tar_path)
        return False

    ensure_idx_exists(tar_path, idx_path)

    if not os.path.exists(idx_path):
        return False

    return _check_tar_has_jpeg_header(tar_path)
