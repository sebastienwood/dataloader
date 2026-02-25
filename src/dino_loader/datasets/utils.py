import os
import subprocess
import logging
from typing import List

log = logging.getLogger(__name__)

def _extract_jpegs(tar_view: memoryview) -> List[bytes]:
    """
    Highly optimized custom tar extractor directly from a memoryview.
    Skips tarfile module overhead, reading headers directly and extracting JPEGs.
    """
    results: List[bytes] = []
    
    offset = 0
    total_len = len(tar_view)
    
    while offset + 512 <= total_len:
        # Check for end of archive (null block)
        if tar_view[offset] == 0:
            break
            
        # Parse name
        name_end = offset
        while name_end < offset + 100 and tar_view[name_end] != 0:
            name_end += 1
        name = bytes(tar_view[offset:name_end]).decode("utf-8", "ignore").lower()
        
        # Parse size (octal string at offset 124, length 12)
        size_str = bytes(tar_view[offset + 124 : offset + 135]).strip(b" \0")
        try:
            file_size = int(size_str, 8) if size_str else 0
        except ValueError:
            file_size = 0
            
        data_offset = offset + 512
        typeflag = tar_view[offset + 156]
        
        # Check if normal file (0 or '0' = 48) and jpg/jpeg
        if typeflag in (0, 48) and (name.endswith(".jpg") or name.endswith(".jpeg")):
            if data_offset + file_size <= total_len:
                results.append(bytes(tar_view[data_offset : data_offset + file_size]))
                
        # Advance offset (512-byte aligned)
        blocks = (file_size + 511) // 512
        offset += 512 + blocks * 512

    if not results:
        raise RuntimeError(
            "Shard contained no JPEG files. "
            "Check that shards are WebDataset tars with .jpg/.jpeg members."
        )
    return results

def ensure_idx_exists(tar_path: str, idx_path: str):
    """
    Checks if the idx file exists and is newer than the tar file.
    If not, it regenerates the idx file using webdataset's CLI.
    """
    needs_generation = False
    
    if not os.path.exists(idx_path):
        needs_generation = True
        log.info(f"Missing .idx file for {tar_path}, generating...")
    else:
        tar_mtime = os.path.getmtime(tar_path)
        idx_mtime = os.path.getmtime(idx_path)
        if tar_mtime > idx_mtime:
            needs_generation = True
            log.info(f"Stale .idx file for {tar_path} (tar is newer), regenerating...")

    if needs_generation:
        try:
            import sys
            # webdataset provides the `wds2idx` CLI tool.
            # `wds2idx input.tar > output.idx` is the standard usage.
            with open(idx_path, 'wb') as f_out:
                subprocess.run(
                    [sys.executable, "-m", "webdataset.wds2idx", tar_path],
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    check=True
                )
            log.info(f"✅ Generated .idx for {tar_path}")
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to generate idx for {tar_path}: {e.stderr.decode('utf-8', 'ignore')}")
        except Exception as e:
            log.error(f"Failed to execute wds2idx for {tar_path}: {e}")

def validate_webdataset_shard(tar_path: str, idx_path: str) -> bool:
    """
    Validates a webdataset shard by checking:
    1. The tar file exists.
    2. Its .idx is auto-generated and up-to-date.
    3. The tar file contains valid JPEG members by testing extraction on the first few entries.
       (We just do a full parse, but fast enough).
    """
    if not os.path.exists(tar_path):
        return False
        
    # Auto-generate or update the idx file if necessary
    ensure_idx_exists(tar_path, idx_path)
    
    # After generation attempt, verify it actually exists
    if not os.path.exists(idx_path):
        return False

    try:
        with open(tar_path, 'rb') as f:
            tar_bytes = f.read()
            _extract_jpegs(memoryview(tar_bytes))
        return True
    except Exception as e:
        log.warning(f"Shard validation failed for {tar_path}: {e}")
        return False
