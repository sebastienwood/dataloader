# dino_loader

**HPC-grade DINOv3 data pipeline for B200 / GB200 NVL72 clusters.**

`dino_loader` is a production-ready PyTorch dataloader designed so that GPU
training is *never* bottlenecked by data ingestion.  It was built to train
DINO-style self-supervised vision models at petascale — hundreds of GPUs,
hundreds of millions of images, weeks of wall-clock training — on modern SLURM
HPC clusters backed by Lustre parallel filesystems.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Architecture overview](#architecture-overview)
3. [Multi-node data-flow diagram](#multi-node-data-flow-diagram)
4. [Design decisions (annotated)](#design-decisions-annotated)
5. [Configuration reference](#configuration-reference)
6. [Dataset hub — IDE integration](#dataset-hub--ide-integration)
7. [Real-time monitoring](#real-time-monitoring)
8. [Checkpointing and resume](#checkpointing-and-resume)
9. [Known bugs & fixes applied](#known-bugs--fixes-applied)
10. [Extending the loader](#extending-the-loader)
11. [Installation](#installation)

---

## Quick start

```python
from dino_loader import DatasetSpec, DINOAugConfig, DINODataLoader, LoaderConfig, slurm_init

env = slurm_init()          # reads SLURM env vars, sets NCCL knobs, init process group

specs = [
    DatasetSpec("laion2b",   shards=[...], weight=0.5),
    DatasetSpec("datacomp1b",shards=[...], weight=0.3),
    DatasetSpec("imagenet22k",shards=[...],weight=0.2),
]

loader = DINODataLoader(
    specs            = specs,
    batch_size       = 512,
    aug_cfg          = DINOAugConfig(n_local_crops=8),
    config           = LoaderConfig(node_shm_gb=256, shard_prefetch_window=128),
    device_id        = env.local_rank,
    steps_per_epoch  = 200_000,
    resume           = True,            # auto-load latest checkpoint
)

for epoch in range(100):
    loader.set_epoch(epoch)             # re-shuffles shards — must be called!
    for step, batch in enumerate(loader):
        # batch.global_crops : list of 2  tensors (BF16 or FP8 on GPU)
        # batch.local_crops  : list of 8  tensors
        train_step(batch)
        loader.checkpoint(step)         # saves every N steps (rank 0 only)

    # Dynamic curriculum — shift toward curated data over time
    if epoch == 10:
        loader.set_weights([0.4, 0.4, 0.2])
```

### SLURM submission

```bash
# GB200 NVL72 (72 GPUs per rack, 4 racks = 288 GPUs total)
sbatch --nodes=4 --ntasks-per-node=72 --gres=gpu:72 \
       --cpus-per-task=4 --mem=2048G \
       --wrap="python train.py"

# B200 PCIe (8 GPUs per node, 32 nodes = 256 GPUs)
sbatch --nodes=32 --ntasks-per-node=8 --gres=gpu:8 \
       --cpus-per-task=16 --mem=512G \
       --wrap="python train.py"
```

---

## Architecture overview

The pipeline is a **5-stage assembly line**, each stage running concurrently
with the others.  No stage waits for the next; the only back-pressure is
intentional.

```
Stage 1  ──  Shard I/O          (Lustre → /dev/shm, async, node-master only)
Stage 2  ──  JPEG Extraction    (tar parsing → JPEG byte lists, thread pool)
Stage 3  ──  DALI Augmentation  (decode + augment on GPU, pipeline_def)
Stage 4  ──  H2D Transfer       (pinned host → GPU, dedicated CUDA stream)
Stage 5  ──  FP8 Quantisation   (BF16 → FP8 E4M3 with TE metadata, optional)
```

---

## Multi-node data-flow diagram

The diagram below shows a **4-node SLURM job** (e.g. GB200 NVL72, 4 × 72 GPUs
= 288 ranks total).  Only two nodes are drawn; all others are identical.

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  LUSTRE PARALLEL FILESYSTEM  (shared across all nodes)                          ║
║                                                                                  ║
║   /lustre/laion2b/shard-000000.tar  shard-000001.tar  shard-000002.tar  ...     ║
║   /lustre/datacomp/shard-000000.tar ...                                          ║
╚══════════════╤═══════════════════════════════════╤═══════════════════════════════╝
               │  TCP/IB (one stream per node)     │
               ▼                                   ▼
╔══════════════════════════════╗   ╔══════════════════════════════╗
║         NODE  0              ║   ║         NODE  1              ║
║                              ║   ║                              ║
║  ┌─────────────────────────┐ ║   ║  ┌─────────────────────────┐ ║
║  │  local_rank 0           │ ║   ║  │  local_rank 0           │ ║
║  │  ┌───────────────────┐  │ ║   ║  │  ┌───────────────────┐  │ ║
║  │  │ NodeSharedShard   │  │ ║   ║  │  │ NodeSharedShard   │  │ ║
║  │  │ Cache             │  │ ║   ║  │  │ Cache             │  │ ║
║  │  │ (node master)     │  │ ║   ║  │  │ (node master)     │  │ ║
║  │  │                   │  │ ║   ║  │  │                   │  │ ║
║  │  │  asyncio loop  ◄──┼──┼─╫──►│  │  asyncio loop     │  │ ║
║  │  │  aiofiles read    │  │ ║   ║  │  aiofiles read    │  │ ║
║  │  │  LRU eviction     │  │ ║   ║  │  LRU eviction     │  │ ║
║  │  └────────┬──────────┘  │ ║   ║  └────────┬──────────┘  │ ║
║  │           │ writes      │ ║   ║           │ writes      │ ║
║  │           ▼             │ ║   ║           ▼             │ ║
║  │  ┌────────────────────┐ │ ║   ║  ┌────────────────────┐ │ ║
║  │  │  /dev/shm/<job>/   │ │ ║   ║  │  /dev/shm/<job>/   │ │ ║
║  │  │  (tmpfs, ~256 GB)  │ │ ║   ║  │  (tmpfs, ~256 GB)  │ │ ║
║  │  │                    │ │ ║   ║  │                    │ │ ║
║  │  │  abc123.shm  ◄─────┼─┼─╫──►│  inotify wait      │ │ ║
║  │  │  def456.shm        │ │ ║   ║  │  (no busy-spin)   │ │ ║
║  │  │  ...               │ │ ║   ║  │  ...              │ │ ║
║  │  └─────────┬──────────┘ │ ║   ║  └─────────┬─────────┘ │ ║
║  └────────────┼────────────┘ ║   ║  └──────────┼──────────┘ ║
║               │ mmap         ║   ║             │ mmap       ║
║               ▼              ║   ║             ▼            ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │         ALL LOCAL RANKS  (e.g. 72 on NVL72)           │ ║
║  │                                                        │ ║
║  │  ShardIterator (per dataset, per rank)                 │ ║
║  │  ┌──────────────────────────────────────────────────┐  │ ║
║  │  │  ThreadPoolExecutor (4 workers default)          │  │ ║
║  │  │  ┌──────────────────────────────────────────┐    │  │ ║
║  │  │  │  _fetch_and_extract(shard_path)          │    │  │ ║
║  │  │  │  cache.get_view() → _extract_jpegs()     │    │  │ ║
║  │  │  │  → deque[bytes]  (JPEG byte lists)       │    │  │ ║
║  │  │  └──────────────────────────────────────────┘    │  │ ║
║  │  └──────────────────────────────────────────────────┘  │ ║
║  │                       │                                 │ ║
║  │  MixingSource  ◄──────┘  (weighted random dataset pick) │ ║
║  │  (DALI ExternalSource callback)                         │ ║
║  └────────────────────────────┬───────────────────────────┘ ║
║                               │  list[np.ndarray]           ║
║                               ▼                             ║
║  ┌─────────────────────────────────────────────────────────┐║
║  │  NVIDIA DALI Pipeline  (GPU)                            │║
║  │                                                         │║
║  │  HW JPEG decode (nvjpeg ASIC)                          │║
║  │  Random resized crop  ──►  Resize (cubic)              │║
║  │  H-flip  ──►  Color jitter  ──►  Grayscale             │║
║  │  Gaussian blur  ──►  Solarise (2nd global crop only)   │║
║  │  Normalise  ──►  HWC→CHW transpose                     │║
║  │                                                         │║
║  │  Output: 2 global crops + 8 local crops (BF16 GPU)     │║
║  └───────────────────────────┬─────────────────────────────┘║
║                              │                              ║
║                              ▼                              ║
║  ┌─────────────────────────────────────────────────────────┐║
║  │  AsyncPrefetchIterator  ──►  H2DStream                  │║
║  │  (CUDA stream, non-blocking, overlapped with compute)   │║
║  └───────────────────────────┬─────────────────────────────┘║
║                              │                              ║
║                              ▼                              ║
║  ┌─────────────────────────────────────────────────────────┐║
║  │  FP8Formatter  (Transformer Engine)        [optional]   │║
║  │  BF16 → FP8 E4M3  + FP8TensorMeta                      │║
║  │  Rolling amax window (length 16, TE convention)        │║
║  └───────────────────────────┬─────────────────────────────┘║
║                              │  Batch(global_crops, local_crops)║
║                              ▼                              ║
║           TRAINING LOOP  (ViT forward / backward)           ║
╚══════════════════════════════════════════════════════════════╝

Legend
──────
  ◄──►  Shared-memory access (same host, zero-copy)
  ────  Data flow (in-process)
  ═══   Network boundary (IB / TCP between nodes)
  [/dev/shm]  Linux tmpfs — lives in DRAM, kernel-managed,
              appears as a regular filesystem path
```

---

## Design decisions (annotated)

### Why `/dev/shm` as the shard cache?

`/dev/shm` is a **Linux tmpfs** (temporary filesystem backed entirely by
DRAM).  It is accessible by all processes on the same node as a regular
filesystem path, which means:

- **One network read per shard, not one per GPU.** On an NVL72 node with 72
  GPUs, only `local_rank 0` reads each shard from Lustre over InfiniBand.  The
  other 71 ranks read from `/dev/shm` — which is as fast as reading from RAM
  (~200 GB/s via cache) and generates zero network traffic.

- **Cross-process sharing without custom IPC.** Because it looks like a
  filesystem, any rank can `open()` and `mmap()` the cached file.  No shared
  memory handles, semaphores, or pipes are needed.

- **LRU eviction is trivial.** Eviction is `unlink()` — the kernel reclaims
  the pages when no process has the file open.

- **Atomic visibility via `rename()`.** The master writes to a `.tmp` file
  and calls `rename()` (POSIX-atomic) to publish the file.  Readers either
  see the complete file or nothing.  A magic sentinel in the header (`0xDEADBEEFCAFEF00D`) provides a second layer of integrity checking.

- **inotify instead of polling.** Non-master ranks call `inotify_add_watch()`
  on the shard's parent directory and block in `select()`.  This uses zero CPU
  while waiting and wakes up within microseconds of the rename.  This is
  critical on NVL72 where 71 ranks could otherwise spin-wait, wasting 71 CPU
  cores.

### Why async I/O with `aiofiles` for Lustre reads?

Lustre metadata latency is high (~1–5 ms per operation), and stripe I/O
bandwidth is maximised with many concurrent readers.  A single synchronous
reader would saturate at ~1 shard/second.  The node master uses an `asyncio`
event loop with up to `shard_prefetch_window` (default 128) concurrent
downloads — limited by an `asyncio.Semaphore` to avoid Lustre overload —
achieving ~10–50 GB/s aggregate node bandwidth on a well-configured IB fabric.

### Why NVIDIA DALI instead of PyTorch's `DataLoader`?

PyTorch's default `DataLoader` decodes images on CPU and then transfers to
GPU.  On B200/GB200, this path is limited by:

- **CPU decoding throughput**: ~1000 images/sec per worker with Pillow/libjpeg.
- **H2D bandwidth**: PCIe 5.0 ×16 peaks at ~64 GB/s but is shared across all
  workers.

DALI decodes JPEG images directly on the GPU using the **nvjpeg hardware ASIC**
on Blackwell (B200/GB200), which achieves >3× the throughput of CPU decoding
while keeping the PCIe bus free.  Augmentations (crop, flip, color jitter,
blur, normalisation) are also fused into a single GPU kernel pass, eliminating
intermediate CPU↔GPU round-trips.

The `prefetch_queue_depth` (`cpu_size=8, gpu_size=6`) ensures the GPU
augmentation pipeline is never starved by the CPU-side JPEG feed.

### Why `ThreadPoolExecutor` for tar extraction?

Parsing tar headers and copying JPEG bytes from a `memoryview` is
CPU-bound but very fast (~100 MB/s per thread on modern hardware).  A thread
pool allows overlapping extraction of multiple shards — while one shard's
JPEGs are being consumed by DALI, the next shard is already being parsed in
the background.

The `_EXTRACTION_DEPTH = 2` constant means two shards are always in-flight.
The default `num_workers = 4` ensures that even when 2 workers are blocked
waiting for cold shards (not yet in `/dev/shm`), the remaining 2 can process
warm shards, preventing DALI starvation.

### Why `memoryview` for zero-copy tar parsing?

`_extract_jpegs()` receives a `memoryview` from an `mmap`'d shard file.
Slicing a `memoryview` does **not copy bytes** — it creates a view into the
same memory.  This means parsing a 4 GB shard tar produces no intermediate
copies until the final `bytes(tar_view[start:end])` for each individual JPEG.

### Why weighted mixing with `random.choices()`?

`random.choices()` with `k=batch_size` selects `batch_size` dataset indices in
a single call, proportional to the normalised weights.  This is O(k) and
thread-safe once the weight list is read atomically.  The mixing weights can be
updated at any time from the training thread while DALI's prefetch thread
continues consuming — changes take effect on the next batch boundary.

### Why BF16 throughout the augmentation pipeline?

BF16 (Brain Float 16) has the same 8-bit exponent range as FP32, making it
safe for pixel arithmetic (0–255 range) without overflow.  Blackwell tensor
cores natively execute BF16 math, so all DALI ops run at hardware peak.
The final FP8 quantisation uses a rolling amax window identical to Transformer
Engine's internal convention, meaning the `FP8TensorMeta` objects can be passed
directly into `te.fp8_autocast()` blocks without conversion.

### Why per-epoch shard reshuffling?

Without per-epoch reshuffling, every epoch sees the exact same shard order,
causing the model to see the same image at the same training step in every
epoch.  This degrades convergence — the model learns dataset-ordering
correlations rather than generalising.  `set_epoch(epoch)` propagates to all
`ShardIterator.reset_epoch()` calls, which re-shuffle using
`(base_seed + rank + epoch * 1_000_003)`, guaranteeing:
- Reproducibility (given same seed and epoch number).
- Per-rank diversity (different ranks see different shards).
- Per-epoch diversity (different epochs see different orders).

### Why JSON for checkpoints (not pickle)?

On HPC clusters, different nodes may run different Python or library versions
(different conda environments, different MPI wrappers).  Python pickle embeds
class metadata and can silently break when unpickling in a different
environment.  JSON is a stable, version-independent text format — any Python
3.x process can read a checkpoint written by any other.  The checkpoint stores
only the minimal state needed: step, epoch, dataset names, and mixing weights.

### Why NCCL topology-aware configuration?

NCCL defaults are designed for a generic PCIe cluster.  On NVL72:
- InfiniBand is absent (all-reduce happens over NVLink-C2C).
- `NCCL_IB_DISABLE=1` prevents NCCL from trying to use IB and falling back
  slowly.
- `NCCL_NVLS_ENABLE=1` enables NVLink Switch reductions, which are 4× faster
  than ring-allreduce for large all-reduces.
- `NCCL_PROTO=LL128` uses low-latency 128-byte protocol, optimal for the
  low-latency NVLink fabric.

The `verify_interconnect()` health check runs a canary all-reduce at startup to
detect degraded links (e.g. a flaky IB cable or a down NVLink lane) before the
long training run starts.

---

## Configuration reference

All knobs live in `LoaderConfig`.  Sensible defaults are provided for GB200
NVL72; adjust for your cluster.

| Field | Default | Description |
|---|---|---|
| `node_shm_gb` | `128.0` | `/dev/shm` budget per node in GB.  Set to ~50% of node RAM. |
| `shard_prefetch_window` | `64` | Max concurrent Lustre → `/dev/shm` downloads. |
| `shard_timeout_s` | `300.0` | Max seconds a non-master rank waits for a shard to appear. |
| `shard_extraction_workers` | `4` | Thread-pool workers for tar → JPEG extraction.  Set to `≥ prefetch_window / 16`, min 4. |
| `dali_cpu_queue` | `8` | DALI CPU-side prefetch queue depth. |
| `dali_gpu_queue` | `6` | DALI GPU-side prefetch queue depth. |
| `dali_num_threads` | `8` | DALI CPU worker threads for pre-decode ops. |
| `hw_decoder_load` | `0.90` | Fraction of JPEG decode sent to nvjpeg HW ASIC (0–1). |
| `use_fp8_output` | `True` | Quantise output to FP8 E4M3 with Transformer Engine metadata. |
| `output_dtype` | `"bf16"` | Intermediate dtype (`"bf16"` or `"fp32"`). |
| `checkpoint_dir` | `"/checkpoint/dino/dl"` | Where to write `.json` checkpoint files. |
| `checkpoint_every_steps` | `500` | Checkpoint frequency (rank 0 only). |
| `force_topology` | `None` | Override topology detection: `"nvl72"` or `"pcie"`. |
| `seed` | `0` | Base random seed for shard shuffling and augmentation. |
| `shuffle_buffer` | `2000` | (Reserved) In-memory shuffle buffer depth. |

### Augmentation: `DINOAugConfig`

Defaults match the DINOv2 paper (Oquab et al., 2023), §A.1.

| Field | Default | Description |
|---|---|---|
| `n_global_crops` | `2` | Number of large (224px) crops per image. |
| `n_local_crops` | `8` | Number of small (96px) crops per image. |
| `global_crop_size` | `224` | Pixel size of global crops. |
| `local_crop_size` | `96` | Pixel size of local crops. |
| `global_crops_scale` | `(0.32, 1.0)` | Area fraction range for global crops. |
| `local_crops_scale` | `(0.05, 0.32)` | Area fraction range for local crops. |
| `solarize_prob` | `0.2` | Probability of solarisation (2nd global crop only). |
| `mean / std` | ImageNet | Per-channel normalisation statistics. |

---

## Dataset hub — IDE integration

Datasets are discovered automatically from the filesystem hierarchy:

```
$DINO_DATASETS_ROOT/
  <confidentiality>/          (e.g. "public", "private")
    <modality>/               (e.g. "rgb", "multispectral")
      <dataset_name>/
        <split>/              (e.g. "train", "val")
          shard-000000.tar
          shard-000000.idx
          ...
```

### Configuring the root path

Resolution order (first match wins):
1. `root_path` argument to `Dataset(name, root_path=...)` or CLI `--root`
2. `tool.dino_loader.datasets.root` in `pyproject.toml`
3. `$DINO_DATASETS_ROOT` environment variable
4. `~/.dinoloader/`

### CLI

```bash
# Preview all datasets in a tree view
python -m dino_loader.datasets preview

# Count items in a dataset (uses .idx line counts)
python -m dino_loader.datasets count imagenet

# Scaffold a new dataset directory
python -m dino_loader.datasets add public rgb my_dataset train

# Regenerate IDE stubs (hub.py)
python -m dino_loader.datasets stubs
```

### IDE stubs (`hub.py`)

After running `stubs`, you get autocomplete-friendly dataset references:

```python
from dino_loader.datasets.hub import imagenet, custom

spec = imagenet.to_spec(
    global_filter=GlobalDatasetFilter(allowed_splits=["train"]),
)
```

---

## Real-time monitoring

A live terminal UI (requires `rich`) shows per-rank throughput, shard cache
utilisation, and pipeline stall times:

```bash
python -m dino_loader.monitor.cli --job $SLURM_JOB_ID
```

The monitor connects to the shared-memory metrics block (`/dev/shm`) written by
the dataloader without acquiring any locks (read-only, tolerates torn reads for
display purposes).

---

## Checkpointing and resume

`loader.checkpoint(step)` is a no-op on all ranks except rank 0, and a no-op
unless `step % checkpoint_every_steps == 0`.  It writes a JSON file
atomically:

```
/checkpoint/dino/dl/dl_state_000001000.json
```

Only the 3 most recent checkpoints are retained to bound Lustre usage.

To resume:
```python
loader = DINODataLoader(..., resume=True)
```

The loader restores `epoch`, `step`, and `mixing_weights`.  Note: within-epoch
position is not restored (DALI pipeline state is not checkpointed by default).
DALI 1.30+ supports pipeline serialisation via `pipeline.serialize()` —
integrate that for sub-epoch resume on very large datasets.

---

## Known bugs & fixes applied

The following issues were identified during the pre-production code review and
**require patching before deployment**.

### 🔴 Critical

**BUG-A — `loader.py` `_restore()` does not call `set_epoch()`**

After restoring a checkpoint, `self._epoch` is set to the saved value but
`self._source.set_epoch(state.epoch)` is never called.  All `ShardIterator`
instances start from epoch 0 regardless.  On resume, the model sees the same
shard order as epoch 0.

```python
# In _restore(), after setting self._epoch = state.epoch, add:
self._source.set_epoch(state.epoch)
```

---

**BUG-B — `datasets/cli.py` `count_elements()` reads `.idx` as text**

`webdataset.wds2idx` writes a **binary** index file, not newline-delimited
text.  `sum(1 for _ in f)` counts newline characters, not dataset items, and
may raise a `UnicodeDecodeError` on binary data.

```python
# Replace the text-mode open with binary and proper idx parsing.
# The wds2idx format: 8-byte little-endian int64 per entry (byte offset).
import struct

with open(idx_path, 'rb') as f:
    data = f.read()
count = len(data) // 8   # each entry is one int64
total_count += count
```

---

### 🟡 Medium

**BUG-C — `loader.py` `__iter__` creates multiple consumers on the same DALI iterator**

`__iter__` constructs a new `AsyncPrefetchIterator` wrapping `self._dali_collate()`
which in turn iterates `self._dali_iter`.  If `__iter__` is called twice
within one epoch (e.g. in a progress-bar wrapper that peeks at the iterator),
two consumers interleave reads from the same DALI output stream, producing
silently corrupted batches.

```python
def __iter__(self) -> Iterator[Batch]:
    if getattr(self, "_active_iter", None) is not None:
        raise RuntimeError(
            "DINODataLoader: __iter__ called while a previous iteration "
            "is still active. Call set_epoch() to reset before re-iterating."
        )
    self._active_iter = AsyncPrefetchIterator(
        source=self._dali_collate(), h2d=self._h2d, te_fmt=self._fp8_fmt
    )
    return self._active_iter
# Clear _active_iter in set_epoch() and __del__().
```

---

**BUG-D — `monitor/cli.py` column labels do not match `MetricsStruct` fields**

The "Net Stall (ms)" column displays `m.lustre_read_time_ms` (Lustre I/O time),
not `m.network_stall_time_ms`.  The "Mutex Wait (ms)" column displays
`m.shard_wait_time_ms`, which is labelled "Shard Cache Wait" in the struct —
this one is arguably acceptable but should be renamed for clarity.

```python
# Fix the workers_table.add_row() call:
workers_table.add_row(
    str(i),
    f"{m.loader_batches_yielded}",
    f"{m.network_stall_time_ms}",   # was m.lustre_read_time_ms
    f"{m.shard_wait_time_ms}",
    f"{m.pipeline_yield_time_ms}",
)
```

---

**BUG-E — `mixing_source.py` `_init_epoch()` deduplication skips first shard from prefetch window**

When `n_shards_per_rank < _EXTRACTION_DEPTH` (tiny datasets), the deduplication
branch increments `self._idx` but does not submit a future.  The subsequent
`_prefetch_window()` call then starts from `self._idx + 1`, silently skipping
prefetch for the first shard.

```python
# In _init_epoch(), replace the else branch:
else:
    # Same shard would be double-submitted. Don't advance _idx here;
    # just skip the second submission. _prefetch_window will cover it.
    pass
```

---

### 🟢 Minor / Style

**BUG-F — `datasets/stub_gen.py` instantiates a `Dataset` just to call `resolve_datasets_root()`**

```python
# Replace:
dummy = Dataset("dummy", root_path=root_path)
base_dir = dummy.root_path
# With:
from dino_loader.datasets.settings import resolve_datasets_root
base_dir = resolve_datasets_root(root_path)
```

**NOTE — `pipeline.py` coin-flip arithmetic with BOOL tensors**

The pattern `do_jitter * jittered + (1 - do_jitter) * imgs` multiplies a BOOL
tensor by a FLOAT16 tensor.  DALI auto-promotes BOOL to the higher dtype, so
this works correctly today.  The more idiomatic and future-proof form is
`fn.cast(do_jitter, dtype=types.FLOAT16) * jittered + ...`, which should be
considered for robustness.

---

## Extending the loader

### Adding a new dataset

```bash
python -m dino_loader.datasets add private rgb my_new_dataset train
# Drop .tar and .idx files into the printed path, then:
python -m dino_loader.datasets stubs
```

### Custom augmentation

Subclass or replace `DINOAugConfig` and pass it to `DINODataLoader`.  The DALI
pipeline is rebuilt on each `DINODataLoader` construction — there is no caching.

### Dynamic weight scheduling (curriculum learning)

```python
loader.set_weights([0.5, 0.3, 0.2])          # all at once
loader.set_weight_by_name("imagenet22k", 0.4) # one at a time
```

Changes are thread-safe and take effect on the next batch.  Weights are
re-normalised automatically so they need not sum to 1.

---

## Installation

```bash
# Requires Python 3.12+, CUDA 12.8+, NVIDIA DALI ≥ 1.34, Transformer Engine ≥ 2.12

pip install nvidia-dali-cuda120       # or cuda118, cuda121 etc.
pip install transformer-engine

# From source:
git clone https://github.com/your-org/dino_loader
cd dino_loader
pip install -e ".[dev]"
```

### Optional dependencies

| Package | Purpose | Required? |
|---|---|---|
| `nvidia-dali` | GPU augmentation pipeline | **Yes** (training) |
| `transformer-engine` | FP8 output quantisation | No (graceful fallback to BF16) |
| `aiofiles` | Async Lustre reads | No (falls back to executor) |
| `rich` | Terminal monitor UI | No |
| `tomli` | `pyproject.toml` config on Python < 3.11 | Auto-installed |

---

## Project layout

```
src/dino_loader/
├── __init__.py          Public API surface
├── config.py            All dataclasses — DatasetSpec, DINOAugConfig, LoaderConfig
├── loader.py            DINODataLoader — main entry point, composes all subsystems
├── pipeline.py          DALI augmentation graph (build_pipeline)
├── mixing_source.py     MixingSource, ShardIterator, MixingWeights
├── shard_cache.py       NodeSharedShardCache — /dev/shm LRU cache
├── memory.py            Batch, H2DStream, FP8Formatter, AsyncPrefetchIterator
├── checkpoint.py        DataLoaderCheckpointer — JSON checkpoint I/O
├── distributed.py       slurm_init, detect_topology, configure_nccl
├── train.py             Reference training script
├── datasets/
│   ├── dataset.py       Dataset — filesystem discovery and shard resolution
│   ├── settings.py      resolve_datasets_root — config precedence logic
│   ├── utils.py         _extract_jpegs, validate_webdataset_shard
│   ├── stub_gen.py      Auto-generates hub.py for IDE autocomplete
│   ├── hub.py           Auto-generated dataset stubs (do not edit)
│   └── cli.py           CLI: preview / count / add / stubs
└── monitor/
    ├── metrics.py       Shared-memory MetricsRegistry (lock-free counters)
    ├── tracing.py       Chrome trace event recording
    └── cli.py           Live terminal monitor UI
```
