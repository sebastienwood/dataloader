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
9. [Extending the loader](#extending-the-loader)
10. [Installation](#installation)
11. [Project layout](#project-layout)

---

## Quick start

```python
from dino_loader import DINOAugConfig, DINODataLoader, LoaderConfig, slurm_init
from dino_dataset import DatasetSpec

env = slurm_init()          # reads SLURM env vars, sets NCCL knobs, inits process group

specs = [
    DatasetSpec("laion2b",    shards=[...], weight=0.5),
    DatasetSpec("datacomp1b", shards=[...], weight=0.3),
    DatasetSpec("imagenet22k", shards=[...], weight=0.2, shard_sampling="resampled"),
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
        # batch.global_crops : list[Tensor]  — 2  BF16 (or FP8) tensors on GPU
        # batch.local_crops  : list[Tensor]  — 8  tensors
        # batch.metadata     : list[Optional[dict]]  — .json sidecar per sample
        # batch.masks        : Optional[Tensor]  — iBOT patch masks (bool)
        train_step(batch)
        loader.checkpoint(step)         # writes JSON checkpoint (rank 0 only)

    # Dynamic curriculum — shift toward curated data over time
    if epoch == 10:
        loader.set_weights([0.4, 0.4, 0.2])
```

### Fluid post-processing API

`DINODataLoader` exposes a **composable pipeline API** for chaining
post-DALI transforms without modifying the loader itself:

```python
loader = (
    DINODataLoader(specs, batch_size=512, ...)
    .map(apply_ibot_masks)          # fn(Batch) → Batch
    .select(quality_ok)             # predicate(Batch) → bool, drops False batches
    .with_epoch(steps_per_epoch)    # limit steps (overrides steps_per_epoch)
)
```

Each method returns a `PostProcessPipeline`; the base loader is not mutated.
The pipeline is lazy — transforms execute only as batches flow through.

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

See `src/dino_loader/train.py` for a fully-annotated reference training script.

---

## Architecture overview

The pipeline is a **5-stage assembly line**, each stage running concurrently
with the others.  No stage waits for the next; the only back-pressure is
intentional.

```
Stage 1  ──  Shard I/O          (Lustre → /dev/shm, async aiofiles, node-master only)
Stage 2  ──  JPEG Extraction    (tar parsing → JPEG bytes, thread pool)
Stage 3  ──  DALI Augmentation  (decode + multi-crop augmentation on GPU, pipeline_def)
Stage 4  ──  H2D Transfer       (pinned host → GPU, dedicated CUDA stream)
Stage 5  ──  FP8 Quantisation   (BF16 → FP8 E4M3 with TE metadata, optional)
```

### Backend abstraction

The loader is fully backend-agnostic.  All DALI-specific logic lives in
`DALIBackend`; a `CPUBackend` (PIL + torchvision) is also provided:

| Backend | Use case | Requirements |
|---|---|---|
| `"dali"` (default) | Production training | nvidia-dali, CUDA GPU, SLURM |
| `"cpu"` | Unit tests, CI, laptops | Python 3.12+, PyTorch |
| `"auto"` | Fallback logic | uses DALI if available, else CPU |

```python
loader = DINODataLoader(..., backend="cpu")  # force CPU backend
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
║  local_rank 0 (node master)  ║   ║  local_rank 0 (node master)  ║
║  ┌─────────────────────────┐ ║   ║  ┌─────────────────────────┐ ║
║  │ NodeSharedShardCache    │ ║   ║  │ NodeSharedShardCache    │ ║
║  │ asyncio + aiofiles      │ ║   ║  │ asyncio + aiofiles      │ ║
║  │ LRU eviction            │ ║   ║  │ LRU eviction            │ ║
║  │ heartbeat watchdog      │ ║   ║  │ heartbeat watchdog      │ ║
║  └──────────┬──────────────┘ ║   ║  └──────────┬──────────────┘ ║
║             │ atomic rename  ║   ║             │ atomic rename  ║
║             ▼                ║   ║             ▼                ║
║  ┌────────────────────────┐  ║   ║  ┌────────────────────────┐  ║
║  │  /dev/shm/<job>/       │  ║   ║  │  /dev/shm/<job>/       │  ║
║  │  (tmpfs, ~256 GB DRAM) │  ║   ║  │  (tmpfs, ~256 GB DRAM) │  ║
║  │  abc123.shm ◄──────────┼──╫──►│  inotify wait (no spin)│  ║
║  │  def456.shm            │  ║   ║  │  persistent mmap pool  │  ║
║  └──────────┬─────────────┘  ║   ║  └──────────┬─────────────┘  ║
║             │ mmap (zero-copy)║   ║             │ mmap            ║
║             ▼                ║   ║             ▼                ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │         ALL LOCAL RANKS  (e.g. 72 ranks on NVL72)         │  ║
║  │                                                            │  ║
║  │  MixingSource  →  DALI pipeline  →  H2DStream             │  ║
║  │  (thread pool)    (GPU decode)      (pinned CUDA stream)   │  ║
║  │                                                            │  ║
║  │  [optional]  FP8Formatter (Transformer Engine)            │  ║
║  │  BF16 → FP8 E4M3 + FP8TensorMeta (rolling amax)          │  ║
║  │                                                            │  ║
║  │  Batch(global_crops, local_crops, metadata, masks)        │  ║
║  │             ↓                                             │  ║
║  │     TRAINING LOOP  (ViT forward / backward)              │  ║
║  └────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════╝   ╚══════════════════════════════╝

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

`/dev/shm` is a **Linux tmpfs** backed entirely by DRAM.  Accessible to all
processes on the same node as a regular path, which gives:

- **One Lustre read per shard, not one per GPU.**  On NVL72 (72 GPUs/node),
  only `local_rank 0` reads from Lustre.  The other 71 ranks read from
  `/dev/shm` at ~200 GB/s with zero network traffic.
- **Cross-process sharing without custom IPC.**  Any rank can `open()` +
  `mmap()` the cached file.  No semaphores, pipes, or shared memory handles.
- **Persistent mmap pool (PERF-2).**  `_MmapPool` maintains open `(fd, mmap)`
  pairs keyed by shard path, ref-counted and LRU-evicted.  On NVL72 this
  reduces mmap syscall overhead by ~70× for hot shards (hit ratio > 95% at
  steady state).
- **LRU eviction via `unlink()`.**  The kernel reclaims pages when no process
  has the file open.
- **Atomic visibility via `rename()`.**  The master writes to `.tmp` and calls
  `rename()` (POSIX-atomic).  Readers see either the complete file or nothing.
  A magic sentinel (`0xDEADBEEFCAFEF00D`) provides a second integrity layer.
- **inotify instead of polling.**  Non-master ranks call
  `inotify_add_watch()` and block in `select()`.  Zero CPU while waiting,
  microsecond wake-up latency.  Critical on NVL72 where 71 cores would
  otherwise spin-wait.

### Why async I/O with `aiofiles` for Lustre reads?

Lustre metadata latency is high (~1–5 ms/op); stripe I/O bandwidth is
maximised with many concurrent readers.  The node master uses an `asyncio`
event loop with up to `shard_prefetch_window` (default 64) concurrent
downloads, limited by an `asyncio.Semaphore` to avoid Lustre overload.
Typical aggregate node bandwidth: **10–50 GB/s** on a well-configured IB
fabric.

### Why DALI instead of PyTorch DataLoader?

DALI runs the full decode + augment pipeline on the GPU with fused kernels.
Key advantages:
- **nvjpeg hardware decoder:** JPEG decode on dedicated HW ASICs
  (`hw_decoder_load=0.90` routes 90% of decode to HW).
- **Kernel fusion:** `normalize → cast → transpose` compiles into a single
  GPU kernel when `fuse_normalization=True`.
- **Persistent prefetch queues:** `cpu_queue` and `gpu_queue` depth decouple
  I/O latency from training throughput.
- **Zero-copy pipeline output:** DALI tensors live on GPU memory; no H2D
  copy for the augmented images themselves.

### Why FP8 output?

B200 / GB200 ViT training uses `te.fp8_autocast()`.  Delivering batches
already in FP8 E4M3 (with `FP8TensorMeta`) eliminates the first cast in the
forward pass.  Two paths:
- `dali_fp8_output=False` (default): `FP8Formatter` runs post-DALI with
  Transformer Engine metadata (rolling amax, compatible with `te.fp8_autocast`).
- `dali_fp8_output=True`: FP8 cast fused into the DALI graph
  (`normalize → cast FP8 → transpose` in one kernel), but TE metadata is
  not available.

### Why per-dataset normalisation fused in DALI?

When mixing datasets with different normalisation statistics (e.g. ImageNet
vs. satellite imagery), each sample needs different `(mean, std)` tensors.
`NormSource` emits per-sample `(mean, std)` pairs as DALI `ExternalSource`
tensors, allowing the DALI compiler to fuse them with the decode kernel.
This eliminates one GPU kernel launch and one BF16 intermediate buffer per
view per batch (`fuse_normalization=True`, default).

### Why a heartbeat file instead of `squeue` for orphan detection?

Calling `squeue` from the dataloader on every rank is dangerous on large
clusters (thousands of concurrent calls saturate the SLURM controller;
`subprocess.run()` forks under heavy load, exhausting file descriptors).
The node master now writes `/dev/shm/<job_id>/heartbeat` (PID + mtime),
refreshed every few seconds by a daemon thread.  Orphan detection becomes an
`os.kill(pid, 0)` + mtime check — O(1) and purely local.

### Why `wds.ResampledShards` for small datasets?

Small curated datasets (e.g. `imagenet22k` at 5k shards) need
over-sampling without duplicating shards on disk.  `shard_sampling="resampled"`
delegates to `wds.ResampledShards` for infinite with-replacement sampling,
controlled by the mixing `weight`.  The default `"epoch"` mode does a
deterministic shuffled full pass per epoch.

### Why iBOT masking stays outside DALI?

`MaskingGenerator` operates on ViT **patch-level indices** (a bool grid of
shape `grid × grid` where `grid = img_size // patch_size`), not on image
pixels.  DALI's computation graph only processes dense image tensors.  The
CPU overhead is ~0.3 ms per batch — negligible vs. ~40 ms DALI decode.

---

## Configuration reference

### `LoaderConfig`

All knobs live in `LoaderConfig`.  Sensible defaults target GB200 NVL72;
adjust for your cluster.

| Field | Default | Description |
|---|---|---|
| `node_shm_gb` | `128.0` | `/dev/shm` budget per node in GB (~50% of node RAM). |
| `shard_prefetch_window` | `64` | Max concurrent Lustre → `/dev/shm` downloads. |
| `shard_timeout_s` | `300.0` | Max seconds a non-master rank waits for a shard. |
| `shard_extraction_workers` | `4` | Thread-pool workers for tar → JPEG extraction. |
| `shuffle_buffer_size` | `512` | In-memory shuffle reservoir depth per `ShardIterator`. |
| `dali_cpu_queue` | `8` | DALI CPU-side prefetch queue depth. |
| `dali_gpu_queue` | `6` | DALI GPU-side prefetch queue depth. |
| `dali_num_threads` | `8` | DALI CPU worker threads for pre-decode ops. |
| `hw_decoder_load` | `0.90` | Fraction of JPEG decode sent to nvjpeg HW ASIC (0–1). |
| `use_fp8_output` | `True` | Quantise output to FP8 E4M3. |
| `dali_fp8_output` | `False` | Fuse FP8 cast into DALI graph (no TE metadata). |
| `fuse_normalization` | `True` | Fuse per-dataset norm into DALI kernel. |
| `output_dtype` | `"bf16"` | Intermediate dtype (`"bf16"` or `"fp32"`). |
| `stall_timeout_s` | `600.0` | Watchdog timeout before raising on no batches. Set to `0` to disable. |
| `checkpoint_dir` | `"/checkpoint/dino/dl"` | Where `.json` checkpoint files are written. |
| `checkpoint_every_steps` | `500` | Checkpoint frequency (rank 0 only). |
| `stateful_dataloader` | `False` | Enable `state_dict()` / `load_state_dict()` interface. |
| `force_topology` | `None` | Override topology detection: `"nvl72"` or `"pcie"`. |
| `seed` | `0` | Base random seed for shard shuffling and augmentation. |
| `debug_log_keys` | `None` | Path to append per-sample key audit log (disable in production). |
| `shm_warn_threshold` | `0.90` | `/dev/shm` utilisation fraction that triggers a warning. |

### `DINOAugConfig`

Defaults match DINOv2 (Oquab et al., 2023, §A.1).

| Field | Default | Description |
|---|---|---|
| `n_global_crops` | `2` | Number of large crops per image. |
| `n_local_crops` | `8` | Number of small crops per image. |
| `global_crop_size` | `224` | Pixel size of global crops (changeable at runtime). |
| `local_crop_size` | `96` | Pixel size of local crops (changeable at runtime). |
| `global_crops_scale` | `(0.32, 1.0)` | Area fraction range for global crops. |
| `local_crops_scale` | `(0.05, 0.32)` | Area fraction range for local crops. |
| `preserve_aspect_ratio` | `True` | Resize shorter side then centre-crop (avoids distortion). |
| `solarize_prob` | `0.2` | Probability of solarisation (2nd global crop only). |
| `resolution_schedule` | `[]` | List of `(epoch, global_crop_size)` pairs applied by `set_epoch()`. |
| `max_global_crop_size` | `global_crop_size` | nvjpeg pre-allocation ceiling. |
| `max_local_crop_size` | `local_crop_size` | nvjpeg pre-allocation ceiling. |
| `mean` | `(0.485, 0.456, 0.406)` | Per-channel normalisation mean (ImageNet). |
| `std` | `(0.229, 0.224, 0.225)` | Per-channel normalisation std (ImageNet). |

### `DatasetSpec`

| Field | Default | Description |
|---|---|---|
| `name` | — | Human-readable identifier, used in logs and checkpoints. |
| `shards` | — | List of absolute `.tar` shard paths on Lustre. |
| `weight` | `1.0` | Mixing weight (re-normalised automatically). |
| `prob` | `None` | Alias for `weight` aligned with `wds.RandomMix` API. |
| `shard_sampling` | `"epoch"` | `"epoch"` (deterministic full pass) or `"resampled"` (infinite with-replacement). |
| `shard_quality_scores` | `None` | Per-shard quality scores; biases shard selection proportionally. |
| `min_sample_quality` | `None` | Hard filter: drops samples whose `.json` `quality_score` is below this. |
| `metadata_key` | `"json"` | WebDataset sidecar extension to extract. `None` disables sidecar extraction. |
| `mean` / `std` | `None` | Per-dataset normalisation stats override (uses `DINOAugConfig` globals if `None`). |

---

## Dataset hub — IDE integration

Datasets are discovered automatically from the filesystem hierarchy:

```
$DINO_DATASETS_ROOT/
  <confidentiality>/          (e.g. "public", "private")
    <modality>/               (e.g. "rgb", "multispectral")
      <dataset_name>/
        outputs/
          <strategy>/         (e.g. "default")
            <split>/          (e.g. "train", "val")
              shard-000000.tar
              shard-000000.idx
```

The `datasets` sub-package is **self-contained** — it can be imported without
DALI, CUDA, or the loader.  Safe to use in cataloguing tools and CI pipelines.

### Confidentiality registry

Multiple confidentiality roots (on different Lustre mount points) can be
registered simultaneously:

```python
from dino_loader.datasets import register_confidentiality

register_confidentiality("public",  "/lustre/public_data")
register_confidentiality("private", "/lustre/private_data")
```

Resolution order (first match wins):
1. `root_path` argument to `Dataset(name, root_path=...)`
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

### IDE stubs (`hub/`)

After running `stubs`, autocomplete-friendly dataset references are available:

```python
from dino_loader.datasets.hub import imagenet, custom

spec = imagenet.to_spec(
    global_filter=GlobalDatasetFilter(allowed_splits=["train"]),
)
```

Hub staleness is detected via a **registry hash** stored in
`hub/_registry_hash.txt`.  The check is O(n_mounts) dict operations with a
single tiny file read — zero filesystem stat calls against HPC mount points.

---

## Real-time monitoring

A live terminal UI (requires `rich`) shows per-rank throughput, shard cache
utilisation, and pipeline stall times at ~4 Hz:

```bash
python -m dino_loader.monitor.cli --job $SLURM_JOB_ID
```

The monitor connects **read-only** to the `/dev/shm` metrics block
(`MetricsRegistry`) written lock-free by the dataloader.  It tolerates torn
reads (display jitter is acceptable; training is unaffected).

Metrics tracked per rank include: Lustre bytes read, batches yielded,
pipeline yield time, H2D transfer time, shard cache wait time, network stall
time, and heartbeat timestamp (staleness detection).

Chrome trace events are also recorded to `tracing.py` for offline profiling
with `chrome://tracing`.

---

## Checkpointing and resume

`loader.checkpoint(step)` is a no-op on all ranks except rank 0, and a no-op
unless `step % checkpoint_every_steps == 0`.  It writes a JSON file
atomically (write-to-tmp + rename):

```
/checkpoint/dino/dl/dl_state_000001000.json
```

A `LATEST` pointer file is also maintained for fast discovery.  Only the 3
most recent checkpoints are retained to bound Lustre usage.  Stale `.tmp`
files are cleaned up on failure.

To resume:

```python
loader = DINODataLoader(..., resume=True)
```

The loader restores `epoch`, `step`, and `mixing_weights`.  Within-epoch
position is not restored by default (DALI pipeline state is not checkpointed).

**StatefulDataLoader interface** (`stateful_dataloader=True` in `LoaderConfig`):

```python
sd = loader.state_dict()
loader.load_state_dict(sd)
```

---

## Extending the loader

### Adding a new dataset

```bash
python -m dino_loader.datasets add private rgb my_new_dataset train
# Drop .tar and .idx files into the printed path, then:
python -m dino_loader.datasets stubs
```

### Custom augmentation

Subclass `DINOAugConfig` and pass it to `DINODataLoader`.  The DALI pipeline
is rebuilt on each construction.

### Dynamic weight scheduling (curriculum learning)

```python
loader.set_weights([0.5, 0.3, 0.2])            # all at once (re-normalised)
loader.set_weight_by_name("imagenet22k", 0.4)  # one at a time
```

Changes are thread-safe and take effect on the next batch.

### Dynamic resolution schedule

Pass `resolution_schedule` to `DINOAugConfig`:

```python
aug_cfg = DINOAugConfig(
    global_crop_size   = 224,
    resolution_schedule = [(0, 224), (20, 280), (50, 336)],
)
```

`set_epoch()` applies the correct size automatically — **no pipeline rebuild**
required.  Call `loader.set_resolution(global_size, local_size)` for manual
control.

### Custom backend

Implement `BackendProtocol` and pass an instance:

```python
loader = DINODataLoader(..., backend=MyCustomBackend())
```

The protocol requires: `build_shard_cache`, `build_pipeline`,
`build_pipeline_iterator`, `build_h2d_stream`, `build_fp8_formatter`,
`init_distributed`.

---

## Installation

```
Requires Python ≥ 3.12, CUDA ≥ 12.8, NVIDIA DALI ≥ 1.34, Transformer Engine ≥ 2.12
```

```bash
pip install nvidia-dali-cuda120       # or cuda118, cuda121 etc.
pip install transformer-engine~=2.12

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
| `aiofiles` | Async Lustre reads | No (falls back to thread executor) |

---

## Project layout

```
src/dino_loader/
├── __init__.py              Public API surface
├── config.py                Dataclasses: DatasetSpec (re-export), DINOAugConfig, LoaderConfig
├── loader.py                DINODataLoader, PostProcessPipeline — main entry point
├── pipeline.py              DALI augmentation graph (build_pipeline, NormSource)
├── mixing_source.py         MixingSource, ShardIterator, MixingWeights
├── shard_cache.py           NodeSharedShardCache — /dev/shm LRU cache, mmap pool, heartbeat
├── memory.py                Batch, H2DStream, FP8Formatter, AsyncPrefetchIterator
├── checkpoint.py            DataLoaderCheckpointer — atomic JSON I/O, LATEST pointer
├── distributed.py           slurm_init, detect_topology, configure_nccl, ClusterTopology
├── train.py                 Reference training script (fully annotated)
│
├── backends/
│   ├── __init__.py          get_backend() factory, BackendName type alias
│   ├── protocol.py          BackendProtocol — abstract interface for all backends
│   ├── dali_backend.py      DALIBackend — production DALI + CUDA path
│   └── cpu.py               CPUBackend — PIL + torchvision, for tests and CI
│
├── datasets/
│   ├── __init__.py          Self-contained sub-package; re-exports public API
│   ├── spec.py              DatasetSpec — canonical home (re-exported by config.py)
│   ├── dataset.py           Dataset — filesystem discovery and shard resolution
│   ├── settings.py          ConfidentialityRegistry, resolve_datasets_root
│   ├── shard_writer.py      ShardWriter — WebDataset tar writing utility
│   ├── utils.py             _extract_jpegs, validate_webdataset_shard, ensure_idx_exists
│   ├── stub_gen.py          Generates hub/ package; registry-hash staleness detection
│   ├── cli.py               CLI: preview / count / add / stubs
│   └── hub/                 Auto-generated; do not edit — run `stubs` to regenerate
│       ├── __init__.py
│       ├── _registry_hash.txt
│       └── <modality>.py    (e.g. rgb.py, multispectral.py)
│
└── monitor/
    ├── metrics.py           MetricsRegistry — lock-free /dev/shm counters, MetricField StrEnum
    ├── tracing.py           Chrome trace event recording
    └── cli.py               Live terminal monitor UI (Rich, 4 Hz)

tests/
├── fixtures.py              write_shard, scaffold_dataset_dir helpers
├── test_mixing_source.py    ShardIterator + MixingSource unit tests
├── test_improvements.py     Regression tests for all perf/maintainability improvements
└── ...
```