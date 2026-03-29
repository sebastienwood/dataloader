# dino_loader — Code Conventions

> This document is read automatically at the start of every working session on this project.

---

## Python version and style

- **Required version: Python ≥ 3.12.** Use `tomllib` (stdlib), never the backport `tomli`.
- **No `from __future__ import annotations`** — use the native 3.12 syntax directly.
  All type annotations use the built-in generics (`list[str]`, `dict[str, int]`, `X | Y`).
- **PEP 695 type aliases** (`type Foo = ...`) are preferred over `TypeAlias` from `typing`.
- **`match` / `case`** structural pattern matching is preferred over long `isinstance` chains
  when dispatching on type hierarchies (e.g. `AugmentationSpec` subtypes).
- **`pathlib.Path`** is preferred throughout. When `os.*` or `str` paths are unavoidable,
  add `# noqa: PTH<code>` with the exact code.
- **Annotations**: every public function and method is annotated, including `__iter__`
  (returns `Iterator[...]`). Complex local variables are also annotated when the type is
  not obvious.

---

## Imports

- **Top-level only.** Conditional heavy imports (`torch.distributed`, `nvidia.dali`, etc.)
  stay local to their function and carry `# noqa: PLC0415` with a justification comment.
- **No circular imports.** The dependency order is:
  `config → augmentation → mixing_source → pipeline → backends → loader`.
  `monitor.*` is imported only from the layers that need it (backends, loader).
- `TYPE_CHECKING` blocks are allowed for forward references in type annotations only.

---

## Exceptions and error messages

- **No f-strings in `raise` expressions** (EM102). Assign the message to a variable first.
  ```python
  # ✗
  raise ValueError(f"Unknown spec type {type(spec)!r}")
  # ✓
  msg = f"Unknown spec type {type(spec)!r}"
  raise ValueError(msg)
  ```
- **Long messages outside the exception class** (TRY003): always use a variable.
- **`except Exception`** is reserved for genuine catch-all situations. Annotate with
  `# noqa: BLE001` and add a comment explaining why a broad catch is intentional.

---

## Concurrency

- **Threading locks** are named `_<resource>_lock` and are always `threading.Lock()` or
  `threading.RLock()` — never bare `threading.Semaphore` for mutual exclusion.
- **`threading.Event`** is preferred over polling for inter-thread signalling.
- **`contextvars.ContextVar`** propagates context to `ThreadPoolExecutor` workers
  automatically (Python 3.12 guarantee). Use `contextvars.copy_context()` only for
  long-lived threads that need to track changes made after thread start.
- **Thread-safe copy-on-write**: when a lock protects shared state, build the new value
  fully outside the lock, then swap atomically inside.

### Thread budget — ShardIterator and CPU oversubscription

Each `ShardIterator` instance creates one I/O daemon thread plus a
`ThreadPoolExecutor(max_workers=num_workers)` for extraction. When mixing N datasets
with W workers each, the total thread count is `N × (1 + W)`.

On HPC nodes where CPU cores are tightly allocated per GPU (B200 / H100 / NVL72
configurations), excess kernel threads cause context-switching overhead and NUMA
thrashing that can reduce I/O throughput by 10–30 %.

**Current approach (accepted trade-off):**
- Each `ShardIterator` manages its own pool — simple, no inter-dataset coordination.
- Workers are I/O-bound (tar parsing + queue operations), so the impact is lower than
  for compute-bound workers.
- Recommended values: `shard_extraction_workers=4` (default) per dataset, capped at 4
  concurrent datasets per rank. Total extraction threads: ≤ 16 for typical configs.

**Known limitation / future work:**
- A global `ThreadPoolExecutor` shared across all `ShardIterator` instances would bound
  the total thread count regardless of dataset count. This would help when mixing 8+
  datasets simultaneously. Implementation must handle per-dataset priority and avoid
  starvation.
- Track as: *"[PERF] Shared extraction pool across ShardIterators"*. Do not implement
  inline without benchmarking; the current design is correct and acceptable for ≤ 6
  datasets.

---

## Dataclasses

- **`@dataclass(frozen=True)`** for immutable value objects (`DatasetSpec`, `LaunchContext`,
  `ClusterTopology`, `NormStats`, …). These must be genuinely hashable: all fields must be `tuple` or
  scalar, never `list`.
- **`@dataclass`** (mutable) for configuration objects (`LoaderConfig`, `DINOAugConfig`,
  `CheckpointState`). `__post_init__` handles validation — keep it focused and fast.
- **`dataclasses.replace()`** (aliased as `_dc_replace`) is the only way to produce
  modified copies of frozen dataclasses.

---

## Normalisation statistics

All normalisation statistics are stored and passed in **[0, 1] scale** throughout the
codebase. The single source of truth is `NormStats` (in `config.py`).

Conversions to other scales happen **only at the point of use**, via the `NormStats`
helper methods:

| Method | Scale | Used by |
|---|---|---|
| `to_dali_scale()` | [0, 255] `list[float]` | `pipeline.py`, `dynamic_pipeline.py`, `cpu.py` |
| `to_numpy()` | [0, 1] `np.ndarray` | CPU augmentation helpers |

**Never multiply mean/std by 255 inline.** Always call `to_dali_scale()`. This rule
is enforced by code review: any literal `* 255` or `/ 255` on a mean/std array outside
`NormStats` is a bug.

---

## Complexity and size limits

- Functions: ≤ 10 branches (`C901`), ≤ 12 branches (`PLR0912`), ≤ 50 statements
  (`PLR0915`). Extract private helpers when exceeded.
- Public functions: ≤ 5 arguments (`PLR0913`). Group extra parameters into a config
  dataclass.
- Lines: ≤ 120 characters (`E501`).

---

## Docstrings

- **Module docstrings**: one-line summary ending with `.`, blank line, then description.
  (`D205`, `D400`, `D415`)
- **Class docstrings**: required for all public classes.
- **Method/function docstrings**: required for all public methods and functions (`D102`).
  One line is sufficient for simple accessors.
- **`__init__`**: docstring required when the class has a docstring (`D107`).
- **Style**: Google style. Use `Args:`, `Returns:`, `Raises:` sections.

---

## Ruff — full reference

| Code | Rule | Action |
|------|------|--------|
| EM102 | f-string in raise | Use intermediate variable |
| TRY003 | Long message outside class | Use intermediate variable |
| TRY300 | `return` in `try` body | Move to `else` block |
| TRY400 | `log.error` in `except` | Use `log.exception` |
| BLE001 | Broad `except Exception` | `# noqa: BLE001` + comment |
| PTH | `os.path.*` / `os.*` file ops | Use `Path.*` or `# noqa: PTHxxx` |
| PLC0415 | Non-top-level import | `# noqa: PLC0415` + justification |
| PLW0603 | `global` statement | Avoid — encapsulate state in a class |
| PLW1508 | Non-string env default | `int(os.environ.get("X", "0"))` |
| PLR0912 | Too many branches (> 12) | Extract helpers |
| PLR0913 | Too many arguments (> 5) | Config dataclass |
| PLR0915 | Too many statements (> 50) | Extract helpers |
| PLR2004 | Magic constant | Name the constant |
| PERF401 | `list.append` in loop | `list.extend(x for x in ...)` |
| PIE810 | Multiple `endswith` args | `endswith((".a", ".b"))` |
| SIM105 | `try`-`except`-`pass` | `contextlib.suppress(...)` |
| S603 | `subprocess` input | `# noqa: S603` when input is controlled |
| ANN | Missing annotation | Annotate all functions incl. private |
| ARG001 | Unused argument | Remove; don't keep dead args for future use |
| D102 | Missing method docstring | Add (one line is fine) |
| D107 | Missing `__init__` docstring | Add if class is documented |
| D205 | Blank line after summary | Format: `summary.` + blank + description |
| D400/D415 | Summary missing `.` | End summary with `.` |
| E501 | Line > 120 chars | Reformat |
| C901 | Complexity > 10 | Extract helpers |
| N801 | Class name not CapWords | Rename |

---

## ty — static type checking

Run `uvx ty check src/` before every PR.

### Known patterns to watch

| ty error | Cause | Fix |
|----------|-------|-----|
| `invalid-argument-type` on `sorted(..., key=len)` | `key=len` widens inferred element type to `Sized` | Annotate the loop variable: `item: str` before the loop |
| `unresolved-attribute` on inspection results | Confusion between `.n_bad` (ShardInspectionResult) and `.total_bad` (DatasetInspectionResult) | Use the correct attribute for each type |

### General rule

When `sorted()` is called with a non-trivial `key=`, annotate the loop variable rather
than inserting a cast:

```python
# ✗ — ty infers item: Sized
for item in sorted(my_list, key=len, reverse=True): ...

# ✓
item: str
for item in sorted(my_list, key=len, reverse=True): ...
```

---

## Architecture — responsibility per file

| File | Responsibility |
|------|---------------|
| `config.py` | Pure dataclasses: `DINOAugConfig`, `LoaderConfig`, `CheckpointState`, `NormStats` |
| `augmentation.py` | `AugmentationSpec` hierarchy + `SamplePredicate` protocol |
| `mixing_source.py` | `MixingSource`, `ShardIterator`, `MixingWeights`, `ResolutionSource` |
| `pipeline.py` | DALI static-graph pipeline builder + `NormSource` |
| `memory.py` | `Batch`, `H2DStream`, `FP8Formatter`, `allocate_buffers` |
| `checkpoint.py` | `DataLoaderCheckpointer` — atomic JSON I/O, LATEST pointer |
| `loader.py` | `DINODataLoader` — main entry point; no post-processing logic |
| `masking.py` | `MaskingGenerator` — pure iBOT patch-mask generator |
| `nodes.py` | `torchdata.nodes` wrappers: `ShardReaderNode`, `MetadataNode`, `MaskMapNode` |
| `pipeline_graph.py` | `NodePipeline`, `BatchMapNode`, `BatchFilterNode`, `wrap_loader` |
| `backends/` | Pluggable backend abstraction (DALI, CPU) |
| `monitor/` | Metrics, tracing, OTEL, CLI monitor |
| `experimental/` | `dynamic_pipeline` — DALI v2 dynamic-mode (not production) |

### Key invariants

- `loader.py` contains **no augmentation logic** and **no post-processing logic**. All
  augmentation lives in `augmentation.py`, `pipeline.py`, and the `backends/`. All
  post-DALI transforms live in `pipeline_graph.py`.
- `nodes.py` does **not** import from `loader.py`. The dependency flows one way:
  `loader.py → nodes.py`, never the reverse.
- `masking.py` is a **pure module** with no torch.distributed or DALI dependency.
  `MaskMapNode` in `nodes.py` wraps it for the torchdata graph.
- `config.py` imports **nothing** from `dino_loader`. It may import from `dino_datasets`
  only for `DatasetSpec` re-export.
- `monitor/` modules are imported **lazily** inside functions with `# noqa: PLC0415`.
- All normalisation stats pass through `NormStats`. No `× 255` inline conversions.

### Backend abstraction — current state and known limitations

The `BackendProtocol` + `CPUBackend` / `DALIBackend` split is intentional and should
be preserved. Its primary value is the `CPUBackend` which enables the full test suite
to run without GPU, DALI, or SLURM.

**Known design smell in `DALIBackend`**: each method in `dali_backend.py` is a thin
shim that does a local import and delegates directly to `pipeline.py` or
`shard_cache.py`. There is no logic of its own. This is acceptable for now but means
the abstraction overhead is one-sided — `CPUBackend` gains a lot from it, `DALIBackend`
less so.

**Recommended future action**: flatten `dali_backend.py` by inlining its methods
directly into `pipeline.py` and `shard_cache.py` factory functions, keeping the
`BackendProtocol` interface but removing the intermediate class. This reduces
indirection without breaking the CPU backend or tests. Do not do this without
updating all import sites.

---

## Post-processing pipeline

`PostProcessPipeline` has been removed. **`wrap_loader()` from
`dino_loader.pipeline_graph` is the single post-processing entry point.**

```python
from dino_loader.pipeline_graph import wrap_loader

pipeline = (
    wrap_loader(DINODataLoader(...))
    .map(fn)
    .select(pred)
    .with_epoch(n)
)
```

`DINODataLoader` does not expose `.map()`, `.select()`, or `.with_epoch()`. Do not
re-add them. `NodePipeline` provides full `state_dict` support across the whole graph,
which `PostProcessPipeline` did not.

---

## Dynamic pipeline (`experimental/dynamic_pipeline.py`)

The dynamic pipeline is **experimental** in the sense that it depends on
`nvidia.dali.experimental.dynamic`, an NVIDIA API that may change between DALI
versions. It is **not** experimental in quality — it is expected to be correct and
performant, and is used to benchmark the static pipeline.

### Randomness contract

All stochastic parameters inside dynamic batch functions **must** use `ndd.random.*`
operators, never Python `random` or `numpy.random`. This is a hard requirement:

```python
# ✗ — scalar broadcast: every sample gets the same value
brightness = float(np.random.uniform(0.6, 1.4))
crop = ndd.color_twist(crop, brightness=brightness)

# ✓ — per-sample: each sample gets an independent draw
crop = ndd.color_twist(crop, brightness=ndd.random.uniform(range=(0.6, 1.4)))
```

Python RNG calls inside a DALI dynamic function return a *single scalar* that DALI
broadcasts to all samples in the batch. `ndd.random.uniform` produces one independent
draw per sample. The distinction is critical for contrastive self-supervised learning,
where diversity within a batch directly affects representation quality.

### Per-sample normalisation

Dataset index callbacks must be wired via
`source.register_dataset_index_callback(_capture_indices)` so the aug function
receives per-sample dataset indices. This enables per-sample `mean`/`std` lookups
matching the `NormSource` semantics of the static pipeline. Never use only the first
sample's index for the whole batch.

---

## torchdata.nodes integration (Phase 1 + 3)

The preferred API for composing pipeline stages is `torchdata.nodes`. New stages should
be implemented as `BaseNode` subclasses. The key contracts:

- **`reset(initial_state)`**: called before every epoch; must be idempotent.
- **`next()`**: returns one item; raises `StopIteration` when the epoch ends.
- **`get_state()`**: returns a JSON-serialisable dict for checkpointing.

`wrap_loader(dino_loader)` bridges a `DINODataLoader` into this graph. Prefer
`NodePipeline` (from `pipeline_graph.py`) over any custom iterator for new code.

---

## Tests

- **TDD**: tests are written before or alongside the code they cover.
- **Isolation**: each test is independent. Singletons and global state are patched or
  reset in fixtures.
- **Naming**: `test_<what>_<condition>_<result>` when the name would otherwise be unclear.
- **Fixtures**: session-scoped for stateless config objects (`DINOAugConfig`,
  `LoaderConfig`); function-scoped for anything that touches the filesystem or threads.
- **Slow tests**: any test that spins up real `ShardIterator` threads, builds a full
  `tn.Loader` graph, or runs multiple shard I/O operations must be decorated with
  `@pytest.mark.slow`. These are excluded from the default `pytest` run:
  ```
  pytest -m "not slow"   # fast CI
  pytest                 # full suite including slow tests
  ```
- **No hub pollution**: never call `generate_stubs()` (no `hub_dir`) in tests.
- **Direct attribute access**: `r.n_shards`, never via an alias that doesn't exist yet.
- **No `from __future__ import annotations`** in test files either.

---

## Performance / HPC invariants

- **No per-file `stat()`** during shard resolution (`runtime_mode=True`).
- **`_pivot_stats`**: `scandir` depth-1 only — never `os.walk`.
- **Metadata storm**: audit any new filesystem traversal before merging.
- **DALI queues replace AsyncPrefetchIterator**: `dali_cpu_queue ≥ 16` is the
  compensating measure. Do not reintroduce application-level prefetch threading.
- **`NormSource` copy-on-write**: `set_dataset_indices()` builds the new list outside
  the lock and swaps atomically. `__call__()` returns explicit numpy copies.
- **Thread budget**: see the *Thread budget* section under *Concurrency* above.

---

## Documentation

- **README.md**: keep runtime dependencies up to date; no backports if stdlib suffices.
- **Docstrings**: modules, public classes, all public methods. Google style.
- **Inline comments**: *why*, never *what*.
- **`# noqa` comments**: always include the specific code (e.g. `# noqa: PTH112`),
  never bare `# noqa`.