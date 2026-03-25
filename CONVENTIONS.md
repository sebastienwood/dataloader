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

---

## Dataclasses

- **`@dataclass(frozen=True)`** for immutable value objects (`DatasetSpec`, `LaunchContext`,
  `ClusterTopology`, …). These must be genuinely hashable: all fields must be `tuple` or
  scalar, never `list`.
- **`@dataclass`** (mutable) for configuration objects (`LoaderConfig`, `DINOAugConfig`,
  `CheckpointState`). `__post_init__` handles validation — keep it focused and fast.
- **`dataclasses.replace()`** (aliased as `_dc_replace`) is the only way to produce
  modified copies of frozen dataclasses.

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
| `config.py` | Pure dataclasses: `DINOAugConfig`, `LoaderConfig`, `CheckpointState` |
| `augmentation.py` | `AugmentationSpec` hierarchy + `SamplePredicate` protocol |
| `mixing_source.py` | `MixingSource`, `ShardIterator`, `MixingWeights`, `ResolutionSource` |
| `pipeline.py` | DALI static-graph pipeline builder + `NormSource` |
| `memory.py` | `Batch`, `H2DStream`, `FP8Formatter`, `allocate_buffers` |
| `checkpoint.py` | `DataLoaderCheckpointer` — atomic JSON I/O, LATEST pointer |
| `loader.py` | `DINODataLoader`, `PostProcessPipeline` — main entry point |
| `masking.py` | `MaskingGenerator` — pure iBOT patch-mask generator |
| `nodes.py` | `torchdata.nodes` wrappers: `ShardReaderNode`, `MetadataNode`, `MaskMapNode` |
| `pipeline_graph.py` | `NodePipeline`, `BatchMapNode`, `BatchFilterNode` — composable post-processing |
| `backends/` | Pluggable backend abstraction (DALI, CPU) |
| `monitor/` | Metrics, tracing, OTEL, CLI monitor |
| `experimental/` | `dynamic_pipeline` — DALI v2 dynamic-mode (not production) |

### Key invariants

- `loader.py` contains **no augmentation logic**. All augmentation lives in `augmentation.py`,
  `pipeline.py`, and the `backends/`.
- `nodes.py` does **not** import from `loader.py`. The dependency flows one way:
  `loader.py → nodes.py`, never the reverse.
- `masking.py` is a **pure module** with no torch.distributed or DALI dependency.
  `MaskMapNode` in `nodes.py` wraps it for the torchdata graph.
- `config.py` imports **nothing** from `dino_loader`. It may import from `dino_datasets`
  only for `DatasetSpec` re-export.
- `monitor/` modules are imported **lazily** inside functions with `# noqa: PLC0415`.

---

## torchdata.nodes integration (Phase 1 + 3)

The preferred API for composing pipeline stages is `torchdata.nodes`. New stages should
be implemented as `BaseNode` subclasses. The key contracts:

- **`reset(initial_state)`**: called before every epoch; must be idempotent.
- **`next()`**: returns one item; raises `StopIteration` when the epoch ends.
- **`get_state()`**: returns a JSON-serialisable dict for checkpointing.

`wrap_loader(dino_loader)` bridges a `DINODataLoader` into this graph. Prefer
`NodePipeline` (from `pipeline_graph.py`) over `PostProcessPipeline` for new code.

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

---

## Documentation

- **README.md**: keep runtime dependencies up to date; no backports if stdlib suffices.
- **Docstrings**: modules, public classes, all public methods. Google style.
- **Inline comments**: *why*, never *what*.
- **`# noqa` comments**: always include the specific code (e.g. `# noqa: PTH112`),
  never bare `# noqa`.
