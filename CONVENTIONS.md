# dino_loader — Code Conventions

> Ce document est lu automatiquement au début de chaque session de travail sur ce projet.

---

## Version Python et style

- **Version requise : Python ≥ 3.12.** Utiliser `tomllib` (stdlib), jamais le backport `tomli`.
- **Pas de `from __future__ import annotations`** — utiliser la syntaxe native 3.12.
  Toutes les annotations de type utilisent les génériques built-in (`list[str]`, `dict[str, int]`, `X | Y`).
- **Alias de type PEP 695** (`type Foo = ...`) préférés à `TypeAlias` de `typing`.
- **`match` / `case`** préféré aux longues chaînes `isinstance` dans le code générique ; dans les backends,
  `isinstance` reste acceptable pour les dispatch sur les specs (le pattern Strategy y est plus lisible).
- **`pathlib.Path`** utilisé partout. Quand `os.*` ou les chemins `str` sont inévitables, ajouter `# noqa: PTH<code>`.
- **Annotations** : toutes les fonctions et méthodes publiques sont annotées, y compris `__iter__`.

---

## Imports

- **Niveau module uniquement.** Les imports lourds conditionnels (`torch.distributed`, `nvidia.dali`, etc.)
  restent locaux à leur fonction avec `# noqa: PLC0415` et un commentaire de justification.
- **Pas d'imports circulaires.** L'ordre de dépendance est :
  `config → augmentation → sources → shard_reader → dali_node → pipeline_graph → backends → loader`.
  `monitor.*` est importé seulement depuis les couches qui en ont besoin.
- Les blocs `TYPE_CHECKING` sont autorisés uniquement pour les références forward dans les annotations.

---

## Exceptions et messages d'erreur

- **Pas d'f-strings dans les expressions `raise`** (EM102). Assigner le message à une variable d'abord.
- **`except Exception`** réservé aux situations genuinement catch-all. Annoter avec `# noqa: BLE001`.

---

## Concurrence

- **Locks de threading** nommés `_<resource>_lock`, toujours `threading.Lock()` ou `threading.RLock()`.
- **`threading.Event`** préféré au polling pour la signalisation inter-thread.
- **Copy-on-write thread-safe** : construire la nouvelle valeur entièrement hors du lock, puis swap atomique dedans.

---

## Dataclasses

- **`@dataclass(frozen=True)`** pour les value objects immutables (`DatasetSpec`, `ClusterTopology`, `NormStats`, `PipelineConfig`).
- **`@dataclass`** (mutable) pour les objets de configuration (`LoaderConfig`, `DINOAugConfig`).
- **`CheckpointState`** est un pur dataclass **sans méthodes I/O** — `save` et `load` vivent dans `checkpoint.py`.
- **`dataclasses.replace()`** est la seule façon de produire des copies modifiées des dataclasses frozen.

---

## Statistiques de normalisation

Toutes les statistiques de normalisation sont stockées et transmises en **échelle [0, 1]** dans tout le codebase.
La source de vérité unique est `NormStats` (dans `config.py`).

La construction de tables de normalisation et des tableaux par batch est centralisée dans `norm_utils.py` :

| Fonction | Usage |
|---|---|
| `build_norm_table(aug_cfg, specs)` | Construit la table indexée par dataset |
| `build_norm_arrays(indices, table, fallback)` | Construit les arrays `(B, 3)` par batch |

Les conversions vers d'autres échelles se font **uniquement au point d'utilisation**, via les méthodes helper de `NormStats` :

| Méthode | Échelle | Utilisé par |
|---|---|---|
| `to_dali_scale()` | [0, 255] `list[float]` | `pipeline.py`, `dynamic_pipeline.py`, `cpu.py` |
| `to_numpy()` | [0, 1] `np.ndarray` | helpers d'augmentation CPU |

**Ne jamais multiplier mean/std par 255 en ligne.** Toujours appeler `to_dali_scale()`.

---

## Architecture — responsabilité par fichier

| Fichier | Responsabilité |
|------|---------------|
| `config.py` | Dataclasses pures : `DINOAugConfig`, `LoaderConfig`, `CheckpointState`, `NormStats`, `PipelineConfig` |
| `augmentation.py` | `SampleRecord`, `SampleMeta`, `SamplePredicate` + hiérarchie `AugmentationSpec` + `split_views` par spec |
| `norm_utils.py` | `build_norm_table`, `build_norm_arrays` — partagé entre pipelines DALI statique et dynamique |
| `checkpoint.py` | `save_checkpoint(path, state)`, `load_checkpoint`, `DataLoaderCheckpointer` — toute la logique I/O + SHA-256 |
| `dali_node.py` | `_DALINode` — pilote l'itérateur backend, assemble `Batch`, métriques, stall watchdog |
| `sources/protocol.py` | `SourceProtocol` — interface commune pour toutes les sources de données |
| `sources/_weights.py` | `MixingWeights` — vecteur de poids normalisé thread-safe |
| `sources/resolution.py` | `ResolutionSource` — holder thread-safe de la résolution de crop |
| `sources/hpc_source.py` | `MixingSource`, `ShardIterator` — source de production HPC (Lustre + /dev/shm) |
| `sources/wds_source.py` | `WDSSource` — source alternative basée webdataset |
| `shard_reader.py` | `ShardReaderNode`, `_ReaderAdapter`, `build_reader_graph` — stages 1-2 : I/O shards + mixing |
| `pipeline_graph.py` | `MetadataNode`, `MaskMapNode`, `BatchMapNode`, `BatchFilterNode`, `NodePipeline`, `wrap_loader` |
| `pipeline.py` | Constructeur de pipeline DALI statique + `NormSource` |
| `memory.py` | `Batch`, `H2DStream`, `FP8Formatter`, `allocate_buffers` |
| `masking.py` | `MaskingGenerator` — générateur pur de masques de patches iBOT |
| `backends/` | Abstraction backend pluggable (DALI, CPU) |
| `monitor/` | Métriques, tracing, OTEL, CLI monitor |
| `experimental/` | `dynamic_pipeline` — mode dynamique DALI v2 (pas en production) |
| `loader.py` | `DINODataLoader` — point d'entrée principal ; orchestration sans logique métier |

### Invariants clés

- `loader.py` ne contient **aucune logique d'augmentation** ni **aucune logique de post-traitement**.
  Toute l'augmentation est dans `augmentation.py`, `pipeline.py` et les `backends/`.
  Tous les transforms post-DALI sont dans `pipeline_graph.py`.

- **`split_views` et `initial_sizes` vivent sur les specs**, pas dans `loader.py`.
  Chaque `AugmentationSpec` sait comment séparer ses propres vues et déclarer ses dimensions initiales.
  Il n'y a **pas de dispatch `isinstance`** dans `loader.py` pour ces opérations.

- **`CheckpointState` est un pur dataclass** sans `save()` ni `load()`.
  Ces méthodes sont dans `checkpoint.py` (`save_checkpoint(path, state)`, `load_checkpoint(path)`).
  **La path est toujours le premier argument** (cohérent avec les conventions Python standard).

- **`_DALINode` est dans `dali_node.py`** — il pilote l'itérateur backend et assemble les Batch.
  Importer directement depuis `dino_loader.dali_node`.

- **`_ReaderAdapter` est dans `shard_reader.py`** — il est le bridge entre `ShardReaderNode`
  et le callable attendu par les backends.  Il n'a aucune dépendance sur `loader.py`.

- **`SampleRecord`, `SampleMeta`, `SamplePredicate` sont dans `augmentation.py`** —
  ils constituent le contrat entre les sources (qui produisent des `SampleRecord`) et le
  filtrage/augmentation (qui les consomme).

- `shard_reader.py` ne connaît **pas** `pipeline_graph.py`. La dépendance est unidirectionnelle :
  `loader.py → shard_reader.py → sources/`, jamais l'inverse.

- `pipeline_graph.py` ne connaît **pas** `shard_reader.py`. Les deux sont des feuilles
  importées par `loader.py`.

- `masking.py` est un **module pur** sans dépendance à torch.distributed ou DALI.
  `MaskMapNode` dans `pipeline_graph.py` l'enveloppe pour le graphe torchdata.

- `config.py` n'importe **rien** de `dino_loader`. Il peut importer de `dino_datasets`
  uniquement pour le ré-export de `DatasetSpec`.

- Les modules `monitor/` sont importés **paresseusement** avec `# noqa: PLC0415`.

- Toutes les statistiques de normalisation passent par `NormStats`. Pas de conversion `× 255` en ligne.
  La logique de construction des arrays par batch est dans `norm_utils.py`.

---

## Sources de données — stratégie

Deux implémentations de source, toutes deux conformes à `SourceProtocol` :

### `MixingSource` (HPC, production) — `sources/hpc_source.py`
- Cache /dev/shm + double-buffering strict I/O + extraction
- Pool d'extraction partagé (`SharedExtractionPoolConfig`) — borne le budget de threads
- Recommandée sur B200 / GB200 NVL72 avec Lustre lent (≥ 8 rangs/nœud)
- `ShardIterator` est interne à `hpc_source.py` — ne pas l'importer directement depuis `loader.py`

### `WDSSource` (simple, alternative) — `sources/wds_source.py`
- Délègue cycling, shuffle et mixing à `webdataset`
- Recommandée sur NVMe local ou Lustre MDS rapide (≤ 8 rangs/nœud)

### Règle d'or
Typer les arguments de source avec `SourceProtocol`, pas avec une implémentation concrète.

---

## Checkpoint I/O

La signature canonique est **`save_checkpoint(path, state)`** — la path est le premier argument,
cohérent avec les conventions Python standard (pathlib, open, json.dump, etc.).

```python
from dino_loader.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(path, state)   # path en premier
state = load_checkpoint(path)
```

---

## Pipeline de post-traitement

`wrap_loader()` de `dino_loader.pipeline_graph` est **le seul point d'entrée** pour composer des transforms post-DALI.

```python
from dino_loader.pipeline_graph import wrap_loader

pipeline = (
    wrap_loader(DINODataLoader(...))
    .map(fn)
    .select(pred)
    .with_epoch(n)
)
```

`DINODataLoader` expose aussi des raccourcis `.map()`, `.select()`, `.with_epoch()`.

---

## Nœuds du graphe torchdata

L'API préférée pour composer des stages de pipeline est `torchdata.nodes`.
Les nouveaux stages doivent être implémentés comme sous-classes de `BaseNode`.
Les contrats clés :

- **`reset(initial_state)`** : appelé avant chaque époque ; doit être idempotent.
- **`next()`** : retourne un item ; lève `StopIteration` en fin d'époque.
- **`get_state()`** : retourne un dict JSON-sérialisable pour le checkpointing.

---

## Dispatch sur AugmentationSpec

Le dispatch sur les sous-types de `AugmentationSpec` doit être **limité aux backends** (`cpu.py`, `dali_backend.py`).  Dans le reste du codebase (loader, pipeline_graph), utiliser les méthodes polymorphes :

| Besoin | Méthode / propriété |
|---|---|
| Dimensions initiales | `spec.initial_global_size`, `spec.initial_local_size` |
| Séparer global/local | `spec.split_views(views)` |
| Noms des vues | `spec.output_map` |
| Stats de normalisation | `spec.norm_stats` |

---

## Parallélisme CPU [PERF-CPU]

`CPUAugPipeline.run_one_batch()` parallélise le décodage + augmentation JPEG via un `ThreadPoolExecutor` interne.  Le nombre de workers est `min(batch_size, cpu_count, 16)`.  La reproductibilité des tests est assurée par un seed initial commun ; les workers utilisent leurs propres états RNG thread-locaux.

`CPUAugPipeline` expose une méthode `close()` — appeler explicitement via `CPUBackend` ou dans un context manager pour éviter les fuites de threads.

---

## Tests

- **TDD** : les tests sont écrits avant ou en parallèle du code.
- **Isolation** : chaque test est indépendant. Les singletons et l'état global sont patchés dans les fixtures.
- **Tests lents** : tout test qui démarre de vrais threads `ShardIterator`, construit un graphe `tn.Loader` complet,
  ou exécute plusieurs opérations d'I/O de shards doit être décoré avec `@pytest.mark.slow`.

### Organisation des fichiers de tests

La convention suit la structure de `src/dino_loader/` :

- Un module de premier niveau `src/dino_loader/foo.py` → `tests/test_foo.py`
- Un sous-package `src/dino_loader/sources/` → `tests/sources/test_hpc_source.py`, `tests/sources/test_wds_source.py`, etc.
- Les fixtures partagées vivent dans `tests/fixtures/__init__.py` (fonctions pures) et `tests/conftest.py` (fixtures pytest).

`make_spec` est défini **uniquement** dans `tests/fixtures/__init__.py` et importé par `conftest.py` — pas de doublon.

### Imports dans les tests

- Importer directement depuis `shard_reader` et `pipeline_graph`.
- Importer `_DALINode` depuis `dino_loader.dali_node`.
- Importer `_ReaderAdapter` depuis `dino_loader.shard_reader`.

---

## Performance / Invariants HPC

- **Pas de `stat()` par fichier** pendant la résolution des shards (`runtime_mode=True`).
- **DALI queues remplacent AsyncPrefetchIterator** : `dali_cpu_queue ≥ 16` est la mesure compensatoire.
- **`NormSource` copy-on-write** : `set_dataset_indices()` construit la nouvelle liste hors du lock et swap atomiquement.
- **Budget de threads** : le pool d'extraction est partagé entre tous les `ShardIterator` via `SharedExtractionPoolConfig`.
- **Parallélisme CPU** : `CPUAugPipeline` utilise un `ThreadPoolExecutor` pour paralléliser le décodage JPEG.
- **`build_norm_arrays`** est factorisé dans `norm_utils.py` — ne pas dupliquer la logique de lookup par batch.

---

## Documentation

- **README.md** : garder les dépendances runtime et la liste des modules à jour.
- **Docstrings** : modules, classes publiques, toutes les méthodes publiques. Style Google.
- **Commentaires en ligne** : *pourquoi*, jamais *quoi*.
- **`# noqa` commentaires** : toujours inclure le code spécifique (ex : `# noqa: PTH112`).