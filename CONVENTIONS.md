# dino_loader — Code Conventions

> Ce document est lu automatiquement au début de chaque session de travail sur ce projet.

---

## Version Python et style

- **Version requise : Python ≥ 3.12.** Utiliser `tomllib` (stdlib), jamais le backport `tomli`.
- **Pas de `from __future__ import annotations`** — utiliser la syntaxe native 3.12.
  Toutes les annotations de type utilisent les génériques built-in (`list[str]`, `dict[str, int]`, `X | Y`).
- **Alias de type PEP 695** (`type Foo = ...`) préférés à `TypeAlias` de `typing`.
- **`match` / `case`** préféré aux longues chaînes `isinstance` pour le dispatch sur les hiérarchies de types.
- **`pathlib.Path`** utilisé partout. Quand `os.*` ou les chemins `str` sont inévitables, ajouter `# noqa: PTH<code>`.
- **Annotations** : toutes les fonctions et méthodes publiques sont annotées, y compris `__iter__`.

---

## Imports

- **Niveau module uniquement.** Les imports lourds conditionnels (`torch.distributed`, `nvidia.dali`, etc.)
  restent locaux à leur fonction avec `# noqa: PLC0415` et un commentaire de justification.
- **Pas d'imports circulaires.** L'ordre de dépendance est :
  `config → augmentation → sources → shard_reader → pipeline_graph → backends → loader`.
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

- **`@dataclass(frozen=True)`** pour les value objects immutables (`DatasetSpec`, `ClusterTopology`, `NormStats`, …).
- **`@dataclass`** (mutable) pour les objets de configuration (`LoaderConfig`, `DINOAugConfig`, `CheckpointState`).
- **`dataclasses.replace()`** (alias `_dc_replace`) est la seule façon de produire des copies modifiées des dataclasses frozen.

---

## Statistiques de normalisation

Toutes les statistiques de normalisation sont stockées et transmises en **échelle [0, 1]** dans tout le codebase.
La source de vérité unique est `NormStats` (dans `config.py`).

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
| `config.py` | Dataclasses pures : `DINOAugConfig`, `LoaderConfig`, `CheckpointState`, `NormStats` |
| `augmentation.py` | Hiérarchie `AugmentationSpec` + protocole `SamplePredicate` |
| `sources/protocol.py` | `SourceProtocol` — interface commune pour toutes les sources de données |
| `sources/_weights.py` | `MixingWeights` — vecteur de poids normalisé thread-safe |
| `sources/resolution.py` | `ResolutionSource` — holder thread-safe de la résolution de crop |
| `sources/hpc_source.py` | `MixingSource`, `ShardIterator` — source de production HPC (Lustre + /dev/shm) |
| `sources/wds_source.py` | `WDSSource` — source alternative basée webdataset |
| `shard_reader.py` | `ShardReaderNode`, `build_reader_graph` — stages 1-2 : I/O shards + mixing |
| `pipeline_graph.py` | `_DALINode`, `MetadataNode`, `MaskMapNode`, `BatchMapNode`, `BatchFilterNode`, `NodePipeline`, `wrap_loader` |
| `pipeline.py` | Constructeur de pipeline DALI statique + `NormSource` |
| `memory.py` | `Batch`, `H2DStream`, `FP8Formatter`, `allocate_buffers` |
| `checkpoint.py` | `DataLoaderCheckpointer` — I/O JSON atomique, pointeur LATEST |
| `loader.py` | `DINODataLoader` — point d'entrée principal ; pas de logique de post-traitement |
| `masking.py` | `MaskingGenerator` — générateur pur de masques de patches iBOT |
| `nodes.py` | **Shim de compatibilité uniquement** — ré-exporte depuis `shard_reader` et `pipeline_graph` |
| `backends/` | Abstraction backend pluggable (DALI, CPU) |
| `monitor/` | Métriques, tracing, OTEL, CLI monitor |
| `experimental/` | `dynamic_pipeline` — mode dynamique DALI v2 (pas en production) |

### Invariants clés

- `loader.py` ne contient **aucune logique d'augmentation** ni **aucune logique de post-traitement**.
  Toute l'augmentation est dans `augmentation.py`, `pipeline.py` et les `backends/`.
  Tous les transforms post-DALI sont dans `pipeline_graph.py`.

- `shard_reader.py` ne connaît **pas** `loader.py`. La dépendance est unidirectionnelle :
  `loader.py → shard_reader.py → sources/`, jamais l'inverse.

- `pipeline_graph.py` ne connaît **pas** `shard_reader.py`. Les deux sont des feuilles
  importées par `loader.py`. Cela évite tout couplage entre l'I/O et les transforms de batch.

- `nodes.py` est **uniquement un shim de compatibilité**. Ne pas y ajouter de logique.
  Les nouveaux modules doivent importer directement depuis `shard_reader` et `pipeline_graph`.

- `masking.py` est un **module pur** sans dépendance à torch.distributed ou DALI.
  `MaskMapNode` dans `pipeline_graph.py` l'enveloppe pour le graphe torchdata.

- `config.py` n'importe **rien** de `dino_loader`. Il peut importer de `dino_datasets`
  uniquement pour le re-export de `DatasetSpec`.

- Les modules `monitor/` sont importés **paresseusement** dans les fonctions avec `# noqa: PLC0415`.

- Toutes les statistiques de normalisation passent par `NormStats`. Pas de conversion `× 255` en ligne.

---

## Sources de données — stratégie

Deux implémentations de source, toutes deux conformes à `SourceProtocol` :

### `MixingSource` (HPC, production)
- Cache /dev/shm + double-buffering strict I/O + extraction
- Pool d'extraction partagé (`SharedExtractionPoolConfig`) — borne le budget de threads
- Recommandée sur B200 / GB200 NVL72 avec Lustre lent (≥ 8 rangs/nœud)

### `WDSSource` (simple, alternative)
- Délègue cycling, shuffle et mixing à `webdataset`
- Recommandée sur NVMe local ou Lustre MDS rapide (≤ 8 rangs/nœud)
- Plus simple à déboguer

### Règle d'or

Typer les arguments de source avec `SourceProtocol`, pas avec une implémentation concrète.
`ShardReaderNode` accepte une source injectée via `source=` ; sans injection, `MixingSource` est utilisée par défaut.

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

`DINODataLoader` expose aussi des raccourcis `.map()`, `.select()`, `.with_epoch()` qui délèguent à son `NodePipeline` interne.
`NodePipeline` fournit un `state_dict` complet sur tout le graphe.

---

## Nœuds du graphe torchdata

L'API préférée pour composer des stages de pipeline est `torchdata.nodes`.
Les nouveaux stages doivent être implémentés comme sous-classes de `BaseNode`.
Les contrats clés :

- **`reset(initial_state)`** : appelé avant chaque époque ; doit être idempotent.
- **`next()`** : retourne un item ; lève `StopIteration` en fin d'époque.
- **`get_state()`** : retourne un dict JSON-sérialisable pour le checkpointing.

`wrap_loader(dino_loader)` bridge un `DINODataLoader` dans ce graphe.

---

## Pipeline dynamique (`experimental/dynamic_pipeline.py`)

Le pipeline dynamique est **expérimental** au sens où il dépend de
`nvidia.dali.experimental.dynamic`, une API NVIDIA susceptible de changer.

### Contrat de randomness

Tous les paramètres stochastiques dans les fonctions de batch dynamiques **doivent** utiliser
`ndd.random.*`, jamais Python `random` ou `numpy.random`.

---

## Tests

- **TDD** : les tests sont écrits avant ou en parallèle du code.
- **Isolation** : chaque test est indépendant. Les singletons et l'état global sont patchés dans les fixtures.
- **Tests lents** : tout test qui démarre de vrais threads `ShardIterator`, construit un graphe `tn.Loader` complet,
  ou exécute plusieurs opérations d'I/O de shards doit être décoré avec `@pytest.mark.slow`.
- **Imports directs** : les nouveaux tests importent depuis `shard_reader` et `pipeline_graph`,
  pas depuis `nodes` (shim de compatibilité).
- **Pas de `from __future__ import annotations`** dans les fichiers de test.

---

## Performance / Invariants HPC

- **Pas de `stat()` par fichier** pendant la résolution des shards (`runtime_mode=True`).
- **DALI queues remplacent AsyncPrefetchIterator** : `dali_cpu_queue ≥ 16` est la mesure compensatoire.
- **`NormSource` copy-on-write** : `set_dataset_indices()` construit la nouvelle liste hors du lock et swap atomiquement.
- **Budget de threads** : le pool d'extraction est partagé entre tous les `ShardIterator` via `SharedExtractionPoolConfig`.

---

## Documentation

- **README.md** : garder les dépendances runtime à jour.
- **Docstrings** : modules, classes publiques, toutes les méthodes publiques. Style Google.
- **Commentaires en ligne** : *pourquoi*, jamais *quoi*.
- **`# noqa` commentaires** : toujours inclure le code spécifique (ex : `# noqa: PTH112`).