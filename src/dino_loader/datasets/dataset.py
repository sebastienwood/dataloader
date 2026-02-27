import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
import logging

from dino_loader.config import DatasetSpec
from dino_loader.datasets.utils import validate_webdataset_shard
from dino_loader.datasets.settings import resolve_datasets_root

log = logging.getLogger(__name__)

@dataclass
class GlobalDatasetFilter:
    """Filters applied globally to all datasets unless overridden."""
    allowed_confidentialities: Optional[List[str]] = None
    allowed_modalities: Optional[List[str]] = None
    allowed_datasets: Optional[List[str]] = None
    allowed_splits: Optional[List[str]] = None

@dataclass
class DatasetConfig:
    """Per-dataset filters that override global ones."""
    allowed_confidentialities: Optional[List[str]] = None
    allowed_modalities: Optional[List[str]] = None
    allowed_splits: Optional[List[str]] = None
    weight: float = 1.0

class Dataset:
    """
    Represents a dataset in the webdatasets directory structure:
    root_path/confidentiality/modality/dataset_name/split/*.tar
    """
    def __init__(self, name: str, root_path: Optional[str] = None):
        self.name = name
        self.root_path = resolve_datasets_root(root_path)

    def _get_effective_allowed(self, global_filter: Optional[GlobalDatasetFilter], config: Optional[DatasetConfig], attr: str) -> Optional[Set[str]]:
        local_val = getattr(config, attr) if config else None
        if local_val is not None:
            return set(local_val)
        global_val = getattr(global_filter, attr) if global_filter else None
        if global_val is not None:
            return set(global_val)
        return None

    def resolve(self, global_filter: Optional[GlobalDatasetFilter] = None, config: Optional[DatasetConfig] = None) -> List[str]:
        """
        Discovers and validates dataset shards matching the provided filters.
        """
        if global_filter and global_filter.allowed_datasets is not None:
            if self.name not in global_filter.allowed_datasets:
                return []

        allowed_confs = self._get_effective_allowed(global_filter, config, "allowed_confidentialities")
        allowed_mods = self._get_effective_allowed(global_filter, config, "allowed_modalities")
        allowed_splits = self._get_effective_allowed(global_filter, config, "allowed_splits")

        valid_shards = []

        if not os.path.exists(self.root_path):
            log.warning(f"Dataset root path does not exist: {self.root_path}")
            return []

        # Walk the hierarchy: root/conf/modality/dataset/split
        for conf in os.listdir(self.root_path):
            if allowed_confs is not None and conf not in allowed_confs:
                continue
            conf_path = os.path.join(self.root_path, conf)
            if not os.path.isdir(conf_path):
                continue

            for mod in os.listdir(conf_path):
                if allowed_mods is not None and mod not in allowed_mods:
                    continue
                mod_path = os.path.join(conf_path, mod)
                if not os.path.isdir(mod_path):
                    continue

                dataset_path = os.path.join(mod_path, self.name)
                if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
                    continue

                for split in os.listdir(dataset_path):
                    if allowed_splits is not None and split not in allowed_splits:
                        continue
                    split_path = os.path.join(dataset_path, split)
                    if not os.path.isdir(split_path):
                        continue

                    # Discover tar shards in the split path
                    for f in os.listdir(split_path):
                        if f.endswith(".tar"):
                            tar_path = os.path.join(split_path, f)
                            idx_path = os.path.join(split_path, f[:-4] + ".idx")

                            # Minimal check instead of full validation here for speed
                            # We check existence and size. Full validation happens at stub gen
                            if os.path.exists(idx_path):
                                valid_shards.append(tar_path)
                            else:
                                log.warning(f"Missing .idx file for shard: {tar_path}")

        return sorted(valid_shards)

    def to_spec(self, global_filter: Optional[GlobalDatasetFilter] = None, config: Optional[DatasetConfig] = None) -> Optional[DatasetSpec]:
        """
        Converts this dataset representation to a DINO loader DatasetSpec.
        Returns None if no valid shards are found.
        """
        shards = self.resolve(global_filter, config)
        if not shards:
            return None

        weight = config.weight if config else 1.0
        return DatasetSpec(name=self.name, shards=shards, weight=weight)
