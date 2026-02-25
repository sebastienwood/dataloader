import os
import sys
import logging
from typing import Dict, Set
from dino_loader.datasets.dataset import Dataset
from dino_loader.datasets.utils import validate_webdataset_shard

log = logging.getLogger(__name__)

def generate_stubs(root_path: str = None, output_file: str = None):
    """
    Scans the webdatasets directory and generates a hub.py file with IDE stubs.
    Checks the validity of .tar shards and .idx files during discovery.
    """
    # Instantiate a dummy dataset just to get the default root path logic
    dummy = Dataset("dummy", root_path=root_path)
    base_dir = dummy.root_path
    
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "hub.py")

    if not os.path.exists(base_dir):
        log.warning(f"Root dataset directory not found: {base_dir}")
        with open(output_file, 'w') as f:
            f.write("# Auto-generated stubs\n")
            f.write("from dino_loader.datasets.dataset import Dataset\n\n")
            f.write("# No dataset directory found at generation time.\n")
        return

    # dataset_name -> {"confidentialities": set(), "modalities": set(), "splits": set()}
    datasets_info: Dict[str, Dict[str, Set[str]]] = {}

    for conf in os.listdir(base_dir):
        conf_path = os.path.join(base_dir, conf)
        if not os.path.isdir(conf_path): continue
        
        for mod in os.listdir(conf_path):
            mod_path = os.path.join(conf_path, mod)
            if not os.path.isdir(mod_path): continue
            
            for dname in os.listdir(mod_path):
                dataset_path = os.path.join(mod_path, dname)
                if not os.path.isdir(dataset_path): continue
                
                if dname not in datasets_info:
                    datasets_info[dname] = {
                        "confidentialities": set(),
                        "modalities": set(),
                        "splits": set()
                    }
                
                datasets_info[dname]["confidentialities"].add(conf)
                datasets_info[dname]["modalities"].add(mod)
                
                for split in os.listdir(dataset_path):
                    split_path = os.path.join(dataset_path, split)
                    if not os.path.isdir(split_path): continue
                    
                    has_valid_shard = False
                    for f in os.listdir(split_path):
                        if f.endswith(".tar"):
                            tar_path = os.path.join(split_path, f)
                            idx_path = os.path.join(split_path, f[:-4] + ".idx")
                            if validate_webdataset_shard(tar_path, idx_path):
                                has_valid_shard = True
                                break # Just need one valid shard to confirm split
                            else:
                                log.warning(f"Corrupted or invalid shard found: {tar_path}")
                    
                    if has_valid_shard:
                        datasets_info[dname]["splits"].add(split)

    # Generate the stub file
    with open(output_file, 'w') as f:
        f.write("# Auto-generated dataset stubs by dino_loader.datasets.stub_gen\n")
        f.write("# Do not edit manually.\n\n")
        f.write("from dino_loader.datasets.dataset import Dataset\n\n")
        
        for dname, info in sorted(datasets_info.items()):
            confs = sorted(list(info["confidentialities"]))
            mods = sorted(list(info["modalities"]))
            splits = sorted(list(info["splits"]))
            
            f.write(f"{dname}: Dataset = Dataset('{dname}')\n")
            f.write(f'"""\n')
            f.write(f'Dataset: {dname}\n')
            f.write(f'Supported Confidentialities: {", ".join(confs)}\n')
            f.write(f'Supported Modalities: {", ".join(mods)}\n')
            f.write(f'Available Splits: {", ".join(splits)}\n')
            f.write(f'"""\n\n')

if __name__ == "__main__":
    generate_stubs()
