import os
import argparse
from typing import Dict, Set

from dino_loader.datasets.dataset import Dataset
from dino_loader.datasets.stub_gen import generate_stubs

def preview_datasets(root_path: str = None):
    """
    Prints a tree-like view of available datasets organized by confidentiality/modality.
    """
    dummy = Dataset("dummy", root_path=root_path)
    base_dir = dummy.root_path
    if not os.path.exists(base_dir):
        print(f"Error: Dataset root {base_dir} does not exist.")
        return
        
    print(f"Dataset Root: {base_dir}\n")
    
    for conf in os.listdir(base_dir):
        conf_path = os.path.join(base_dir, conf)
        if not os.path.isdir(conf_path): continue
        print(f"📂 {conf}/")
        
        for mod in os.listdir(conf_path):
            mod_path = os.path.join(conf_path, mod)
            if not os.path.isdir(mod_path): continue
            print(f"  📂 {mod}/")
            
            for dname in os.listdir(mod_path):
                dataset_path = os.path.join(mod_path, dname)
                if not os.path.isdir(dataset_path): continue
                print(f"    📦 {dname}")
                
                for split in os.listdir(dataset_path):
                    split_path = os.path.join(dataset_path, split)
                    if not os.path.isdir(split_path): continue
                    
                    # count shards
                    shards = [f for f in os.listdir(split_path) if f.endswith(".tar")]
                    print(f"      └── {split} ({len(shards)} shards)")

def count_elements(dataset_name: str, root_path: str = None):
    """
    Approximates count of images in a dataset by counting lines in its .idx files.
    """
    dataset = Dataset(dataset_name, root_path=root_path)
    shards = dataset.resolve()
    if not shards:
        print(f"No valid shards found for dataset '{dataset_name}'.")
        return
        
    total_count = 0
    for tar_path in shards:
        idx_path = tar_path[:-4] + ".idx"
        if os.path.exists(idx_path):
            try:
                # We assume each line in the idx file corresponds to a dataset item
                with open(idx_path, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                    total_count += count
            except Exception as e:
                print(f"Warning: could not read {idx_path} ({e})")
                
    print(f"Dataset '{dataset_name}': ~{total_count} items across {len(shards)} valid shards.")

def add_dataset(conf: str, mod: str, name: str, split: str, root_path: str = None):
    """
    Scaffolds the directory structure for a new dataset split.
    """
    dummy = Dataset("dummy", root_path=root_path)
    base_dir = dummy.root_path
    
    target_dir = os.path.join(base_dir, conf, mod, name, split)
    os.makedirs(target_dir, exist_ok=True)
    print(f"✅ Scaffolded empty dataset directory at:\n  {target_dir}")
    print("  You can now drop your .tar and .idx files here.")

def main():
    parser = argparse.ArgumentParser(description="DINO Dataloader Datasets Hub CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # preview
    preview_parser = subparsers.add_parser("preview", help="Preview available datasets")
    preview_parser.add_argument("--root", type=str, help="Dataset root path override")
    
    # count
    count_parser = subparsers.add_parser("count", help="Count items in a dataset")
    count_parser.add_argument("name", type=str, help="Name of the dataset")
    count_parser.add_argument("--root", type=str, help="Dataset root path override")
    
    # add
    add_parser = subparsers.add_parser("add", help="Scaffold a new dataset directory")
    add_parser.add_argument("conf", type=str, help="Confidentiality level (e.g. public, private)")
    add_parser.add_argument("mod", type=str, help="Modality (e.g. rgb, multispectral)")
    add_parser.add_argument("name", type=str, help="Dataset name")
    add_parser.add_argument("split", type=str, help="Split name (e.g. train, val)")
    add_parser.add_argument("--root", type=str, help="Dataset root path override")
    
    # stubs
    stubs_parser = subparsers.add_parser("stubs", help="Generate IDE stubs (hub.py)")
    stubs_parser.add_argument("--root", type=str, help="Dataset root path override")

    args = parser.parse_args()
    
    if args.command == "preview":
        preview_datasets(args.root)
    elif args.command == "count":
        count_elements(args.name, args.root)
    elif args.command == "add":
        add_dataset(args.conf, args.mod, args.name, args.split, args.root)
    elif args.command == "stubs":
        generate_stubs(args.root)
        print("✅ Stubs generated at src/dino_loader/datasets/hub.py")

if __name__ == "__main__":
    main()
