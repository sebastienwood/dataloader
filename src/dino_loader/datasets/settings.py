import os
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

def get_default_datasets_root() -> str:
    """Fallback default dataset root path relative to the module installation."""
    return "~/.dinoloader/"

def _load_toml_datasets_root() -> str | None:
    """Attempts to read the DINO_DATASETS_ROOT from a pyproject.toml in the current working directory."""
    try:
        pyproject_path = os.path.join(os.getcwd(), "pyproject.toml")
        if os.path.exists(pyproject_path):
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                
            # Path: tool.dino_loader.datasets.root
            tool = data.get("tool", {})
            dino = tool.get("dino_loader", {})
            datasets = dino.get("datasets", {})
            root = datasets.get("root")
            
            if root:
                return os.path.abspath(root)
    except Exception:
        pass
    return None

def resolve_datasets_root(arg_path: str | None = None) -> str:
    """
    Resolves the root datasets path by following strictly ordered precedence:
    1. Direct Argument Override (arg_path)
    2. TOML Configuration (tool.dino_loader.datasets.root in pyproject.toml)
    3. Environment Variable (DINO_DATASETS_ROOT)
    4. Code Default Fallback (../webdatasets relative to this file's grand-parent tree)
    """
    if arg_path is not None:
        return arg_path
        
    toml_path = _load_toml_datasets_root()
    if toml_path is not None:
        return toml_path
        
    env_path = os.environ.get("DINO_DATASETS_ROOT")
    if env_path is not None:
        return env_path
        
    return get_default_datasets_root()
