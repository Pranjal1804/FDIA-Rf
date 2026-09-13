import yaml
import argparse
from pathlib import Path

def load_config(config_path: str = "configs/default.yaml"):
    """
    Loads YAML config from the given path.
    If full path is given and exists, uses it; otherwise assumes it's relative to the project root.
    """
    path_obj = Path(config_path)
    if not path_obj.is_absolute():
        base_path = Path(__file__).parent.parent
        path_obj = base_path / config_path
        
    with open(path_obj, "r") as f:
        return yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser(description="FDIA Selective Verification")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    return parser.parse_args()
