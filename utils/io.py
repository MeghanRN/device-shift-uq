import os, json
from typing import Any, Dict

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
