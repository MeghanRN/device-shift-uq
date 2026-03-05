import os
from dataclasses import dataclass

@dataclass
class DataPaths:
    root: str
    splits: str

def default_paths(dataset: str) -> DataPaths:
    root = os.path.join("data", dataset)
    return DataPaths(root=root, splits=os.path.join(root, "splits"))
