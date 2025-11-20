import sys
from pathlib import Path


def _ensure_useful_on_path():
    """Allow test modules to import the utilities as top-level modules."""
    useful_path = Path(__file__).resolve().parents[1]
    path_str = str(useful_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_useful_on_path()
