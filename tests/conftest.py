import sys
from pathlib import Path

import pytest


def _ensure_useful_on_path():
    """Allow test modules to import the utilities as top-level modules."""
    useful_path = Path(__file__).resolve().parents[1]
    path_str = str(useful_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_useful_on_path()


_CATEGORY_MARKERS = {
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
    "property": pytest.mark.property,
    "regression": pytest.mark.regression,
}


def pytest_collection_modifyitems(config, items):
    """Auto-tag tests by top-level folder category."""
    for item in items:
        path_parts = Path(str(item.fspath)).parts
        for category, marker in _CATEGORY_MARKERS.items():
            if category in path_parts:
                item.add_marker(marker)
                break
        if "test_smoke_pipeline.py" in path_parts:
            item.add_marker(pytest.mark.smoke)
