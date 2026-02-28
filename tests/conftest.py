from pathlib import Path

import pytest

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
