"""Marker 'slow': tests that take more than 60 s on a CPU (model ports, full held-out selection on Ped2). They are
skipped unless VADKIT_SLOW=1:  VADKIT_SLOW=1 python -m pytest tests/vadkit"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: more than 60 s on a CPU; runs only with VADKIT_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("VADKIT_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow (> 60 s on a CPU); set VADKIT_SLOW=1 to run")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip)
