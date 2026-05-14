"""Mark every test in tests/stress/ with the 'stress' marker automatically."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "stress" in item.nodeid:
            item.add_marker(pytest.mark.stress)
