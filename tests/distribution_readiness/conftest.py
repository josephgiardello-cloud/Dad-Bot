"""conftest for Distribution Readiness Gate tests.

Applies the distribution_readiness marker to every test collected under this
folder.  Tier sub-markers are applied per subfolder via their own conftest.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    gate_prefix = "tests/distribution_readiness"
    # Normalize to forward slashes for Windows compatibility
    for item in items:
        node_path = item.nodeid.replace("\\", "/")
        if gate_prefix in node_path:
            item.add_marker(pytest.mark.distribution_readiness)
