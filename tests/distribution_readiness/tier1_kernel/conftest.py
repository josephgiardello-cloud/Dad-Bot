"""Tier 1 conftest — applies the tier1 marker to all kernel tests."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    tier1_prefix = "tests/distribution_readiness/tier1_kernel"
    for item in items:
        node_path = item.nodeid.replace("\\", "/")
        if tier1_prefix in node_path:
            item.add_marker(pytest.mark.tier1)
