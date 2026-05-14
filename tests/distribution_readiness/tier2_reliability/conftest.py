"""Tier 2 conftest — applies the tier2 marker to all reliability tests."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    tier2_prefix = "tests/distribution_readiness/tier2_reliability"
    for item in items:
        node_path = item.nodeid.replace("\\", "/")
        if tier2_prefix in node_path:
            item.add_marker(pytest.mark.tier2)
