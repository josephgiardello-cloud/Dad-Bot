"""Tier 3 conftest — applies the tier3 marker to all advisory tests."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    tier3_prefix = "tests/distribution_readiness/tier3_advisory"
    for item in items:
        node_path = item.nodeid.replace("\\", "/")
        if tier3_prefix in node_path:
            item.add_marker(pytest.mark.tier3)
