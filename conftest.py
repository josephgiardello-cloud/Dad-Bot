"""Root conftest.py — pytest session configuration."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-stress",
        action="store_true",
        default=False,
        help="Include stress/integration tests (skipped by default).",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Include slow/soak tests (skipped by default).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-stress"):
        return
    skip_stress = pytest.mark.skip(reason="Stress tests skipped — pass --run-stress to enable.")
    skip_slow = pytest.mark.skip(reason="Slow tests skipped — pass --run-slow to enable.")
    run_slow = config.getoption("--run-slow")
    for item in items:
        if "stress" in item.nodeid or item.get_closest_marker("stress"):
            item.add_marker(skip_stress)
        elif not run_slow and item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
