"""Root conftest.py — pytest session configuration."""

from __future__ import annotations

import asyncio
import platform

import pytest

pytest_plugins = ("dadbot.core.kernel_boundary_enforcer",)


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


@pytest.fixture(autouse=True)
def _drain_event_loop_windows() -> None:
    """Drain the Windows Proactor event loop after each test.
    
    On Windows, the Proactor transport may hold pending socket operations.
    This fixture ensures the loop processes any final TCP FIN packets and
    cleanup operations before the test ends, preventing unclosed transport
    and socket warnings under warning-as-error mode.
    """
    yield

    if platform.system() != "Windows":
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None or loop.is_closed():
        return

    # Allow the loop to process pending callbacks and transport cleanup
    try:
        # Schedule a tiny yielding operation to drain the loop
        async def _drain_pending() -> None:
            await asyncio.sleep(0)

        # Run it synchronously if we're not in an async context
        if not loop.is_running():
            loop.run_until_complete(_drain_pending())
    except Exception:
        pass
