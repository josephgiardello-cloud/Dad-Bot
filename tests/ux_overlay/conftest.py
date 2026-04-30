from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_experimental_runtime(monkeypatch):
    """Allow ux_overlay tests to run the experimental runtime path."""
    monkeypatch.setenv("DADBOT_ENABLE_EXPERIMENTAL_RUNTIME", "1")
