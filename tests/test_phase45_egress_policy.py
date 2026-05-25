from __future__ import annotations

from types import SimpleNamespace

import pytest

 
from dadbot.core.egress_policy import evaluate_url

pytestmark = pytest.mark.phase4


def test_evaluate_url_allowlist_decision() -> None:
    allowed = evaluate_url("https://api.duckduckgo.com/?q=dad", allowlist=("api.duckduckgo.com",))
    blocked = evaluate_url("https://example.com", allowlist=("api.duckduckgo.com",))
    assert allowed.allowed is True
    assert blocked.allowed is False


