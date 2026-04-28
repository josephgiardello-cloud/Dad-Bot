from __future__ import annotations

from types import SimpleNamespace

from dadbot.agentic import AgenticHandler, ToolRegistry
from dadbot.core.egress_policy import evaluate_url


def test_evaluate_url_allowlist_decision() -> None:
    allowed = evaluate_url("https://api.duckduckgo.com/?q=dad", allowlist=("api.duckduckgo.com",))
    blocked = evaluate_url("https://example.com", allowlist=("api.duckduckgo.com",))
    assert allowed.allowed is True
    assert blocked.allowed is False


def test_agentic_lookup_web_blocked_by_allowlist() -> None:
    bot = SimpleNamespace(config=SimpleNamespace(egress_allowlist=("localhost",)))
    handler = AgenticHandler(bot=bot, tool_registry=ToolRegistry(bot))
    assert handler.lookup_web("weather") is None
