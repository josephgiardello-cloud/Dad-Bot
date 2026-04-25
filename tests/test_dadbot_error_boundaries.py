"""Regression tests for DadBot error boundaries and fallback semantics."""
from __future__ import annotations

from dadbot.core.dadbot import DadBot
from dadbot.core.observability import CorrelationContext


def test_process_user_message_returns_finalized_turn_result_on_graph_failure():
    """Graph failures in strict mode must raise and never fallback."""
    bot = DadBot.__new__(DadBot)
    bot._turn_graph_enabled = True
    bot._strict_graph_mode = True
    bot.finalize_reply = lambda text: text

    def boom(user_input, attachments=None):
        raise ConnectionError("Ollama unreachable")

    bot._run_graph_turn_sync = boom
    bot._emit_graph_failure_event = lambda **_kw: None
    bot._append_signoff_compat = lambda text: text

    with CorrelationContext.bind("test-corr-001"):
        try:
            bot.process_user_message("Hello")
            assert False, "Expected strict graph failure"
        except RuntimeError as exc:
            assert "strict mode" in str(exc)
