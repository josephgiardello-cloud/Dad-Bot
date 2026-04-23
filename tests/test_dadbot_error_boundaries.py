"""Regression tests for DadBot error boundaries and fallback semantics."""
from __future__ import annotations

from dadbot.core.dadbot import DadBot


class DummyTurnService:
    def process_user_message(self, user_input, attachments=None):
        raise ConnectionError("Ollama unreachable")


def test_process_user_message_returns_finalized_turn_result_on_ollama_failure():
    bot = DadBot.__new__(DadBot)
    bot._turn_graph_enabled = False
    bot.turn_service = DummyTurnService()
    bot.finalize_reply = lambda text: text
    bot.ollama_retryable_errors = staticmethod(lambda: (ConnectionError, TimeoutError, OSError))

    reply, should_end = bot.process_user_message("Hello")

    assert "connection seems to be down" in reply
    assert should_end is False
