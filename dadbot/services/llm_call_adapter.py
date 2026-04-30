"""llm_call_adapter.py — Single boundary for all LLM I/O in TurnService.

Wraps bot.call_ollama_chat / call_ollama_chat_async.
Receives payload, returns response dict.  No state mutation.
"""
from __future__ import annotations

from typing import Any


class LLMCallAdapter:
    """Adapts all LLM model calls behind a single explicit boundary.

    TurnService receives this via constructor injection.
    No direct call_ollama_* calls may appear in turn_service.py.
    """

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    def call(
        self,
        *,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        response_format: str | None = None,
        purpose: str = "",
    ) -> dict[str, Any]:
        return self._bot.call_ollama_chat(
            messages=messages,
            options=options or {},
            response_format=response_format,
            purpose=purpose,
        )

    async def call_async(
        self,
        *,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        response_format: str | None = None,
        purpose: str = "",
    ) -> dict[str, Any]:
        return await self._bot.call_ollama_chat_async(
            messages=messages,
            options=options or {},
            response_format=response_format,
            purpose=purpose,
        )
