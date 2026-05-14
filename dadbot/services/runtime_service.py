from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dadbot.infrastructure.llm import ModelAdapter, OllamaModelAdapter


class RuntimeService:
    """Service wrapper around runtime client readiness/model checks."""

    def __init__(self, bot: Any, model_adapter: ModelAdapter | None = None):
        self.bot = bot
        self.model_adapter = model_adapter or OllamaModelAdapter()

    def model_is_available(self, models: list[dict[str, Any]], model_name: str) -> bool:
        return OllamaModelAdapter.model_is_available(models, model_name)

    def ensure_ollama_ready(
        self,
        status_callback: Callable[[str], Any] | None = None,
    ) -> bool:
        selected = self.model_adapter.ensure_ready(
            self.bot.model_candidates(),
            status_callback=status_callback,
            deliver_status=self.bot.deliver_status_message,
            finalize_reply=self.bot.finalize_reply,
            retryable_errors=self.bot.ollama_retryable_errors(),
        )
        if selected:
            self.bot.ACTIVE_MODEL = selected
            return True
        return False
