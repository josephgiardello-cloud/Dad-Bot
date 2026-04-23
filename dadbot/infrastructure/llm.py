from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol

import ollama


class ModelAdapter(Protocol):
    """Contract for model backends."""

    def list_models(self) -> list[dict[str, Any]]: ...

    def pull_model(self, model_name: str) -> None: ...


class OllamaModelAdapter:
    """Ollama implementation of the ModelAdapter contract."""

    def list_models(self) -> list[dict[str, Any]]:
        response = ollama.list()
        return list(response.get("models", []))

    def pull_model(self, model_name: str) -> None:
        ollama.pull(model_name)

    @staticmethod
    def model_is_available(models: Iterable[dict[str, Any]], model_name: str) -> bool:
        """Determines if a model is ready, handling Ollama tag quirks."""
        available_names = {
            model.get("model") or model.get("name")
            for model in models
            if model.get("model") or model.get("name")
        }

        if model_name in available_names:
            return True

        if ":" not in model_name:
            return any(str(name).startswith(f"{model_name}:") for name in available_names)

        return False

    def ensure_ready(
        self,
        model_candidates: Iterable[str],
        *,
        status_callback: Callable[[str], Any] | None = None,
        deliver_status: Callable[[str, Callable[[str], Any] | None], Any] | None = None,
        finalize_reply: Callable[[str], str] | None = None,
        retryable_errors: tuple[type[BaseException], ...] = (Exception,),
    ) -> str | None:
        try:
            models = self.list_models()
        except retryable_errors:
            if deliver_status and finalize_reply:
                deliver_status(finalize_reply("I can't reach Ollama right now. Make sure the Ollama app is open, then try again."), status_callback)
            return None

        for candidate in model_candidates:
            if self.model_is_available(models, candidate):
                return candidate

        for candidate in model_candidates:
            if deliver_status:
                deliver_status(f"I don't have {candidate} downloaded yet. Give me a minute to get it ready for you.", status_callback)
            try:
                self.pull_model(candidate)
                if deliver_status and finalize_reply:
                    deliver_status(finalize_reply(f"{candidate} is ready. Let's talk."), status_callback)
                return candidate
            except retryable_errors:
                continue

        if deliver_status and finalize_reply:
            deliver_status(finalize_reply("I couldn't download my brain just yet. Make sure Ollama is online, then try again."), status_callback)
        return None
