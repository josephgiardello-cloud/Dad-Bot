from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import ollama


class ModelAdapter(Protocol):
    """Contract for model backends."""

    def list_models(self) -> list[dict[str, Any]]: ...

    def pull_model(self, model_name: str) -> None: ...


class OllamaModelAdapter:
    """Ollama implementation of the ModelAdapter contract."""

    def list_models(self) -> list[dict[str, Any]]:
        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", None)
        if not isinstance(models, list):
            raise ValueError("Ollama returned an unsupported model-list response.")

        normalized = []
        for model in models:
            if isinstance(model, dict):
                normalized.append(model)
                continue
            model_dump = getattr(model, "model_dump", None)
            if callable(model_dump):
                normalized.append(model_dump())
                continue
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if not name:
                raise ValueError("Ollama returned a model without a name.")
            normalized.append({"model": name})
        return normalized

    def pull_model(self, model_name: str) -> None:
        ollama.pull(model_name)

    @staticmethod
    def model_is_available(models: Iterable[dict[str, Any]], model_name: str) -> bool:
        """Determines if a model is ready, handling Ollama tag quirks."""
        available_names = set()
        for model in models:
            if isinstance(model, dict):
                name = model.get("model") or model.get("name")
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                available_names.add(str(name).strip().lower())

        normalized_model_name = str(model_name or "").strip().lower()

        if normalized_model_name in available_names:
            return True

        if ":" not in normalized_model_name:
            return any(name.startswith(f"{normalized_model_name}:") for name in available_names)

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
                deliver_status(
                    finalize_reply(
                        "I can't reach Ollama right now. Make sure the Ollama app is open, then try again.",
                    ),
                    status_callback,
                )
            return None

        for candidate in model_candidates:
            if self.model_is_available(models, candidate):
                return candidate

        for candidate in model_candidates:
            if deliver_status:
                deliver_status(
                    f"I don't have {candidate} downloaded yet. Give me a minute to get it ready for you.",
                    status_callback,
                )
            try:
                self.pull_model(candidate)
                if deliver_status and finalize_reply:
                    deliver_status(
                        finalize_reply(f"{candidate} is ready. Let's talk."),
                        status_callback,
                    )
                return candidate
            except retryable_errors:
                continue

        if deliver_status and finalize_reply:
            deliver_status(
                finalize_reply(
                    "I couldn't download my brain just yet. Make sure Ollama is online, then try again.",
                ),
                status_callback,
            )
        return None
