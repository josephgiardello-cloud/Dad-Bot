"""Ollama-backed implementation of ModelPort."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable

from dadbot.core.egress_policy import enforce_url
from dadbot.core.execution_boundary import ModelGatewayScope
from dadbot.core.execution_trace_context import (
    record_execution_step,
    record_external_system_call,
)

from dadbot.runtime.model.model_call_port import (
    DeterminismContext,
    ModelConfig,
    ModelCallError,
    ModelPort,
)


class OllamaModelAdapter(ModelPort):
    """Deterministic model port adapter backed by Ollama via runtime_client.
    
    This adapter wraps existing DadBot infrastructure:
    - self.runtime_client for generation
    - self.model_runtime for tokenization
    
    No logic changes — pure interface boundary enforcement.
    """

    def __init__(
        self,
        runtime_client: Any,  # DadBot.runtime_client
        model_runtime: Any,  # DadBot.model_runtime
        config: ModelConfig,
    ):
        """Initialize adapter.
        
        Args:
            runtime_client: Existing DadBot.runtime_client manager
            model_runtime: Existing DadBot.model_runtime manager
            config: Model configuration (active model, temperature, etc.)
        """
        self.runtime_client = runtime_client
        self.model_runtime = model_runtime
        self.config = config

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        determinism_context: DeterminismContext | None = None,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
        purpose: str = "chat",
        **kwargs: Any,
    ) -> str:
        """Generate via runtime_client.call_llm()."""
        try:
            self._enforce_model_egress()
            input_hash = self._messages_hash(messages)
            record_execution_step(
                "model_call",
                payload={
                    "mode": "sync",
                    "provider": "ollama",
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "chat"),
                    "message_count": len(list(messages or [])),
                    "input_hash": input_hash,
                    "stream": bool(stream),
                },
                required=True,
            )
            with ModelGatewayScope.bind("ModelPort"):
                raw_output = self.runtime_client.call_llm(
                    messages,
                    caller="ModelPort",
                    model=model,
                    temperature=temperature,
                    response_format=response_format,
                    determinism_context=determinism_context,
                    stream=stream,
                    chunk_callback=chunk_callback,
                    purpose=purpose,
                    **kwargs,
                )
            record_external_system_call(
                operation="model_generation",
                system="ollama",
                request_payload={
                    "mode": "sync",
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "chat"),
                    "message_count": len(list(messages or [])),
                    "input_hash": input_hash,
                },
                response_payload={"status": "ok"},
                status="ok",
                source="model_adapter",
                required=True,
            )
            normalized = self.normalize_output_for_lock(
                raw_output,
                determinism_context=determinism_context,
            )
            if determinism_context is not None and str(determinism_context.lock_hash or "").strip():
                self._last_lock_normalized_output_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "lock_hash": str(determinism_context.lock_hash),
                            "normalized_output": normalized,
                        },
                        sort_keys=True,
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest()
            record_execution_step(
                "model_output",
                payload={
                    "mode": "sync",
                    "output_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    "output_length": len(normalized),
                },
                required=True,
            )
            return normalized
        except Exception as exc:
            record_external_system_call(
                operation="model_generation",
                system="ollama",
                request_payload={
                    "mode": "sync",
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "chat"),
                },
                response_payload={"status": "error", "error": str(exc)},
                status="error",
                source="model_adapter",
                required=False,
            )
            raise ModelCallError(f"Generation failed: {exc}") from exc

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        determinism_context: DeterminismContext | None = None,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
        purpose: str = "chat",
        **kwargs: Any,
    ) -> str:
        """Generate async via runtime_client.call_llm_async()."""
        try:
            self._enforce_model_egress()
            input_hash = self._messages_hash(messages)
            record_execution_step(
                "model_call",
                payload={
                    "mode": "async",
                    "provider": "ollama",
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "chat"),
                    "message_count": len(list(messages or [])),
                    "input_hash": input_hash,
                    "stream": bool(stream),
                },
                required=True,
            )
            with ModelGatewayScope.bind("ModelPort"):
                raw_output = await self.runtime_client.call_llm_async(
                    messages,
                    caller="ModelPort",
                    model=model,
                    temperature=temperature,
                    response_format=response_format,
                    determinism_context=determinism_context,
                    stream=stream,
                    chunk_callback=chunk_callback,
                    purpose=purpose,
                    **kwargs,
                )
            record_external_system_call(
                operation="model_generation",
                system="ollama",
                request_payload={
                    "mode": "async",
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "chat"),
                    "message_count": len(list(messages or [])),
                    "input_hash": input_hash,
                },
                response_payload={"status": "ok"},
                status="ok",
                source="model_adapter",
                required=True,
            )
            normalized = self.normalize_output_for_lock(
                raw_output,
                determinism_context=determinism_context,
            )
            if determinism_context is not None and str(determinism_context.lock_hash or "").strip():
                self._last_lock_normalized_output_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "lock_hash": str(determinism_context.lock_hash),
                            "normalized_output": normalized,
                        },
                        sort_keys=True,
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest()
            record_execution_step(
                "model_output",
                payload={
                    "mode": "async",
                    "output_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    "output_length": len(normalized),
                },
                required=True,
            )
            return normalized
        except Exception as exc:
            record_external_system_call(
                operation="model_generation",
                system="ollama",
                request_payload={
                    "mode": "async",
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "chat"),
                },
                response_payload={"status": "error", "error": str(exc)},
                status="error",
                source="model_adapter",
                required=False,
            )
            raise ModelCallError(f"Async generation failed: {exc}") from exc

    @staticmethod
    def _messages_hash(messages: list[dict[str, str]]) -> str:
        normalized = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or ""),
            }
            for item in list(messages or [])
            if isinstance(item, dict)
        ]
        return hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        purpose: str = "semantic_retrieval",
    ) -> list[list[float]]:
        """Embed via model_runtime (typically delegated to memory service)."""
        try:
            # Delegate to existing infrastructure
            # (Usually memory.embed_texts or dedicated embedding service)
            embedder = getattr(self.model_runtime, "embed_texts", None)
            if embedder is None:
                raise ModelCallError("No embedding service configured")
            values = embedder(texts, purpose=purpose)
            record_external_system_call(
                operation="embedding_generation",
                system="ollama",
                request_payload={
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "semantic_retrieval"),
                    "text_count": len(list(texts or [])),
                },
                response_payload={"status": "ok", "vector_count": len(list(values or []))},
                status="ok",
                source="model_adapter",
                required=False,
            )
            return values
        except Exception as exc:
            record_external_system_call(
                operation="embedding_generation",
                system="ollama",
                request_payload={
                    "model": str(model or self.config.active_model or ""),
                    "purpose": str(purpose or "semantic_retrieval"),
                    "text_count": len(list(texts or [])),
                },
                response_payload={"status": "error", "error": str(exc)},
                status="error",
                source="model_adapter",
                required=False,
            )
            raise ModelCallError(f"Embedding failed: {exc}") from exc

    def estimate_token_count(self, text: str, *, model: str | None = None) -> int:
        """Estimate tokens via tokenizer or heuristic."""
        if not text:
            return 0

        normalized = str(text)
        target_model = model or self.config.active_model

        # Try tokenizer first
        tokenizer = self.get_tokenizer(target_model)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(normalized))
            except Exception:
                pass

        # Fallback to character-based heuristic
        chars_per_token = self.model_runtime.model_chars_per_token_estimate(target_model)
        return max(1, int((len(normalized) + chars_per_token - 1) // chars_per_token))

    def initialize_tokenizer(self, model_name: str | None = None) -> None:
        """Initialize tokenizer via model_runtime."""
        self.model_runtime.initialize_tokenizer(model_name)

    def get_tokenizer(self, model_name: str | None = None) -> Any:
        """Get tokenizer via model_runtime."""
        return self.model_runtime.current_tokenizer(model_name)

    def _enforce_model_egress(self) -> None:
        provider = str(getattr(self.config, "provider", "ollama") or "ollama").strip().lower()
        if provider not in {"ollama", ""}:
            return
        endpoint = str(os.environ.get("DADBOT_OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip()
        allowlist = tuple(getattr(self.config, "egress_allowlist", ()) or ())
        if not allowlist:
            allowlist = tuple(
                host.strip()
                for host in str(os.environ.get("DADBOT_EGRESS_ALLOWLIST", "localhost,127.0.0.1")).split(",")
                if host.strip()
            )
        enforce_url(endpoint, allowlist=allowlist)
