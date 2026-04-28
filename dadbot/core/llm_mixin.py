"""DadBotLlmMixin — LLM/model call forwarding for the DadBot facade.

Extracted from the DadBot god-class. Owns:
- Token-count and tokenizer helpers (delegating to model_runtime)
- Unified LLM call entry-points (call_llm, call_llm_async)
- Convenience Ollama-named wrappers kept for backward compatibility
- embed_texts (routes through model_port)
- chat_loop / chat_loop_via_service (routes through runtime_interface)
- Background task recording shim
"""
from __future__ import annotations

from dadbot.contracts import ChunkCallback


class DadBotLlmMixin:
    """LLM/model call forwarding for the DadBot facade.

    All methods are thin delegators — no logic lives here.
    The single source of truth for each capability is the corresponding
    manager (model_runtime, model_port, runtime_interface, etc.).
    """

    # ------------------------------------------------------------------
    # Tokenizer / token-count helpers
    # ------------------------------------------------------------------

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count via deterministic model port."""
        return self.model_port.estimate_token_count(text, model=self.ACTIVE_MODEL)

    def estimate_tokens(self, text: str) -> int:
        return self.estimate_token_count(text)

    def initialize_tokenizer(self, model_name=None):
        """Initialize tokenizer via model runtime manager."""
        return self.model_runtime.initialize_tokenizer(model_name)

    def current_tokenizer(self, model_name=None):
        """Get tokenizer via model runtime manager."""
        return self.model_runtime.current_tokenizer(model_name)

    def model_chars_per_token_estimate(self, model_name=None) -> float:
        return self.model_runtime.model_chars_per_token_estimate(model_name)

    # ------------------------------------------------------------------
    # Unified LLM entry-points
    # ------------------------------------------------------------------

    def call_llm(
        self,
        messages,
        *,
        model=None,
        temperature=None,
        stream=False,
        purpose="chat",
        options=None,
        response_format=None,
        chunk_callback: ChunkCallback | None = None,
        **kwargs,
    ):
        """Main unified LLM entry-point via deterministic model port.

        Deprecated: prefer model_port.generate() directly.
        Routes through: self.model_port.generate()
        """
        return self.model_port.generate(
            messages,
            model=model,
            temperature=temperature,
            stream=stream,
            purpose=purpose,
            response_format=response_format,
            chunk_callback=chunk_callback,
            **kwargs,
        )

    async def call_llm_async(
        self,
        messages,
        *,
        model=None,
        temperature=None,
        stream=False,
        purpose="chat",
        options=None,
        response_format=None,
        chunk_callback: ChunkCallback | None = None,
        **kwargs,
    ):
        """Async unified LLM entry-point via deterministic model port.

        Deprecated: prefer model_port.generate_async() directly.
        Routes through: self.model_port.generate_async()
        """
        return await self.model_port.generate_async(
            messages,
            model=model,
            temperature=temperature,
            stream=stream,
            purpose=purpose,
            response_format=response_format,
            chunk_callback=chunk_callback,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Backward-compatible Ollama-named wrappers
    # ------------------------------------------------------------------

    def call_ollama_chat(self, messages, options=None, response_format=None, purpose="chat"):
        content = self.model_port.generate(
            messages,
            purpose=purpose,
            response_format=response_format,
            options=options,
        )
        return {"message": {"content": str(content or "")}}

    async def call_ollama_chat_async(
        self, messages, options=None, response_format=None, purpose="chat"
    ):
        content = await self.model_port.generate_async(
            messages,
            purpose=purpose,
            response_format=response_format,
            options=options,
        )
        return {"message": {"content": str(content or "")}}

    def call_ollama_chat_with_model(
        self, model_name, messages, options=None, response_format=None, purpose="chat"
    ):
        content = self.model_port.generate(
            messages,
            model=model_name,
            purpose=purpose,
            options=options,
            response_format=response_format,
        )
        return {"message": {"content": str(content or "")}}

    def call_ollama_chat_stream(
        self, messages, options=None, purpose="chat", chunk_callback=None
    ):
        return self.model_port.generate(
            messages,
            stream=True,
            purpose=purpose,
            options=options,
            chunk_callback=chunk_callback,
        )

    async def call_ollama_chat_stream_async(
        self, messages, options=None, purpose="chat", chunk_callback=None
    ):
        return await self.model_port.generate_async(
            messages,
            stream=True,
            purpose=purpose,
            options=options,
            chunk_callback=chunk_callback,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_texts(self, texts, purpose="semantic retrieval"):
        """Embed texts via memory manager, with model-port fallback.

        MemoryManager owns embedding-cache behavior used throughout tests and
        runtime code paths. Keep model_port as a safe fallback when memory
        services are not yet available.
        """
        memory_embedder = getattr(getattr(self, "memory", None), "embed_texts", None)
        if callable(memory_embedder):
            return memory_embedder(texts, purpose=purpose)
        return self.model_port.embed(texts, purpose=purpose)

    # ------------------------------------------------------------------
    # Chat-loop entry-points (delegate to runtime_interface)
    # ------------------------------------------------------------------

    def chat_loop(self):
        return self.runtime_interface.chat_loop()

    def chat_loop_via_service(self, service_client, session_id=None):
        return self.runtime_interface.chat_loop_via_service(
            service_client, session_id=session_id
        )

    # ------------------------------------------------------------------
    # Background task recording shim
    # ------------------------------------------------------------------

    def _record_background_task(
        self, task_id, *, task_kind, status, metadata=None, error=""
    ):
        return self.runtime_orchestration.record_background_task(
            task_id,
            task_kind=task_kind,
            status=status,
            metadata=metadata,
            error=error,
        )
