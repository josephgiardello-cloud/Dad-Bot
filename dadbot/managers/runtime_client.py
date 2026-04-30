from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import time

import ollama

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_boundary import ModelGatewayScope, enforce_model_gateway

logger = logging.getLogger(__name__)
litellm = importlib.import_module("litellm") if importlib.util.find_spec("litellm") else None


class RuntimeClientManager:
    """Owns Ollama request execution, streaming helpers, and local model readiness checks."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    @staticmethod
    def _is_event_loop_closed_error(exc: BaseException) -> bool:
        return isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc)

    def _ensure_event_loop_context(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    @staticmethod
    def _call_supports_kwarg(func, name: str) -> bool:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return True
        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return name in signature.parameters

    def _invoke_ollama_sync(self, fn, *args, caller=None, **kwargs):
        call_kwargs = dict(kwargs)
        if caller is not None and self._call_supports_kwarg(fn, "caller"):
            call_kwargs["caller"] = caller
        try:
            return fn(*args, **call_kwargs)
        except TypeError as exc:
            if "caller" in call_kwargs and "unexpected keyword argument 'caller'" in str(exc):
                call_kwargs.pop("caller", None)
                return fn(*args, **call_kwargs)
            raise

    async def _invoke_ollama_async(self, fn, *args, caller=None, **kwargs):
        call_kwargs = dict(kwargs)
        if caller is not None and self._call_supports_kwarg(fn, "caller"):
            call_kwargs["caller"] = caller
        try:
            return await fn(*args, **call_kwargs)
        except TypeError as exc:
            if "caller" in call_kwargs and "unexpected keyword argument 'caller'" in str(exc):
                call_kwargs.pop("caller", None)
                return await fn(*args, **call_kwargs)
            raise

    def call_llm(
        self,
        messages,
        *,
        caller=None,
        model=None,
        temperature=None,
        stream=False,
        purpose="chat",
        options=None,
        response_format=None,
        chunk_callback=None,
        **kwargs,
    ):
        """Unified LLM entrypoint; prefers LiteLLM and falls back to Ollama."""
        enforce_model_gateway(caller=str(caller or ""))
        provider = self.bot.model_runtime.normalized_llm_provider()
        selected_model = self.bot.model_runtime.normalized_llm_model(model)
        temp = temperature if temperature is not None else self.bot.model_runtime.resolve_temperature(options)

        guarded_messages = self.bot.guard_chat_request_messages(
            messages,
            purpose=purpose,
        )

        if provider == "ollama" or litellm is None:
            ollama_options = dict(options) if isinstance(options, dict) else {}
            ollama_options.setdefault("temperature", temp)
            if stream:
                return self._invoke_ollama_sync(
                    self.call_ollama_chat_stream,
                    guarded_messages,
                    caller="ModelPort",
                    options=ollama_options,
                    purpose=purpose,
                    chunk_callback=chunk_callback,
                )
            return self._invoke_ollama_sync(
                self.call_ollama_chat_with_model,
                selected_model,
                guarded_messages,
                caller="ModelPort",
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
            )

        litellm_model = self.bot.model_runtime.litellm_model_identifier(selected_model)
        try:
            litellm_kwargs = {
                "model": litellm_model,
                "messages": guarded_messages,
                "temperature": temp,
                "stream": stream,
                **kwargs,
            }
            if response_format is not None:
                litellm_kwargs["response_format"] = response_format

            if stream:
                response_stream = litellm.completion(**litellm_kwargs)
                full_content = []
                for chunk in response_stream:
                    content = self.bot.model_runtime.extract_stream_chunk_content(chunk)
                    if content:
                        full_content.append(content)
                        if chunk_callback:
                            chunk_callback(content)
                return "".join(full_content).strip()

            return litellm.completion(**litellm_kwargs)
        except Exception as exc:
            logger.warning(
                "LiteLLM call failed (%s/%s): %s",
                provider,
                selected_model,
                exc,
            )
            logger.info("Falling back to local Ollama...")
            ollama_options = dict(options) if isinstance(options, dict) else {}
            ollama_options.setdefault("temperature", temp)
            if stream:
                return self._invoke_ollama_sync(
                    self.call_ollama_chat_stream,
                    guarded_messages,
                    caller="ModelPort",
                    options=ollama_options,
                    purpose=purpose,
                    chunk_callback=chunk_callback,
                )
            return self._invoke_ollama_sync(
                self.call_ollama_chat_with_model,
                selected_model,
                guarded_messages,
                caller="ModelPort",
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
            )

    async def call_llm_async(
        self,
        messages,
        *,
        caller=None,
        model=None,
        temperature=None,
        stream=False,
        purpose="chat",
        options=None,
        response_format=None,
        chunk_callback=None,
        **kwargs,
    ):
        """Async unified LLM entrypoint; prefers LiteLLM and falls back to Ollama."""
        enforce_model_gateway(caller=str(caller or ""))
        provider = self.bot.model_runtime.normalized_llm_provider()
        selected_model = self.bot.model_runtime.normalized_llm_model(model)
        temp = temperature if temperature is not None else self.bot.model_runtime.resolve_temperature(options)

        guarded_messages = self.bot.guard_chat_request_messages(
            messages,
            purpose=purpose,
        )
        ollama_options = dict(options) if isinstance(options, dict) else {}
        ollama_options.setdefault("temperature", temp)

        if provider == "ollama" or litellm is None:
            if stream:
                return await self._invoke_ollama_async(
                    self.call_ollama_chat_stream_async,
                    guarded_messages,
                    caller="ModelPort",
                    options=ollama_options,
                    purpose=purpose,
                    chunk_callback=chunk_callback,
                )
            return await self.call_ollama_chat_async(
                guarded_messages,
                caller="ModelPort",
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
            )

        litellm_model = self.bot.model_runtime.litellm_model_identifier(selected_model)
        try:
            litellm_kwargs = {
                "model": litellm_model,
                "messages": guarded_messages,
                "temperature": temp,
                "stream": stream,
                **kwargs,
            }
            if response_format is not None:
                litellm_kwargs["response_format"] = response_format

            if stream:
                response_stream = await litellm.acompletion(**litellm_kwargs)
                full_content = []
                async for chunk in response_stream:
                    content = self.bot.model_runtime.extract_stream_chunk_content(chunk)
                    if content:
                        full_content.append(content)
                        if chunk_callback:
                            callback_result = chunk_callback(content)
                            if inspect.isawaitable(callback_result):
                                await callback_result
                return "".join(full_content).strip()

            return await litellm.acompletion(**litellm_kwargs)
        except Exception as exc:
            logger.warning(
                "Async LiteLLM call failed (%s/%s): %s",
                provider,
                selected_model,
                exc,
            )
            logger.info("Falling back to local Ollama...")
            if stream:
                return await self._invoke_ollama_async(
                    self.call_ollama_chat_stream_async,
                    guarded_messages,
                    caller="ModelPort",
                    options=ollama_options,
                    purpose=purpose,
                    chunk_callback=chunk_callback,
                )
            return await self.call_ollama_chat_async(
                guarded_messages,
                caller="ModelPort",
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
            )

    def ollama_async_client(self):
        loop = self._ensure_event_loop_context()
        loop_id = id(loop)
        client = getattr(self.bot, "_ollama_async_client", None)
        client_loop_id = getattr(self.bot, "_ollama_async_client_loop_id", None)
        if client is None or client_loop_id != loop_id:
            if hasattr(ollama, "AsyncClient"):
                client = ollama.AsyncClient()
                self.bot._ollama_async_client = client
                self.bot._ollama_async_client_loop_id = loop_id
            else:
                client = None
                self.bot._ollama_async_client = None
                self.bot._ollama_async_client_loop_id = None
        return client

    def call_ollama_chat(
        self,
        messages,
        caller=None,
        options=None,
        response_format=None,
        purpose="chat",
    ):
        normalized_caller = str(caller or "").strip()
        if normalized_caller == "ModelPort" and ModelGatewayScope.current() != "ModelPort":
            with ModelGatewayScope.bind("ModelPort"):
                return self.call_ollama_chat(
                    messages,
                    caller="ModelPort",
                    options=options,
                    response_format=response_format,
                    purpose=purpose,
                )
        enforce_model_gateway(caller=normalized_caller)
        last_error = None

        for candidate in [
            self.bot.ACTIVE_MODEL,
            *[model for model in self.bot.model_candidates() if model != self.bot.ACTIVE_MODEL],
        ]:
            try:
                kwargs = {
                    "model": candidate,
                    "messages": messages,
                }
                if options is not None:
                    kwargs["options"] = options
                if response_format is not None:
                    kwargs["format"] = response_format
                response = ollama.chat(**kwargs)
                self.bot.ACTIVE_MODEL = candidate
                return response
            except self.bot.ollama_retryable_errors() as exc:
                last_error = exc

        error_summary = self.bot.ollama_error_summary(last_error)
        raise RuntimeError(
            f"Ollama {purpose} failed for all configured models ({error_summary})",
        ) from last_error

    async def call_ollama_chat_async(
        self,
        messages,
        caller=None,
        options=None,
        response_format=None,
        purpose="chat",
    ):
        normalized_caller = str(caller or "").strip()
        if normalized_caller == "ModelPort" and ModelGatewayScope.current() != "ModelPort":
            with ModelGatewayScope.bind("ModelPort"):
                return await self.call_ollama_chat_async(
                    messages,
                    caller="ModelPort",
                    options=options,
                    response_format=response_format,
                    purpose=purpose,
                )
        enforce_model_gateway(caller=normalized_caller)
        last_error = None
        self._ensure_event_loop_context()

        for candidate in [
            self.bot.ACTIVE_MODEL,
            *[model for model in self.bot.model_candidates() if model != self.bot.ACTIVE_MODEL],
        ]:
            try:
                kwargs = {
                    "model": candidate,
                    "messages": messages,
                }
                if options is not None:
                    kwargs["options"] = options
                if response_format is not None:
                    kwargs["format"] = response_format
                client = self.ollama_async_client()
                if client is None:
                    response = await asyncio.to_thread(ollama.chat, **kwargs)
                else:
                    try:
                        response = await client.chat(**kwargs)
                    except RuntimeError as exc:
                        if not self._is_event_loop_closed_error(exc):
                            raise
                        logger.info(
                            "Recreating async Ollama client after closed event loop",
                        )
                        self.bot._ollama_async_client = None
                        self.bot._ollama_async_client_loop_id = None
                        refreshed_client = self.ollama_async_client()
                        if refreshed_client is None:
                            response = await asyncio.to_thread(ollama.chat, **kwargs)
                        else:
                            response = await refreshed_client.chat(**kwargs)
                self.bot.ACTIVE_MODEL = candidate
                return response
            except self.bot.ollama_retryable_errors() as exc:
                last_error = exc
            except RuntimeError as exc:
                if self._is_event_loop_closed_error(exc):
                    last_error = exc
                    continue
                raise

        error_summary = self.bot.ollama_error_summary(last_error)
        raise RuntimeError(
            f"Ollama {purpose} failed for all configured models ({error_summary})",
        ) from last_error

    def call_ollama_chat_with_model(
        self,
        model_name,
        messages,
        caller=None,
        options=None,
        response_format=None,
        purpose="chat",
    ):
        enforce_model_gateway(caller=str(caller or ""))
        kwargs = {
            "model": model_name,
            "messages": messages,
        }
        if options is not None:
            kwargs["options"] = options
        if response_format is not None:
            kwargs["format"] = response_format
        try:
            return ollama.chat(**kwargs)
        except self.bot.ollama_retryable_errors() as exc:
            error_summary = self.bot.ollama_error_summary(exc)
            raise RuntimeError(
                f"Ollama {purpose} failed for model {model_name} ({error_summary})",
            ) from exc

    def call_ollama_chat_stream(
        self,
        messages,
        caller=None,
        options=None,
        purpose="chat",
        chunk_callback=None,
    ):
        enforce_model_gateway(caller=str(caller or ""))
        last_error = None

        for candidate in [
            self.bot.ACTIVE_MODEL,
            *[model for model in self.bot.model_candidates() if model != self.bot.ACTIVE_MODEL],
        ]:
            chunks: list[str] = []
            try:
                kwargs = {"model": candidate, "messages": messages, "stream": True}
                if options is not None:
                    kwargs["options"] = options
                stream = ollama.chat(**kwargs)
                chunks, truncated = self._collect_stream_chunks(stream, purpose=purpose, chunk_callback=chunk_callback)
                final_reply = "".join(chunks).strip()
                if final_reply and truncated:
                    final_reply = final_reply.rstrip(" ,.;:") + "..."
                if final_reply:
                    self.bot.ACTIVE_MODEL = candidate
                    return final_reply
                last_error = RuntimeError(f"Ollama {purpose} stream on {candidate} produced no content")
            except self.bot.ollama_retryable_errors() as exc:
                if chunks:
                    partial_reply = "".join(chunks).strip().rstrip(" ,.;:")
                    if partial_reply:
                        logger.warning(
                            "Ollama %s stream on %s ended early after partial content: %s",
                            purpose,
                            candidate,
                            self.bot.ollama_error_summary(exc),
                        )
                        self.bot.ACTIVE_MODEL = candidate
                        return partial_reply + "..."
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - stream iterators may raise non-retryable transport errors
                if chunks:
                    partial_reply = "".join(chunks).strip().rstrip(" ,.;:")
                    if partial_reply:
                        logger.warning(
                            "Ollama %s stream on %s aborted after partial content: %s",
                            purpose,
                            candidate,
                            self.bot.ollama_error_summary(exc),
                        )
                        self.bot.ACTIVE_MODEL = candidate
                        return partial_reply + "..."
                last_error = exc

        error_summary = self.bot.ollama_error_summary(last_error)
        raise RuntimeError(
            f"Ollama {purpose} failed for all configured models ({error_summary})",
        ) from last_error

    async def deliver_stream_chunk_async(self, chunk_callback, content):
        if chunk_callback is None:
            return
        callback_result = chunk_callback(content)
        if inspect.isawaitable(callback_result):
            await callback_result

    def _collect_stream_chunks(
        self,
        stream_iter,
        *,
        purpose: str,
        chunk_callback,
    ) -> tuple[list[str], bool]:
        """Iterate a synchronous Ollama stream, enforcing timeout and character limits.

        Returns ``(chunks, truncated)``.  ``truncated`` is True when streaming was
        stopped early due to the timeout or character-limit guards.
        """
        chunks: list[str] = []
        total_chars = 0
        truncated = False
        deadline = time.monotonic() + max(1, int(self.bot.STREAM_TIMEOUT_SECONDS or 1))

        try:
            for response in stream_iter:
                if time.monotonic() > deadline:
                    logger.warning("Stopping %s stream after %s seconds", purpose, self.bot.STREAM_TIMEOUT_SECONDS)
                    truncated = True
                    break
                content = self.bot.extract_ollama_message_content(response)
                if not content:
                    continue
                remaining_chars = max(0, int(self.bot.STREAM_MAX_CHARS or 0) - total_chars)
                if remaining_chars <= 0:
                    logger.warning("Stopping %s stream after %s characters", purpose, self.bot.STREAM_MAX_CHARS)
                    truncated = True
                    break
                if len(content) > remaining_chars:
                    content = content[:remaining_chars]
                    truncated = True
                chunks.append(content)
                total_chars += len(content)
                if chunk_callback is not None:
                    chunk_callback(content)
                if total_chars >= self.bot.STREAM_MAX_CHARS:
                    logger.warning("Stopping %s stream after %s characters", purpose, self.bot.STREAM_MAX_CHARS)
                    truncated = True
                    break
        except Exception as exc:  # noqa: BLE001 - preserve partial stream content on interruption
            if chunks:
                logger.warning("%s stream interrupted after partial output: %s", purpose, self.bot.ollama_error_summary(exc))
                truncated = True
            else:
                raise

        return chunks, truncated

    async def _collect_stream_chunks_async(
        self,
        stream_iter,
        *,
        purpose: str,
        chunk_callback,
    ) -> tuple[list[str], bool]:
        """Async variant of ``_collect_stream_chunks`` for async Ollama streams."""
        chunks: list[str] = []
        total_chars = 0
        truncated = False
        deadline = time.monotonic() + max(1, int(self.bot.STREAM_TIMEOUT_SECONDS or 1))

        try:
            async for response in stream_iter:
                if time.monotonic() > deadline:
                    logger.warning("Stopping %s stream after %s seconds", purpose, self.bot.STREAM_TIMEOUT_SECONDS)
                    truncated = True
                    break
                content = self.bot.extract_ollama_message_content(response)
                if not content:
                    continue
                remaining_chars = max(0, int(self.bot.STREAM_MAX_CHARS or 0) - total_chars)
                if remaining_chars <= 0:
                    logger.warning("Stopping %s stream after %s characters", purpose, self.bot.STREAM_MAX_CHARS)
                    truncated = True
                    break
                if len(content) > remaining_chars:
                    content = content[:remaining_chars]
                    truncated = True
                chunks.append(content)
                total_chars += len(content)
                await self.deliver_stream_chunk_async(chunk_callback, content)
                if total_chars >= self.bot.STREAM_MAX_CHARS:
                    logger.warning("Stopping %s stream after %s characters", purpose, self.bot.STREAM_MAX_CHARS)
                    truncated = True
                    break
        except Exception as exc:  # noqa: BLE001 - preserve partial stream content on interruption
            if chunks:
                logger.warning(
                    "%s async stream interrupted after partial output: %s",
                    purpose,
                    self.bot.ollama_error_summary(exc),
                )
                truncated = True
            else:
                raise

        return chunks, truncated

    async def call_ollama_chat_stream_async(
        self,
        messages,
        caller=None,
        options=None,
        purpose="chat",
        chunk_callback=None,
    ):
        enforce_model_gateway(caller=str(caller or ""))
        last_error = None
        client = self.ollama_async_client()

        if client is None:
            return await asyncio.to_thread(
                self.call_ollama_chat_stream,
                messages,
                "ModelPort",
                options,
                purpose,
                chunk_callback,
            )

        for candidate in [
            self.bot.ACTIVE_MODEL,
            *[model for model in self.bot.model_candidates() if model != self.bot.ACTIVE_MODEL],
        ]:
            chunks: list[str] = []
            try:
                kwargs = {"model": candidate, "messages": messages, "stream": True}
                if options is not None:
                    kwargs["options"] = options
                stream = await client.chat(**kwargs)
                chunks, truncated = await self._collect_stream_chunks_async(stream, purpose=purpose, chunk_callback=chunk_callback)
                final_reply = "".join(chunks).strip()
                if final_reply and truncated:
                    final_reply = final_reply.rstrip(" ,.;:") + "..."
                if final_reply:
                    self.bot.ACTIVE_MODEL = candidate
                    return final_reply
                last_error = RuntimeError(f"Ollama {purpose} stream on {candidate} produced no content")
            except self.bot.ollama_retryable_errors() as exc:
                if chunks:
                    partial_reply = "".join(chunks).strip().rstrip(" ,.;:")
                    if partial_reply:
                        logger.warning(
                            "Ollama %s stream on %s ended early after partial content: %s",
                            purpose,
                            candidate,
                            self.bot.ollama_error_summary(exc),
                        )
                        self.bot.ACTIVE_MODEL = candidate
                        return partial_reply + "..."
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - async stream iterators may raise non-retryable transport errors
                if chunks:
                    partial_reply = "".join(chunks).strip().rstrip(" ,.;:")
                    if partial_reply:
                        logger.warning(
                            "Ollama %s async stream on %s aborted after partial content: %s",
                            purpose,
                            candidate,
                            self.bot.ollama_error_summary(exc),
                        )
                        self.bot.ACTIVE_MODEL = candidate
                        return partial_reply + "..."
                last_error = exc

        error_summary = self.bot.ollama_error_summary(last_error)
        raise RuntimeError(
            f"Ollama {purpose} failed for all configured models ({error_summary})",
        ) from last_error

    def available_model_names(self):
        try:
            response = ollama.list()
        except self.bot.ollama_retryable_errors():
            return []
        names = []
        for model in response.get("models", []):
            name = str(model.get("model") or model.get("name") or "").strip().lower()
            if name and name not in names:
                names.append(name)
        return names

    def find_available_vision_model(self):
        active_model = str(self.bot.ACTIVE_MODEL or "").strip().lower()
        if self.bot.model_supports_image_input(active_model):
            return active_model

        available = self.available_model_names()
        for hint in self.bot.runtime_config.preferred_vision_model_hints:
            for name in available:
                if hint in name:
                    return name
        return None

    def vision_fallback_status(self):
        if self.bot.model_supports_image_input(self.bot.ACTIVE_MODEL):
            return True, f"{self.bot.ACTIVE_MODEL} already accepts direct image input."
        model_name = self.find_available_vision_model()
        if model_name:
            return (
                True,
                f"{model_name} is available as a fallback photo-understanding model.",
            )
        return (
            False,
            "No local vision model is available, so photos will rely only on any note you attach.",
        )

    def ensure_ollama_ready(self, status_callback=None):
        try:
            response = ollama.list()
            models = response.get("models", [])
        except self.bot.ollama_retryable_errors():
            self.bot.deliver_status_message(
                self.bot.reply_finalization.append_signoff(
                    "I can't reach Ollama right now. Make sure the Ollama app is open, then try again.",
                ),
                status_callback,
            )
            return False

        for candidate in self.bot.model_candidates():
            if self.bot.model_runtime.model_is_available(models, candidate):
                self.bot.ACTIVE_MODEL = candidate
                return True

        for candidate in self.bot.model_candidates():
            self.bot.deliver_status_message(
                f"I don't have {candidate} downloaded yet. Give me a minute to get it ready for you.",
                status_callback,
            )

            try:
                ollama.pull(candidate)
                self.bot.ACTIVE_MODEL = candidate
                self.bot.deliver_status_message(
                    self.bot.reply_finalization.append_signoff(
                        f"{candidate} is ready. Let's talk.",
                    ),
                    status_callback,
                )
                return True
            except self.bot.ollama_retryable_errors():
                continue

        self.bot.deliver_status_message(
            self.bot.reply_finalization.append_signoff(
                "I couldn't download my brain just yet. Make sure Ollama is online, then try again.",
            ),
            status_callback,
        )
        return False


__all__ = ["RuntimeClientManager"]
