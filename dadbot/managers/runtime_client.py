from __future__ import annotations

# ruff: noqa: C901, PLR0913
import asyncio
import atexit
import importlib
import importlib.util
import inspect
import logging
import time

import ollama  # type: ignore[reportMissingImports]

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_boundary import ModelGatewayScope, enforce_model_gateway
from dadbot.core.otel_bridge import start_span

logger = logging.getLogger(__name__)
litellm = importlib.import_module("litellm") if importlib.util.find_spec("litellm") else None
OLLAMA_REQUEST_TIMEOUT_SECONDS = 120.0


def _close_ollama_module_http_client() -> None:
    module_client = getattr(ollama, "_client", None)
    http_client = getattr(module_client, "_client", None)
    close = getattr(http_client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - shutdown path must not propagate transport errors.
            return


atexit.register(_close_ollama_module_http_client)


async def safe_close_ollama_async(client) -> None:
    """Canonical async cleanup for Ollama AsyncClient.

    Handles the internal httpx.AsyncClient (_client attribute) which is
    the actual resource holder. Uses robust introspection to handle
    library API changes gracefully.
    """
    if client is None:
        return

    # Ollama.AsyncClient wraps httpx.AsyncClient in _client; that's the resource.
    close_target = getattr(client, "_client", None) or client

    aclose = getattr(close_target, "aclose", None)
    if callable(aclose):
        try:
            close_result = aclose()
            if inspect.isawaitable(close_result):
                await close_result
        except Exception:  # noqa: BLE001 - cleanup should remain best-effort.
            pass
        return

    close = getattr(close_target, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - cleanup should remain best-effort.
            pass


class RuntimeClientManager:
    """Owns Ollama request execution, streaming helpers, and local model readiness checks."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def _prepare_ollama_request(self, model, messages, options, purpose):
        request_options = dict(options or {})
        runtime = self.bot.model_runtime
        context_budget = int(request_options.get("num_ctx", runtime.effective_context_token_budget(model)))
        model_limit = runtime.model_context_length(model)
        if isinstance(model_limit, int) and model_limit > 0:
            context_budget = min(context_budget, model_limit)
        if context_budget < 256:
            raise ValueError("Ollama num_ctx must allow at least 256 tokens")
        # Bucket allocation, not prompt capacity: the guard retains the requested budget.
        allocation = max(512, 1 << (context_budget - 1).bit_length())
        if isinstance(model_limit, int) and model_limit > 0:
            allocation = min(allocation, model_limit)
        request_options["num_ctx"] = allocation
        reserved = max(64, int(self.bot.RESERVED_RESPONSE_TOKENS))
        capabilities = runtime.ollama_show_payload(model).get("capabilities") or []
        if "thinking" in capabilities:
            reserved = max(reserved, 3072)
        reserved = min(reserved, context_budget // 2)
        guarded = self.bot.guard_chat_request_messages(
            messages, purpose=purpose, context_budget=context_budget,
            reserved_response_tokens=reserved,
        )
        return guarded, request_options

    def _record_ollama_response(self, response, *, model, purpose, started, stream_content=None):
        payload = response if isinstance(response, dict) else response.model_dump()
        logger.info(
            "Ollama purpose=%s model=%s seconds=%.3f load_ms=%.1f prompt_tokens=%s "
            "prompt_ms=%.1f generated_tokens=%s generation_ms=%.1f done_reason=%s",
            purpose, model, time.perf_counter() - started,
            float(payload.get("load_duration") or 0) / 1e6,
            payload.get("prompt_eval_count"), float(payload.get("prompt_eval_duration") or 0) / 1e6,
            payload.get("eval_count"), float(payload.get("eval_duration") or 0) / 1e6,
            payload.get("done_reason"),
        )
        visible = self.bot.extract_ollama_message_content(response) if stream_content is None else stream_content
        if payload.get("done_reason") == "length" and not visible.strip():
            raise RuntimeError(
                f"Ollama {purpose} exhausted its context/generation budget without a visible answer "
                f"(prompt_tokens={payload.get('prompt_eval_count')}, generated_tokens={payload.get('eval_count')})",
            )

    @staticmethod
    def _is_event_loop_closed_error(exc: BaseException) -> bool:
        return isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc)

    @staticmethod
    def _resolve_close_target(client):
        if client is None:
            return None
        nested_http_client = getattr(client, "_client", None)
        if nested_http_client is not None:
            return nested_http_client
        return client

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

    def _close_cached_ollama_async_client(self) -> None:
        shutdown_cached_client = getattr(self.bot, "_shutdown_ollama_async_client", None)
        if callable(shutdown_cached_client):
            shutdown_cached_client()
            return

        client = getattr(self.bot, "_ollama_async_client", None)
        self.bot._ollama_async_client = None
        self.bot._ollama_async_client_loop_id = None
        if client is None:
            return

        close_target = self._resolve_close_target(client)

        aclose = getattr(close_target, "aclose", None)
        if callable(aclose):
            close_result = aclose()
            if inspect.isawaitable(close_result):
                async def _await_close_result() -> None:
                    await close_result

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    _task = loop.create_task(_await_close_result())

                    def _consume_aclose_exc(t: asyncio.Task) -> None:
                        exc = t.exception() if not t.cancelled() else None
                        if exc is not None:
                            logger.debug(
                                "Async client close error (ignored): %s", exc
                            )

                    _task.add_done_callback(_consume_aclose_exc)
                else:
                    asyncio.run(_await_close_result())
            return

        close = getattr(close_target, "close", None)
        if callable(close):
            close()

    @staticmethod
    async def _close_ollama_async_client_instance(client) -> None:
        """Close an async Ollama client instance using canonical cleanup."""
        await safe_close_ollama_async(client)

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
        thinking = {"think": kwargs["think"]} if "think" in kwargs else {}

        guarded_messages = messages
        if provider != "ollama" and litellm is not None:
            guarded_messages = self.bot.guard_chat_request_messages(messages, purpose=purpose)

        if provider == "ollama" or litellm is None:
            ollama_options = dict(options) if isinstance(options, dict) else {}
            ollama_options.setdefault("temperature", temp)
            if stream:
                return self._invoke_ollama_sync(
                    self.call_ollama_chat_stream,
                    guarded_messages,
                    caller="ModelPort",
                    model=selected_model,
                    options=ollama_options,
                    purpose=purpose,
                    chunk_callback=chunk_callback,
                    **thinking,
                )
            return self._invoke_ollama_sync(
                self.call_ollama_chat_with_model,
                selected_model,
                guarded_messages,
                caller="ModelPort",
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
                **thinking,
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
        except Exception as exc:  # noqa: BLE001 - provider exceptions are intentionally normalized.
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
                    **thinking,
                )
            return self._invoke_ollama_sync(
                self.call_ollama_chat,
                guarded_messages,
                caller="ModelPort",
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
                **thinking,
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
        thinking = {"think": kwargs["think"]} if "think" in kwargs else {}

        guarded_messages = messages
        if provider != "ollama" and litellm is not None:
            guarded_messages = self.bot.guard_chat_request_messages(messages, purpose=purpose)
        ollama_options = dict(options) if isinstance(options, dict) else {}
        ollama_options.setdefault("temperature", temp)

        if provider == "ollama" or litellm is None:
            if stream:
                return await self._invoke_ollama_async(
                    self.call_ollama_chat_stream_async,
                    guarded_messages,
                    caller="ModelPort",
                    model=selected_model,
                    options=ollama_options,
                    purpose=purpose,
                    chunk_callback=chunk_callback,
                    **thinking,
                )
            return await self.call_ollama_chat_async(
                guarded_messages,
                caller="ModelPort",
                model=selected_model,
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
                **thinking,
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
        except Exception as exc:  # noqa: BLE001 - provider exceptions are intentionally normalized.
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
                    **thinking,
                )
            return await self.call_ollama_chat_async(
                guarded_messages,
                caller="ModelPort",
                options=ollama_options,
                response_format=response_format,
                purpose=purpose,
                **thinking,
            )

    def ollama_async_client(self):
        self._ensure_event_loop_context()
        if hasattr(ollama, "AsyncClient"):
            return ollama.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS)
        return None

    @staticmethod
    def _call_ollama_sync(**kwargs):
        client = ollama.Client(timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS)
        if kwargs.get("stream"):
            def bounded_stream():
                with client._client:
                    with start_span(
                        "gen_ai.chat", **{"gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama"}
                    ):
                        yield from client.chat(**kwargs)

            return bounded_stream()
        with client._client:
            with start_span(
                "gen_ai.chat", **{"gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama"}
            ):
                return client.chat(**kwargs)

    def call_ollama_chat(
        self,
        messages,
        caller=None,
        options=None,
        response_format=None,
        purpose="chat",
        think: bool | None = None,
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
                    think=think,
                )
        enforce_model_gateway(caller=normalized_caller)
        last_error = None

        for candidate in [
            self.bot.ACTIVE_MODEL,
            *[model for model in self.bot.model_candidates() if model != self.bot.ACTIVE_MODEL],
        ]:
            try:
                guarded, request_options = self._prepare_ollama_request(candidate, messages, options, purpose)
                kwargs = {
                    "model": candidate,
                    "messages": guarded,
                    "options": request_options,
                    "keep_alive": self.bot.runtime_config.ollama_keep_alive_seconds,
                }
                if response_format is not None:
                    kwargs["format"] = response_format
                if think is not None:
                    kwargs["think"] = think
                started = time.perf_counter()
                response = self._call_ollama_sync(**kwargs)
                self._record_ollama_response(response, model=candidate, purpose=purpose, started=started)
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
        model=None,
        think: bool | None = None,
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
                    model=model,
                    think=think,
                )
        enforce_model_gateway(caller=normalized_caller)
        last_error = None
        self._ensure_event_loop_context()

        selected_model = str(model or self.bot.ACTIVE_MODEL or "").strip()
        for candidate in [
            selected_model,
            *[name for name in self.bot.model_candidates() if name != selected_model],
        ]:
            try:
                guarded, request_options = self._prepare_ollama_request(candidate, messages, options, purpose)
                kwargs = {
                    "model": candidate,
                    "messages": guarded,
                    "options": request_options,
                    "keep_alive": self.bot.runtime_config.ollama_keep_alive_seconds,
                }
                if response_format is not None:
                    kwargs["format"] = response_format
                if think is not None:
                    kwargs["think"] = think
                started = time.perf_counter()
                client = self.ollama_async_client()
                if client is None:
                    response = await asyncio.to_thread(self._call_ollama_sync, **kwargs)
                else:
                    try:
                        with start_span(
                            "gen_ai.chat", **{"gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama"}
                        ):
                            response = await client.chat(**kwargs)
                    except RuntimeError as exc:
                        if not self._is_event_loop_closed_error(exc):
                            raise
                        logger.info("Falling back to sync Ollama call after closed event loop")
                        response = await asyncio.to_thread(self._call_ollama_sync, **kwargs)
                    finally:
                        await self._close_ollama_async_client_instance(client)
                self._record_ollama_response(response, model=candidate, purpose=purpose, started=started)
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
        think: bool | None = None,
    ):
        enforce_model_gateway(caller=str(caller or ""))
        guarded, request_options = self._prepare_ollama_request(model_name, messages, options, purpose)
        kwargs = {
            "model": model_name,
            "messages": guarded,
            "options": request_options,
            "keep_alive": self.bot.runtime_config.ollama_keep_alive_seconds,
        }
        if response_format is not None:
            kwargs["format"] = response_format
        if think is not None:
            kwargs["think"] = think
        try:
            started = time.perf_counter()
            response = self._call_ollama_sync(**kwargs)
            self._record_ollama_response(response, model=model_name, purpose=purpose, started=started)
            return response
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
        model=None,
        think: bool | None = None,
    ):
        enforce_model_gateway(caller=str(caller or ""))
        last_error = None

        selected_model = str(model or self.bot.ACTIVE_MODEL or "").strip()
        for candidate in [
            selected_model,
            *[name for name in self.bot.model_candidates() if name != selected_model],
        ]:
            chunks: list[str] = []
            try:
                guarded, request_options = self._prepare_ollama_request(candidate, messages, options, purpose)
                kwargs = {
                    "model": candidate, "messages": guarded, "stream": True, "options": request_options,
                    "keep_alive": self.bot.runtime_config.ollama_keep_alive_seconds,
                }
                if think is not None:
                    kwargs["think"] = think
                stream = self._call_ollama_sync(**kwargs)
                try:
                    chunks, truncated = self._collect_stream_chunks(stream, purpose=purpose, chunk_callback=chunk_callback)
                finally:
                    stream.close()
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
        started = time.perf_counter()

        try:
            for response in stream_iter:
                done_reason = (
                    response.get("done_reason") if isinstance(response, dict) else getattr(response, "done_reason", None)
                )
                if done_reason:
                    self._record_ollama_response(
                        response, model=self.bot.ACTIVE_MODEL, purpose=purpose, started=started,
                        stream_content="".join(chunks) + self.bot.extract_ollama_message_content(response),
                    )
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
        except Exception as exc:
            if chunks:
                logger.warning(
                    "%s stream interrupted after partial output: %s",
                    purpose,
                    self.bot.ollama_error_summary(exc),
                )
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
        started = time.perf_counter()

        try:
            async for response in stream_iter:
                done_reason = (
                    response.get("done_reason") if isinstance(response, dict) else getattr(response, "done_reason", None)
                )
                if done_reason:
                    self._record_ollama_response(
                        response, model=self.bot.ACTIVE_MODEL, purpose=purpose, started=started,
                        stream_content="".join(chunks) + self.bot.extract_ollama_message_content(response),
                    )
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
        except Exception as exc:
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
        model=None,
        think: bool | None = None,
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
                model,
                think,
            )

        try:
            selected_model = str(model or self.bot.ACTIVE_MODEL or "").strip()
            for candidate in [
                selected_model,
                *[name for name in self.bot.model_candidates() if name != selected_model],
            ]:
                chunks: list[str] = []
                try:
                    guarded, request_options = self._prepare_ollama_request(candidate, messages, options, purpose)
                    kwargs = {
                        "model": candidate, "messages": guarded, "stream": True, "options": request_options,
                        "keep_alive": self.bot.runtime_config.ollama_keep_alive_seconds,
                    }
                    if think is not None:
                        kwargs["think"] = think
                    with start_span(
                        "gen_ai.chat", **{"gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama"}
                    ):
                        stream = await client.chat(**kwargs)
                        chunks, truncated = await self._collect_stream_chunks_async(
                            stream,
                            purpose=purpose,
                            chunk_callback=chunk_callback,
                        )
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
        finally:
            await self._close_ollama_async_client_instance(client)

        error_summary = self.bot.ollama_error_summary(last_error)
        raise RuntimeError(
            f"Ollama {purpose} failed for all configured models ({error_summary})",
        ) from last_error

    def available_model_names(self):
        try:
            response = ollama.list()
            return self.bot.model_runtime.model_names_from_response(response)
        except self.bot.ollama_retryable_errors():
            return []

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
            models = self.bot.model_runtime.models_from_response(response)
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
