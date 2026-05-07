"""DadBotTurnMixin — turn execution and graph failure handling for DadBot.

Extracted from the DadBot god-class. Owns:
- Turn orchestrator resolution (_get_turn_orchestrator)
- Graph turn execution (sync/async, coro-in-thread bridge)
- Graph failure event emission and strict-mode error propagation
- Public turn entry-points: process_user_message*, handle_turn_*, stream variants
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Iterable, Mapping
from typing import Any, cast
from uuid import uuid4

from dadbot.contracts import AttachmentList, ChunkCallback, FinalizedTurnResult
from dadbot.core.execution_contract import (
    AgentState,
    ExecutionMode,
    TurnDelivery,
    TurnRequest,
    TurnResponse,
    TurnResult,
    UserInput,
    live_turn_request,
)
from dadbot.core.kernel_locks import KernelReplaySequenceLock
from dadbot.core.kernel_signals import CorrelationContext, TracingContext

logger = logging.getLogger(__name__)


class DadBotTurnMixin:
    """Turn execution and graph failure handling for the DadBot facade."""

    # These attributes are provided by the concrete DadBot facade at runtime.
    _turn_graph_config_path: str
    _turn_execution_lock: Any
    services: Any
    turn_orchestrator: Any

    # ------------------------------------------------------------------
    # Orchestrator resolution
    # ------------------------------------------------------------------

    def _get_turn_orchestrator(self):
        services = getattr(self, "services", None)
        if services is not None:
            return services.turn_orchestrator
        orchestrator = getattr(self, "_turn_orchestrator", None)
        if orchestrator is not None:
            return orchestrator
        # Lazy construction: deferred import avoids a circular import at module
        # load time (orchestrator → graph → dadbot would be circular).
        from dadbot.core.orchestrator import DadBotOrchestrator

        orchestrator = DadBotOrchestrator(
            config_path=self._turn_graph_config_path,
            bot=self,
            strict=bool(getattr(self, "_strict_graph_mode", True)),
        )
        self._turn_orchestrator = orchestrator
        return orchestrator

    # ------------------------------------------------------------------
    # Graph turn execution
    # ------------------------------------------------------------------

    async def _run_graph_turn_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        orchestrator = self._get_turn_orchestrator()
        session_id = str(getattr(self, "_execute_turn_session_id", "") or "").strip() or str(
            getattr(self, "active_thread_id", "") or "default",
        )
        submit_turn = getattr(orchestrator, "_submit_turn_via_control_plane", None)
        if callable(submit_turn):
            return await cast(
                Awaitable[FinalizedTurnResult],
                submit_turn(
                    user_input,
                    attachments=attachments,
                    session_id=session_id,
                ),
            )
        return await orchestrator.control_plane.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            metadata={},
        )

    @staticmethod
    def _run_coro_in_thread(coro):
        """Run a coroutine in a worker thread with its own event loop.

        Used when the coroutine must be awaited from within an already-running
        loop (e.g. Streamlit, Jupyter) where asyncio.run() is forbidden.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def _run_graph_turn_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        """Run the graph turn synchronously regardless of the calling context.

        When called from within a running event loop (e.g., Streamlit), the
        coroutine is dispatched to a dedicated thread with its own event loop
        rather than falling back to the legacy TurnProcessingManager.
        """
        with self._turn_execution_lock:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            coro = self._run_graph_turn_async(
                user_input,
                attachments=attachments,
                chunk_callback=chunk_callback,
            )
            if loop is not None and loop.is_running():
                return self._run_coro_in_thread(coro)
            return asyncio.run(coro)

    def _run_turn_request_sync(
        self,
        request: TurnRequest,
        *,
        state: AgentState | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResponse:
        coro = self._execute_turn_async(
            request,
            state=state,
            chunk_callback=chunk_callback,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            return self._run_coro_in_thread(coro)
        return asyncio.run(coro)

    def _validate_managers(self, *, smoke: bool = False) -> None:
        services = getattr(self, "services", None)
        if services is not None:
            services.validate_facade(smoke=smoke)

    # ------------------------------------------------------------------
    # EventTap boundary integration (TURN/NODE/TOOL lifecycle events)
    # ------------------------------------------------------------------

    def _resolve_event_tap(self):
        direct = getattr(self, "event_tap", None) or getattr(self, "_event_tap", None)
        if direct is not None:
            return direct
        services = getattr(self, "services", None)
        if services is not None:
            return getattr(services, "event_tap", None)
        return None

    @staticmethod
    def _safe_event_payload(value: Any, *, limit: int = 240) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:limit]
        if isinstance(value, dict):
            return {
                str(k)[:80]: DadBotTurnMixin._safe_event_payload(v, limit=limit)
                for k, v in list(value.items())[:16]
            }
        if isinstance(value, (list, tuple, set)):
            return [DadBotTurnMixin._safe_event_payload(item, limit=limit) for item in list(value)[:16]]
        return str(value)[:limit]

    def _event_tap_context(self) -> tuple[str, str, str]:
        session_id = str(getattr(self, "active_thread_id", "") or "").strip() or "default"
        tenant_id = str(getattr(self, "tenant_id", "") or "").strip() or "default"
        trace_id = str(TracingContext.current_trace_id() or CorrelationContext.current() or uuid4().hex)
        return session_id, tenant_id, trace_id

    def _begin_turn_event_run(self) -> str:
        tap = self._resolve_event_tap()
        if tap is None or not callable(getattr(tap, "begin_run", None)):
            return ""
        session_id, tenant_id, trace_id = self._event_tap_context()
        run_id = str(getattr(self, "_active_turn_run_id", "") or "").strip()
        if run_id:
            return run_id
        run_id = str(
            tap.begin_run(
                session_id=session_id,
                tenant_id=tenant_id,
                run_id=trace_id,
                contract_version="1.0",
            )
        )
        self._active_turn_run_id = run_id
        return run_id

    def _emit_turn_event(self, event_type: str, **payload: Any) -> None:
        tap = self._resolve_event_tap()
        emit = getattr(tap, "emit", None)
        if not callable(emit):
            return
        run_id = str(getattr(self, "_active_turn_run_id", "") or "").strip()
        if not run_id:
            run_id = self._begin_turn_event_run()
        if not run_id:
            return
        emit(
            event_type,
            run_id=run_id,
            **{k: self._safe_event_payload(v) for k, v in payload.items()},
        )

    def _snapshot_kernel_state(self) -> dict[str, Any]:
        snapshot_fn = getattr(self, "snapshot_session_state", None)
        if callable(snapshot_fn):
            try:
                raw = snapshot_fn()
                if isinstance(raw, Mapping):
                    return {str(k): v for k, v in raw.items()}
                return {}
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _restore_kernel_state(self, snapshot: dict[str, Any]) -> None:
        restore_fn = getattr(self, "load_session_state_snapshot", None)
        if callable(restore_fn):
            restore_fn(dict(snapshot or {}))

    def _checkpoint_turn_state(self, *, state_hash: str = "") -> None:
        tap = self._resolve_event_tap()
        if tap is None:
            return
        maybe_checkpoint = getattr(tap, "maybe_checkpoint", None)
        if callable(maybe_checkpoint):
            maybe_checkpoint(
                state_snapshot={"kernel_state": self._snapshot_kernel_state()},
                state_hash=str(state_hash or ""),
            )

    # ------------------------------------------------------------------
    # Graph failure handling
    # ------------------------------------------------------------------

    def _graph_failure_session_id(self) -> str:
        candidate = getattr(self, "active_thread_id", "") or getattr(
            self,
            "tenant_id",
            "",
        )
        normalized = str(candidate or "").strip()
        return normalized or "default"

    @staticmethod
    def _safe_graph_failure_payload(value: Any, *, limit: int = 240) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:limit]
        if isinstance(value, dict):
            return {
                str(key)[:80]: DadBotTurnMixin._safe_graph_failure_payload(
                    item,
                    limit=limit,
                )
                for key, item in list(value.items())[:12]
            }
        if isinstance(value, (list, tuple, set)):
            return [DadBotTurnMixin._safe_graph_failure_payload(item, limit=limit) for item in list(value)[:12]]
        return str(value)[:limit]

    def _emit_graph_failure_event(
        self,
        *,
        mode: str,
        correlation_id: str,
        trace_id: str,
        user_input: str,
        attachments: AttachmentList | None,
        exc: Exception,
    ) -> None:
        orchestrator = getattr(self, "_turn_orchestrator", None)
        if orchestrator is None:
            try:
                orchestrator = self._get_turn_orchestrator()
            except Exception:  # noqa: BLE001
                orchestrator = None
        control_plane = getattr(orchestrator, "control_plane", None)
        ledger_writer = getattr(control_plane, "ledger_writer", None)
        write_event = getattr(ledger_writer, "write_event", None)
        if not callable(write_event):
            return

        payload = {
            "mode": mode,
            "graph_enabled": True,
            "strict_graph_mode": bool(getattr(self, "_strict_graph_mode", True)),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "user_input": self._safe_graph_failure_payload(user_input),
            "attachment_count": len(attachments or []),
            "attachments": self._safe_graph_failure_payload(
                list(attachments or []),
                limit=120,
            ),
            "recorded_at": str(
                getattr(getattr(self, "_current_turn_time_base", None), "wall_time", ""),
            ),
        }
        write_event(
            event_type="GRAPH_EXECUTION_FAILED",
            session_id=self._graph_failure_session_id(),
            trace_id=trace_id or correlation_id,
            kernel_step_id=f"graph.failure.{mode}",
            payload={
                **payload,
                "correlation_id": correlation_id,
                "trace_id": trace_id or correlation_id,
            },
            committed=True,
        )

    def _graph_failure_reply(self, correlation_id: str) -> str:
        return self._append_signoff_compat(
            "I hit an internal graph error and stopped before touching memory or state. "
            f"Please try again. Reference ID: {correlation_id}",
        )

    def _raise_graph_execution_failure(
        self,
        exc: Exception,
        *,
        mode: str,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> None:
        correlation_id = CorrelationContext.current() or CorrelationContext.ensure()
        trace_id = TracingContext.current_trace_id() or correlation_id
        logger.exception(
            "Graph execution failed in %s mode; strict mode forbids alternate execution paths",
            mode,
            extra={
                "correlation_id": correlation_id,
                "trace_id": trace_id,
                "graph_enabled": True,
                "strict_graph_mode": bool(getattr(self, "_strict_graph_mode", True)),
                "attachment_count": len(attachments or []),
            },
        )
        self._emit_graph_failure_event(
            mode=mode,
            correlation_id=correlation_id,
            trace_id=trace_id,
            user_input=user_input,
            attachments=attachments,
            exc=exc,
        )
        raise RuntimeError(
            "Graph execution failed in strict mode; legacy path is disabled",
        ) from exc

    # ------------------------------------------------------------------
    # Stream helper
    # ------------------------------------------------------------------

    def _deliver_buffered_stream_chunks(
        self,
        reply: str,
        chunk_callback: ChunkCallback | None,
    ) -> None:
        if callable(chunk_callback) and reply:
            chunk_callback(reply)

    # ------------------------------------------------------------------
    # Signoff compat helper (used by failure reply + stream variants)
    # ------------------------------------------------------------------

    def _append_signoff_compat(self, text: str) -> str:
        """Apply reply signoff using manager-first compatibility resolution."""
        finalization = getattr(self, "reply_finalization", None)
        append_signoff = getattr(finalization, "append_signoff", None)
        if callable(append_signoff):
            return str(append_signoff(text))
        compat_finalize = getattr(self, "finalize_reply", None)
        if callable(compat_finalize):
            return str(compat_finalize(text))
        return str(text or "")

    @staticmethod
    def _response_from_result(request: TurnRequest, result: TurnResult) -> TurnResponse:
        reply, should_end = result
        return TurnResponse(
            reply=reply,
            should_end=bool(should_end),
            mode=request.mode,
            delivery=request.delivery,
        )

    @staticmethod
    def _delivery_event_mode(delivery: TurnDelivery) -> str:
        if delivery in {TurnDelivery.ASYNC, TurnDelivery.STREAM_ASYNC}:
            return "async"
        return "sync"

    async def _execute_live_turn_async(
        self,
        request: TurnRequest,
        *,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResponse:
        user_input = request.input.text
        attachments = list(request.input.attachments or [])
        request_session_id = str(request.session_id or "").strip() or str(
            getattr(self, "active_thread_id", "") or "default",
        )
        event_mode = self._delivery_event_mode(request.delivery)
        previous_session_id = str(getattr(self, "_execute_turn_session_id", "") or "")
        self._execute_turn_session_id = request_session_id
        self._begin_turn_event_run()
        self._emit_turn_event(
            "TURN_START",
            mode=event_mode,
            user_input=user_input,
            attachment_count=len(attachments),
        )
        try:
            if request.delivery in {TurnDelivery.SYNC, TurnDelivery.STREAM}:
                result = self._run_graph_turn_sync_compat(
                    user_input,
                    attachments,
                    chunk_callback,
                )
            else:
                result = await self._run_graph_turn_async_compat(
                    user_input,
                    attachments,
                    chunk_callback,
                )
            self._emit_turn_event(
                "TURN_END",
                mode=event_mode,
                status="ok",
                should_end=bool(result[1]),
            )
            self._checkpoint_turn_state()
            return self._response_from_result(request, result)
        except Exception as exc:  # noqa: BLE001
            self._emit_turn_event(
                "TURN_END",
                mode=event_mode,
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._raise_graph_execution_failure(
                exc,
                mode=event_mode,
                user_input=user_input,
                attachments=attachments,
            )
            raise AssertionError("unreachable")
        finally:
            self._execute_turn_session_id = previous_session_id
            self._active_turn_run_id = ""

    async def _execute_turn_async(
        self,
        request: TurnRequest,
        *,
        state: AgentState | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResponse:
        if request.mode == ExecutionMode.LIVE:
            return await self._execute_live_turn_async(
                request,
                chunk_callback=chunk_callback,
            )

        if state is None:
            raise RuntimeError(f"{request.mode.value} mode requires AgentState")

        if request.mode == ExecutionMode.REPLAY:
            replay_handler = getattr(self, "_run_turn_replay", None)
            if not callable(replay_handler):
                raise RuntimeError("Replay mode requires _run_turn_replay handler")
            result = cast(
                TurnResult,
                replay_handler(
                    request.input,
                    state,
                    chunk_callback=chunk_callback,
                ),
            )
            return self._response_from_result(request, result)

        if request.mode == ExecutionMode.RECOVERY:
            recovery_handler = getattr(self, "_run_turn_recovery", None)
            if callable(recovery_handler):
                result = cast(
                    TurnResult,
                    recovery_handler(
                        request.input,
                        state,
                        chunk_callback=chunk_callback,
                    ),
                )
            else:
                result = self._run_turn_recovery_default(
                    request.input,
                    state,
                    chunk_callback=chunk_callback,
                )
            return self._response_from_result(request, result)

        raise ValueError(f"Unsupported execution mode: {request.mode}")

    def execute_turn(
        self,
        request: TurnRequest,
        *,
        state: AgentState | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResponse | Awaitable[TurnResponse]:
        if request.delivery in {TurnDelivery.ASYNC, TurnDelivery.STREAM_ASYNC}:
            return self._execute_turn_async(
                request,
                state=state,
                chunk_callback=chunk_callback,
            )
        return self._run_turn_request_sync(
            request,
            state=state,
            chunk_callback=chunk_callback,
        )

    # ------------------------------------------------------------------
    # Public turn entry-points
    # ------------------------------------------------------------------

    @staticmethod
    def _callable_accepts_chunk_callback(fn: Any) -> bool:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return True

        parameters = signature.parameters.values()
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or parameter.name == "chunk_callback"
            for parameter in parameters
        )

    @classmethod
    def _invoke_chunk_callback_compat(cls, fn: Any, /, *args: Any, **kwargs: Any):
        if cls._callable_accepts_chunk_callback(fn):
            return fn(*args, **kwargs)
        trimmed_kwargs = {key: value for key, value in kwargs.items() if key != "chunk_callback"}
        return fn(*args, **trimmed_kwargs)

    def _run_graph_turn_sync_compat(
        self,
        user_input: str,
        attachments: AttachmentList | None,
        chunk_callback: ChunkCallback | None,
    ) -> FinalizedTurnResult:
        return self._invoke_chunk_callback_compat(
            self._run_graph_turn_sync,
            user_input,
            attachments=attachments,
            chunk_callback=chunk_callback,
        )

    async def _run_graph_turn_async_compat(
        self,
        user_input: str,
        attachments: AttachmentList | None,
        chunk_callback: ChunkCallback | None,
    ) -> FinalizedTurnResult:
        result = self._invoke_chunk_callback_compat(
            self._run_graph_turn_async,
            user_input,
            attachments=attachments,
            chunk_callback=chunk_callback,
        )
        return await result

    def process_user_message(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        response = cast(
            TurnResponse,
            self.execute_turn(
                live_turn_request(
                    user_input,
                    attachments=list(attachments or []),
                    delivery=TurnDelivery.SYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
                chunk_callback=chunk_callback,
            ),
        )
        dad_reply, should_end = response.as_result()
        if should_end:
            if dad_reply is None or isinstance(dad_reply, str):
                return dad_reply, should_end
            return str(dad_reply), should_end
        if not isinstance(dad_reply, str):
            dad_reply = None if dad_reply is None else str(dad_reply)
            return dad_reply, should_end

        memory = getattr(self, "memory", None)
        should_do_daily_checkin = getattr(memory, "should_do_daily_checkin", None)
        tone_context = getattr(self, "tone_context", None)
        blend_daily_checkin_reply = getattr(tone_context, "blend_daily_checkin_reply", None)
        if callable(should_do_daily_checkin) and callable(blend_daily_checkin_reply):
            try:
                should_blend = bool(getattr(self, "_pending_daily_checkin_context", False)) or bool(
                    should_do_daily_checkin(),
                )
                if should_blend:
                    self._pending_daily_checkin_context = True
                    mood = "neutral"
                    last_ctx = getattr(self, "_last_turn_context", None)
                    if last_ctx is not None:
                        mood = str(getattr(last_ctx, "state", {}).get("mood") or "neutral")
                    dad_reply = blend_daily_checkin_reply(dad_reply, mood)
            except Exception as exc:
                logger.debug("Daily check-in blend failed (non-fatal): %s", exc)
                if not isinstance(dad_reply, str):
                    dad_reply = None if dad_reply is None else str(dad_reply)
        return cast(str | None, dad_reply), should_end

    def run_turn(
        self,
        input: UserInput,
        state: AgentState,
        *,
        chunk_callback: ChunkCallback | None = None,
        mode: ExecutionMode = ExecutionMode.LIVE,
    ) -> TurnResult:
        """Canonical deterministic turn execution contract.

        LIVE delegates to the graph-backed process_user_message path.
        REPLAY/RECOVERY require explicit handlers on the facade.
        """
        state.recompute_invariance_hash()

        response = cast(
            TurnResponse,
            self.execute_turn(
                TurnRequest(
                    input=input,
                    mode=mode,
                    delivery=TurnDelivery.SYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
                state=state,
                chunk_callback=chunk_callback,
            ),
        )
        result = response.as_result()

        state.step_id = int(state.step_id) + 1
        state.current_node = "turn.complete"
        state.recompute_invariance_hash()
        self._checkpoint_turn_state(state_hash=state.invariance_hash)
        return result

    def _run_turn_recovery_default(
        self,
        input: UserInput,
        state: AgentState,
        *,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResult:
        tap = self._resolve_event_tap()
        if tap is None:
            raise RuntimeError("Recovery mode requires configured EventTap")

        latest_checkpoint = getattr(tap, "latest_checkpoint", None)
        events_after_cursor = getattr(tap, "events_after_cursor", None)
        if not callable(latest_checkpoint) or not callable(events_after_cursor):
            raise RuntimeError("Recovery mode requires checkpoint-capable EventTap")

        checkpoint = latest_checkpoint()
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("Recovery mode requires a durable checkpoint (replay-only policy)")

        apply_event = getattr(self, "_apply_recovery_event", None)
        if not callable(apply_event):
            raise RuntimeError("Recovery mode requires _apply_recovery_event handler")

        replayed = 0
        snapshot = dict(checkpoint.get("state") or {})
        self._restore_kernel_state(dict(snapshot.get("kernel_state") or snapshot))
        if str(checkpoint.get("state_hash") or "").strip():
            state.invariance_hash = str(checkpoint.get("state_hash"))

        cursor = int(checkpoint.get("event_sequence_id") or 0)
        self._emit_turn_event("RECOVERY_REPLAY_START", cursor=cursor)
        raw_events = events_after_cursor(cursor)
        events_iter = list(raw_events) if isinstance(raw_events, Iterable) else []
        run_id = str(checkpoint.get("run_id") or "").strip() or str(
            getattr(self, "_active_turn_run_id", "") or "",
        ).strip()
        digest, canonical = KernelReplaySequenceLock.strict_hash(
            trace_id=run_id,
            events=[dict(event) for event in events_iter],
        )
        for event in events_iter:
            replayed += 1
            apply_event(dict(event))
        self._emit_turn_event(
            "RECOVERY_REPLAY_END",
            replayed_events=replayed,
            strict_sequence_hash=digest,
            strict_sequence_count=len(canonical),
        )

        response = cast(
            TurnResponse,
            self.execute_turn(
                live_turn_request(
                    input.text,
                    attachments=list(input.attachments or []),
                    delivery=TurnDelivery.SYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
                chunk_callback=chunk_callback,
            ),
        )
        return response.as_result()

    async def process_user_message_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        response = await cast(
            Awaitable[TurnResponse],
            self.execute_turn(
                live_turn_request(
                    user_input,
                    attachments=list(attachments or []),
                    delivery=TurnDelivery.ASYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
                chunk_callback=chunk_callback,
            ),
        )
        return response.as_result()

    def process_user_message_stream(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        streamed = False

        def _wrapped_chunk_callback(chunk: str) -> None:
            nonlocal streamed
            if not chunk:
                return
            streamed = True
            if callable(chunk_callback):
                chunk_callback(chunk)

        response = cast(
            TurnResponse,
            self.execute_turn(
                live_turn_request(
                    user_input,
                    attachments=list(attachments or []),
                    delivery=TurnDelivery.STREAM,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
                chunk_callback=_wrapped_chunk_callback,
            ),
        )
        reply, should_end = response.as_result()
        if not streamed:
            self._deliver_buffered_stream_chunks(str(reply or ""), chunk_callback)
        return reply, should_end

    async def process_user_message_stream_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        streamed = False

        def _wrapped_chunk_callback(chunk: str) -> None:
            nonlocal streamed
            if not chunk:
                return
            streamed = True
            if callable(chunk_callback):
                chunk_callback(chunk)

        response = await cast(
            Awaitable[TurnResponse],
            self.execute_turn(
                live_turn_request(
                    user_input,
                    attachments=list(attachments or []),
                    delivery=TurnDelivery.STREAM_ASYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
                chunk_callback=_wrapped_chunk_callback,
            ),
        )
        reply, should_end = response.as_result()
        if not streamed:
            self._deliver_buffered_stream_chunks(str(reply or ""), chunk_callback)
        return reply, should_end

    async def handle_turn_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        """Canonical async turn entry-point."""
        response = await cast(
            Awaitable[TurnResponse],
            self.execute_turn(
                live_turn_request(
                    user_input,
                    attachments=list(attachments or []),
                    delivery=TurnDelivery.ASYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
            ),
        )
        return response.as_result()

    def handle_turn_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        """Canonical sync turn entry-point."""
        response = cast(
            TurnResponse,
            self.execute_turn(
                live_turn_request(
                    user_input,
                    attachments=list(attachments or []),
                    delivery=TurnDelivery.SYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                ),
            ),
        )
        return response.as_result()
