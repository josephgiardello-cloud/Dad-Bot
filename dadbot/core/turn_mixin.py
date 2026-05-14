"""DadBotTurnMixin — turn execution and graph failure handling for DadBot.

Extracted from the DadBot god-class. Owns:
- Turn orchestrator resolution (_get_turn_orchestrator)
- Graph turn execution (sync/async, coro-in-thread bridge)
- Graph failure event emission and strict-mode error propagation
- Public turn entry-points: process_user_message*, handle_turn_*, stream variants
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, cast
from uuid import uuid4

from dadbot.contracts import AttachmentList, ChunkCallback, FinalizedTurnResult
from dadbot.core.execution_contract import (
    AgentState,
    ExecutionMode,
    SovereignContext,
    TurnDelivery,
    TurnRequest,
    TurnResponse,
    TurnResult,
    UserInput,
    live_turn_request,
)
from dadbot.core.graph_failure_handler import DadBotGraphFailureHandlerMixin
from dadbot.core.kernel_locks import KernelReplaySequenceLock
from dadbot.core.kernel_signals import CorrelationContext, TracingContext
from dadbot.memory.ledger import MemoryLedger
from dadbot.core.policy_store import DadPolicyStore
from dadbot.core.runtime_service_provider import DefaultCoreRuntimeServices
from dadbot.core.runtime_errors import (
    CanonicalInvariantViolation,
    ConfigurationError,
    TransientExecutionError,
)
from dadbot.core.turn_handler import TurnContext, TurnHandler
from dadbot.core.world_model import WorldModelStore

logger = logging.getLogger(__name__)

SYNC_DELIVERIES: set[TurnDelivery] = {TurnDelivery.SYNC, TurnDelivery.STREAM}
ASYNC_DELIVERIES: set[TurnDelivery] = {TurnDelivery.ASYNC, TurnDelivery.STREAM_ASYNC}


class DadBotTurnMixin(DadBotGraphFailureHandlerMixin):
    """Turn execution and graph failure handling for the DadBot facade."""

    # These attributes are provided by the concrete DadBot facade at runtime.
    _turn_graph_config_path: str
    _turn_execution_lock: Any
    services: Any
    turn_orchestrator: Any

    def _get_runtime_services(self) -> Any:
        runtime_services = getattr(self, "runtime_services", None)
        if runtime_services is not None:
            return runtime_services
        runtime_services = DefaultCoreRuntimeServices(self)
        self.runtime_services = runtime_services
        return runtime_services

    def _get_policy_store(self) -> DadPolicyStore:
        services = self._get_runtime_services()
        getter = getattr(services, "get_policy_store", None)
        if callable(getter):
            policy_store = getter()
            if isinstance(policy_store, DadPolicyStore):
                return policy_store
        raise ConfigurationError("Core runtime services must provide a valid policy store.")

    def _get_prompt_builder(self) -> Callable[[], str] | None:
        relationship_manager = getattr(self, "relationship_manager", None)
        builder = getattr(relationship_manager, "build_prompt_context", None)
        if callable(builder):
            return cast("Callable[[], str]", builder)
        return None

    def _get_relationship_snapshotter(self) -> Callable[[], dict[str, Any] | None] | None:
        relationship_manager = getattr(self, "relationship_manager", None)
        snapshotter = getattr(relationship_manager, "current_state", None)
        if callable(snapshotter):
            return cast("Callable[[], dict[str, Any] | None]", snapshotter)
        return None

    def _get_memory_ledger(self) -> MemoryLedger:
        services = self._get_runtime_services()
        getter = getattr(services, "get_memory_ledger", None)
        if callable(getter):
            ledger = getter()
            if isinstance(ledger, MemoryLedger):
                return ledger
        raise ConfigurationError("Core runtime services must provide a valid memory ledger.")

    def _get_world_model_store(self) -> WorldModelStore:
        services = self._get_runtime_services()
        getter = getattr(services, "get_world_model_store", None)
        if callable(getter):
            store = getter()
            if isinstance(store, WorldModelStore):
                return store
        raise ConfigurationError("Core runtime services must provide a valid world model store.")

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

    @staticmethod
    def _derive_confluence_key(
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None,
    ) -> str:
        payload = {
            "session_id": str(session_id or "default"),
            "user_input": str(user_input or ""),
            "attachments": list(attachments or []),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8"),
        ).hexdigest()[:24]
        return f"turn:{digest}"

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
        confluence_key = self._derive_confluence_key(
            session_id=session_id,
            user_input=str(user_input or ""),
            attachments=attachments,
        )
        submit_turn = getattr(orchestrator, "_submit_turn_via_control_plane", None)
        if not callable(submit_turn):
            raise ConfigurationError(
                "Canonical execution gate violation: control-plane submit entrypoint missing.",
            )

        handler = TurnHandler(
            submit_turn=cast(Any, submit_turn),
            policy_store=self._get_policy_store(),
            prompt_builder=self._get_prompt_builder(),
            memory_ledger=self._get_memory_ledger(),
            relationship_snapshotter=self._get_relationship_snapshotter(),
            world_model_store=self._get_world_model_store(),
        )
        return await handler.process_turn(
            TurnContext(
                user_input=str(user_input or ""),
                attachments=attachments,
                session_id=session_id,
                confluence_key=confluence_key,
            ),
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
        return self._event_tap_context_from_context(None)

    def _event_tap_context_from_context(self, context: SovereignContext | None) -> tuple[str, str, str]:
        if context is not None:
            return (
                str(context.session_id or "default"),
                str(context.tenant_id or "default"),
                str(context.trace_id or ""),
            )
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
    # Signoff helper used by failure reply + stream variants
    # ------------------------------------------------------------------

    def _append_signoff(self, text: str) -> str:
        """Apply reply signoff via reply finalization manager when available."""
        finalization = getattr(self, "reply_finalization", None)
        append_signoff = getattr(finalization, "append_signoff", None)
        if callable(append_signoff):
            return str(append_signoff(text))
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
        if delivery in ASYNC_DELIVERIES:
            return "async"
        if delivery in SYNC_DELIVERIES:
            return "sync"
        raise ValueError(f"Unsupported turn delivery: {delivery}")

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
            if request.delivery in SYNC_DELIVERIES:
                result = self._run_graph_turn_sync(
                    user_input,
                    attachments=attachments,
                    chunk_callback=chunk_callback,
                )
            elif request.delivery in ASYNC_DELIVERIES:
                result = await self._run_graph_turn_async(
                    user_input,
                    attachments=attachments,
                    chunk_callback=chunk_callback,
                )
            else:
                raise ValueError(f"Unsupported turn delivery: {request.delivery}")
            self._emit_turn_event(
                "TURN_END",
                mode=event_mode,
                status="ok",
                should_end=bool(result[1]),
            )
            self._checkpoint_turn_state()
            return self._response_from_result(request, result)
        except Exception as exc:
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
                context=request.context,
            )
            raise TransientExecutionError(
                "Graph execution failed after failure handler dispatch.",
            ) from exc
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

        if request.mode == ExecutionMode.REPLAY:
            if state is None:
                raise RuntimeError("replay mode requires AgentState")
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
            # Recovery reuses the canonical live path; resume/hydration decisions
            # are resolved by checkpoint and lifecycle infrastructure upstream.
            normalized_request = request.model_copy(
                update={
                    "mode": ExecutionMode.LIVE,
                    "context": self._build_sovereign_context(
                        mode=ExecutionMode.LIVE,
                        base_context=request.context,
                    ),
                },
            )
            return await self._execute_live_turn_async(
                normalized_request,
                chunk_callback=chunk_callback,
            )

        raise ValueError(f"Unsupported execution mode: {request.mode}")

    def execute_turn(
        self,
        request: TurnRequest,
        *,
        state: AgentState | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResponse | Awaitable[TurnResponse]:
        if request.delivery in ASYNC_DELIVERIES:
            return self._execute_turn_async(
                request,
                state=state,
                chunk_callback=chunk_callback,
            )
        if request.delivery in SYNC_DELIVERIES:
            return self._run_turn_request_sync(
                request,
                state=state,
                chunk_callback=chunk_callback,
            )
        raise ValueError(f"Unsupported turn delivery: {request.delivery}")

    # ------------------------------------------------------------------
    # Public turn entry-points
    # ------------------------------------------------------------------

    def _get_pre_turn_checkin_due(self) -> bool:
        memory = getattr(self, "memory", None)
        should_do_daily_checkin = getattr(memory, "should_do_daily_checkin", None)
        if callable(should_do_daily_checkin):
            try:
                return bool(should_do_daily_checkin())
            except Exception:
                return False
        return False

    def _blend_reply_with_daily_checkin(
        self,
        dad_reply: str | None,
        pre_turn_checkin_due: bool,
    ) -> str | None:
        memory = getattr(self, "memory", None)
        should_do_daily_checkin = getattr(memory, "should_do_daily_checkin", None)
        tone_context = getattr(self, "tone_context", None)
        blend_daily_checkin_reply = getattr(tone_context, "blend_daily_checkin_reply", None)
        if not callable(should_do_daily_checkin) or not callable(blend_daily_checkin_reply):
            return dad_reply
        try:
            should_blend = (
                bool(pre_turn_checkin_due)
                or bool(getattr(self, "_pending_daily_checkin_context", False))
                or bool(getattr(self, "_last_should_offer_daily_checkin", False))
                or bool(should_do_daily_checkin())
            )
            if should_blend:
                self._pending_daily_checkin_context = True
                self._last_should_offer_daily_checkin = False
                mood = "neutral"
                last_ctx = getattr(self, "_last_turn_context", None)
                if last_ctx is not None:
                    mood = str(getattr(last_ctx, "state", {}).get("mood") or "neutral")
                dad_reply = cast(str | None, blend_daily_checkin_reply(dad_reply, mood))
        except Exception as exc:
            logger.debug("Daily check-in blend failed (non-fatal): %s", exc)
            if not isinstance(dad_reply, str):
                dad_reply = None if dad_reply is None else str(dad_reply)
        return dad_reply

    def process_user_message(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        pre_turn_daily_checkin_due = self._get_pre_turn_checkin_due()

        context = self._build_sovereign_context(mode=ExecutionMode.LIVE)
        response = cast(
            TurnResponse,
            self.execute_turn(
                live_turn_request(
                    user_input,
                    attachments=list(attachments or []),
                    delivery=TurnDelivery.SYNC,
                    context=context,
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

        dad_reply = self._blend_reply_with_daily_checkin(dad_reply, pre_turn_daily_checkin_due)
        return cast(str | None, dad_reply), should_end

    def run_turn(
        self,
        input: UserInput,
        state: AgentState,
        *,
        chunk_callback: ChunkCallback | None = None,
        mode: ExecutionMode = ExecutionMode.LIVE,
        context: SovereignContext | None = None,
    ) -> TurnResult:
        """Canonical deterministic turn execution contract.

        LIVE delegates to the graph-backed process_user_message path.
        REPLAY requires an explicit replay handler on the facade.
        """
        state.recompute_invariance_hash()
        sovereign_context = context or self._build_sovereign_context(
            mode=mode,
            base_context=context,
        )

        response = cast(
            TurnResponse,
            self.execute_turn(
                TurnRequest(
                    input=input,
                    mode=mode,
                    delivery=TurnDelivery.SYNC,
                    session_id=str(getattr(self, "active_thread_id", "") or "default"),
                    context=sovereign_context,
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

    def _build_sovereign_context(
        self,
        *,
        mode: ExecutionMode,
        base_context: SovereignContext | None = None,
    ) -> SovereignContext:
        resolved_context = base_context or SovereignContext()
        active_trace_id = str(TracingContext.current_trace_id() or "").strip()
        active_correlation_id = str(CorrelationContext.current() or "").strip()
        resolved_trace_id = (
            str(resolved_context.trace_id or "").strip()
            or active_trace_id
            or active_correlation_id
            or uuid4().hex
        )
        resolved_request_id = str(resolved_context.request_id or "").strip() or resolved_trace_id
        return SovereignContext(
            session_id=str(resolved_context.session_id or getattr(self, "active_thread_id", "") or "default"),
            tenant_id=str(resolved_context.tenant_id or getattr(self, "tenant_id", "") or "default"),
            trace_id=resolved_trace_id,
            request_id=resolved_request_id,
            execution_mode=mode,
            policy_scope=str(resolved_context.policy_scope or "default"),
        )

    def _load_recovery_tap_state(self) -> tuple:
        tap = self._resolve_event_tap()
        if tap is None:
            raise RuntimeError("Recovery mode requires configured EventTap")
        latest_checkpoint = getattr(tap, "latest_checkpoint", None)
        events_after_cursor_fn = getattr(tap, "events_after_cursor", None)
        if not callable(latest_checkpoint) or not callable(events_after_cursor_fn):
            raise RuntimeError("Recovery mode requires checkpoint-capable EventTap")
        checkpoint = latest_checkpoint()
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("Recovery mode requires a durable checkpoint (replay-only policy)")
        apply_event = getattr(self, "_apply_recovery_event", None)
        if not callable(apply_event):
            raise RuntimeError("Recovery mode requires _apply_recovery_event handler")
        return checkpoint, events_after_cursor_fn, apply_event

    def _run_turn_recovery_default(
        self,
        input: UserInput,
        state: AgentState,
        *,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResult:
        checkpoint, events_after_cursor, apply_event = self._load_recovery_tap_state()
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
            trace_token=run_id,
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
                    context=self._build_sovereign_context(mode=ExecutionMode.LIVE),
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
                    context=self._build_sovereign_context(mode=ExecutionMode.LIVE),
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
                    context=self._build_sovereign_context(mode=ExecutionMode.LIVE),
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
                    context=self._build_sovereign_context(mode=ExecutionMode.LIVE),
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
                    context=self._build_sovereign_context(mode=ExecutionMode.LIVE),
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
                    context=self._build_sovereign_context(mode=ExecutionMode.LIVE),
                ),
            ),
        )
        return response.as_result()
