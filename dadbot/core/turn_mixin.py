"""DadBotTurnMixin — turn execution and graph failure handling for DadBot.

Extracted from the DadBot god-class. Owns:
- Turn orchestrator resolution (_get_turn_orchestrator)
- Graph turn execution (sync/async, coro-in-thread bridge)
- Graph failure event emission and strict-mode error propagation
- Public turn entry-points: process_user_message*, handle_turn_*, stream variants
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dadbot.contracts import AttachmentList, ChunkCallback, FinalizedTurnResult
from dadbot.core.observability import CorrelationContext, TracingContext

logger = logging.getLogger(__name__)


class DadBotTurnMixin:
    """Turn execution and graph failure handling for the DadBot facade."""

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
    ) -> FinalizedTurnResult:
        return await self.turn_orchestrator.handle_turn(
            user_input,
            attachments=attachments,
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

            coro = self._run_graph_turn_async(user_input, attachments=attachments)
            if loop is not None and loop.is_running():
                return self._run_coro_in_thread(coro)
            return asyncio.run(coro)

    def _validate_managers(self, *, smoke: bool = False) -> None:
        self.services.validate_facade(smoke=smoke)

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
            return append_signoff(text)
        compat_finalize = getattr(self, "finalize_reply", None)
        if callable(compat_finalize):
            return compat_finalize(text)
        return str(text or "")

    # ------------------------------------------------------------------
    # Public turn entry-points
    # ------------------------------------------------------------------

    def process_user_message(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        try:
            return self._run_graph_turn_sync(user_input, attachments=attachments)
        except Exception as exc:  # noqa: BLE001
            self._raise_graph_execution_failure(
                exc,
                mode="sync",
                user_input=user_input,
                attachments=attachments,
            )

    async def process_user_message_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        try:
            return await self._run_graph_turn_async(user_input, attachments=attachments)
        except Exception as exc:  # noqa: BLE001
            self._raise_graph_execution_failure(
                exc,
                mode="async",
                user_input=user_input,
                attachments=attachments,
            )

    def process_user_message_stream(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        reply, should_end = self.process_user_message(
            user_input,
            attachments=attachments,
        )
        self._deliver_buffered_stream_chunks(reply, chunk_callback)
        return reply, should_end

    async def process_user_message_stream_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        reply, should_end = await self.process_user_message_async(
            user_input,
            attachments=attachments,
        )
        self._deliver_buffered_stream_chunks(reply, chunk_callback)
        return reply, should_end

    async def handle_turn_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        """Canonical async turn entry-point."""
        return await self.process_user_message_async(
            user_input,
            attachments=attachments,
        )

    def handle_turn_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        """Canonical sync turn entry-point."""
        return self.process_user_message(user_input, attachments=attachments)
