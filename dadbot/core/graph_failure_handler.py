"""DadBotGraphFailureHandlerMixin — graph failure detection, event emission, and recovery.

Extracted execution boundary from DadBotTurnMixin for clarity and testability.
Handles:
- Failure payload sanitization and safe emission
- Kernel boundary-aware event recording (fallback to logging when out-of-kernel)
- Failure response formatting
- Exception classification and re-raising (IntegrityBreachError vs. runtime errors)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dadbot.contracts import AttachmentList
from dadbot.core.execution_contract import SovereignContext
from dadbot.core.kernel_signals import CorrelationContext, TracingContext

if TYPE_CHECKING:
    from dadbot.core.mixin_protocols import GraphFailureHandlerProvider

logger = logging.getLogger(__name__)


class DadBotGraphFailureHandlerMixin:
    """Handles graph execution failures, event emission, and response formatting.
    
    Methods in this mixin are called during graph execution failure paths and are
    responsible for:
    1. Formatting failure payloads safely (truncating, sanitizing)
    2. Emitting events to the kernel boundary with fallback to logging
    3. Generating user-facing failure responses
    4. Classifying exceptions and re-raising appropriately
    """

    def _graph_failure_session_id(self) -> str:
        """Resolve the session/tenant ID for failure reporting."""
        candidate = getattr(self, "active_thread_id", "") or getattr(
            self,
            "tenant_id",
            "",
        )
        normalized = str(candidate or "").strip()
        return normalized or "default"

    @staticmethod
    def _safe_graph_failure_payload(value: Any, *, limit: int = 240) -> Any:
        """Recursively sanitize and truncate failure payload for safe emission."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:limit]
        if isinstance(value, dict):
            return {
                str(key)[:80]: DadBotGraphFailureHandlerMixin._safe_graph_failure_payload(
                    item,
                    limit=limit,
                )
                for key, item in list(value.items())[:12]
            }
        if isinstance(value, (list, tuple, set)):
            return [
                DadBotGraphFailureHandlerMixin._safe_graph_failure_payload(item, limit=limit)
                for item in list(value)[:12]
            ]
        return str(value)[:limit]

    def _emit_graph_failure_event(
        self: GraphFailureHandlerProvider,
        *,
        mode: str,
        correlation_id: str,
        trace_id: str,
        user_input: str,
        attachments: AttachmentList | None,
        exc: Exception,
    ) -> None:
        """Emit a structured failure event to the ledger (best-effort, never blocks primary failure)."""
        orchestrator = getattr(self, "_turn_orchestrator", None)
        if orchestrator is None:
            try:
                orchestrator = self._get_turn_orchestrator()
            except Exception:  # noqa: BLE001
                orchestrator = None
        control_plane = getattr(orchestrator, "control_plane", None)

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
        self._safe_emit_graph_failure_event(
            control_plane,
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

    @staticmethod
    def _safe_emit_graph_failure_event(control_plane: Any, **event: Any) -> None:
        """Best-effort graph-failure emission that never masks the primary failure.
        
        - If in kernel boundary: writes to ledger.
        - If out-of-kernel: logs warning and returns (never throws).
        - If control_plane is None: returns silently.
        """
        if control_plane is None:
            return

        try:
            ledger_writer = getattr(control_plane, "ledger_writer", None)
            write_event = getattr(ledger_writer, "write_event", None)
            if not callable(write_event):
                return
            write_event(**event)
        except RuntimeError as exc:
            if "Kernel boundary violation" in str(exc):
                logger.warning(
                    "GRAPH_FAILURE_EVENT_DROPPED (out-of-kernel): %s",
                    event,
                )
                return
            raise

    def _graph_failure_reply(self: GraphFailureHandlerProvider, correlation_id: str) -> str:
        """Format a user-facing failure message with reference ID."""
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
        context: SovereignContext | None = None,
    ) -> None:
        """Classify, log, emit event for, and re-raise a graph execution failure.
        
        - IntegrityBreachError: re-raise original (already hard-stopped by kernel)
        - Other exceptions: re-raise as RuntimeError (legacy path disabled in strict mode)
        """
        context_trace_id = str(context.trace_id or "") if context is not None else ""
        correlation_id = str(CorrelationContext.current() or CorrelationContext.ensure())
        trace_id = str(context_trace_id or TracingContext.current_trace_id() or correlation_id)
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
        try:
            from dadbot.core.execution_ledger import IntegrityBreachError

            if isinstance(exc, IntegrityBreachError):
                raise exc
        except ImportError:
            pass
        raise RuntimeError(
            "Graph execution failed in strict mode; legacy path is disabled",
        ) from exc
