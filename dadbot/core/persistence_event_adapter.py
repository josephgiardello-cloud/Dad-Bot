from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any


class GraphPersistenceEventAdapter:
    """Adapter for persistence event emission from graph execution.

    This keeps persistence payload shaping out of TurnGraph orchestration logic.
    The persistence service is the authority boundary; it must append runtime
    events to ExecutionLedger, and any filesystem emission is derived-only.
    """

    def __init__(self, *, json_safe: Callable[[Any], Any]) -> None:
        self._json_safe = json_safe

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        if is_dataclass(value):
            return dict(asdict(value))
        if isinstance(value, dict):
            return dict(value)
        return {}

    def emit_graph_checkpoint(
        self,
        *,
        service: Any,
        turn_context: Any,
        stage: str,
        status: str,
        error: str,
        active_stage: str,
        determinism_lock: dict[str, Any],
        checkpoint: dict[str, Any],
    ) -> None:
        save_checkpoint = getattr(service, "save_graph_checkpoint", None)
        if callable(save_checkpoint):
            save_checkpoint(checkpoint, _skip_turn_event=True)

        save_event = getattr(service, "save_turn_event", None)
        if not callable(save_event):
            return

        turn_context.event_sequence += 1
        save_event(
            {
                "event_type": "graph_checkpoint",
                "trace_id": turn_context.trace_id,
                "sequence": turn_context.event_sequence,
                "occurred_at": turn_context.temporal.wall_time,
                "stage": str(stage or ""),
                "status": str(status or ""),
                "error": str(error or "").strip(),
                "phase": turn_context.phase.value,
                "active_stage": str(active_stage or ""),
                "determinism_lock": self._json_safe(dict(determinism_lock or {})),
                "checkpoint": checkpoint,
            },
        )

    def emit_phase_transition(
        self,
        *,
        service: Any,
        turn_context: Any,
        stage: str,
        transition: dict[str, Any],
        determinism_lock: dict[str, Any],
    ) -> None:
        save_event = getattr(service, "save_turn_event", None)
        if not callable(save_event):
            return

        turn_context.event_sequence += 1
        save_event(
            {
                "event_type": "phase_transition",
                "trace_id": turn_context.trace_id,
                "sequence": turn_context.event_sequence,
                "occurred_at": turn_context.temporal.wall_time,
                "stage": str(stage or ""),
                "phase": turn_context.phase.value,
                "transition": dict(transition or {}),
                "determinism_lock": self._json_safe(dict(determinism_lock or {})),
            },
        )

    def emit_kernel_rejection(
        self,
        *,
        service: Any,
        turn_context: Any,
        stage: str,
        reason: str,
        semantics: Any,
    ) -> None:
        save_event = getattr(service, "save_turn_event", None)
        if not callable(save_event):
            return

        turn_context.event_sequence += 1
        save_event(
            {
                "event_type": "kernel_rejection",
                "trace_id": turn_context.trace_id,
                "sequence": turn_context.event_sequence,
                "occurred_at": turn_context.temporal.wall_time,
                "stage": str(stage or ""),
                "reason": str(reason or ""),
                "phase": turn_context.phase.value,
                "semantics": self._json_safe(self._to_dict(semantics)),
            },
        )

    def emit_execution_identity(
        self,
        *,
        service: Any,
        turn_context: Any,
        identity: Any,
    ) -> None:
        """Emit the canonical execution identity as a durable turn event.

        This makes the execution fingerprint a first-class replay artifact so
        validators can verify equivalence from the event stream alone.
        """
        save_event = getattr(service, "save_turn_event", None)
        if not callable(save_event):
            return

        turn_context.event_sequence += 1
        identity_dict = identity.to_dict() if hasattr(identity, "to_dict") else dict(identity or {})
        save_event(
            {
                "event_type": "execution_identity",
                "trace_id": turn_context.trace_id,
                "sequence": turn_context.event_sequence,
                "occurred_at": turn_context.temporal.wall_time,
                "phase": turn_context.phase.value,
                "identity": identity_dict,
            },
        )
