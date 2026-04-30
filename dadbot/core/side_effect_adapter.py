"""
SideEffectAdapter — pure emission layer extracted from TurnGraph.

This module owns ONLY record-and-emit operations.  It has zero decision-making,
zero branching logic, zero recovery logic, and zero policy enforcement.

Strict interface contract
-------------------------
record_receipt()       — append signed execution receipt to the chain
record_witness()       — invoke the registered execution witness emitter
freeze_capabilities()  — freeze the capability set at turn start
verify_capability_freeze() — verify no escalation on a resumed turn
emit_execution_event() — append a deterministic event to the execution trace

Nothing else.  If logic needs to decide *whether* to call one of these methods,
that decision stays in TurnGraph.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from dadbot.core.capability_registry import freeze_capabilities as _freeze, verify_capability_freeze as _verify

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Recursively sanitize a value for JSON serialisation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


class SideEffectAdapter:
    """Pure record-and-emit adapter.  No decisions, no branching, no policy.

    Architectural role
    ------------------
    Layer               Responsibility
    ──────────────────────────────────────────────────────
    StageEntryGate      "Can I enter this stage?"
    SideEffectAdapter   "Record what happened"
    TurnGraph           "Execute workflow"
    ControlPlane        "Schedule execution"
    """

    # ------------------------------------------------------------------
    # Receipt chain
    # ------------------------------------------------------------------

    def record_receipt(
        self,
        context: Any,
        stage_name: str,
        *,
        signer: Any,
        stage_call_id: str = "",
        checkpoint_hash: str = "",
    ) -> None:
        """Append a signed execution receipt to the chain stored in context.state."""
        from dadbot.core.execution_receipt import ReceiptChain

        chain = ReceiptChain.from_state(context.state)
        receipt = signer.sign(
            turn_id=str(context.trace_id or ""),
            stage=stage_name,
            sequence=chain.next_sequence,
            stage_call_id=str(stage_call_id or ""),
            checkpoint_hash=str(checkpoint_hash or ""),
            prev_receipt_sig=chain.last_signature,
        )
        chain.append(receipt)
        chain.write_to_state(context.state)

    # ------------------------------------------------------------------
    # Execution witness
    # ------------------------------------------------------------------

    def record_witness(
        self,
        component: str,
        context: Any,
        *,
        emitter: Callable[[str, Any], None] | None,
    ) -> None:
        """Invoke the registered execution witness emitter (no-op if None)."""
        if callable(emitter):
            emitter(str(component or ""), context)

    # ------------------------------------------------------------------
    # Capability freeze / verify
    # ------------------------------------------------------------------

    def freeze_capabilities(
        self,
        context: Any,
        *,
        policy: Any,
        session_id: str,
    ) -> None:
        """Freeze the current capability set at turn start."""
        _freeze(context, policy=policy, session_id=session_id)

    def verify_capability_freeze(
        self,
        context: Any,
        *,
        policy: Any,
        session_id: str,
    ) -> None:
        """Verify no privilege escalation since the turn's capability set was frozen."""
        _verify(context, policy=policy, session_id=session_id)

    # ------------------------------------------------------------------
    # Execution trace event emission
    # ------------------------------------------------------------------

    def emit_execution_event(
        self,
        context: Any,
        *,
        event_type: str,
        stage: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append a deterministic event to context.state['execution_trace'].

        Reads the phase from context.phase (TurnPhase enum or plain string).
        Increments the monotonic sequence counter in context.state.
        """
        sequence = int(context.state.get("_execution_trace_sequence", 0) or 0) + 1
        context.state["_execution_trace_sequence"] = sequence

        phase_raw = getattr(context, "phase", None)
        phase_value = str(getattr(phase_raw, "value", phase_raw) or "")

        event_dict = {
            "sequence": sequence,
            "event_type": str(event_type or ""),
            "stage": str(stage or ""),
            "phase": phase_value,
            "trace_id": str(getattr(context, "trace_id", "") or ""),
            "detail": _json_safe(dict(detail or {})),
        }

        trace = context.state.setdefault("execution_trace", [])
        if not isinstance(trace, list):
            trace = []
            context.state["execution_trace"] = trace
        trace.append(event_dict)

    def record_failure_taxonomy(
        self,
        context: Any,
        *,
        severity_str: str,
        error_str: str,
    ) -> None:
        """Write the standardized failure taxonomy payload to context.state.

        Called only on turn failure, after classification.  No decision logic —
        the caller (TurnGraph) already holds the classified severity; this method
        only persists the result.
        """
        context.state["failure_taxonomy"] = {
            "severity": str(severity_str or ""),
            "error": str(error_str or ""),
        }
