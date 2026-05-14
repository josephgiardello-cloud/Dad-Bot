from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from dadbot.core.execution_result_schema import ExecutionResultEnvelopeCompat


class CheckpointEnvelope(TypedDict):
    trace_id: str
    state: dict[str, Any]
    manifest: dict[str, Any]
    checksum: str


class TraceEventEnvelope(TypedDict):
    event_type: str
    session_id: str
    trace_id: str
    timestamp: float
    sequence: int
    payload: dict[str, Any]


class DecisionExplanationEnvelope(TypedDict):
    available: bool
    timeline_events: int
    last_plan: dict[str, Any]
    recent_events: list[str]
    semantic_memory_context_size: int
    active_hypothesis: dict[str, Any]
    belief_calibration: dict[str, Any]
    tool_self_model: dict[str, Any]
    interactive_cognition_ui: dict[str, Any]
    alignment_policy: dict[str, Any]
    tool_ecosystem: dict[str, Any]
    swarm_health: dict[str, Any]
    reason: NotRequired[str]


class LedgerEntryEnvelope(TypedDict):
    type: str
    session_id: str
    trace_id: str
    kernel_step_id: str
    payload: dict[str, Any]
    timestamp: float
    committed: NotRequired[bool]


_EXECUTION_RESULT_REQUIRED_KEYS: frozenset[str] = frozenset({"status", "degradation", "failure", "timeout", "outputs"})
_CHECKPOINT_REQUIRED_KEYS: frozenset[str] = frozenset({"trace_id", "state", "manifest", "checksum"})
_TRACE_EVENT_REQUIRED_KEYS: frozenset[str] = frozenset({"event_type", "session_id", "trace_id", "timestamp", "sequence", "payload"})
_DECISION_EXPLANATION_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "available",
        "timeline_events",
        "last_plan",
        "recent_events",
        "semantic_memory_context_size",
        "active_hypothesis",
        "belief_calibration",
        "tool_self_model",
        "interactive_cognition_ui",
        "alignment_policy",
        "tool_ecosystem",
        "swarm_health",
    },
)
_LEDGER_ENTRY_REQUIRED_KEYS: frozenset[str] = frozenset({"type", "session_id", "trace_id", "kernel_step_id", "payload", "timestamp"})
_KNOWN_FAILURE_TYPES: frozenset[str] = frozenset({"retryable", "non_retryable", "poison", "partial_commit", "unknown_state"})
_KNOWN_FAILURE_ACTIONS: frozenset[str] = frozenset({"manual_retry", "fail_fast", "quarantine", "reconcile"})


def validate_execution_result_contract(payload: dict[str, Any]) -> ExecutionResultEnvelopeCompat:
    envelope = dict(payload or {})
    missing = sorted(key for key in _EXECUTION_RESULT_REQUIRED_KEYS if key not in envelope)
    if missing:
        raise RuntimeError(f"execution_result contract violation: missing keys {', '.join(missing)}")
    return envelope  # type: ignore[return-value]


def validate_checkpoint_contract(payload: dict[str, Any]) -> CheckpointEnvelope:
    envelope = dict(payload or {})
    missing = sorted(key for key in _CHECKPOINT_REQUIRED_KEYS if key not in envelope)
    if missing:
        raise RuntimeError(f"checkpoint contract violation: missing keys {', '.join(missing)}")
    extra = sorted(key for key in envelope.keys() if key not in _CHECKPOINT_REQUIRED_KEYS)
    if extra:
        raise RuntimeError(f"checkpoint contract violation: unexpected keys {', '.join(extra)}")

    trace_id = str(envelope.get("trace_id") or "").strip()
    checksum = str(envelope.get("checksum") or "").strip()
    state = envelope.get("state")
    manifest = envelope.get("manifest")
    if not trace_id:
        raise RuntimeError("checkpoint contract violation: trace_id is required")
    if not checksum:
        raise RuntimeError("checkpoint contract violation: checksum is required")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint contract violation: state must be a dict")
    if not isinstance(manifest, dict):
        raise RuntimeError("checkpoint contract violation: manifest must be a dict")
    return {
        "trace_id": trace_id,
        "state": dict(state),
        "manifest": dict(manifest),
        "checksum": checksum,
    }


def validate_failure_contract(*, failure_type: str, failure_action: str) -> None:
    if str(failure_type or "") not in _KNOWN_FAILURE_TYPES:
        raise RuntimeError(f"failure contract violation: unknown failure_type {failure_type!r}")
    if str(failure_action or "") not in _KNOWN_FAILURE_ACTIONS:
        raise RuntimeError(f"failure contract violation: unknown failure_action {failure_action!r}")


def validate_trace_event_contract(payload: dict[str, Any]) -> TraceEventEnvelope:
    envelope = dict(payload or {})
    missing = sorted(key for key in _TRACE_EVENT_REQUIRED_KEYS if key not in envelope)
    if missing:
        raise RuntimeError(f"trace event contract violation: missing keys {', '.join(missing)}")
    if not isinstance(envelope.get("payload"), dict):
        raise RuntimeError("trace event contract violation: payload must be a dict")
    return {
        "event_type": str(envelope.get("event_type") or "runtime.unknown"),
        "session_id": str(envelope.get("session_id") or "default"),
        "trace_id": str(envelope.get("trace_id") or ""),
        "timestamp": float(envelope.get("timestamp") or 0.0),
        "sequence": int(envelope.get("sequence") or 0),
        "payload": dict(envelope.get("payload") or {}),
    }


def validate_decision_explanation_contract(payload: dict[str, Any]) -> DecisionExplanationEnvelope:
    envelope = dict(payload or {})
    missing = sorted(key for key in _DECISION_EXPLANATION_REQUIRED_KEYS if key not in envelope)
    if missing:
        raise RuntimeError(f"decision explanation contract violation: missing keys {', '.join(missing)}")
    if not isinstance(envelope.get("last_plan"), dict):
        raise RuntimeError("decision explanation contract violation: last_plan must be a dict")
    if not isinstance(envelope.get("recent_events"), list):
        raise RuntimeError("decision explanation contract violation: recent_events must be a list")
    return {
        "available": bool(envelope.get("available", False)),
        "timeline_events": int(envelope.get("timeline_events") or 0),
        "last_plan": dict(envelope.get("last_plan") or {}),
        "recent_events": [str(item) for item in list(envelope.get("recent_events") or [])],
        "semantic_memory_context_size": int(envelope.get("semantic_memory_context_size") or 0),
        "active_hypothesis": dict(envelope.get("active_hypothesis") or {}),
        "belief_calibration": dict(envelope.get("belief_calibration") or {}),
        "tool_self_model": dict(envelope.get("tool_self_model") or {}),
        "interactive_cognition_ui": dict(envelope.get("interactive_cognition_ui") or {}),
        "alignment_policy": dict(envelope.get("alignment_policy") or {}),
        "tool_ecosystem": dict(envelope.get("tool_ecosystem") or {}),
        "swarm_health": dict(envelope.get("swarm_health") or {}),
        **({"reason": str(envelope.get("reason") or "")} if "reason" in envelope else {}),
    }


def validate_ledger_entry_contract(payload: dict[str, Any]) -> LedgerEntryEnvelope:
    envelope = dict(payload or {})
    missing = sorted(key for key in _LEDGER_ENTRY_REQUIRED_KEYS if key not in envelope)
    if missing:
        raise RuntimeError(f"ledger entry contract violation: missing keys {', '.join(missing)}")
    if not isinstance(envelope.get("payload"), dict):
        raise RuntimeError("ledger entry contract violation: payload must be a dict")
    return {
        "type": str(envelope.get("type") or ""),
        "session_id": str(envelope.get("session_id") or "default"),
        "trace_id": str(envelope.get("trace_id") or ""),
        "kernel_step_id": str(envelope.get("kernel_step_id") or ""),
        "payload": dict(envelope.get("payload") or {}),
        "timestamp": float(envelope.get("timestamp") or 0.0),
        **({"committed": bool(envelope.get("committed", False))} if "committed" in envelope else {}),
    }
