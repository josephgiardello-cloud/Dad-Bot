from __future__ import annotations

import copy
from typing import Any

PERSISTENCE_SCHEMA_VERSION = "persistence-record.v1"
TRACE_EVENT_SCHEMA_VERSION = "trace-event.v1"
REPLAY_RECORD_SCHEMA_VERSION = "replay-record.v1"
IO_CAPTURE_SCHEMA_VERSION = "io-capture.v1"
STATE_SNAPSHOT_SCHEMA_VERSION = "state-snapshot.v1"
CHECKPOINT_RECORD_SCHEMA_VERSION = "checkpoint-record.v1"


class PersistenceSchemaError(ValueError):
    """Raised when a persisted record violates canonical schema constraints."""


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return bool(default)


def normalize_io_capture(record: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(record or {})
    tool_name = str(payload.get("tool") or payload.get("tool_name") or "").strip().lower()
    version = str(payload.get("version") or "v1").strip() or "v1"
    status = str(payload.get("status") or "ok").strip().lower() or "ok"
    response = copy.deepcopy(payload.get("response"))
    if response is None and "output" in payload:
        response = copy.deepcopy(payload.get("output"))

    return {
        "schema_version": IO_CAPTURE_SCHEMA_VERSION,
        "tool": tool_name,
        "version": version,
        "status": status,
        "response": response,
    }


def _trace_event_base_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TRACE_EVENT_SCHEMA_VERSION,
        "trace_id": str(payload.get("trace_id") or "unknown").strip() or "unknown",
        "event_id": str(payload.get("event_id") or "").strip(),
        "sequence": _as_int(payload.get("sequence"), default=0),
        "event_type": str(payload.get("event_type") or "").strip().lower(),
        "stage": str(payload.get("stage") or "").strip().lower(),
        "status": str(payload.get("status") or "").strip().lower(),
        "occurred_at": str(payload.get("occurred_at") or "").strip(),
    }


def _apply_trace_event_structured_fields(normalized: dict[str, Any], payload: dict[str, Any]) -> None:
    if "determinism_lock" in payload and isinstance(payload.get("determinism_lock"), dict):
        lock = dict(payload.get("determinism_lock") or {})
        normalized["determinism_lock"] = {
            "lock_hash": str(lock.get("lock_hash") or "").strip(),
            "enforced": _as_bool(lock.get("enforced"), default=False),
        }

    if "checkpoint" in payload and isinstance(payload.get("checkpoint"), dict):
        normalized["checkpoint"] = copy.deepcopy(payload.get("checkpoint"))

    if "session_state" in payload and isinstance(payload.get("session_state"), dict):
        normalized["session_state"] = copy.deepcopy(payload.get("session_state"))

    io_capture = payload.get("io_capture")
    if isinstance(io_capture, dict):
        normalized["io_capture"] = normalize_io_capture(io_capture)


def normalize_trace_event(record: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(record or {})
    normalized = _trace_event_base_fields(payload)
    _apply_trace_event_structured_fields(normalized, payload)

    # Preserve any additional fields for backward compatibility.
    known = set(normalized.keys())
    for key, value in payload.items():
        if key in known:
            continue
        normalized[key] = copy.deepcopy(value)

    return normalized


def normalize_state_snapshot(record: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(record or {})
    lock = dict(payload.get("determinism_lock") or {})
    return {
        "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
        "trace_id": str(payload.get("trace_id") or "unknown").strip() or "unknown",
        "created_at": str(payload.get("created_at") or "").strip(),
        "last_sequence": _as_int(payload.get("last_sequence"), default=0),
        "phase": str(payload.get("phase") or "PLAN").strip() or "PLAN",
        "strict_sequence_hash": str(payload.get("strict_sequence_hash") or "").strip(),
        "event_count": _as_int(payload.get("event_count"), default=0),
        "determinism_lock": {
            "lock_hash": str(lock.get("lock_hash") or "").strip(),
            "consistent": _as_bool(lock.get("consistent"), default=True),
        },
    }


def _base_checkpoint_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_RECORD_SCHEMA_VERSION,
        "trace_id": str(payload.get("trace_id") or "unknown").strip() or "unknown",
        "session_id": str(payload.get("session_id") or "default").strip() or "default",
        "stage": str(payload.get("stage") or "unknown").strip().lower() or "unknown",
        "status": str(payload.get("status") or "unknown").strip().lower() or "unknown",
        "phase": str(payload.get("phase") or "").strip(),
        "checkpoint_hash": str(payload.get("checkpoint_hash") or "").strip(),
        "prev_checkpoint_hash": str(payload.get("prev_checkpoint_hash") or "").strip(),
        "event_sequence_id": _as_int(payload.get("event_sequence_id"), default=0),
        "occurred_at": str(payload.get("occurred_at") or "").strip(),
        "execution_mode": str(payload.get("execution_mode") or "live").strip().lower() or "live",
    }


def _normalize_checkpoint_lock(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "lock_hash": str(lock.get("lock_hash") or "").strip(),
        "enforced": _as_bool(lock.get("enforced"), default=False),
    }


def _normalize_execution_state(execution_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": copy.deepcopy(dict(execution_state.get("state") or {})),
        "metadata": copy.deepcopy(dict(execution_state.get("metadata") or {})),
        "phase_history": copy.deepcopy(list(execution_state.get("phase_history") or [])),
        "stage_traces": copy.deepcopy(list(execution_state.get("stage_traces") or [])),
        "event_sequence": _as_int(execution_state.get("event_sequence"), default=0),
    }


def _normalize_checkpoint_continuity(continuity: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_fingerprint": str(continuity.get("execution_fingerprint") or "").strip(),
        "strict_sequence_hash": str(continuity.get("strict_sequence_hash") or "").strip(),
        "event_count": _as_int(continuity.get("event_count"), default=0),
    }


def normalize_checkpoint_record(record: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(record or {})
    lock = dict(payload.get("determinism_lock") or {})
    execution_state = dict(payload.get("execution_state") or {})
    continuity = dict(payload.get("continuity") or {})
    out = _base_checkpoint_record(payload)
    if lock:
        out["determinism_lock"] = _normalize_checkpoint_lock(lock)
    if execution_state:
        out["execution_state"] = _normalize_execution_state(execution_state)
    if continuity:
        out["continuity"] = _normalize_checkpoint_continuity(continuity)
    return out


def normalize_replay_record(
    *,
    trace_token: str = "",
    events: list[dict[str, Any]],
    strict_sequence_hash: str,
    strict_sequence: list[dict[str, Any]],
    phase: str,
    replayed_state: dict[str, Any],
    replayed_metadata: dict[str, Any],
    determinism: dict[str, Any],
    **legacy_kwargs: Any,
) -> dict[str, Any]:
    legacy_trace = legacy_kwargs.pop("trace_id", "")
    if legacy_kwargs:
        unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    normalized_events = [normalize_trace_event(item) for item in list(events or [])]
    output = {
        "phase": str(phase or "PLAN").strip() or "PLAN",
        "replayed_state": copy.deepcopy(dict(replayed_state or {})),
        "replayed_metadata": copy.deepcopy(dict(replayed_metadata or {})),
        "determinism": copy.deepcopy(dict(determinism or {})),
    }
    return {
        "schema_version": REPLAY_RECORD_SCHEMA_VERSION,
        "persistence_schema_version": PERSISTENCE_SCHEMA_VERSION,
        "trace_id": str(trace_token or legacy_trace or "unknown").strip() or "unknown",
        "strict_sequence_hash": str(strict_sequence_hash or "").strip(),
        "strict_sequence": copy.deepcopy(list(strict_sequence or [])),
        "event_count": len(normalized_events),
        "events": normalized_events,
        "replay_output": output,
        # Backward-compatible top-level mirrors.
        "phase": output["phase"],
        "replayed_state": output["replayed_state"],
        "replayed_metadata": output["replayed_metadata"],
        "determinism": output["determinism"],
    }


def trace_event_errors(record: dict[str, Any] | None) -> list[str]:
    payload = dict(record or {})
    errors: list[str] = []
    if str(payload.get("schema_version") or "").strip() != TRACE_EVENT_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if not str(payload.get("trace_id") or "").strip():
        errors.append("trace_id missing")
    if not str(payload.get("event_id") or "").strip():
        errors.append("event_id missing")
    if _as_int(payload.get("sequence"), default=0) <= 0:
        errors.append("sequence must be >= 1")
    if not str(payload.get("event_type") or "").strip():
        errors.append("event_type missing")
    if not str(payload.get("occurred_at") or "").strip():
        errors.append("occurred_at missing")
    return errors


def _append_replay_record_header_errors(payload: dict[str, Any], errors: list[str]) -> None:
    if str(payload.get("schema_version") or "").strip() != REPLAY_RECORD_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if str(payload.get("persistence_schema_version") or "").strip() != PERSISTENCE_SCHEMA_VERSION:
        errors.append("persistence_schema_version mismatch")
    if not str(payload.get("trace_id") or "").strip():
        errors.append("trace_id missing")


def _append_replay_output_errors(replay_output: Any, errors: list[str]) -> None:
    if not isinstance(replay_output, dict):
        errors.append("replay_output must be a dict")
        return
    if not str(replay_output.get("phase") or "").strip():
        errors.append("replay_output.phase missing")
    if not isinstance(replay_output.get("replayed_state"), dict):
        errors.append("replay_output.replayed_state must be a dict")
    if not isinstance(replay_output.get("replayed_metadata"), dict):
        errors.append("replay_output.replayed_metadata must be a dict")
    if not isinstance(replay_output.get("determinism"), dict):
        errors.append("replay_output.determinism must be a dict")


def replay_record_errors(record: dict[str, Any] | None) -> list[str]:
    payload = dict(record or {})
    errors: list[str] = []
    _append_replay_record_header_errors(payload, errors)

    events = payload.get("events")
    if not isinstance(events, list):
        errors.append("events must be a list")
        events = []
    event_count = _as_int(payload.get("event_count"), default=-1)
    if event_count != len(events):
        errors.append("event_count mismatch")

    for idx, event in enumerate(events):
        event_errors = trace_event_errors(event if isinstance(event, dict) else {})
        if event_errors:
            errors.append(f"events[{idx}] invalid: {', '.join(event_errors)}")

    _append_replay_output_errors(payload.get("replay_output"), errors)
    return errors


def _append_checkpoint_required_field_errors(payload: dict[str, Any], errors: list[str]) -> None:
    if str(payload.get("schema_version") or "").strip() != CHECKPOINT_RECORD_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if not str(payload.get("trace_id") or "").strip():
        errors.append("trace_id missing")
    if not str(payload.get("stage") or "").strip():
        errors.append("stage missing")
    if not str(payload.get("status") or "").strip():
        errors.append("status missing")
    if not str(payload.get("occurred_at") or "").strip():
        errors.append("occurred_at missing")


def _append_checkpoint_execution_mode_errors(payload: dict[str, Any], errors: list[str]) -> None:
    execution_mode = str(payload.get("execution_mode") or "").strip().lower()
    if execution_mode and execution_mode not in {"live", "recovery", "replay"}:
        errors.append("execution_mode invalid")


def _append_checkpoint_execution_state_errors(payload: dict[str, Any], errors: list[str]) -> None:
    execution_state = payload.get("execution_state")
    if execution_state is not None and not isinstance(execution_state, dict):
        errors.append("execution_state must be a dict")
    if not isinstance(execution_state, dict):
        return
    if not isinstance(execution_state.get("state"), dict):
        errors.append("execution_state.state must be a dict")
    if not isinstance(execution_state.get("metadata"), dict):
        errors.append("execution_state.metadata must be a dict")
    if not isinstance(execution_state.get("phase_history"), list):
        errors.append("execution_state.phase_history must be a list")
    if not isinstance(execution_state.get("stage_traces"), list):
        errors.append("execution_state.stage_traces must be a list")


def _append_checkpoint_continuity_errors(payload: dict[str, Any], errors: list[str]) -> None:
    continuity = payload.get("continuity")
    if continuity is not None and not isinstance(continuity, dict):
        errors.append("continuity must be a dict")


def checkpoint_record_errors(record: dict[str, Any] | None) -> list[str]:
    payload = dict(record or {})
    errors: list[str] = []
    _append_checkpoint_required_field_errors(payload, errors)
    _append_checkpoint_execution_mode_errors(payload, errors)
    _append_checkpoint_execution_state_errors(payload, errors)
    _append_checkpoint_continuity_errors(payload, errors)
    return errors


def assert_valid_trace_event(record: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_trace_event(record)
    errors = trace_event_errors(normalized)
    if errors:
        raise PersistenceSchemaError("Invalid trace event record: " + "; ".join(errors))
    return normalized


def assert_valid_replay_record(record: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(record or {})
    errors = replay_record_errors(payload)
    if errors:
        raise PersistenceSchemaError("Invalid replay record: " + "; ".join(errors))
    return payload


def assert_valid_checkpoint_record(record: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_checkpoint_record(record)
    errors = checkpoint_record_errors(normalized)
    if errors:
        raise PersistenceSchemaError("Invalid checkpoint record: " + "; ".join(errors))
    return normalized


__all__ = [
    "PERSISTENCE_SCHEMA_VERSION",
    "TRACE_EVENT_SCHEMA_VERSION",
    "REPLAY_RECORD_SCHEMA_VERSION",
    "IO_CAPTURE_SCHEMA_VERSION",
    "STATE_SNAPSHOT_SCHEMA_VERSION",
    "CHECKPOINT_RECORD_SCHEMA_VERSION",
    "normalize_io_capture",
    "normalize_trace_event",
    "normalize_state_snapshot",
    "normalize_checkpoint_record",
    "normalize_replay_record",
    "PersistenceSchemaError",
    "trace_event_errors",
    "replay_record_errors",
    "checkpoint_record_errors",
    "assert_valid_trace_event",
    "assert_valid_checkpoint_record",
    "assert_valid_replay_record",
]
