from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SCHEMA_VERSION = 1
_MAX_GOAL_UPDATES = 16
_MAX_GOALS = 64
_MAX_KEY_LENGTH = 80
_MAX_VALUE_LENGTH = 300
_MAX_TOOL_OUTCOMES = 16
_MAX_TOOL_NAME_LENGTH = 128
_MAX_TOOL_STATUS_LENGTH = 32
_MAX_AUTHORITY_VALUE_LENGTH = 32
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_EVENT_CHECKSUM_RE = re.compile(r"^evtchk-[0-9a-f]{64}$")

_TOOL_OUTCOME_FIELDS = (
    "event_id",
    "event_checksum",
    "previous_checksum",
    "trace_id",
    "tool_name",
    "status",
    "authorization_decision",
    "authority_mode",
    "input_hash",
    "output_hash",
)


@dataclass(frozen=True, slots=True)
class AgentLoopCommitIdentity:
    tenant_id: str
    session_id: str
    trace_id: str
    job_id: str


def _normalize_goal_map(value: Any, *, limit: int) -> dict[str, str]:
    if type(value) is not dict or len(value) > limit:
        raise ValueError("Invalid agent-loop goal state")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or type(item) is not str:
            raise ValueError("Agent-loop goal keys and values must be strings")
        clean_key = key.strip()
        clean_value = item.strip()
        if (
            not clean_key or len(clean_key) > _MAX_KEY_LENGTH
            or not clean_value or len(clean_value) > _MAX_VALUE_LENGTH
        ):
            raise ValueError("Invalid agent-loop goal key or value")
        if clean_key in normalized:
            raise ValueError("Agent-loop goal keys must be unique after trimming")
        normalized[clean_key] = clean_value
    return {key: normalized[key] for key in sorted(normalized)}


def normalize_goal_updates(value: Any) -> dict[str, str]:
    return _normalize_goal_map(value, limit=_MAX_GOAL_UPDATES)


def normalize_goal_blackboard(value: Any) -> dict[str, str]:
    return _normalize_goal_map(value, limit=_MAX_GOALS)


def normalize_tool_outcome_evidence(
    value: Any,
    *,
    expected_trace_id: str | None = None,
) -> dict[str, Any]:
    if value is None:
        return {
            "schema_version": _SCHEMA_VERSION,
            "trace_id": str(expected_trace_id or ""),
            "outcomes": [],
            "truncated": False,
        }
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "trace_id", "outcomes", "truncated"}
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != _SCHEMA_VERSION
        or type(value.get("trace_id")) is not str
        or not value["trace_id"].strip()
        or type(value.get("outcomes")) is not list
        or len(value["outcomes"]) > _MAX_TOOL_OUTCOMES
        or type(value.get("truncated")) is not bool
    ):
        raise ValueError("Invalid hash-only tool outcome evidence")
    trace_id = value["trace_id"].strip()
    if expected_trace_id is not None and trace_id != expected_trace_id:
        raise ValueError("Tool outcome evidence belongs to a different trace")

    outcomes: list[dict[str, str]] = []
    for outcome in value["outcomes"]:
        if type(outcome) is not dict or set(outcome) != set(_TOOL_OUTCOME_FIELDS):
            raise ValueError("Invalid hash-only tool outcome")
        if any(type(outcome[field]) is not str for field in _TOOL_OUTCOME_FIELDS):
            raise ValueError("Hash-only tool outcome fields must be strings")
        normalized = {field: outcome[field].strip() for field in _TOOL_OUTCOME_FIELDS}
        if (
            _EVENT_ID_RE.fullmatch(normalized["event_id"]) is None
            or _EVENT_CHECKSUM_RE.fullmatch(normalized["event_checksum"]) is None
            or (
                normalized["previous_checksum"]
                and _EVENT_CHECKSUM_RE.fullmatch(normalized["previous_checksum"]) is None
            )
            or normalized["trace_id"] != trace_id
            or not normalized["tool_name"]
            or len(normalized["tool_name"]) > _MAX_TOOL_NAME_LENGTH
            or not normalized["status"]
            or len(normalized["status"]) > _MAX_TOOL_STATUS_LENGTH
            or not normalized["authorization_decision"]
            or len(normalized["authorization_decision"]) > _MAX_AUTHORITY_VALUE_LENGTH
            or not normalized["authority_mode"]
            or len(normalized["authority_mode"]) > _MAX_AUTHORITY_VALUE_LENGTH
            or _HASH_RE.fullmatch(normalized["input_hash"]) is None
            or _HASH_RE.fullmatch(normalized["output_hash"]) is None
        ):
            raise ValueError("Invalid hash-only tool outcome fields")
        outcomes.append(normalized)
    return {
        "schema_version": _SCHEMA_VERSION,
        "trace_id": trace_id,
        "outcomes": outcomes,
        "truncated": value["truncated"],
    }


def read_agent_loop_goal_blackboard(
    session_state: dict[str, Any],
    *,
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    progress = session_state.get("agent_loop_progress")
    if progress is None:
        return {}
    if (
        type(progress) is not dict
        or set(progress) not in (
            {"schema_version", "goal_blackboard", "last_committed"},
            {"schema_version", "goal_blackboard", "last_committed", "last_tool_outcome_evidence"},
        )
        or type(progress.get("schema_version")) is not int
        or progress["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError("Invalid persisted agent-loop progress")
    board = normalize_goal_blackboard(progress["goal_blackboard"])
    last_committed = progress["last_committed"]
    required = {
        "tenant_id", "session_id", "trace_id", "job_id", "request_id", "loop_run_id", "turn_index",
    }
    if (
        type(last_committed) is not dict
        or set(last_committed) != required
        or any(type(last_committed[name]) is not str or not last_committed[name].strip()
               for name in required - {"turn_index"})
        or type(last_committed["turn_index"]) is not int
        or last_committed["turn_index"] < 1
    ):
        raise ValueError("Invalid persisted agent-loop completion identity")
    if tenant_id is not None and last_committed["tenant_id"] != tenant_id:
        raise ValueError("Persisted agent-loop progress belongs to a different tenant")
    if session_id is not None and last_committed["session_id"] != session_id:
        raise ValueError("Persisted agent-loop progress belongs to a different session")
    if "last_tool_outcome_evidence" in progress:
        normalize_tool_outcome_evidence(
            progress["last_tool_outcome_evidence"],
            expected_trace_id=last_committed["trace_id"],
        )
    return board


def read_agent_loop_tool_outcome_evidence(
    session_state: dict[str, Any],
    *,
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    read_agent_loop_goal_blackboard(
        session_state,
        tenant_id=tenant_id,
        session_id=session_id,
    )
    progress = session_state.get("agent_loop_progress")
    if progress is None:
        return normalize_tool_outcome_evidence(None)
    return normalize_tool_outcome_evidence(
        progress.get("last_tool_outcome_evidence"),
        expected_trace_id=str(progress["last_committed"]["trace_id"]),
    )


def reconcile_committed_agent_loop_progress(
    session_state: dict[str, Any],
    *,
    completed_entries: list[dict[str, Any]],
    tenant_id: str,
    session_id: str,
) -> bool:
    if type(completed_entries) is not list or any(
        type(entry) is not dict
        or entry.get("tenant_id") != tenant_id
        or entry.get("session_id") != session_id
        for entry in completed_entries
    ):
        raise ValueError("Invalid or cross-scope canonical agent-loop completion history")
    progress = session_state.get("agent_loop_progress")
    if progress is None:
        board: dict[str, str] = {}
        start_index = 0
    else:
        board = read_agent_loop_goal_blackboard(
            session_state,
            tenant_id=tenant_id,
            session_id=session_id,
        )
        last_request_id = str(progress["last_committed"]["request_id"])
        marker_index = next(
            (
                index for index, entry in enumerate(completed_entries)
                if entry.get("request_id") == last_request_id
            ),
            None,
        )
        if marker_index is None:
            raise ValueError("Persisted agent-loop commit is missing from canonical completion history")
        last_committed = progress["last_committed"]
        marker = completed_entries[marker_index]
        marker_metadata = marker.get("metadata")
        marker_loop = marker_metadata.get("agent_loop") if type(marker_metadata) is dict else None
        if (
            marker.get("trace_id") != last_committed["trace_id"]
            or marker.get("job_id") != last_committed["job_id"]
            or type(marker_loop) is not dict
            or marker_loop.get("loop_run_id") != last_committed["loop_run_id"]
            or marker_loop.get("turn_index") != last_committed["turn_index"]
        ):
            raise ValueError("Persisted agent-loop identity does not match canonical completion history")
        start_index = marker_index + 1

    pending = completed_entries[start_index:]
    for entry in pending:
        if any(
            type(entry.get(name)) is not str or not entry[name].strip()
            for name in ("request_id", "trace_id", "job_id")
        ):
            raise ValueError("Invalid canonical agent-loop completion identity")
        metadata = entry.get("metadata")
        loop = metadata.get("agent_loop") if type(metadata) is dict else None
        if type(loop) is not dict:
            raise ValueError("Canonical agent-loop completion metadata is malformed")
        snapshot = normalize_goal_blackboard(loop.get("goal_blackboard"))
        if snapshot != board:
            raise ValueError("Cannot safely reconcile agent-loop progress from a stale goal snapshot")
        committed_metadata = dict(metadata)
        committed_metadata["request_id"] = entry["request_id"]
        apply_committed_agent_loop_progress(
            session_state,
            metadata=committed_metadata,
            identity=AgentLoopCommitIdentity(
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=entry["trace_id"],
                job_id=entry["job_id"],
            ),
        )
        board = read_agent_loop_goal_blackboard(session_state)
    return bool(pending)


def apply_committed_agent_loop_progress(
    session_state: dict[str, Any],
    *,
    metadata: dict[str, Any],
    identity: AgentLoopCommitIdentity,
) -> bool:
    loop = metadata.get("agent_loop")
    if not isinstance(loop, dict) or "schema_version" not in loop:
        return False
    if type(loop.get("schema_version")) is not int or loop["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Unsupported agent-loop progress metadata")
    run_id = loop.get("loop_run_id")
    turn_index = loop.get("turn_index")
    request_id = metadata.get("request_id")
    if (
        type(run_id) is not str or not run_id.strip()
        or type(turn_index) is not int or turn_index < 1
        or type(request_id) is not str or not request_id.strip()
        or type(identity.job_id) is not str or not identity.job_id.strip()
        or type(identity.trace_id) is not str or not identity.trace_id.strip()
        or type(identity.tenant_id) is not str or not identity.tenant_id.strip()
        or type(identity.session_id) is not str or not identity.session_id.strip()
    ):
        raise ValueError("Invalid committed agent-loop identity")
    updates = normalize_goal_updates(loop.get("goal_updates"))
    prior = session_state.get("agent_loop_progress")
    board = read_agent_loop_goal_blackboard({"agent_loop_progress": prior}) if prior is not None else {}
    board.update(updates)
    if len(board) > _MAX_GOALS:
        raise ValueError("Agent-loop goal blackboard cannot contain more than 64 entries")
    tool_outcome_evidence = normalize_tool_outcome_evidence(
        metadata.get("tool_outcome_evidence"),
        expected_trace_id=identity.trace_id,
    )
    session_state["agent_loop_progress"] = {
        "schema_version": _SCHEMA_VERSION,
        "goal_blackboard": normalize_goal_blackboard(board),
        "last_committed": {
            "tenant_id": identity.tenant_id,
            "session_id": identity.session_id,
            "trace_id": identity.trace_id,
            "job_id": identity.job_id,
            "request_id": request_id,
            "loop_run_id": run_id,
            "turn_index": turn_index,
        },
        "last_tool_outcome_evidence": tool_outcome_evidence,
    }
    return True


__all__ = [
    "AgentLoopCommitIdentity",
    "apply_committed_agent_loop_progress",
    "normalize_goal_blackboard",
    "normalize_goal_updates",
    "normalize_tool_outcome_evidence",
    "read_agent_loop_goal_blackboard",
    "read_agent_loop_tool_outcome_evidence",
    "reconcile_committed_agent_loop_progress",
]
