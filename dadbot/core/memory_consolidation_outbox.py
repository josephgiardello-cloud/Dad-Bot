"""Ledger-native outbox records for trusted-local memory consolidation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

MEMORY_CONSOLIDATION_POLICY_VERSION = "tenant-memory-consolidation-v1"
MEMORY_CONSOLIDATION_COMPLETED = "TENANT_MEMORY_CONSOLIDATION_COMPLETED"
MEMORY_CONSOLIDATION_TURN_OUTCOME = "TENANT_TURN_OUTCOME_RECORDED"


def build_memory_consolidation_work(
    *,
    tenant_id: str,
    session_id: str,
    trace_id: str,
    source_revision: int,
    source_operation_id: str,
) -> dict[str, Any]:
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in (tenant_id, session_id, trace_id, source_operation_id)
    ):
        raise ValueError("Memory consolidation work identity must be normalized")
    if type(source_revision) is not int or source_revision < 1:
        raise ValueError("Memory consolidation source revision must be positive")
    identity = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "source_revision": source_revision,
        "source_operation_id": source_operation_id,
        "policy_version": MEMORY_CONSOLIDATION_POLICY_VERSION,
    }
    work_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    return {**identity, "work_id": f"memory-consolidation:{work_id}"}


def _validated_completion(event: dict[str, Any], payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    work_id = payload.get("work_id")
    source_revision = payload.get("source_revision")
    source_operation_id = payload.get("source_operation_id")
    policy_version = payload.get("policy_version")
    session_id = payload.get("session_id")
    trace_id = payload.get("trace_id")
    if (
        event.get("committed") is not True
        or not isinstance(work_id, str)
        or not work_id
        or payload.get("status") not in {"succeeded", "skipped"}
        or type(source_revision) is not int
        or source_revision < 1
        or not isinstance(source_operation_id, str)
        or not source_operation_id
        or policy_version != MEMORY_CONSOLIDATION_POLICY_VERSION
        or not isinstance(session_id, str)
        or not isinstance(trace_id, str)
        or event.get("session_id") != session_id
        or event.get("trace_id") != trace_id
    ):
        raise ValueError("Invalid memory consolidation completion record")
    return work_id, payload


def _validated_turn_outcome(
    event: dict[str, Any],
    payload: dict[str, Any],
    *,
    tenant_id: str,
) -> tuple[tuple[str, str, str], dict[str, str]]:
    session_id = payload.get("session_id")
    trace_id = payload.get("trace_id")
    status = payload.get("status")
    commit_id = payload.get("commit_id")
    if (
        event.get("committed") is not True
        or event.get("tenant_id") != tenant_id
        or payload.get("tenant_id") != tenant_id
        or not isinstance(session_id, str)
        or not session_id
        or not isinstance(trace_id, str)
        or not trace_id
        or event.get("session_id") != session_id
        or event.get("trace_id") != trace_id
        or not isinstance(status, str)
        or status not in {"succeeded", "failed"}
        or not isinstance(commit_id, str)
    ):
        raise ValueError("Invalid memory consolidation turn outcome")
    return (tenant_id, session_id, trace_id), {
        "status": status,
        "commit_id": commit_id,
    }


def _collect_turn_outcome(
    outcomes: dict[tuple[str, str, str], dict[str, str]],
    event: dict[str, Any],
    payload: dict[str, Any],
    *,
    tenant_id: str,
) -> None:
    if payload.get("tenant_id") != tenant_id and event.get("tenant_id") != tenant_id:
        return
    identity, outcome = _validated_turn_outcome(event, payload, tenant_id=tenant_id)
    previous_outcome = outcomes.setdefault(identity, outcome)
    if previous_outcome != outcome:
        raise ValueError("Memory consolidation turn has conflicting outcomes")


def _collect_source_work(
    sources: dict[str, dict[str, Any]],
    event: dict[str, Any],
    payload: dict[str, Any],
    *,
    tenant_id: str,
) -> None:
    work = _source_work(event, payload, tenant_id=tenant_id)
    if work is None:
        return
    previous = sources.setdefault(work["work_id"], work)
    if any(
        previous[field] != work[field]
        for field in (
            "source_revision",
            "source_operation_id",
            "session_id",
            "trace_id",
            "policy_version",
        )
    ):
        raise ValueError("Memory consolidation work ID has conflicting sources")


def _collect_outbox_event(
    event: dict[str, Any],
    *,
    tenant_id: str,
    completed: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    outcomes: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return
    event_type = event.get("type")
    if event_type == MEMORY_CONSOLIDATION_TURN_OUTCOME:
        _collect_turn_outcome(outcomes, event, payload, tenant_id=tenant_id)
        return
    if payload.get("tenant_id") != tenant_id:
        return
    if event_type == MEMORY_CONSOLIDATION_COMPLETED:
        work_id, completion = _validated_completion(event, payload)
        previous_completion = completed.setdefault(work_id, completion)
        if previous_completion != completion:
            raise ValueError("Memory consolidation work has conflicting completion records")
        return
    if event_type == "TENANT_MEMORY_COMMITTED":
        _collect_source_work(sources, event, payload, tenant_id=tenant_id)


def _source_work(
    event: dict[str, Any],
    payload: dict[str, Any],
    *,
    tenant_id: str,
) -> dict[str, Any] | None:
    marker = payload.get("consolidation_work")
    if marker is None:
        return None
    if event.get("committed") is not True:
        raise ValueError("Memory consolidation source is not committed")
    if not isinstance(marker, dict):
        raise ValueError("Invalid memory consolidation work marker")
    if marker.get("policy_version") != MEMORY_CONSOLIDATION_POLICY_VERSION:
        raise ValueError("Unsupported memory consolidation source policy")
    try:
        work = build_memory_consolidation_work(
            tenant_id=tenant_id,
            session_id=str(event.get("session_id") or ""),
            trace_id=str(event.get("trace_id") or ""),
            source_revision=payload["revision"],
            source_operation_id=payload["operation_id"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid memory consolidation source identity") from exc
    if (
        work["work_id"] != marker.get("work_id")
        or marker.get("source_revision") != work["source_revision"]
        or marker.get("source_operation_id") != work["source_operation_id"]
        or not work["session_id"]
        or not work["trace_id"]
    ):
        raise ValueError("Memory consolidation marker does not match its committed source")
    work["source_timestamp"] = str(event.get("timestamp") or "")
    work["source_event"] = event
    return work


def _validate_completion_sources(
    completed: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    for work_id, completion in completed.items():
        source = sources.get(work_id)
        if source is None:
            raise ValueError("Memory consolidation completion has no committed source")
        if any(
            completion.get(field) != source[field]
            for field in ("source_revision", "source_operation_id", "policy_version", "session_id", "trace_id")
        ):
            raise ValueError("Memory consolidation completion does not match its source")


def pending_memory_consolidation_work(
    events: list[dict[str, Any]],
    *,
    tenant_id: str,
) -> list[dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    outcomes: dict[tuple[str, str, str], dict[str, str]] = {}
    for event in events:
        _collect_outbox_event(
            event,
            tenant_id=tenant_id,
            completed=completed,
            sources=sources,
            outcomes=outcomes,
        )

    _validate_completion_sources(completed, sources)

    for work in sources.values():
        outcome = outcomes.get((tenant_id, work["session_id"], work["trace_id"]))
        work["turn_outcome_status"] = None if outcome is None else outcome["status"]
        work["turn_outcome_commit_id"] = "" if outcome is None else outcome["commit_id"]

    return sorted(
        (work for work_id, work in sources.items() if work_id not in completed),
        key=lambda work: (work["source_revision"], work["work_id"]),
    )


def record_memory_consolidation_completion(
    ledger: Any,
    work: dict[str, Any],
    *,
    status: str,
) -> None:
    if status not in {"succeeded", "skipped"}:
        raise ValueError("Memory consolidation completion status is invalid")
    from dadbot.core.tenant_session import tenant_session_scope

    with tenant_session_scope(work["tenant_id"], work["session_id"], allow_tenant_switch=True):
        ledger.write({
            "type": MEMORY_CONSOLIDATION_COMPLETED,
            "tenant_id": work["tenant_id"],
            "session_id": work["session_id"],
            "trace_id": work["trace_id"],
            "kernel_step_id": "memory-consolidation-outbox",
            "timestamp": datetime.now(UTC).isoformat(),
            "committed": True,
            "payload": {
                "tenant_id": work["tenant_id"],
                "session_id": work["session_id"],
                "trace_id": work["trace_id"],
                "work_id": work["work_id"],
                "source_revision": work["source_revision"],
                "source_operation_id": work["source_operation_id"],
                "policy_version": work["policy_version"],
                "status": status,
            },
        })


def record_memory_consolidation_turn_outcome(
    ledger: Any,
    identity: tuple[str, str, str],
    *,
    status: str,
    commit_id: str = "",
) -> None:
    if not isinstance(identity, tuple) or len(identity) != 3:
        raise ValueError("Memory consolidation turn outcome identity must have three parts")
    tenant_id, session_id, trace_id = identity
    if not isinstance(status, str) or status not in {"succeeded", "failed"}:
        raise ValueError("Memory consolidation turn outcome status is invalid")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in (tenant_id, session_id, trace_id)
    ):
        raise ValueError("Memory consolidation turn outcome identity must be normalized")
    if not isinstance(commit_id, str):
        raise ValueError("Memory consolidation turn outcome commit ID must be a string")
    from dadbot.core.tenant_session import tenant_session_scope

    with tenant_session_scope(tenant_id, session_id, allow_tenant_switch=True):
        ledger.write({
            "type": MEMORY_CONSOLIDATION_TURN_OUTCOME,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "kernel_step_id": "memory-consolidation-turn-outcome",
            "timestamp": datetime.now(UTC).isoformat(),
            "committed": True,
            "payload": {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "trace_id": trace_id,
                "status": status,
                "commit_id": commit_id,
            },
        })
