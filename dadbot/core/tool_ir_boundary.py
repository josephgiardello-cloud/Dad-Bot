"""L3-P1 — Formal Tool IR Closure.

Makes Tool IR a fully closed typed system with:

1. ToolRequest schema validation at the boundary (type-safe entry point)
2. ToolResult strict typing enforcement
3. ToolEvent → ToolResult bijection guarantee (every executed event maps to
   exactly one result; no orphan events remain in the log)

These are enforceable invariants, not just documentation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.tool_ir import (
    ToolEventLog,
    ToolEventType,
    ToolRequest,
    ToolResult,
    deterministic_tool_id,
)

# ---------------------------------------------------------------------------
# Allowed values (single source of truth at the IR boundary)
# ---------------------------------------------------------------------------

_ALLOWED_TOOLS: frozenset[str] = frozenset({"memory_lookup"})
_ALLOWED_INTENTS: frozenset[str] = frozenset({"goal_lookup", "session_memory_fetch"})
_ALLOWED_STATUSES: frozenset[str] = frozenset({"ok", "error", "cached", "skipped"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolSchemaError(ValueError):
    """Raised when a Tool IR object fails schema validation at the boundary."""

    def __init__(self, field: str, reason: str, value: Any = None) -> None:
        self.field = field
        self.reason = reason
        self.value = value
        super().__init__(
            f"ToolSchemaError: field={field!r} reason={reason!r} value={value!r}",
        )


class ToolIRBijectionError(ValueError):
    """Raised when the ToolEvent → ToolResult bijection invariant is violated."""

    def __init__(self, message: str, orphan_ids: list[str] | None = None) -> None:
        self.orphan_ids = orphan_ids or []
        super().__init__(f"ToolIRBijectionError: {message}")


# ---------------------------------------------------------------------------
# L3-P1a — ToolRequest schema validation
# ---------------------------------------------------------------------------


def validate_tool_request(raw: dict[str, Any]) -> ToolRequest:
    """Validate and construct a ToolRequest from a raw dict at the IR boundary.

    Raises ``ToolSchemaError`` for any validation failure.  This is the single
    enforcement point — no raw dicts should escape this function unvalidated.
    """
    if not isinstance(raw, dict):
        raise ToolSchemaError("raw", "must be a dict", raw)

    tool_name = str(raw.get("tool_name") or "").strip().lower()
    if not tool_name:
        raise ToolSchemaError("tool_name", "required and non-empty")
    if tool_name not in _ALLOWED_TOOLS:
        raise ToolSchemaError(
            "tool_name",
            f"not in allowed set {sorted(_ALLOWED_TOOLS)}",
            tool_name,
        )

    args = raw.get("args")
    if not isinstance(args, dict):
        raise ToolSchemaError("args", "must be a dict", args)

    intent = str(raw.get("intent") or "").strip().lower()
    if not intent:
        raise ToolSchemaError("intent", "required and non-empty")
    if intent not in _ALLOWED_INTENTS:
        raise ToolSchemaError(
            "intent",
            f"not in allowed set {sorted(_ALLOWED_INTENTS)}",
            intent,
        )

    expected_output = str(raw.get("expected_output") or "").strip()
    if not expected_output:
        raise ToolSchemaError("expected_output", "required and non-empty")

    raw_priority = raw.get("priority")
    if raw_priority is None:
        priority = 100
    else:
        try:
            priority = int(raw_priority)
        except (TypeError, ValueError):
            raise ToolSchemaError("priority", "must be an integer", raw_priority)
        if priority < 0:
            raise ToolSchemaError("priority", "must be non-negative", priority)

    return ToolRequest(
        tool_name=tool_name,
        args=dict(args),
        intent=intent,
        expected_output=expected_output,
        priority=priority,
    )


def validate_tool_requests_batch(
    raws: list[Any],
) -> tuple[list[ToolRequest], list[dict[str, Any]]]:
    """Validate a batch of raw dicts.

    Returns (valid_requests, rejections) where each rejection has keys
    {index, reason}.  Never raises — validation errors become rejections.
    """
    valid: list[ToolRequest] = []
    rejections: list[dict[str, Any]] = []
    for idx, raw in enumerate(raws or []):
        try:
            req = validate_tool_request(raw)
            valid.append(req)
        except ToolSchemaError as exc:
            rejections.append({"index": idx, "field": exc.field, "reason": exc.reason})
    return valid, rejections


# ---------------------------------------------------------------------------
# L3-P1b — ToolResult strict typing enforcement
# ---------------------------------------------------------------------------


def validate_tool_result(raw: dict[str, Any]) -> ToolResult:
    """Validate and construct a ToolResult from a raw dict.

    Raises ``ToolSchemaError`` for any validation failure.
    """
    if not isinstance(raw, dict):
        raise ToolSchemaError("raw", "must be a dict", raw)

    tool_name = str(raw.get("tool_name") or "").strip().lower()
    if not tool_name:
        raise ToolSchemaError("tool_name", "required and non-empty")

    status = str(raw.get("status") or "ok").strip().lower()
    if status not in _ALLOWED_STATUSES:
        raise ToolSchemaError(
            "status",
            f"must be one of {sorted(_ALLOWED_STATUSES)}",
            status,
        )

    output = raw.get("output")

    det_id = str(raw.get("deterministic_id") or "").strip()
    if not det_id:
        # Derive it from tool_name + any args present.
        args = raw.get("args")
        det_id = deterministic_tool_id(
            tool_name,
            dict(args) if isinstance(args, dict) else {},
        )

    return ToolResult(
        tool_name=tool_name,
        status=status,
        output=output,
        deterministic_id=det_id,
    )


# ---------------------------------------------------------------------------
# L3-P1c — ToolEvent → ToolResult bijection guarantee
# ---------------------------------------------------------------------------


def assert_bijection(
    event_log: ToolEventLog,
    results: list[dict[str, Any] | ToolResult],
) -> None:
    """Assert that every EXECUTED/FAILED event in the log maps to exactly one result.

    Bijection contract:
      - Every EXECUTED/FAILED event has a corresponding result (no orphans).
      - Every result has a corresponding EXECUTED/FAILED event (no phantoms).
      - The mapping is one-to-one (no result is mapped from multiple events).

    Raises ``ToolIRBijectionError`` on any violation.
    """
    # Collect event tool_ids for EXECUTED and FAILED events.
    executed_ids: list[str] = []
    for event in event_log.events or []:
        if event.event_type in (ToolEventType.EXECUTED, ToolEventType.FAILED):
            executed_ids.append(event.tool_id)

    # Collect result tool_ids.
    result_ids: list[str] = []
    for r in results or []:
        if isinstance(r, ToolResult):
            result_ids.append(r.deterministic_id)
        else:
            result_ids.append(str(r.get("deterministic_id") or r.get("tool_id") or ""))

    # Check: every executed event has exactly one result.
    event_id_set = set(executed_ids)
    result_id_set = set(result_ids)

    orphan_events = event_id_set - result_id_set
    phantom_results = result_id_set - event_id_set

    if orphan_events:
        raise ToolIRBijectionError(
            f"orphan events (no corresponding result): {sorted(orphan_events)}",
            orphan_ids=sorted(orphan_events),
        )
    if phantom_results:
        raise ToolIRBijectionError(
            f"phantom results (no corresponding event): {sorted(phantom_results)}",
        )
    if len(executed_ids) != len(result_ids):
        raise ToolIRBijectionError(
            f"cardinality mismatch: {len(executed_ids)} executed events vs {len(result_ids)} results",
        )


def assert_no_orphan_events(
    event_log: ToolEventLog,
    results: list[dict[str, Any] | ToolResult],
) -> None:
    """Assert that no EXECUTED/FAILED event in the log is without a result.

    Subset of the full bijection check — only validates orphan direction.
    """
    executed_ids = {
        e.tool_id for e in (event_log.events or []) if e.event_type in (ToolEventType.EXECUTED, ToolEventType.FAILED)
    }
    result_ids = set()
    for r in results or []:
        if isinstance(r, ToolResult):
            result_ids.add(r.deterministic_id)
        else:
            result_ids.add(str(r.get("deterministic_id") or r.get("tool_id") or ""))

    orphans = executed_ids - result_ids
    if orphans:
        raise ToolIRBijectionError(
            f"orphan events detected: {sorted(orphans)}",
            orphan_ids=sorted(orphans),
        )


def build_bijection_proof(
    event_log: ToolEventLog,
    results: list[dict[str, Any] | ToolResult],
) -> dict[str, Any]:
    """Non-raising form: return a bijection proof dict rather than raising.

    Keys: ok (bool), orphan_events, phantom_results, cardinality_match,
    event_count, result_count.
    """
    executed_ids: list[str] = [
        e.tool_id for e in (event_log.events or []) if e.event_type in (ToolEventType.EXECUTED, ToolEventType.FAILED)
    ]
    result_ids: list[str] = []
    for r in results or []:
        if isinstance(r, ToolResult):
            result_ids.append(r.deterministic_id)
        else:
            result_ids.append(str(r.get("deterministic_id") or r.get("tool_id") or ""))

    event_id_set = set(executed_ids)
    result_id_set = set(result_ids)

    orphan_events = sorted(event_id_set - result_id_set)
    phantom_results = sorted(result_id_set - event_id_set)
    cardinality_match = len(executed_ids) == len(result_ids)
    ok = (not orphan_events) and (not phantom_results) and cardinality_match

    return {
        "ok": ok,
        "orphan_events": orphan_events,
        "phantom_results": phantom_results,
        "cardinality_match": cardinality_match,
        "event_count": len(executed_ids),
        "result_count": len(result_ids),
        "bijection_hash": hashlib.sha256(
            json.dumps(
                {"executed": sorted(executed_ids), "results": sorted(result_ids)},
                sort_keys=True,
            ).encode(),
        ).hexdigest()[:16],
    }


__all__ = [
    "ToolIRBijectionError",
    "ToolSchemaError",
    "assert_bijection",
    "assert_no_orphan_events",
    "build_bijection_proof",
    "validate_tool_request",
    "validate_tool_requests_batch",
    "validate_tool_result",
]
