from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Canonical event boundary — system-wide policy
# ---------------------------------------------------------------------------

# Fields that uniquely identify an event in the causal chain and determine
# replay equivalence.  These are the ONLY fields projected into canonical
# hashes (replay_hash, trace_hash).
CANONICAL_EVENT_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "session_id",
        "session_index",
        "event_id",
        "parent_event_id",
        "kernel_step_id",
    },
)

# Operational / wall-clock payload fields that MUST NOT appear in canonical
# hashes.  These values are machine-local timestamps that vary between runs
# but carry no semantic meaning for replay equivalence.
#
# Enforcement:
#   - canonicalize_event_payload() strips these from any payload dict.
#   - validate_trace() raises AssertionError if any survive into a trace.
NON_CANONICAL_PAYLOAD_FIELDS: frozenset[str] = frozenset(
    {
        "submitted_at",
        "acquired_at",
        "expires_at",
        "last_checked_at",
        "occurred_at",
        "created_at",
        "updated_at",
        "duration_ms",
        "elapsed_ms",
        "checkpoint",
        "identity",
        "execution_trace_contract",
        "leaf_hash",
        "merkle_root",
        "inclusion_proof",
        "trace_id",
        "correlation_id",
        "job_id",
        "request_id",
        "lease_id",
    },
)

FORBIDDEN_TRACE_FIELDS: frozenset[str] = frozenset(
    {
        "submitted_at",
        "acquired_at",
        "expires_at",
        "last_checked_at",
    },
)

# Event types that are persisted for observability/reporting only and must not
# participate in replay equivalence hashing.
NON_REPLAY_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "CAPABILITY_AUDIT_EVENT",
        "PolicyTraceEvent",
    },
)


def _strip_non_canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_non_canonical(item) for key, item in value.items() if key not in NON_CANONICAL_PAYLOAD_FIELDS
        }
    if isinstance(value, list):
        return [_strip_non_canonical(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_non_canonical(item) for item in value]
    return value


def canonicalize_event_payload(payload: Any) -> dict[str, Any]:
    """Return a canonical copy of *payload* with all non-canonical fields removed.

    This is the single enforcement point for replay-determinism at the payload
    level.  All hash-producing code paths (``replay_hash``, ``trace_hash``)
    MUST route through this function rather than performing their own ad-hoc
    field exclusion.

    Args:
        payload: The raw event payload dict (or any object — non-dicts return
            an empty dict so callers never need to guard against None/wrong type).

    Returns:
        A new ``dict`` containing only the canonical keys.

    """
    if not isinstance(payload, dict):
        return {}
    return _strip_non_canonical(payload)


def validate_trace(trace: list[dict[str, Any]]) -> None:
    """Assert that no event payload in *trace* carries forbidden non-canonical fields.

    Raises:
        AssertionError: On the first forbidden field found, with a message that
            identifies the field name, event index, type, and kernel_step_id.

    Usage: hook this into stress tests and turn-graph invariant assertions to
    catch regressions before they reach production.

    """
    for i, event in enumerate(trace or []):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        for field in FORBIDDEN_TRACE_FIELDS:
            if field in payload:
                raise AssertionError(
                    f"Non-canonical field {field!r} found in event[{i}] payload "
                    f"(type={event.get('type')!r}, "
                    f"kernel_step_id={event.get('kernel_step_id')!r})",
                )
