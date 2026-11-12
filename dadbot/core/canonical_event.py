from __future__ import annotations

import hashlib
import json
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
        "tenant_id",
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
        "timestamp",
        "submitted_at",
        "submitted_timestamp",
        "submitted_monotonic",
        "acquired_at",
        "expires_at",
        "lease_expiry",
        "last_checked_at",
        "retry_not_before_monotonic",
        "occurred_at",
        "created_at",
        "updated_at",
        "duration_ms",
        "elapsed_ms",
        "execution_timestamp_ms",
        "checkpoint",
        "checkpoint_history_delta",
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
        "PROJECTION_STATUS_UPDATED",
    },
)


def _strip_non_canonical(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if depth == 0 and key in NON_CANONICAL_PAYLOAD_FIELDS:
                continue
            cleaned[key] = _strip_non_canonical(item, depth=depth + 1)
        return cleaned
    if isinstance(value, list):
        return [_strip_non_canonical(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return [_strip_non_canonical(item, depth=depth + 1) for item in value]
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


def _canonical_trace_payload(event_type: str, payload: Any) -> dict[str, Any]:
    if str(event_type or "") in NON_REPLAY_EVENT_TYPES:
        return {}
    return canonicalize_event_payload(payload)


def _deterministic_event_id(payload: dict[str, Any]) -> str:
    seed = {
        "type": str(payload.get("type") or ""),
        "session_id": str(payload.get("session_id") or ""),
        "session_index": int(payload.get("session_index") or 0),
        "kernel_step_id": str(payload.get("kernel_step_id") or ""),
        "payload": _canonical_trace_payload(
            str(payload.get("type") or ""),
            payload.get("payload"),
        ),
    }
    if "tenant_id" in payload:
        seed["tenant_id"] = str(payload.get("tenant_id") or "")
    return (
        "evt-"
        + hashlib.sha256(
            json.dumps(seed, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:24]
    )


def _canonical_event_for_hash(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("type") or "")
    canonical = {
        "type": event_type,
        "session_id": str(event.get("session_id") or ""),
        "session_index": int(event.get("session_index") or 0),
        "kernel_step_id": str(event.get("kernel_step_id") or ""),
        "parent_event_id": str(event.get("parent_event_id") or ""),
        "event_id": str(event.get("event_id") or ""),
        "payload": _canonical_trace_payload(event_type, event.get("payload")),
    }
    if "tenant_id" in event:
        canonical["tenant_id"] = str(event.get("tenant_id") or "")
    return canonical


def _event_sha256(event: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_canonical_event_for_hash(event), sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _chain_hash(prev_chain_hash: str, event_sha256: str) -> str:
    return hashlib.sha256(
        f"{prev_chain_hash or ''!s}:{event_sha256 or ''!s}".encode(),
    ).hexdigest()


def _verify_persisted_chain(
    events: list[dict[str, Any]], *, previous_chain_hash: str = "", chain_started: bool = False,
) -> list[str]:
    violations: list[str] = []
    prev_chain = previous_chain_hash
    for idx, event in enumerate(list(events or [])):
        has_chain_fields = any(
            key in event for key in ("event_sha256", "prev_chain_hash", "chain_hash")
        )
        canonical_event = dict(event)
        if not has_chain_fields and not str(canonical_event.get("event_id") or ""):
            canonical_event["event_id"] = _deterministic_event_id(canonical_event)
        expected_event_sha = _event_sha256(canonical_event)
        expected_prev = prev_chain
        expected_chain = _chain_hash(expected_prev, expected_event_sha)

        actual_event_sha = str(event.get("event_sha256") or "")
        actual_prev = str(event.get("prev_chain_hash") or "")
        actual_chain = str(event.get("chain_hash") or "")

        if has_chain_fields:
            chain_started = True
        if chain_started and (
            not actual_event_sha or not actual_chain or "prev_chain_hash" not in event
        ):
            violations.append(f"event[{idx}].chain_fields_missing")
        if actual_event_sha and actual_event_sha != expected_event_sha:
            violations.append(f"event[{idx}].event_sha256_mismatch")
        if has_chain_fields and actual_prev != expected_prev:
            violations.append(f"event[{idx}].prev_chain_hash_mismatch")
        if actual_chain and actual_chain != expected_chain:
            violations.append(f"event[{idx}].chain_hash_mismatch")

        prev_chain = expected_chain
    return violations


def validate_trace(
    trace: list[dict[str, Any]],
    *,
    enforce_full_policy: bool = False,
) -> None:
    """Assert that no event payload in *trace* carries forbidden non-canonical fields.

    Raises:
        AssertionError: On the first forbidden field found, with a message that
            identifies the field name, event index, type, and kernel_step_id.

    Usage: hook this into stress tests and turn-graph invariant assertions to
    catch regressions before they reach production.

    """
    disallowed_fields = NON_CANONICAL_PAYLOAD_FIELDS if enforce_full_policy else FORBIDDEN_TRACE_FIELDS
    for i, event in enumerate(trace or []):
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        for field in disallowed_fields:
            if field in payload:
                raise AssertionError(
                    f"Non-canonical field {field!r} found in event[{i}] payload "
                    f"(type={event.get('type')!r}, "
                    f"kernel_step_id={event.get('kernel_step_id')!r})",
                )
