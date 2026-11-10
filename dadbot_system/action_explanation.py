from __future__ import annotations

from typing import Any

_MAX_ACTIONS = 50
_HASH_LENGTH = 64


def explain_last_actions(records: Any, *, limit: int = 10) -> dict[str, Any]:
    """Build a bounded explanation from persisted, checksum-linked tool records."""
    safe_limit = max(1, min(int(limit or 10), _MAX_ACTIONS))
    if not isinstance(records, list):
        records = []

    actions: list[dict[str, str]] = []
    for record in records[-safe_limit:]:
        if not isinstance(record, dict):
            continue
        tool_name = str(record.get("tool_name") or "").strip()[:128]
        status = str(record.get("status") or "").strip().lower()[:32]
        decision = str(record.get("authorization_decision") or "unknown").strip().lower()[:32]
        authority_mode = str(record.get("authority_mode") or "unknown").strip().lower()[:32]
        input_hash = str(record.get("input_hash") or "")
        output_hash = str(record.get("output_hash") or "")
        if not tool_name:
            continue
        input_fingerprint = input_hash if _is_sha256(input_hash) else ""
        output_fingerprint = output_hash if _is_sha256(output_hash) else ""
        if status in {"succeeded", "completed"}:
            outcome = "completed"
        elif status == "cached":
            outcome = "reused a previously recorded result"
        elif status == "failed":
            outcome = "failed"
        else:
            outcome = f"reported status {status or 'unknown'}"

        if decision == "verified":
            authorization = "after verified delegated authority"
        elif decision == "trusted_local":
            authorization = "in trusted-local mode"
        else:
            authorization = "with authorization details unavailable"

        action: dict[str, str] = {
            "event_id": str(record.get("event_id") or "")[:64],
            "event_checksum": str(record.get("event_checksum") or "")[:80],
            "previous_checksum": str(record.get("previous_checksum") or "")[:80],
            "tool_name": tool_name,
            "status": status or "unknown",
            "authorization_decision": decision,
            "authority_mode": authority_mode,
            "trace_id": str(record.get("trace_id") or "")[:128],
            "timestamp": str(record.get("timestamp") or "")[:64],
            "explanation": f"{tool_name} {outcome} {authorization}.",
        }
        if input_fingerprint:
            action["input_fingerprint"] = input_fingerprint
        if output_fingerprint:
            action["output_fingerprint"] = output_fingerprint
        actions.append(action)

    return {
        "scope": "latest_turn",
        "actions": actions,
        "summary": (
            "No tool actions were recorded for the latest turn."
            if not actions
            else " ".join(action["explanation"] for action in actions)
        ),
    }


def _is_sha256(value: str) -> bool:
    return len(value) == _HASH_LENGTH and all(char in "0123456789abcdef" for char in value)
