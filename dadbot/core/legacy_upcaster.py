from __future__ import annotations

from typing import Any

from dadbot.core.canonical_event import canonicalize_event_payload
from dadbot.core.execution_schema import migrate_trace_contract


def upcast_trace_contract(contract: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(contract or {})
    payload.setdefault("version", str(payload.get("version") or "1.0"))
    return migrate_trace_contract(payload)


def upcast_event_record(event: dict[str, Any] | None) -> dict[str, Any]:
    item = dict(event or {})

    if "type" not in item and "event_type" in item:
        item["type"] = str(item.pop("event_type") or "")
    if "sequence" not in item and "sequence_id" in item:
        try:
            item["sequence"] = int(item.pop("sequence_id") or 0)
        except (TypeError, ValueError):
            item["sequence"] = 0

    item["type"] = str(item.get("type") or "").strip() or "LEGACY_EVENT"
    item["sequence"] = int(item.get("sequence") or 0)
    item["kernel_step_id"] = str(item.get("kernel_step_id") or item.get("stage") or "legacy.unknown")
    item["session_id"] = str(item.get("session_id") or "legacy")
    item["session_index"] = int(item.get("session_index") or item["sequence"] or 0)
    item["event_id"] = str(item.get("event_id") or "")
    item["parent_event_id"] = str(item.get("parent_event_id") or "")
    item["payload"] = canonicalize_event_payload(item.get("payload") or {})
    return item


def upcast_event_log(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    upgraded: list[dict[str, Any]] = []
    for event in list(events or []):
        if isinstance(event, dict):
            upgraded.append(upcast_event_record(event))
    return upgraded
