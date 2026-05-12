from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _stable_sha256(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonical_state_hash(state: Mapping[str, Any] | dict[str, Any]) -> str:
    return _stable_sha256(dict(state or {}))


def build_state_snapshot_entry(
    *,
    session_id: str,
    trace_token: str = "",
    trace_id: str = "",
    version: int,
    prev_snapshot_hash: str,
    state_hash: str,
    reason: str,
) -> dict[str, Any]:
    resolved_trace_token = str(trace_token or trace_id or "")
    entry: dict[str, Any] = {
        "session_id": str(session_id or ""),
        "trace_id": resolved_trace_token,
        "version": int(version),
        "prev_snapshot_hash": str(prev_snapshot_hash or ""),
        "state_hash": str(state_hash or ""),
        "reason": str(reason or ""),
    }
    entry["snapshot_hash"] = _stable_sha256(entry)
    return entry
