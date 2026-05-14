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


def build_causal_fork_entry(
    *,
    session_id: str,
    source_snapshot_hash: str,
    fork_snapshot_hash: str,
    branch_id: str,
    reason: str = "",
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "session_id": str(session_id or ""),
        "branch_id": str(branch_id or ""),
        "source_snapshot_hash": str(source_snapshot_hash or ""),
        "fork_snapshot_hash": str(fork_snapshot_hash or ""),
        "reason": str(reason or ""),
    }
    entry["fork_hash"] = _stable_sha256(entry)
    return entry


def diff_state_snapshots(
    before: Mapping[str, Any] | dict[str, Any],
    after: Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    before_dict = dict(before or {})
    after_dict = dict(after or {})
    before_keys = set(before_dict.keys())
    after_keys = set(after_dict.keys())

    added_keys = sorted(after_keys - before_keys)
    removed_keys = sorted(before_keys - after_keys)
    changed_keys = sorted(
        key
        for key in (before_keys & after_keys)
        if before_dict.get(key) != after_dict.get(key)
    )

    return {
        "before_hash": canonical_state_hash(before_dict),
        "after_hash": canonical_state_hash(after_dict),
        "added_keys": added_keys,
        "removed_keys": removed_keys,
        "changed_keys": changed_keys,
        "changed_count": len(changed_keys),
    }
