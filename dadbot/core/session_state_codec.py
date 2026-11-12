"""Lossless append references for cumulative session histories."""

from __future__ import annotations

import hashlib
import json
import marshal
from typing import Any

from dadbot.core.ledger_backend import _copy_ledger_value


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, default=str).encode("utf-8")


def _history(state: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = state
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _native_reference_values(before: Any, after: Any) -> bool:
    # Reject custom types in C; discard the binary bytes and compare JSON below.
    try:
        marshal.dumps((before, after))
    except ValueError:
        return False
    return True


def _window_offset(before: list[Any], after: list[Any]) -> int:
    if not before or not after or before == after[:len(before)]:
        return 0
    try:
        return before.index(after[0], max(1, len(before) - len(after)))
    except ValueError:
        return 0


def _encode_field(before: Any, after: Any, path: list[str], fields: list[dict[str, Any]]) -> Any:
    if any(type(key) is not str for key in path):
        return after
    if type(before) is type(after) and type(before) in (list, dict):
        drop = 0
        retained = before
        if type(before) is list:
            drop = _window_offset(before, after)
            retained = before[drop:]
            prefix, tail = after[:len(retained)], after[len(retained):]
        else:
            prefix = {key: after[key] for key in before if key in after}
            tail = {key: value for key, value in after.items() if key not in before}
        previous = _encoded(before) if prefix == retained and _native_reference_values(retained, prefix) else b""
        matched = _encoded(retained) if previous and drop else previous
        if previous and _encoded(prefix) == matched:
            delta = {
                "path": path, "base_length": len(before),
                "base_hash": hashlib.sha256(previous).hexdigest(), "append": tail,
            }
            if drop:
                delta["drop_prefix"] = drop
            full_size = len(matched) + len(_encoded(tail)) - 2 + (2 if retained and tail else 0)
            if len(_encoded(delta)) < full_size:
                fields.append(delta)
                return _OMITTED
    if type(before) is dict and type(after) is dict:
        result = {}
        for key, value in after.items():
            encoded = _encode_field(before[key], value, [*path, key], fields) if key in before else value
            if encoded is not _OMITTED:
                result[key] = encoded
        return result
    return after


_OMITTED = object()


def encode_session_state_update(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    state = {}
    fields: list[dict[str, Any]] = []
    for key, value in patch.items():
        encoded = _encode_field(current[key], value, [key], fields) if key in current else value
        if encoded is not _OMITTED:
            state[key] = encoded
    payload: dict[str, Any] = {"state": state}
    if fields:
        payload["state_history_delta"] = {
            "version": 2 if any("drop_prefix" in field for field in fields) else 1, "fields": fields,
        }
    return payload


def _delta_fields(payload: dict[str, Any]) -> list[dict[str, Any]]:
    envelope = payload["state_history_delta"]
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"version", "fields"}
        or type(envelope["version"]) is not int
        or envelope["version"] not in (1, 2)
        or not isinstance(envelope["fields"], list)
        or not envelope["fields"]
        or (
            envelope["version"] == 1
            and any(isinstance(field, dict) and "drop_prefix" in field for field in envelope["fields"])
        )
    ):
        raise RuntimeError("Invalid session history delta envelope")
    return envelope["fields"]


def _restore_history(before: Any, field: dict[str, Any]) -> Any:
    drop = field.get("drop_prefix", 0)
    if (
        type(before) not in (list, dict)
        or type(field["base_length"]) is not int
        or field["base_length"] != len(before)
        or field["base_hash"] != hashlib.sha256(_encoded(before)).hexdigest()
        or type(field["append"]) is not type(before)
        or type(drop) is not int
        or drop < 0
        or drop > len(before)
        or (type(before) is dict and "drop_prefix" in field)
    ):
        raise RuntimeError("Session history delta base is missing or does not match")
    restored = _copy_ledger_value(before[drop:] if type(before) is list else before)
    if type(before) is list:
        restored.extend(_copy_ledger_value(field["append"]))
    else:
        if any(key in before for key in field["append"]):
            raise RuntimeError("Session history delta cannot overwrite existing keys")
        restored.update(_copy_ledger_value(field["append"]))
    return restored


def apply_session_state_update(current: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    state = dict(current)
    patch = _copy_ledger_value(dict(payload.get("state") or {}))
    fields = _delta_fields(payload) if "state_history_delta" in payload else []
    seen: set[tuple[str, ...]] = set()
    for field in fields:
        required = {"path", "base_length", "base_hash", "append"}
        if not isinstance(field, dict) or set(field) not in (required, required | {"drop_prefix"}):
            raise RuntimeError("Invalid session history delta field")
        raw_path = field["path"]
        if not isinstance(raw_path, list) or not raw_path or not all(type(key) is str for key in raw_path):
            raise RuntimeError("Invalid session history delta path")
        path = tuple(raw_path)
        if any(path[:len(other)] == other or other[:len(path)] == path for other in seen):
            raise RuntimeError("Overlapping session history delta paths")
        seen.add(path)
        before = _history(current, path)
        restored = _restore_history(before, field)
        parent = patch
        for key in path[:-1]:
            if not isinstance(parent.get(key), dict):
                raise RuntimeError("Session history delta parent is missing")
            parent = parent[key]
        if path[-1] in parent:
            raise RuntimeError("Session history delta conflicts with a full state field")
        parent[path[-1]] = restored
    state.update(patch)
    return state


def apply_checkpoint_update(current: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    envelopes = {"checkpoint_history_delta": "checkpoint", "session_state_history_delta": "session_state"}
    update = {"state": {key: value for key, value in payload.items() if key not in envelopes}}
    fields = []
    version = 1
    for envelope, root in envelopes.items():
        if envelope in payload:
            references = _delta_fields({"state_history_delta": payload[envelope]})
            version = max(version, payload[envelope]["version"])
            if any(
                not isinstance(field, dict) or not isinstance(field.get("path"), list)
                or not field["path"] or field["path"][0] != root
                for field in references
            ):
                raise RuntimeError("Checkpoint history delta has an invalid root")
            fields.extend(references)
    if fields:
        update["state_history_delta"] = {"version": version, "fields": fields}
    restored = apply_session_state_update(current if fields else {}, update)
    retained = set(update["state"]) | {field["path"][0] for field in fields}
    return {key: value for key, value in restored.items() if key in retained}
