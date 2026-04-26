from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.canonical_event import NON_REPLAY_EVENT_TYPES
from dadbot.core.event_reducer import CanonicalEventReducer
from dadbot.core.execution_ledger import _canonical_trace_payload


class ReplayVerifier:
    """Deterministic replay equivalence boundary.

    A replay is equivalent only when both the canonical trace hash and reduced
    state hash match.
    """

    def __init__(self, reducer: CanonicalEventReducer | None = None) -> None:
        self.reducer = reducer or CanonicalEventReducer()

    def trace_hash(self, events: list[dict[str, Any]]) -> str:
        ordered = sorted(
            [
                dict(event)
                for event in list(events or [])
                if isinstance(event, dict)
                and str(event.get("type") or "") not in NON_REPLAY_EVENT_TYPES
            ],
            key=lambda event: int(event.get("sequence") or 0),
        )
        canonical_trace = [
            {
                "type": str(event.get("type") or ""),
                "session_id": str(event.get("session_id") or ""),
                "session_index": int(event.get("session_index") or 0),
                "event_id": str(event.get("event_id") or ""),
                "parent_event_id": str(event.get("parent_event_id") or ""),
                "kernel_step_id": str(event.get("kernel_step_id") or ""),
                "payload": _canonical_trace_payload(
                    str(event.get("type") or ""),
                    event.get("payload"),
                ),
            }
            for event in ordered
        ]
        return hashlib.sha256(json.dumps(canonical_trace, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def state_hash(self, events: list[dict[str, Any]]) -> str:
        state = self.reducer.reduce(events)
        return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def verify_equivalence(self, original_events: list[dict[str, Any]], replayed_events: list[dict[str, Any]]) -> dict[str, Any]:
        original_trace_hash = self.trace_hash(original_events)
        replayed_trace_hash = self.trace_hash(replayed_events)
        original_state_hash = self.state_hash(original_events)
        replayed_state_hash = self.state_hash(replayed_events)
        return {
            "ok": bool(original_trace_hash == replayed_trace_hash and original_state_hash == replayed_state_hash),
            "original_trace_hash": original_trace_hash,
            "replayed_trace_hash": replayed_trace_hash,
            "original_state_hash": original_state_hash,
            "replayed_state_hash": replayed_state_hash,
        }
