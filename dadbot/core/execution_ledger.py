from __future__ import annotations

import hashlib
import json
import threading
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Any

from dadbot.core.canonical_event import (
    CANONICAL_EVENT_FIELDS,
    NON_REPLAY_EVENT_TYPES,
    canonicalize_event_payload,
)
from dadbot.core.event_schema import get_migrator, stamp_schema_version
from dadbot.core.ledger_backend import InMemoryLedgerBackend, SequenceValidator


_TAIL_LIMIT = 256


class WriteBoundaryViolationError(RuntimeError):
    """Raised when strict ledger mode rejects a write outside the boundary guard."""


class WriteBoundaryGuard(AbstractContextManager["WriteBoundaryGuard"]):
    """Temporarily allow writes to a strict-mode ledger."""

    def __init__(self, ledger: ExecutionLedger) -> None:
        self._ledger = ledger

    def __enter__(self) -> WriteBoundaryGuard:
        self._ledger._write_guard_depth += 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        self._ledger._write_guard_depth = max(0, self._ledger._write_guard_depth - 1)


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
    return (
        "evt-"
        + hashlib.sha256(
            json.dumps(seed, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:24]
    )


def _canonical_event_for_hash(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("type") or "")
    return {
        "type": event_type,
        "session_id": str(event.get("session_id") or ""),
        "session_index": int(event.get("session_index") or 0),
        "kernel_step_id": str(event.get("kernel_step_id") or ""),
        "parent_event_id": str(event.get("parent_event_id") or ""),
        "event_id": str(event.get("event_id") or ""),
        "payload": _canonical_trace_payload(event_type, event.get("payload")),
    }


def _event_sha256(event: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_canonical_event_for_hash(event), sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def _chain_hash(prev_chain_hash: str, event_sha256: str) -> str:
    return hashlib.sha256(
        f"{str(prev_chain_hash or '')}:{str(event_sha256 or '')}".encode("utf-8"),
    ).hexdigest()


class ExecutionLedger:
    """In-memory append-only execution ledger with replay-hash support."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        strict_writes: bool = False,
    ) -> None:
        self._backend = backend or InMemoryLedgerBackend()
        self._events: list[dict[str, Any]] = []
        self._cache: dict[str, Any] = {
            "sequence_counter": 0,
            "event_count": 0,
            "last_event_hash": None,
            "recent_tail": [],
            "trace_sequence_counters": {},
            "cache_rebuild_count": 0,
            "version": 0,
        }
        self._lock = threading.RLock()
        self._session_heads: dict[str, str] = {}
        self._session_indices: dict[str, int] = {}
        self._write_guard_depth = 0
        self._strict_writes = bool(strict_writes)
        self._replay_filters: list[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = []

    @property
    def sealed_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.read())

    def add_replay_filter(
        self,
        filter_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> None:
        self._replay_filters.append(filter_fn)

    def _ensure_write_allowed(self) -> None:
        if self._strict_writes and self._write_guard_depth <= 0:
            raise WriteBoundaryViolationError(
                "ExecutionLedger strict mode requires WriteBoundaryGuard",
            )

    def get_next_sequence(self) -> int:
        with self._lock:
            return int(self._cache.get("sequence_counter") or 0) + 1

    def get_next_trace_sequence(self, trace_id: str) -> int:
        with self._lock:
            counters = dict(self._cache.get("trace_sequence_counters") or {})
            key = str(trace_id or "").strip()
            return int(counters.get(key, 0)) + 1

    def event_count(self) -> int:
        with self._lock:
            return int(self._cache.get("event_count") or 0)

    def _rebuild_cache(self) -> None:
        trace_sequence_counters: dict[str, int] = {}
        for event in self._events:
            if str(event.get("type") or "") != "TURN_EVENT":
                continue
            payload = dict(event.get("payload") or {})
            trace_id = str(payload.get("trace_id") or "").strip()
            if not trace_id:
                continue
            seq = int(payload.get("sequence") or 0)
            if seq > int(trace_sequence_counters.get(trace_id, 0)):
                trace_sequence_counters[trace_id] = seq

        self._cache["sequence_counter"] = len(self._events)
        self._cache["event_count"] = len(self._events)
        self._cache["last_event_hash"] = (
            str(self._events[-1].get("chain_hash") or "") if self._events else None
        )
        self._cache["recent_tail"] = list(self._events[-_TAIL_LIMIT:])
        self._cache["trace_sequence_counters"] = trace_sequence_counters
        self._cache["cache_rebuild_count"] = int(self._cache.get("cache_rebuild_count") or 0) + 1
        self._cache["version"] = int(self._cache.get("version") or 0) + 1

    def telemetry_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self.validate_cache()
            return {
                "sequence_counter": int(self._cache.get("sequence_counter") or 0),
                "event_count": int(self._cache.get("event_count") or 0),
                "cache_rebuild_count": int(self._cache.get("cache_rebuild_count") or 0),
                "tail_size": int(len(self._cache.get("recent_tail") or [])),
                "version": int(self._cache.get("version") or 0),
            }

    def validate_cache(self) -> None:
        if int(self._cache.get("event_count") or 0) != len(self._events):
            self._rebuild_cache()

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        from dadbot.core.ledger.enforcement import LedgerEnforcer

        self._ensure_write_allowed()
        LedgerEnforcer().validate(dict(event or {}))
        with self._lock:
            payload = dict(event or {})
            cache = self._cache
            session_id = str(payload.get("session_id") or "")
            parent_event_id = str(payload.get("parent_event_id") or "")
            current_head = str(self._session_heads.get(session_id) or "")
            if session_id:
                if parent_event_id:
                    if current_head and parent_event_id != current_head:
                        raise RuntimeError(
                            f"causal chain violation: session_id={session_id!r} parent={parent_event_id!r} head={current_head!r}",
                        )
                else:
                    payload["parent_event_id"] = current_head
                payload["session_index"] = int(self._session_indices.get(session_id, 0)) + 1
            else:
                payload.setdefault("parent_event_id", "")
                payload.setdefault("session_index", 0)

            payload.setdefault("payload", {})
            next_sequence = int(cache.get("sequence_counter") or 0) + 1
            payload["_seq"] = len(self._events)
            payload.setdefault("sequence", next_sequence)
            payload.setdefault("event_id", _deterministic_event_id(payload))
            stamp_schema_version(payload)
            event_sha256 = _event_sha256(payload)
            prev_chain_hash = str(self._events[-1].get("chain_hash") or "") if self._events else ""
            payload["event_sha256"] = event_sha256
            payload["prev_chain_hash"] = prev_chain_hash
            payload["chain_hash"] = _chain_hash(prev_chain_hash, event_sha256)

            event_id = str(payload.get("event_id") or "")
            if session_id:
                self._session_heads[session_id] = event_id
                self._session_indices[session_id] = int(
                    payload.get("session_index") or 0,
                )

            self._events.append(payload)
            self._backend.append(
                deepcopy(payload),
                committed=bool(payload.get("committed", False)),
            )

            cache["sequence_counter"] = int(payload.get("sequence") or next_sequence)
            cache["event_count"] = int(cache.get("event_count") or 0) + 1
            cache["last_event_hash"] = str(payload.get("chain_hash") or "")
            if str(payload.get("type") or "") == "TURN_EVENT":
                turn_payload = dict(payload.get("payload") or {})
                trace_id = str(turn_payload.get("trace_id") or "").strip()
                if trace_id:
                    counters = dict(cache.get("trace_sequence_counters") or {})
                    counters[trace_id] = int(turn_payload.get("sequence") or 0)
                    cache["trace_sequence_counters"] = counters
            tail = list(cache.get("recent_tail") or [])
            tail.append(payload)
            if len(tail) > _TAIL_LIMIT:
                tail = tail[-_TAIL_LIMIT:]
            cache["recent_tail"] = tail
            cache["version"] = int(cache.get("version") or 0) + 1
            self.validate_cache()
            return payload

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        return self.write(event)

    def read(self, full: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            self.validate_cache()
            if full:
                return self._events
            return list(self._cache.get("recent_tail") or [])

    def replay_hash(self) -> str:
        events = [event for event in self.read() if str(event.get("type") or "") not in NON_REPLAY_EVENT_TYPES]
        canonical = [
            {field: event.get(field) for field in CANONICAL_EVENT_FIELDS}
            | {
                "payload": _canonical_trace_payload(
                    str(event.get("type") or ""),
                    event.get("payload"),
                ),
            }
            for event in sorted(events, key=lambda item: int(item.get("sequence") or 0))
        ]
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    def chain_hash(self) -> str:
        events = self.read()
        if not events:
            return ""
        return str(events[-1].get("chain_hash") or "")

    def verify_replay(self, *, mode: str = "basic") -> dict[str, Any]:
        events = list(self.read())
        violations: list[str] = []

        if str(mode or "basic").strip().lower() == "chain":
            prev = ""
            for idx, event in enumerate(events):
                expected_event_sha = _event_sha256(event)
                expected_prev = prev
                expected_chain = _chain_hash(expected_prev, expected_event_sha)

                actual_event_sha = str(event.get("event_sha256") or "")
                actual_prev = str(event.get("prev_chain_hash") or "")
                actual_chain = str(event.get("chain_hash") or "")

                if actual_event_sha != expected_event_sha:
                    violations.append(f"event[{idx}].event_sha256_mismatch")
                if actual_prev != expected_prev:
                    violations.append(f"event[{idx}].prev_chain_hash_mismatch")
                if actual_chain != expected_chain:
                    violations.append(f"event[{idx}].chain_hash_mismatch")
                prev = expected_chain

        return {
            "ok": len(violations) == 0,
            "mode": str(mode or "basic"),
            "event_count": len(events),
            "replay_hash": self.replay_hash(),
            "chain_hash": self.chain_hash(),
            "violations": violations,
        }

    def load_from_backend(self) -> int:
        with self._lock:
            events = list(get_migrator().migrate_all(list(self._backend.load())))
            report = SequenceValidator.validate(events)
            if not bool(report.get("ok")):
                warnings.warn(
                    f"sequence anomaly: {report.get('violations') or []}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            for replay_filter in self._replay_filters:
                events = list(replay_filter(list(events)))
            self._events = deepcopy(events)
            self._session_heads.clear()
            self._session_indices.clear()
            prev_chain = ""
            for event in self._events:
                if not str(event.get("event_id") or ""):
                    event["event_id"] = _deterministic_event_id(event)
                event_sha256 = _event_sha256(event)
                event["event_sha256"] = event_sha256
                event["prev_chain_hash"] = prev_chain
                prev_chain = _chain_hash(prev_chain, event_sha256)
                event["chain_hash"] = prev_chain
                session_id = str(event.get("session_id") or "")
                event_id = str(event.get("event_id") or "")
                if session_id and event_id:
                    self._session_heads[session_id] = event_id
                    self._session_indices[session_id] = int(
                        event.get("session_index") or 0,
                    )
            self._rebuild_cache()
            self.validate_cache()
            return len(self._events)

InMemoryExecutionLedger = ExecutionLedger
