from __future__ import annotations

import hashlib
import json
import threading
from typing import Any


class KernelEventTotalityLock:
    """Global event-before-mutation guardrail for turn execution.

    The lock tracks the most recent durable event witness per run_id.
    Mutation boundaries can require a witness before any durable write.
    """

    _lock = threading.RLock()
    _last_event_by_run: dict[str, dict[str, Any]] = {}
    _mutation_counter_by_run: dict[str, int] = {}

    @classmethod
    def note_event(
        cls,
        *,
        run_id: str,
        sequence_id: int,
        event_id: str,
        event_type: str,
    ) -> None:
        normalized_run = str(run_id or "").strip()
        if not normalized_run:
            return
        with cls._lock:
            cls._last_event_by_run[normalized_run] = {
                "sequence_id": int(sequence_id),
                "event_id": str(event_id or ""),
                "event_type": str(event_type or ""),
            }

    @classmethod
    def require_event_witness(cls, *, run_id: str, source: str = "") -> dict[str, Any]:
        normalized_run = str(run_id or "").strip()
        if not normalized_run:
            raise RuntimeError("Mutation event-totality violation: missing run_id")
        with cls._lock:
            witness = dict(cls._last_event_by_run.get(normalized_run) or {})
            if not witness:
                raise RuntimeError(
                    "Mutation event-totality violation: no prior durable event witness "
                    f"for run_id={normalized_run!r} source={str(source or '').strip()!r}",
                )
            cls._mutation_counter_by_run[normalized_run] = int(
                cls._mutation_counter_by_run.get(normalized_run, 0),
            ) + 1
            witness["mutation_order"] = int(cls._mutation_counter_by_run[normalized_run])
            return witness


class KernelToolIdempotencyRegistry:
    """Process-wide idempotency registry for tool side effects."""

    _lock = threading.RLock()
    _records: dict[str, Any] = {}

    @classmethod
    def deterministic_key(
        cls,
        *,
        tool_name: str,
        payload: dict[str, Any],
        scope: str = "",
    ) -> str:
        normalized_payload = {
            str(k): v
            for k, v in dict(payload or {}).items()
            if not str(k).startswith("_ephemeral_")
        }
        canonical = {
            "scope": str(scope or "").strip(),
            "tool_name": str(tool_name or "").strip().lower(),
            "payload": normalized_payload,
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()
        return digest[:32]

    @classmethod
    def get(cls, key: str) -> Any | None:
        with cls._lock:
            if key not in cls._records:
                return None
            return cls._records[key]

    @classmethod
    def put(cls, key: str, value: Any) -> None:
        with cls._lock:
            cls._records[str(key)] = value

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._records.clear()


class KernelReplaySequenceLock:
    """Strict replay-sequence hashing for deterministic equality checks."""

    @staticmethod
    def canonical_event(event: dict[str, Any], *, trace_id: str) -> dict[str, Any]:
        item = dict(event or {})
        sequence = int(item.get("sequence") or item.get("sequence_id") or 0)
        event_type = str(item.get("event_type") or item.get("type") or "").strip()
        stage = str(item.get("stage") or "").strip()
        expected_event_id = hashlib.sha256(f"{trace_id}:{sequence}".encode("utf-8")).hexdigest()[:16]
        observed_event_id = str(item.get("event_id") or "").strip()
        if observed_event_id and observed_event_id != expected_event_id:
            raise RuntimeError(
                "Replay strict equality violation: event_id mismatch "
                f"trace_id={trace_id!r} sequence={sequence} expected={expected_event_id!r} "
                f"actual={observed_event_id!r}",
            )
        return {
            "sequence": sequence,
            "event_id": observed_event_id or expected_event_id,
            "event_type": event_type,
            "stage": stage,
            "phase": str(item.get("phase") or "").strip(),
            "payload_hash": hashlib.sha256(
                json.dumps(dict(item.get("payload") or {}), sort_keys=True, default=str).encode("utf-8"),
            ).hexdigest()[:24],
        }

    @classmethod
    def strict_hash(cls, *, trace_id: str, events: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        expected = 1
        canonical: list[dict[str, Any]] = []
        ordered_events = sorted(
            list(events or []),
            key=lambda item: int((dict(item or {})).get("sequence") or (dict(item or {})).get("sequence_id") or 0),
        )
        for raw in ordered_events:
            item = cls.canonical_event(raw, trace_id=str(trace_id or ""))
            seq = int(item.get("sequence") or 0)
            if seq < expected:
                continue
            if seq != expected:
                raise RuntimeError(
                    "Replay strict equality violation: non-contiguous sequence "
                    f"expected={expected} actual={seq}",
                )
            canonical.append(item)
            expected += 1
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()
        return digest, canonical
