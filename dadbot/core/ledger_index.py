from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dadbot.contracts import FinalizedTurnResult
from dadbot.core.execution_ledger import ExecutionLedger


@dataclass(slots=True)
class EffectState:
    begun: bool = False
    committed: bool = False
    begin_event: dict[str, Any] | None = None
    commit_event: dict[str, Any] | None = None


class LedgerIndex:
    """Hot-path index layer over ledger events.

    The index rebuilds only when event_count changes, avoiding repeated full scans
    in dedupe and replay decision paths.
    """

    def __init__(self, ledger: ExecutionLedger) -> None:
        self._ledger = ledger
        self._indexed_event_count = -1
        self._completed_by_request: dict[tuple[str, str], FinalizedTurnResult] = {}
        self._started_without_terminal: set[tuple[str, str]] = set()
        self._effect_state: dict[tuple[str, str], EffectState] = {}
        self._active_leases_by_worker: dict[str, set[str]] = {}
        self._trace_roots: dict[str, str] = {}

    @staticmethod
    def _normalize_event(event: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
        event_type = str(event.get("type") or "")
        session_id = str(event.get("session_id") or "default").strip() or "default"
        trace_id = str(event.get("trace_id") or "").strip()
        payload = dict(event.get("payload") or {})
        return event_type, session_id, trace_id, payload

    @staticmethod
    def _index_job_event(
        *,
        event_type: str,
        session_id: str,
        payload: dict[str, Any],
        completed_by_request: dict[tuple[str, str], FinalizedTurnResult],
        started_without_terminal: set[tuple[str, str]],
    ) -> None:
        if event_type not in {"JOB_STARTED", "JOB_COMPLETED", "JOB_FAILED", "JOB_RECONCILED"}:
            return
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            return
        key = (session_id, request_id)
        if event_type == "JOB_STARTED":
            started_without_terminal.add(key)
        else:
            started_without_terminal.discard(key)
        if event_type != "JOB_COMPLETED":
            return
        result = payload.get("result")
        if isinstance(result, tuple) and len(result) == 2:
            completed_by_request[key] = result
        elif isinstance(result, list) and len(result) == 2:
            completed_by_request[key] = (result[0], bool(result[1]))

    @staticmethod
    def _index_effect_event(
        *,
        event: dict[str, Any],
        event_type: str,
        session_id: str,
        payload: dict[str, Any],
        effect_state: dict[tuple[str, str], EffectState],
    ) -> None:
        if event_type not in {"EFFECT_BEGIN", "EFFECT_COMMIT", "EFFECT_RECONCILED"}:
            return
        effect_id = str(payload.get("effect_id") or "").strip()
        if not effect_id:
            return
        key = (session_id, effect_id)
        state = effect_state.get(key) or EffectState()
        if event_type == "EFFECT_BEGIN":
            state.begun = True
            state.begin_event = event
        else:
            state.committed = True
            state.commit_event = event
        effect_state[key] = state

    @staticmethod
    def _index_execution_lifecycle_event(
        *,
        event_type: str,
        payload: dict[str, Any],
        active_leases_by_worker: dict[str, set[str]],
    ) -> None:
        if event_type != "EXECUTION_LIFECYCLE":
            return
        life = dict(payload or {})
        life_type = str(life.get("type") or "").strip()
        execution_id = str(life.get("execution_id") or "").strip()
        worker_id = str(life.get("worker_id") or "").strip()
        if life_type == "Claimed" and worker_id and execution_id:
            active_leases_by_worker.setdefault(worker_id, set()).add(execution_id)
        if life_type in {"Completed", "Failed", "Released", "LeaseExpired"}:
            for leases in active_leases_by_worker.values():
                leases.discard(execution_id)

    def refresh(self, *, force: bool = False) -> None:
        event_count = int(self._ledger.event_count())
        if not force and event_count == self._indexed_event_count:
            return

        completed_by_request: dict[tuple[str, str], FinalizedTurnResult] = {}
        started_without_terminal: set[tuple[str, str]] = set()
        effect_state: dict[tuple[str, str], EffectState] = {}
        active_leases_by_worker: dict[str, set[str]] = {}
        trace_roots: dict[str, str] = {}

        for event in self._ledger.read():
            event_type, session_id, trace_id, payload = self._normalize_event(event)

            if trace_id and trace_id not in trace_roots:
                trace_roots[trace_id] = str(event.get("event_id") or "")

            self._index_job_event(
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                completed_by_request=completed_by_request,
                started_without_terminal=started_without_terminal,
            )
            self._index_effect_event(
                event=event,
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                effect_state=effect_state,
            )
            self._index_execution_lifecycle_event(
                event_type=event_type,
                payload=payload,
                active_leases_by_worker=active_leases_by_worker,
            )

        self._completed_by_request = completed_by_request
        self._started_without_terminal = started_without_terminal
        self._effect_state = effect_state
        self._active_leases_by_worker = active_leases_by_worker
        self._trace_roots = trace_roots
        self._indexed_event_count = event_count

    def completed_result(self, *, session_id: str, request_id: str) -> FinalizedTurnResult | None:
        self.refresh()
        key = (
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        if not key[1]:
            return None
        return self._completed_by_request.get(key)

    def has_ambiguous_request_inflight(self, *, session_id: str, request_id: str) -> bool:
        self.refresh()
        key = (
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        if not key[1]:
            return False
        return key in self._started_without_terminal

    def effect_state(self, *, session_id: str, effect_id: str) -> EffectState:
        self.refresh()
        key = (
            str(session_id or "default").strip() or "default",
            str(effect_id or "").strip(),
        )
        if not key[1]:
            return EffectState()
        return self._effect_state.get(key) or EffectState()

    def is_effect_committed(self, *, session_id: str, effect_id: str) -> bool:
        return self.effect_state(session_id=session_id, effect_id=effect_id).committed

    def has_ambiguous_effect_inflight(self, *, session_id: str, effect_id: str) -> bool:
        state = self.effect_state(session_id=session_id, effect_id=effect_id)
        return bool(state.begun and not state.committed)

    def ambiguous_effect_entries(self) -> list[dict[str, str]]:
        self.refresh()
        entries: list[dict[str, str]] = []
        for (session_id, effect_id), state in self._effect_state.items():
            if state.begun and not state.committed:
                entries.append(
                    {
                        "session_id": str(session_id or "default"),
                        "effect_id": str(effect_id or ""),
                    },
                )
        entries.sort(key=lambda item: (item["session_id"], item["effect_id"]))
        return entries

    def ambiguous_request_entries(self) -> list[dict[str, str]]:
        self.refresh()
        entries = [
            {
                "session_id": str(session_id or "default"),
                "request_id": str(request_id or ""),
            }
            for session_id, request_id in self._started_without_terminal
            if str(request_id or "").strip()
        ]
        entries.sort(key=lambda item: (item["session_id"], item["request_id"]))
        return entries

    def active_leases_for_worker(self, worker_id: str) -> set[str]:
        self.refresh()
        return set(self._active_leases_by_worker.get(str(worker_id or "").strip(), set()))

    def trace_root(self, trace_token: str) -> str:
        self.refresh()
        return str(self._trace_roots.get(str(trace_token or "").strip()) or "")


__all__ = ["EffectState", "LedgerIndex"]
