from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from dadbot.contracts import FinalizedTurnResult
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.tenant_session import current_tenant_id

_INDEX_PAYLOAD_TYPES = frozenset({
    "JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED", "JOB_FAILED", "JOB_RECONCILED",
    "EFFECT_BEGIN", "EFFECT_COMMIT", "EFFECT_RECONCILED", "AGENT_LOOP_NO_EFFECT_CONFIRMED",
    "EXECUTION_LIFECYCLE",
})


@dataclass(slots=True)
class EffectState:
    begun: bool = False
    committed: bool = False
    reconciled_no_effect: bool = False
    begin_event: dict[str, Any] | None = None
    commit_event: dict[str, Any] | None = None


@dataclass(slots=True)
class _JobIndexState:
    completed_by_request: dict[tuple[str, str, str], FinalizedTurnResult] = field(default_factory=dict)
    completed_metadata_by_request: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    started_without_terminal: set[tuple[str, str, str]] = field(default_factory=set)
    requests_with_dispatch_evidence: set[tuple[str, str, str]] = field(default_factory=set)
    operator_confirmed_no_effect: set[tuple[str, str, str, str, int]] = field(default_factory=set)
    operator_confirmed_no_effect_requests: set[tuple[str, str, str]] = field(default_factory=set)
    effect_ids_by_request: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)


class LedgerIndex:
    """Hot-path index layer over ledger events.

    The index rebuilds only when event_count changes, avoiding repeated full scans
    in dedupe and replay decision paths.
    """

    def __init__(self, ledger: ExecutionLedger) -> None:
        self._ledger = ledger
        self._indexed_event_count = -1
        self._completed_by_request: dict[tuple[str, str, str], FinalizedTurnResult] = {}
        self._completed_metadata_by_request: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._started_without_terminal: set[tuple[str, str, str]] = set()
        self._requests_with_dispatch_evidence: set[tuple[str, str, str]] = set()
        self._operator_confirmed_no_effect: set[tuple[str, str, str, str, int]] = set()
        self._operator_confirmed_no_effect_requests: set[tuple[str, str, str]] = set()
        self._effect_ids_by_request: dict[tuple[str, str, str], set[str]] = {}
        self._effect_state: dict[tuple[str, str, str], EffectState] = {}
        self._active_leases_by_worker: dict[str, set[str]] = {}
        self._trace_roots: dict[tuple[str, str], str] = {}

    @staticmethod
    def _normalize_event(event: dict[str, Any]) -> tuple[str, str, str, str, dict[str, Any]]:
        event_type = str(event.get("type") or "")
        session_id = str(event.get("session_id") or "default").strip() or "default"
        tenant_id = str(event.get("tenant_id") or "").strip()
        trace_id = str(event.get("trace_id") or "").strip()
        payload = dict(event.get("payload") or {})
        return event_type, tenant_id, session_id, trace_id, payload

    @staticmethod
    def _index_job_event(event: dict[str, Any], state: _JobIndexState) -> None:
        event_type, tenant_id, session_id, trace_id, payload = LedgerIndex._normalize_event(event)
        if event_type not in {
            "JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED", "JOB_FAILED",
            "JOB_RECONCILED", "AGENT_LOOP_NO_EFFECT_CONFIRMED",
        }:
            return
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            return
        key = (tenant_id, session_id, request_id)
        state.requests_with_dispatch_evidence.add(key)
        if event_type == "JOB_STARTED":
            state.started_without_terminal.add(key)
        elif event_type in {"JOB_COMPLETED", "JOB_FAILED", "JOB_RECONCILED", "AGENT_LOOP_NO_EFFECT_CONFIRMED"}:
            state.started_without_terminal.discard(key)
        if event_type == "AGENT_LOOP_NO_EFFECT_CONFIRMED":
            LedgerIndex._index_operator_attestation(payload, key, state)
        if event_type in {"JOB_SUBMITTED", "JOB_QUEUED", "JOB_STARTED"}:
            return
        if event_type != "JOB_COMPLETED":
            state.completed_by_request.pop(key, None)
            state.completed_metadata_by_request.pop(key, None)
            return
        LedgerIndex._index_completed_job(payload, key, trace_id, state)

    @staticmethod
    def _index_completed_job(
        payload: dict[str, Any],
        request_key: tuple[str, str, str],
        trace_id: str,
        state: _JobIndexState,
    ) -> None:
        state.completed_by_request.pop(request_key, None)
        metadata = payload.get("metadata")
        metadata_summary = {}
        if isinstance(metadata, dict):
            metadata_summary = {
                name: deepcopy(metadata[name])
                for name in (
                    "agent_loop",
                    "tool_outcome_evidence",
                    "execution_receipt_status",
                    "execution_receipt_evidence",
                )
                if name in metadata
            }
        state.completed_metadata_by_request[request_key] = {
            "tenant_id": request_key[0],
            "session_id": request_key[1],
            "request_id": request_key[2],
            "job_id": str(payload.get("job_id") or ""),
            "trace_id": trace_id,
            "metadata": metadata_summary,
        }
        result = payload.get("result")
        if isinstance(result, tuple) and len(result) == 2:
            state.completed_by_request[request_key] = result
        elif isinstance(result, list) and len(result) == 2:
            state.completed_by_request[request_key] = (result[0], bool(result[1]))

    @staticmethod
    def _index_operator_attestation(
        payload: dict[str, Any],
        request_key: tuple[str, str, str],
        state: _JobIndexState,
    ) -> None:
        loop_run_id = str(payload.get("loop_run_id") or "").strip()
        turn_index = payload.get("turn_index")
        if (
            payload.get("confirmed_no_external_effect") is True
            and payload.get("execution_quiesced") is True
            and str(payload.get("operator_ref") or "").strip()
            and loop_run_id
            and type(turn_index) is int
            and turn_index > 0
        ):
            state.operator_confirmed_no_effect.add((*request_key, loop_run_id, turn_index))
            state.operator_confirmed_no_effect_requests.add(request_key)

    @staticmethod
    def _index_effect_event(
        *,
        event: dict[str, Any],
        event_type: str,
        scope: tuple[str, str],
        payload: dict[str, Any],
        effect_state: dict[tuple[str, str, str], EffectState],
    ) -> None:
        if event_type == "AGENT_LOOP_NO_EFFECT_CONFIRMED":
            for effect_id in payload.get("effect_ids", []):
                normalized_effect_id = str(effect_id or "").strip()
                if normalized_effect_id:
                    key = (scope[0], scope[1], normalized_effect_id)
                    state = effect_state.get(key) or EffectState()
                    state.reconciled_no_effect = True
                    effect_state[key] = state
            return
        if event_type not in {"EFFECT_BEGIN", "EFFECT_COMMIT", "EFFECT_RECONCILED"}:
            return
        effect_id = str(payload.get("effect_id") or "").strip()
        if not effect_id:
            return
        tenant_id, session_id = scope
        key = (tenant_id, session_id, effect_id)
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

        job_state = _JobIndexState()
        effect_state: dict[tuple[str, str, str], EffectState] = {}
        active_leases_by_worker: dict[str, set[str]] = {}
        trace_roots: dict[tuple[str, str], str] = {}

        for event in self._ledger.read(payload_types=_INDEX_PAYLOAD_TYPES):
            event_type, tenant_id, session_id, trace_id, payload = self._normalize_event(event)

            trace_key = (tenant_id, trace_id)
            if trace_id and trace_key not in trace_roots:
                trace_roots[trace_key] = str(event.get("event_id") or "")

            self._index_job_event(event, job_state)
            if event_type in {
                "EFFECT_BEGIN", "EFFECT_COMMIT", "EFFECT_RECONCILED", "AGENT_LOOP_NO_EFFECT_CONFIRMED",
            }:
                request_id = str(payload.get("request_id") or "").strip()
                if request_id:
                    request_key = (tenant_id, session_id, request_id)
                    job_state.requests_with_dispatch_evidence.add(request_key)
                    effect_ids = (
                        payload.get("effect_ids", [])
                        if event_type == "AGENT_LOOP_NO_EFFECT_CONFIRMED"
                        else [payload.get("effect_id")]
                    )
                    for effect_id in effect_ids:
                        normalized_effect_id = str(effect_id or "").strip()
                        if normalized_effect_id:
                            job_state.effect_ids_by_request.setdefault(request_key, set()).add(
                                normalized_effect_id,
                            )
            self._index_effect_event(
                event=event,
                event_type=event_type,
                scope=(tenant_id, session_id),
                payload=payload,
                effect_state=effect_state,
            )
            self._index_execution_lifecycle_event(
                event_type=event_type,
                payload=payload,
                active_leases_by_worker=active_leases_by_worker,
            )

        self._completed_by_request = job_state.completed_by_request
        self._completed_metadata_by_request = job_state.completed_metadata_by_request
        self._started_without_terminal = job_state.started_without_terminal
        self._requests_with_dispatch_evidence = job_state.requests_with_dispatch_evidence
        self._operator_confirmed_no_effect = job_state.operator_confirmed_no_effect
        self._operator_confirmed_no_effect_requests = job_state.operator_confirmed_no_effect_requests
        self._effect_ids_by_request = job_state.effect_ids_by_request
        self._effect_state = effect_state
        self._active_leases_by_worker = active_leases_by_worker
        self._trace_roots = trace_roots
        self._indexed_event_count = event_count

    def completed_result(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str | None = None,
    ) -> FinalizedTurnResult | None:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        if not key[1]:
            return None
        return self._completed_by_request.get(key)

    def completed_metadata(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        if not key[1]:
            return None
        metadata = self._completed_metadata_by_request.get(key)
        return deepcopy(metadata) if metadata is not None else None

    def completed_agent_loop_entries(
        self,
        *,
        session_id: str,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.refresh()
        tenant = str(tenant_id or current_tenant_id() or "").strip()
        session = str(session_id or "default").strip() or "default"
        return [
            deepcopy(entry)
            for key, entry in self._completed_metadata_by_request.items()
            if key[0] == tenant
            and key[1] == session
            and isinstance(entry["metadata"].get("agent_loop"), dict)
        ]

    def has_ambiguous_request_inflight(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        if not key[1]:
            return False
        return key in self._started_without_terminal

    def has_request_dispatch_evidence(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        return key in self._requests_with_dispatch_evidence

    def has_operator_confirmed_no_effect(
        self,
        *,
        session_id: str,
        request_id: str,
        loop_run_id: str,
        turn_index: int,
        tenant_id: str | None = None,
    ) -> bool:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
            str(loop_run_id or "").strip(),
            turn_index,
        )
        return key in self._operator_confirmed_no_effect

    def effect_ids_for_request(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str | None = None,
    ) -> tuple[str, ...]:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        return tuple(sorted(self._effect_ids_by_request.get(key, set())))

    def has_operator_confirmed_no_effect_for_request(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(request_id or "").strip(),
        )
        return key in self._operator_confirmed_no_effect_requests

    def effect_state(
        self,
        *,
        session_id: str,
        effect_id: str,
        tenant_id: str | None = None,
    ) -> EffectState:
        self.refresh()
        key = (
            str(tenant_id or current_tenant_id() or "").strip(),
            str(session_id or "default").strip() or "default",
            str(effect_id or "").strip(),
        )
        if not key[1]:
            return EffectState()
        return self._effect_state.get(key) or EffectState()

    def is_effect_committed(self, *, session_id: str, effect_id: str, tenant_id: str | None = None) -> bool:
        return self.effect_state(session_id=session_id, effect_id=effect_id, tenant_id=tenant_id).committed

    def has_ambiguous_effect_inflight(
        self,
        *,
        session_id: str,
        effect_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        state = self.effect_state(session_id=session_id, effect_id=effect_id, tenant_id=tenant_id)
        return bool(state.begun and not state.committed and not state.reconciled_no_effect)

    def ambiguous_effect_entries(self) -> list[dict[str, str]]:
        self.refresh()
        entries: list[dict[str, str]] = []
        for tenant_id, session_id, effect_id in self._effect_state:
            state = self._effect_state[(tenant_id, session_id, effect_id)]
            if state.begun and not state.committed:
                entries.append(
                    {
                        "tenant_id": str(tenant_id or ""),
                        "session_id": str(session_id or "default"),
                        "effect_id": str(effect_id or ""),
                    },
                )
        entries.sort(key=lambda item: (item["tenant_id"], item["session_id"], item["effect_id"]))
        return entries

    def ambiguous_request_entries(self) -> list[dict[str, str]]:
        self.refresh()
        entries = [
            {
                "tenant_id": str(tenant_id or ""),
                "session_id": str(session_id or "default"),
                "request_id": str(request_id or ""),
            }
            for tenant_id, session_id, request_id in self._started_without_terminal
            if str(request_id or "").strip()
        ]
        entries.sort(key=lambda item: (item["tenant_id"], item["session_id"], item["request_id"]))
        return entries

    def active_leases_for_worker(self, worker_id: str) -> set[str]:
        self.refresh()
        return set(self._active_leases_by_worker.get(str(worker_id or "").strip(), set()))

    def trace_root(self, trace_token: str) -> str:
        self.refresh()
        key = (str(current_tenant_id() or "").strip(), str(trace_token or "").strip())
        return str(self._trace_roots.get(key) or "")


__all__ = ["EffectState", "LedgerIndex"]
