from __future__ import annotations

from types import SimpleNamespace

import pytest

import dadbot.services.persistence as persistence_module
from dadbot.core.runtime_errors import PersistenceFailure
from dadbot.services.persistence import PersistenceService

pytestmark = pytest.mark.unit


class _PersistenceManagerStub:
    def __init__(self) -> None:
        self.saved_events: list[dict[str, object]] = []
        self.saved_checkpoints: list[dict[str, object]] = []
        self.replay_payload: dict[str, object] = {
            "schema_version": "replay-record.v1",
            "persistence_schema_version": "persistence-record.v1",
            "trace_id": "trace-ok",
            "strict_sequence_hash": "h",
            "strict_sequence": [{"sequence": 1, "event_id": "e1"}],
            "event_count": 1,
            "events": [
                {
                    "schema_version": "trace-event.v1",
                    "trace_id": "trace-ok",
                    "event_id": "e1",
                    "sequence": 1,
                    "event_type": "graph_checkpoint",
                    "stage": "save",
                    "status": "after",
                    "occurred_at": "2026-01-01T00:00:00",
                },
            ],
            "replay_output": {
                "phase": "SAVE",
                "replayed_state": {},
                "replayed_metadata": {},
                "determinism": {},
            },
            "phase": "SAVE",
            "replayed_state": {},
            "replayed_metadata": {},
            "determinism": {},
        }
        self.latest_checkpoint: dict[str, object] | None = None
        self.handoff_payload: dict[str, object] = {
            "schema_version": "execution-handoff.v1",
            "trace_id": "trace-ok",
            "session_id": "default",
            "worker_id": "worker-a",
            "checkpoint": {
                "schema_version": "checkpoint-record.v1",
                "trace_id": "trace-ok",
                "session_id": "default",
                "stage": "save",
                "status": "after",
                "phase": "RESPOND",
                "checkpoint_hash": "cp-hash",
                "prev_checkpoint_hash": "",
                "event_sequence_id": 1,
                "occurred_at": "2026-01-01T00:00:00",
                "execution_mode": "recovery",
                "execution_state": {
                    "state": {},
                    "metadata": {},
                    "phase_history": [],
                    "stage_traces": [],
                    "event_sequence": 1,
                },
                "continuity": {
                    "execution_fingerprint": "",
                    "strict_sequence_hash": "seq-hash",
                    "event_count": 1,
                },
            },
            "continuity": {
                "execution_fingerprint": "",
                "strict_sequence_hash": "seq-hash",
                "event_count": 1,
            },
            "lease": {
                "status": "claimed",
                "worker_id": "worker-a",
                "lease_expires_at": "2026-01-01T00:01:00",
                "lease_seconds": 60,
                "active": True,
            },
            "handoff_hash": "h",
        }

    def persist_turn_event(self, event: dict[str, object]) -> None:
        self.saved_events.append(dict(event))

    def replay_turn_events(self, trace_id: str) -> dict[str, object]:
        payload = dict(self.replay_payload)
        payload["trace_id"] = str(trace_id or payload.get("trace_id") or "")
        return payload

    def persist_graph_checkpoint(self, checkpoint: dict[str, object], _skip_turn_event: bool = False) -> None:
        _ = _skip_turn_event
        self.saved_checkpoints.append(dict(checkpoint))

    def load_latest_graph_checkpoint(self, trace_id: str = "") -> dict[str, object] | None:
        _ = trace_id
        return None if self.latest_checkpoint is None else dict(self.latest_checkpoint)

    def resume_graph_checkpoint(self, trace_id: str = "") -> dict[str, object] | None:
        _ = trace_id
        return None if self.latest_checkpoint is None else dict(self.latest_checkpoint)

    def export_execution_handoff(self, trace_id: str, worker_id: str = "") -> dict[str, object]:
        _ = trace_id
        _ = worker_id
        return dict(self.handoff_payload)

    def import_execution_handoff(self, handoff: dict[str, object]) -> dict[str, object]:
        _ = handoff
        return dict(self.handoff_payload.get("checkpoint") or {})

    def claim_execution_handoff(
        self,
        trace_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> dict[str, object]:
        _ = trace_id
        return {
            "status": "claimed",
            "worker_id": str(worker_id or ""),
            "lease_seconds": int(lease_seconds),
            "lease_expires_at": "2026-01-01T00:01:00",
            "active": True,
        }

    def renew_execution_handoff_lease(
        self,
        trace_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> dict[str, object]:
        _ = trace_id
        return {
            "status": "claimed",
            "worker_id": str(worker_id or ""),
            "lease_seconds": int(lease_seconds),
            "lease_expires_at": "2026-01-01T00:02:00",
            "active": True,
        }

    def release_execution_handoff_claim(self, trace_id: str, *, worker_id: str) -> dict[str, object]:
        _ = trace_id
        return {
            "status": "released",
            "worker_id": str(worker_id or ""),
            "lease_seconds": 0,
            "lease_expires_at": "",
            "active": False,
        }


class _TurnServiceStub:
    def __init__(self) -> None:
        self.bot = SimpleNamespace()


def test_save_turn_event_strict_mode_rejects_malformed_event() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())
    service.strict_mode = True

    with pytest.raises(PersistenceFailure, match="rejected malformed event"):
        service.save_turn_event(
            {
                "trace_id": "trace-a",
                "event_id": "",
                "sequence": 0,
                "event_type": "",
                "occurred_at": "",
            },
        )


def test_save_turn_event_non_strict_normalizes_and_persists() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())

    service.save_turn_event(
        {
            "trace_id": " trace-b ",
            "event_id": "evt-1",
            "sequence": "2",
            "event_type": "GRAPH_CHECKPOINT",
            "stage": "Save",
            "status": "After",
            "occurred_at": "2026-01-01T00:00:00",
        },
    )

    assert len(pm.saved_events) == 1
    saved = dict(pm.saved_events[0])
    assert saved["schema_version"] == "trace-event.v1"
    assert saved["trace_id"] == "trace-b"
    assert saved["sequence"] == 2
    assert saved["event_type"] == "graph_checkpoint"


def test_replay_turn_events_strict_mode_rejects_schema_drift() -> None:
    pm = _PersistenceManagerStub()
    pm.replay_payload["event_count"] = 2
    service = PersistenceService(pm, turn_service=_TurnServiceStub())
    service.strict_mode = True

    with pytest.raises(PersistenceFailure, match="rejected malformed replay payload"):
        service.replay_turn_events("trace-c")


def test_save_graph_checkpoint_strict_mode_rejects_malformed_checkpoint() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())
    service.strict_mode = True

    with pytest.raises(PersistenceFailure, match="rejected malformed checkpoint"):
        service.save_graph_checkpoint(
            {
                "trace_id": "trace-cp",
                "stage": "save",
                "status": "after",
            },
        )


def test_save_graph_checkpoint_non_strict_normalizes_and_persists() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())

    service.save_graph_checkpoint(
        {
            "trace_id": " trace-cp ",
            "stage": "Save",
            "status": "After",
            "occurred_at": "2026-01-01T00:00:00",
        },
    )

    assert len(pm.saved_checkpoints) == 1
    saved = dict(pm.saved_checkpoints[0])
    assert saved["schema_version"] == "checkpoint-record.v1"
    assert saved["trace_id"] == "trace-cp"
    assert saved["stage"] == "save"
    assert saved["status"] == "after"


def test_load_latest_graph_checkpoint_strict_mode_rejects_malformed_checkpoint() -> None:
    pm = _PersistenceManagerStub()
    pm.latest_checkpoint = {
        "trace_id": "trace-cp",
        "stage": "save",
        "status": "after",
    }
    service = PersistenceService(pm, turn_service=_TurnServiceStub())
    service.strict_mode = True

    with pytest.raises(PersistenceFailure, match="rejected malformed checkpoint"):
        service.load_latest_graph_checkpoint("trace-cp")


def test_export_execution_handoff_returns_payload() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())

    handoff = service.export_execution_handoff("trace-ok", worker_id="worker-a")

    assert handoff["schema_version"] == "execution-handoff.v1"
    assert handoff["trace_id"] == "trace-ok"


def test_import_execution_handoff_returns_valid_checkpoint() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())

    imported = service.import_execution_handoff(pm.handoff_payload)

    assert imported["schema_version"] == "checkpoint-record.v1"
    assert imported["trace_id"] == "trace-ok"
    assert imported["execution_mode"] == "recovery"


def test_claim_execution_handoff_returns_lease() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())

    lease = service.claim_execution_handoff("trace-ok", worker_id="worker-a", lease_seconds=90)

    assert lease["status"] == "claimed"
    assert lease["worker_id"] == "worker-a"
    assert lease["lease_seconds"] == 90


def test_renew_execution_handoff_lease_returns_lease() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())

    lease = service.renew_execution_handoff_lease("trace-ok", worker_id="worker-a", lease_seconds=120)

    assert lease["status"] == "claimed"
    assert lease["worker_id"] == "worker-a"
    assert lease["lease_seconds"] == 120


def test_release_execution_handoff_claim_returns_released() -> None:
    pm = _PersistenceManagerStub()
    service = PersistenceService(pm, turn_service=_TurnServiceStub())

    released = service.release_execution_handoff_claim("trace-ok", worker_id="worker-a")

    assert released["status"] == "released"
    assert released["worker_id"] == "worker-a"


def test_restore_transaction_snapshot_enforces_mutation_entry_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _capture_enforce(**kwargs):
        calls.append(dict(kwargs))

    monkeypatch.setattr(persistence_module, "enforce_mutation_entry_invariants", _capture_enforce)
    monkeypatch.setattr(persistence_module, "require_bound_core_state_for_mutation", lambda **_kwargs: None)

    runtime = SimpleNamespace(
        MEMORY_STORE={},
        _last_turn_pipeline={},
        _background_memory_store_patch_queue=[],
        memory_manager=None,
        load_session_state_snapshot=lambda _snapshot: None,
    )
    turn_context = SimpleNamespace(state={}, metadata={})
    snapshot = {
        "session_state": {},
        "memory_store": {"foo": "bar"},
        "last_turn_pipeline": {"stage": "save"},
        "background_patch_queue": [],
        "turn_state": {"a": 1},
        "metadata": {"b": 2},
        "graph_snapshot": {},
    }

    PersistenceService._restore_transaction_snapshot(runtime, turn_context, snapshot)

    assert len(calls) == 1
    assert calls[0]["mutation_kind"] == "persistence"
    assert calls[0]["source"] == "PersistenceService._restore_transaction_snapshot"
