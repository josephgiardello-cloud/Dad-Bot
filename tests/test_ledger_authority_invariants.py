from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dadbot.core.execution_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.session_store import SessionStore
from dadbot.managers.conversation_persistence import ConversationPersistenceManager


class _FakeBot:
    def __init__(self, ledger: ExecutionLedger, session_log_dir: Path) -> None:
        self.turn_orchestrator = SimpleNamespace(
            control_plane=SimpleNamespace(ledger=ledger),
        )
        self._tenant_document_store = None
        self._graph_commit_active = False
        self._current_turn_time_base = None
        self._io_lock = None
        self.SESSION_LOG_DIR = session_log_dir
        self.config = SimpleNamespace(
            tenant_id="tenant-test",
            active_model="model-test",
            active_embedding_model="embed-test",
            session_log_dir=session_log_dir,
        )
        self.session_summary = ""

    def snapshot_session_state(self) -> dict:
        return {}

    def relationship_state(self) -> dict:
        return {}


def test_runtime_persistence_path_avoids_file_writes(monkeypatch, tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    def _forbid(*_args, **_kwargs):
        raise AssertionError("unexpected filesystem write in runtime authority path")

    monkeypatch.setattr(Path, "write_text", _forbid)
    monkeypatch.setattr(Path, "write_bytes", _forbid)
    monkeypatch.setattr(Path, "open", _forbid)

    recorder = ExecutionTraceRecorder(trace_id="inv-a", prompt="invariants")
    with bind_execution_trace(recorder, required=True):
        manager.persist_turn_event(
            {
                "event_type": "graph_checkpoint",
                "trace_id": "inv-a",
                "stage": "inference",
                "status": "after",
            },
        )
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-a",
                "stage": "inference",
                "status": "after",
                "state": {"candidate": "draft"},
                "metadata": {"determinism": {"lock_hash": "lock-a"}},
            },
        )


def test_persistence_events_are_ledger_authoritative(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-b", prompt="invariants")
    with bind_execution_trace(recorder, required=True):
        manager.persist_turn_event(
            {
                "event_type": "phase_transition",
                "trace_id": "inv-b",
                "phase": "ACT",
            },
        )
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-b",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "lock-b"}},
            },
        )
        events = manager.list_turn_events("inv-b")

    assert any(str(event.get("type") or "") == "TURN_EVENT" for event in ledger.read())
    assert any(str(event.get("type") or "") == "GRAPH_CHECKPOINT" for event in ledger.read())
    assert any(str(event.get("event_type") or "") == "phase_transition" for event in events)


def test_session_store_exposes_no_public_mutators():
    assert not hasattr(SessionStore, "apply_event")
    assert not hasattr(SessionStore, "apply_kernel_mutation")
    assert not hasattr(SessionStore, "set")
    assert not hasattr(SessionStore, "delete")
