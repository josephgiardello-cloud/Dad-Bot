from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dadbot.core.execution_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_writer import LedgerWriter
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


def test_policy_trace_events_are_queryable_historically(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-policy", prompt="policy")
    with bind_execution_trace(recorder, required=True):
        manager.persist_turn_event(
            {
                "event_type": "PolicyTraceEvent",
                "trace_id": "inv-policy",
                "stage": "save",
                "status": "after",
                "payload": {
                    "summary": {"policy": "safety", "decision_action": "handled"},
                },
            },
        )

    writer = LedgerWriter(ledger)
    writer.write_event(
        event_type="PolicyTraceEvent",
        session_id="default",
        trace_id="inv-policy",
        kernel_step_id="save_node.policy_trace",
        payload={"summary": {"policy": "safety", "decision_action": "handled"}},
        committed=False,
    )

    recorder = ExecutionTraceRecorder(trace_id="inv-policy", prompt="policy-query")
    with bind_execution_trace(recorder, required=True):
        policy_events = manager.list_policy_trace_events(trace_id="inv-policy")

    assert len(policy_events) >= 2
    assert all(str(event.get("event_type") or "") == "PolicyTraceEvent" for event in policy_events)


def test_policy_trace_summary_is_queryable_historically(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-policy-summary", prompt="policy")
    with bind_execution_trace(recorder, required=True):
        manager.persist_turn_event(
            {
                "event_type": "PolicyTraceEvent",
                "trace_id": "inv-policy-summary",
                "stage": "save",
                "status": "after",
                "payload": {
                    "summary": {
                        "policy": "safety",
                        "decision_action": "handled",
                        "step_name": "validate",
                    },
                },
            },
        )

    writer = LedgerWriter(ledger)
    writer.write_event(
        event_type="PolicyTraceEvent",
        session_id="default",
        trace_id="inv-policy-summary",
        kernel_step_id="save_node.policy_trace",
        payload={
            "summary": {
                "policy": "safety",
                "decision_action": "passthrough",
                "step_name": "passthrough",
            },
        },
        committed=False,
    )

    recorder = ExecutionTraceRecorder(trace_id="inv-policy-summary", prompt="policy-summary-query")
    with bind_execution_trace(recorder, required=True):
        summary = manager.summarize_policy_trace_events(trace_id="inv-policy-summary")

    assert summary["event_type"] == "PolicyTraceEvent"
    assert int(summary["event_count"] or 0) >= 2
    assert "safety" in list(summary.get("policies") or [])
    action_counts = dict(summary.get("action_counts") or {})
    assert int(action_counts.get("handled") or 0) >= 1
    assert int(action_counts.get("passthrough") or 0) >= 1


def test_session_store_exposes_no_public_mutators():
    assert not hasattr(SessionStore, "apply_event")
    assert not hasattr(SessionStore, "apply_kernel_mutation")
    assert not hasattr(SessionStore, "set")
    assert not hasattr(SessionStore, "delete")
