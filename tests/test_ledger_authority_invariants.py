from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dadbot.contracts import DadBotContext
from dadbot.core.control_plane_projection import ExecutionProjection
from dadbot.core.execution_context import ExecutionTraceRecorder, RuntimeTraceViolation, bind_execution_trace
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.persistence_record_schema import PersistenceSchemaError, normalize_checkpoint_record
from dadbot.core.replay_verifier import ReplayVerifier
from dadbot.core.session_store import SessionStore
from dadbot.core.tenant_session import tenant_session_scope
from dadbot.managers.conversation_persistence import ConversationPersistenceManager

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize(
    "operation",
    [
        "latest_checkpoint",
        "lifecycle_projection",
        "execution_id",
        "session_id",
        "turn_events",
        "policy_events",
        "load_checkpoint",
    ],
)
def test_persistence_lookups_do_not_copy_unrelated_state(
    operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    from dadbot.core import execution_ledger

    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(DadBotContext(_FakeBot(ledger, tmp_path)))
    recorder = ExecutionTraceRecorder(trace_id="selective-read", prompt="selective-read")
    with bind_execution_trace(recorder, required=True):
        manager.persist_turn_event(
            {
                "event_type": "PolicyTraceEvent",
                "trace_id": "selective-read",
                "stage": "safety",
                "status": "after",
                "payload": {"action": "passthrough", "policy": "safety"},
            },
        )
        manager.persist_turn_event(
            {
                "event_type": "phase_transition",
                "trace_id": "selective-read",
                "phase": "ACT",
            },
        )
        manager.persist_graph_checkpoint(
            {
                "trace_id": "selective-read",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "selective-lock"}},
            },
        )
        manager.claim_execution_handoff("selective-read", worker_id="worker-a", lease_seconds=15)
        ledger.write(
            {
                "type": "SESSION_STATE_UPDATED",
                "session_id": "unrelated-session",
                "trace_id": "unrelated-trace",
                "timestamp": "2026-01-01T00:00:00",
                "kernel_step_id": "unrelated-state",
                "payload": {"state": {"large_history": ["unused"] * 100}},
            },
        )
        lookups = {
            "latest_checkpoint": lambda: manager._latest_checkpoint_for_trace(
                "selective-read", session_id="default",
            ),
            "lifecycle_projection": lambda: manager._lifecycle_projection().snapshot(),
            "execution_id": lambda: manager._resolve_execution_id_for_trace("selective-read"),
            "session_id": lambda: manager._resolve_session_id_for_trace("selective-read"),
            "turn_events": lambda: manager.list_turn_events("selective-read"),
            "policy_events": lambda: manager.list_policy_trace_events(trace_token="selective-read"),
            "load_checkpoint": lambda: manager.load_latest_graph_checkpoint("selective-read"),
        }
        original_read = ledger.read
        with monkeypatch.context() as full_read_patch:
            full_read_patch.setattr(
                ledger, "read", lambda full=True, **_selection: original_read(full=full),
            )
            expected = lookups[operation]()
        original_copy = execution_ledger.deepcopy
        copied_lists = []

        def require_selected_copy(value):
            if isinstance(value, list):
                copied_lists.append(value)
                assert not any(
                    event.get("type") == "SESSION_STATE_UPDATED" and "payload" in event
                    for event in value
                ), "Unrelated session-state payload copied during lookup"
            return original_copy(value)

        monkeypatch.setattr(execution_ledger, "deepcopy", require_selected_copy)
        assert lookups[operation]() == expected
        assert copied_lists
        ledger._backend_uncertain = True
        with pytest.raises(RuntimeError, match="reload"):
            lookups[operation]()


def test_graph_checkpoint_ledger_event_carries_session_state_snapshot(tmp_path: Path):
    ledger = ExecutionLedger()
    bot = _FakeBot(ledger, tmp_path)
    bot.snapshot_session_state = lambda: {"memory": {"favorite_color": "blue"}}
    manager = ConversationPersistenceManager(bot)

    recorder = ExecutionTraceRecorder(trace_id="inv-session-state", prompt="invariants")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-session-state",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
            },
        )

    event = next(item for item in ledger.read() if str(item.get("type") or "") == "GRAPH_CHECKPOINT")
    assert dict(event.get("payload") or {}).get("session_state") == {
        "memory": {"favorite_color": "blue"},
    }


def test_graph_checkpoint_snapshot_is_canonical_execution_state(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-canonical", prompt="invariants")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-canonical",
                "stage": "save",
                "status": "after",
                "phase": "RESPOND",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"enforced": True, "lock_hash": "lock-c"}},
                "session_state": {"history": [{"role": "user", "content": "x"}]},
            },
        )
        checkpoint = manager.load_latest_graph_checkpoint(trace_id="inv-canonical")
        replay = manager.replay_turn_events("inv-canonical")

    assert isinstance(checkpoint, dict)
    assert checkpoint.get("schema_version") == "checkpoint-record.v1"
    assert checkpoint.get("trace_id") == "inv-canonical"
    assert checkpoint.get("phase") == "RESPOND"
    execution_state = dict(checkpoint.get("execution_state") or {})
    assert execution_state.get("state") == {"safe_result": "ok"}
    assert isinstance(execution_state.get("metadata"), dict)
    assert "session_state" not in checkpoint
    continuity = dict(checkpoint.get("continuity") or {})
    assert str(continuity.get("strict_sequence_hash") or "")
    assert replay["determinism"]["consistent"] is True
    assert replay["determinism"]["lock_hash"] == "lock-c"


def test_graph_checkpoint_reads_are_scoped_by_tenant_session_and_trace(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))
    for session_id, marker in (("session-a", "owner-a"), ("session-b", "owner-b")):
        recorder = ExecutionTraceRecorder(trace_id="shared-trace", prompt="invariants")
        with tenant_session_scope("tenant-test", session_id), bind_execution_trace(recorder, required=True):
            manager.persist_graph_checkpoint(
                {
                    "trace_id": "shared-trace",
                    "session_id": session_id,
                    "stage": "save",
                    "status": "after",
                    "state": {"marker": marker},
                },
            )

    for session_id, expected_marker in (("session-a", "owner-a"), ("session-b", "owner-b")):
        recorder = ExecutionTraceRecorder(trace_id="shared-trace", prompt="invariants")
        with tenant_session_scope("tenant-test", session_id), bind_execution_trace(recorder, required=True):
            checkpoint = manager.load_latest_graph_checkpoint(
                "shared-trace",
                session_id=session_id,
                tenant_id="tenant-test",
            )
        assert checkpoint is not None
        assert checkpoint["execution_state"]["state"]["marker"] == expected_marker

    recorder = ExecutionTraceRecorder(trace_id="shared-trace", prompt="invariants")
    with tenant_session_scope("tenant-test", "session-a"), bind_execution_trace(recorder, required=True):
        with pytest.raises(RuntimeTraceViolation, match="does not match the active session"):
            manager.load_latest_graph_checkpoint(
                "shared-trace",
                session_id="session-b",
                tenant_id="tenant-test",
            )

    stored_checkpoint_event = next(
        event for event in ledger._events
        if str(event.get("type") or "") == "GRAPH_CHECKPOINT"
        and str(event.get("session_id") or "") == "session-a"
    )
    stored_checkpoint_event["payload"]["checkpoint"]["session_id"] = "session-b"
    recorder = ExecutionTraceRecorder(trace_id="shared-trace", prompt="invariants")
    with tenant_session_scope("tenant-test", "session-a"), bind_execution_trace(recorder, required=True):
        with pytest.raises(PersistenceSchemaError, match="identity does not match"):
            manager.load_latest_graph_checkpoint("shared-trace", session_id="session-a")


def test_normalized_checkpoint_retains_execution_state_in_ledger(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))
    checkpoint = normalize_checkpoint_record(
        {
            "trace_id": "inv-normalized-checkpoint",
            "stage": "save",
            "status": "after",
            "state": {"safe_result": "ok"},
            "metadata": {"session_id": "session-1"},
            "phase_history": [{"phase": "SAVE"}],
            "stage_traces": [{"stage": "save"}],
            "event_sequence": 7,
        },
    )

    recorder = ExecutionTraceRecorder(trace_id="inv-normalized-checkpoint", prompt="invariants")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(checkpoint)
        persisted = manager.load_latest_graph_checkpoint(trace_id="inv-normalized-checkpoint")

    assert persisted is not None
    assert persisted["execution_state"] == checkpoint["execution_state"]


def test_graph_checkpoint_snapshot_is_canonical_and_thin(tmp_path: Path):
    # Contract-anchor compatibility shim: retained for strict manifest stability.
    test_graph_checkpoint_snapshot_is_canonical_execution_state(tmp_path)


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


def test_manager_list_turn_events_strict_rejects_malformed_persisted_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "1")
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    ledger.write(
        {
            "type": "TURN_EVENT",
            "session_id": "default",
            "trace_id": "inv-malformed",
            "timestamp": "2026-01-01T00:00:00",
            "kernel_step_id": "save",
            "payload": {
                "trace_id": "inv-malformed",
                "sequence": 1,
                "event_type": "graph_checkpoint",
                "occurred_at": "2026-01-01T00:00:00",
            },
        },
    )

    recorder = ExecutionTraceRecorder(trace_id="inv-malformed", prompt="strict")
    with bind_execution_trace(recorder, required=True):
        with pytest.raises(PersistenceSchemaError, match="Malformed persisted turn event"):
            manager.list_turn_events("inv-malformed")


def test_manager_replay_turn_events_strict_rejects_malformed_persisted_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "1")
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    ledger.write(
        {
            "type": "TURN_EVENT",
            "session_id": "default",
            "trace_id": "inv-malformed-replay",
            "timestamp": "2026-01-01T00:00:00",
            "kernel_step_id": "save",
            "payload": {
                "trace_id": "inv-malformed-replay",
                "sequence": 1,
                "event_type": "phase_transition",
                "occurred_at": "2026-01-01T00:00:00",
            },
        },
    )

    recorder = ExecutionTraceRecorder(trace_id="inv-malformed-replay", prompt="strict")
    with bind_execution_trace(recorder, required=True):
        with pytest.raises(PersistenceSchemaError, match="Malformed persisted turn event"):
            manager.replay_turn_events("inv-malformed-replay")


def test_manager_load_latest_checkpoint_strict_rejects_malformed_persisted_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "1")
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    ledger.write(
        {
            "type": "GRAPH_CHECKPOINT",
            "session_id": "default",
            "trace_id": "inv-bad-cp",
            "timestamp": "2026-01-01T00:00:00",
            "kernel_step_id": "save",
            "payload": {
                "trace_id": "inv-bad-cp",
                "stage": "save",
                "status": "after",
                "checkpoint": {
                    "trace_id": "inv-bad-cp",
                    "stage": "save",
                    "status": "after",
                },
            },
        },
    )

    recorder = ExecutionTraceRecorder(trace_id="inv-bad-cp", prompt="strict-checkpoint")
    with bind_execution_trace(recorder, required=True):
        with pytest.raises(PersistenceSchemaError, match="Invalid checkpoint record"):
            manager.load_latest_graph_checkpoint("inv-bad-cp")


def test_execution_handoff_round_trip_preserves_checkpoint_continuity(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-handoff", prompt="handoff")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-handoff",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "lock-handoff"}},
                "event_sequence": 5,
            },
        )
        claimed = manager.claim_execution_handoff("inv-handoff", worker_id="worker-a", lease_seconds=30)
        handoff = manager.export_execution_handoff("inv-handoff", worker_id="worker-a")
        imported = manager.import_execution_handoff(handoff)

    assert claimed["status"] == "claimed"
    assert handoff["schema_version"] == "execution-handoff.v1"
    assert handoff["trace_id"] == "inv-handoff"
    assert dict(handoff.get("lease") or {}).get("worker_id") == "worker-a"
    assert imported["trace_id"] == "inv-handoff"
    assert imported["execution_mode"] == "recovery"
    assert dict(imported.get("continuity") or {}).get("strict_sequence_hash")


def test_checkpoint_runtime_transition_rejects_event_sequence_regression(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-transition", prompt="transition")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-transition",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "lock-1"}},
                "event_sequence_id": 10,
            },
        )
        with pytest.raises(RuntimeError, match="event_sequence_id regressed"):
            manager.persist_graph_checkpoint(
                {
                    "trace_id": "inv-transition",
                    "stage": "save",
                    "status": "after",
                    "state": {"safe_result": "ok-2"},
                    "metadata": {"determinism": {"lock_hash": "lock-2"}},
                    "event_sequence_id": 9,
                },
            )


def test_execution_handoff_claim_rejects_other_worker_while_active(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-lease-conflict", prompt="lease")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-lease-conflict",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "lease-lock"}},
            },
        )
        manager.claim_execution_handoff("inv-lease-conflict", worker_id="worker-a", lease_seconds=30)
        with pytest.raises(RuntimeError, match="already claimed by another worker"):
            manager.claim_execution_handoff("inv-lease-conflict", worker_id="worker-b", lease_seconds=30)


def test_execution_handoff_lease_can_be_renewed_and_released(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-lease-renew", prompt="lease")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-lease-renew",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "lease-lock"}},
            },
        )
        claimed = manager.claim_execution_handoff("inv-lease-renew", worker_id="worker-a", lease_seconds=15)
        renewed = manager.renew_execution_handoff_lease("inv-lease-renew", worker_id="worker-a", lease_seconds=45)
        released = manager.release_execution_handoff_claim("inv-lease-renew", worker_id="worker-a")

    assert claimed["status"] == "claimed"
    assert renewed["status"] == "claimed"
    assert renewed["lease_seconds"] == 45
    assert released["status"] == "released"


def test_execution_handoff_import_requires_matching_active_claim(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-import-claim", prompt="lease")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-import-claim",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "lease-lock"}},
            },
        )
        manager.claim_execution_handoff("inv-import-claim", worker_id="worker-a", lease_seconds=30)
        handoff = manager.export_execution_handoff("inv-import-claim", worker_id="worker-a")
        handoff["worker_id"] = "worker-b"
        handoff["lease"] = dict(handoff.get("lease") or {})
        handoff["lease"]["worker_id"] = "worker-b"
        with pytest.raises(RuntimeError, match="owned by another worker"):
            manager.import_execution_handoff(handoff)


def test_execution_handoff_export_rejects_other_worker_while_claim_active(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-export-claim", prompt="lease")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-export-claim",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "lease-lock"}},
            },
        )
        manager.claim_execution_handoff("inv-export-claim", worker_id="worker-a", lease_seconds=30)
        with pytest.raises(RuntimeError, match="owned by another worker"):
            manager.export_execution_handoff("inv-export-claim", worker_id="worker-b")


def test_distributed_crash_takeover_claims_after_expiry(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-dist-crash", prompt="dist")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-dist-crash",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "dist-crash"}},
            },
        )
        manager.claim_execution_handoff("inv-dist-crash", worker_id="worker-a", lease_seconds=1)

    time.sleep(1.1)
    recorder = ExecutionTraceRecorder(trace_id="inv-dist-crash", prompt="dist")
    with bind_execution_trace(recorder, required=True):
        claimed = manager.claim_execution_handoff("inv-dist-crash", worker_id="worker-b", lease_seconds=30)

    assert claimed["status"] == "claimed"
    assert claimed["worker_id"] == "worker-b"

    lifecycle_events = [
        dict(event.get("payload") or {})
        for event in ledger.read()
        if str(event.get("type") or "") == "EXECUTION_LIFECYCLE"
    ]
    event_types = [str(payload.get("event_type") or "") for payload in lifecycle_events]
    assert "LeaseExpired" in event_types
    assert "Redelivered" in event_types


def test_distributed_double_claim_race_allows_single_winner(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-dist-race", prompt="dist")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-dist-race",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "dist-race"}},
            },
        )

    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, str]] = []
    lock = threading.Lock()

    def _attempt(worker: str) -> None:
        local_recorder = ExecutionTraceRecorder(trace_id="inv-dist-race", prompt="dist")
        with bind_execution_trace(local_recorder, required=True):
            barrier.wait()
            try:
                manager.claim_execution_handoff("inv-dist-race", worker_id=worker, lease_seconds=30)
                result = (worker, "claimed")
            except RuntimeError:
                result = (worker, "conflict")
            with lock:
                outcomes.append(result)

    left = threading.Thread(target=_attempt, args=("worker-a",), daemon=True)
    right = threading.Thread(target=_attempt, args=("worker-b",), daemon=True)
    left.start()
    right.start()
    left.join(timeout=5.0)
    right.join(timeout=5.0)

    assert len(outcomes) == 2
    assert sum(1 for _worker, status in outcomes if status == "claimed") == 1
    assert sum(1 for _worker, status in outcomes if status == "conflict") == 1


def test_distributed_restart_mid_flight_handoff_continuity(tmp_path: Path):
    ledger = ExecutionLedger()
    manager_a = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-dist-restart", prompt="dist")
    with bind_execution_trace(recorder, required=True):
        manager_a.persist_graph_checkpoint(
            {
                "trace_id": "inv-dist-restart",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "dist-restart"}},
            },
        )
        manager_a.claim_execution_handoff("inv-dist-restart", worker_id="worker-a", lease_seconds=30)
        handoff = manager_a.export_execution_handoff("inv-dist-restart", worker_id="worker-a")

    manager_b = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))
    recorder = ExecutionTraceRecorder(trace_id="inv-dist-restart", prompt="dist")
    with bind_execution_trace(recorder, required=True):
        imported = manager_b.import_execution_handoff(handoff)

    assert imported["trace_id"] == "inv-dist-restart"
    assert imported["execution_mode"] == "recovery"
    assert str(dict(imported.get("continuity") or {}).get("strict_sequence_hash") or "").strip()


def test_distributed_redelivery_attempt_count_and_replay_determinism(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="inv-dist-redelivery", prompt="dist")
    with bind_execution_trace(recorder, required=True):
        manager.persist_graph_checkpoint(
            {
                "trace_id": "inv-dist-redelivery",
                "stage": "save",
                "status": "after",
                "state": {"safe_result": "ok"},
                "metadata": {"determinism": {"lock_hash": "dist-redelivery"}},
            },
        )
        manager.claim_execution_handoff("inv-dist-redelivery", worker_id="worker-a", lease_seconds=1)

    time.sleep(1.1)
    recorder = ExecutionTraceRecorder(trace_id="inv-dist-redelivery", prompt="dist")
    with bind_execution_trace(recorder, required=True):
        manager.claim_execution_handoff("inv-dist-redelivery", worker_id="worker-b", lease_seconds=30)

    projection = ExecutionProjection()
    events = ledger.read()
    projection.rebuild_from_ledger(events)
    lifecycle_payloads = [
        dict(event.get("payload") or {})
        for event in events
        if str(event.get("type") or "") == "EXECUTION_LIFECYCLE"
        and str(event.get("trace_id") or "") == "inv-dist-redelivery"
    ]
    execution_id = ""
    for payload in lifecycle_payloads:
        candidate = str(payload.get("execution_id") or "").strip()
        if candidate:
            execution_id = candidate
            break
    assert execution_id

    state = projection.get(execution_id)
    assert state is not None
    assert state.attempt_count == 2

    verifier = ReplayVerifier()
    replay_hash_a = verifier.trace_hash(events)
    replay_hash_b = verifier.trace_hash([dict(event) for event in events])
    assert replay_hash_a == replay_hash_b
