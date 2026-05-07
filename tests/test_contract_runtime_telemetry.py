from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dadbot.core.execution_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.managers.conversation_persistence import ConversationPersistenceManager


class _FakeBot:
    def __init__(self, ledger: ExecutionLedger, session_log_dir: Path) -> None:
        self.turn_orchestrator = SimpleNamespace(control_plane=SimpleNamespace(ledger=ledger))
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


def test_execution_ledger_telemetry_snapshot_tracks_cache_rebuilds():
    ledger = ExecutionLedger()
    ledger._cache["event_count"] = -1  # force cache mismatch path for rebuild
    before = int(ledger._cache.get("cache_rebuild_count") or 0)

    telemetry = ledger.telemetry_snapshot()

    after = int(telemetry.get("cache_rebuild_count") or 0)
    assert after >= before + 1
    assert int(telemetry.get("event_count") or 0) == len(ledger.read())


def test_persistence_telemetry_snapshot_exposes_slo_and_policy(tmp_path: Path):
    ledger = ExecutionLedger()
    manager = ConversationPersistenceManager(_FakeBot(ledger, tmp_path))

    recorder = ExecutionTraceRecorder(trace_id="telemetry-trace", prompt="telemetry")
    with bind_execution_trace(recorder, required=True):
        manager.persist_turn_event(
            {
                "event_type": "phase_transition",
                "trace_id": "telemetry-trace",
                "phase": "ACT",
            },
        )

    telemetry = manager.persistence_telemetry_snapshot()

    assert "write_p95_ms" in telemetry
    assert "compaction_p95_ms" in telemetry
    assert isinstance(telemetry.get("slo"), dict)
    assert isinstance(telemetry.get("slo_ok"), dict)
    assert isinstance(telemetry.get("policy"), dict)
    policy = dict(telemetry.get("policy") or {})
    assert int(policy.get("active_compaction_interval_events") or 0) > 0
    assert int(policy.get("recommended_retention_events") or 0) > 0
