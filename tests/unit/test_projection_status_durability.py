from types import SimpleNamespace

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.memory.storage import MemoryStorageBackend


def _reload_projection_status(bot):
    ledger = ExecutionLedger(
        FileWALLedgerBackend(bot.config.session_log_dir / "execution_ledger.jsonl")
    )
    ledger.load_from_backend()
    storage = object.__new__(MemoryStorageBackend)
    storage.bot = SimpleNamespace(
        config=SimpleNamespace(tenant_id=bot.config.tenant_id),
        _execution_ledger=ledger,
    )
    return storage._load_projection_recovery_status()


def test_degraded_projection_status_survives_ledger_restart(
    make_test_dadbot, monkeypatch, tmp_path,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot(
        session_log_dir=tmp_path / "sessions",
        memory_path=tmp_path / "memory.json",
        tenant_id="projection-tenant",
    )

    def fail_semantic_sync(_memories):
        raise RuntimeError("semantic backend unavailable")

    monkeypatch.setattr(bot.memory_manager, "sync_semantic_memory_index", fail_semantic_sync)
    bot.memory_manager._storage.recover_local_projections()

    current = bot.memory_manager._storage._projection_recovery_status
    assert current["semantic"]["status"] == "degraded"
    assert current["graph"]["status"] == "sync_succeeded"

    restored = _reload_projection_status(bot)
    assert restored["semantic"]["status"] == "degraded"
    assert restored["semantic"]["generation"] == current["semantic"]["generation"]
    assert restored["graph"]["status"] == "sync_succeeded"


def test_graph_failure_does_not_prevent_semantic_projection_sync(
    make_test_dadbot, monkeypatch, tmp_path,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot(
        session_log_dir=tmp_path / "sessions",
        memory_path=tmp_path / "memory.json",
        tenant_id="projection-tenant",
    )

    def fail_graph_sync(*, turn_context=None):
        raise RuntimeError("graph backend unavailable")

    monkeypatch.setattr(bot.memory_manager.graph_manager, "sync_graph_store", fail_graph_sync)
    bot.memory_manager._storage.recover_local_projections()

    status = bot.memory_manager._storage._projection_recovery_status
    assert status["graph"]["status"] == "degraded"
    assert status["semantic"]["status"] == "sync_succeeded"


def test_committed_index_changes_make_previous_publication_dirty(make_test_dadbot, monkeypatch):
    from copy import deepcopy

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    snapshot = bot.snapshot_session_state()
    ledger = bot._execution_ledger
    state = deepcopy(bot.MEMORY_STORE)
    state["memories"] = [{"id": "test", "summary": "example"}]
    state = bot.memory_manager.normalize_memory_store(state)
    ledger.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=1,
        operation_id="crash-before-index-repair",
        state=state,
    )
    try:
        status = _reload_projection_status(bot)
        assert all(item["status"] == "dirty" for item in status.values())
        assert all(item["required_revision"] == 2 for item in status.values())
        assert all(item["source_revision"] == 1 for item in status.values())
    finally:
        bot.load_session_state_snapshot(snapshot)


def test_unrelated_memory_commit_does_not_dirty_indexes(make_test_dadbot, monkeypatch):
    from copy import deepcopy

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    snapshot = bot.snapshot_session_state()
    before = _reload_projection_status(bot)
    state = deepcopy(bot.MEMORY_STORE)
    state["last_mood"] = "positive"
    bot._execution_ledger.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=1,
        operation_id="unrelated-mutation",
        state=state,
    )
    try:
        after = _reload_projection_status(bot)
        assert all(item["status"] == "sync_succeeded" for item in after.values())
        assert all(after[name]["source_revision"] == before[name]["source_revision"] for name in after)
    finally:
        bot.load_session_state_snapshot(snapshot)


def test_latest_projection_attempt_wins_when_generation_is_tied(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    storage = bot.memory_manager._storage
    previous = storage._load_projection_recovery_status()["semantic"]
    storage._persist_projection_recovery_status(
        "semantic", {**previous, "status": "degraded", "error": "concurrent attempt failed"},
    )
    assert _reload_projection_status(bot)["semantic"]["status"] == "degraded"
