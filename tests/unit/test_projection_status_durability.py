from types import SimpleNamespace

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.memory.storage import MemoryStorageBackend


def _reload_projection_status(bot):
    ledger = ExecutionLedger(
        FileWALLedgerBackend(bot.config.session_log_dir / "execution_ledger.jsonl")
    )
    ledger.load_from_backend()
    storage = MemoryStorageBackend(SimpleNamespace(
        config=SimpleNamespace(tenant_id=bot.config.tenant_id),
        _execution_ledger=ledger,
    ), bot.memory_manager)
    return storage._load_projection_recovery_status()


def _append_status_event(bot, status, *, tenant_id=None):
    bot._execution_ledger.write({
        "type": "PROJECTION_STATUS_UPDATED",
        "session_id": "",
        "trace_id": "projection-test",
        "timestamp": "0",
        "kernel_step_id": "memory.projection.recovery",
        "committed": True,
        "payload": {
            "tenant_id": tenant_id or bot.config.tenant_id,
            "projection": "semantic",
            "status": status,
        },
    })


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


@pytest.mark.parametrize("projection", ["graph", "semantic"])
@pytest.mark.parametrize("unavailable", [False, True])
def test_recovery_propagates_authority_failures_without_publishing_status(
    make_test_dadbot, monkeypatch, projection, unavailable,
):
    from dadbot_system.delegation import AuthorityUnavailableError
    from dadbot_system.security import AuthorizationError

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    error = AuthorityUnavailableError if unavailable else AuthorizationError

    def reject(*_args, **_kwargs):
        raise error("Projection authority revoked")

    if projection == "graph":
        monkeypatch.setattr(bot.memory_manager.graph_manager, "sync_graph_store", reject)
    else:
        monkeypatch.setattr(bot.memory_manager, "sync_semantic_memory_index", reject)
    before = len(bot._execution_ledger.read())
    with pytest.raises(error, match="Projection authority revoked"):
        bot.memory_manager._storage.recover_local_projections()
    events = bot._execution_ledger.read()[before:]
    assert not any(
        event["type"] == "PROJECTION_STATUS_UPDATED" and event["payload"]["projection"] == projection
        for event in events
    )


def test_failed_rebuild_retains_last_applied_revision(make_test_dadbot, monkeypatch):
    from copy import deepcopy

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    storage = bot.memory_manager._storage
    snapshot = bot.snapshot_session_state()
    previous = storage._load_projection_recovery_status()["semantic"]
    current = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    state = deepcopy(current.state)
    state["memories"] = [{"id": "committed", "summary": "New committed memory"}]
    bot._execution_ledger.commit_tenant_memory(
        tenant_id=bot.config.tenant_id, expected_revision=current.revision,
        operation_id="commit-before-failed-rebuild", state=state,
    )
    storage._tenant_base_revision = current.revision + 1

    def fail(_memories):
        raise RuntimeError("Embedding unavailable")

    original = bot.memory_manager.sync_semantic_memory_index
    try:
        monkeypatch.setattr(bot.memory_manager, "sync_semantic_memory_index", fail)
        storage.recover_local_projections()
        restored = _reload_projection_status(bot)["semantic"]
        assert restored["applied_revision"] == previous["source_revision"]
        assert restored["source_revision"] == restored["required_revision"] == current.revision + 1
        assert restored["schema_version"] == 1
        assert restored["dirty"] is True
        assert restored["status"] == "degraded"
    finally:
        monkeypatch.setattr(bot.memory_manager, "sync_semantic_memory_index", original)
        bot.load_session_state_snapshot(snapshot)


def test_rebuild_records_its_source_before_a_later_commit(make_test_dadbot, monkeypatch):
    from copy import deepcopy

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    storage = bot.memory_manager._storage
    original = bot.memory_manager.graph_manager.sync_graph_store
    source_revision = storage._tenant_base_revision

    def commit_after_graph_sync(*, turn_context=None):
        result = original(turn_context=turn_context)
        current = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
        state = deepcopy(current.state)
        state["last_mood"] = "positive"
        bot._execution_ledger.commit_tenant_memory(
            tenant_id=bot.config.tenant_id, expected_revision=current.revision,
            operation_id="commit-after-graph-rebuild", state=state,
        )
        storage._tenant_base_revision = current.revision + 1
        return result

    monkeypatch.setattr(bot.memory_manager.graph_manager, "sync_graph_store", commit_after_graph_sync)
    storage.recover_local_projections()
    graph = _reload_projection_status(bot)["graph"]
    assert graph["source_revision"] == source_revision
    assert graph["applied_revision"] == source_revision


def test_recovery_uses_the_committed_catalog_not_staged_observation(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    manager = bot.memory_manager
    expected = manager._storage.committed_tenant_projection()["memories"]
    original = manager.sync_semantic_memory_index
    seen = []

    def observe(memories):
        seen.append(memories)
        original(memories)

    monkeypatch.setattr(manager, "memory_projection", lambda: {
        "memories": [{"id": "staged", "summary": "Must not enter the index"}],
    })
    monkeypatch.setattr(manager, "sync_semantic_memory_index", observe)
    manager._storage.recover_local_projections()
    assert seen == [expected]
    assert _reload_projection_status(bot)["semantic"]["status"] == "sync_succeeded"


@pytest.mark.parametrize("outcome", ["sync_succeeded", "degraded"])
def test_legacy_status_infers_only_known_applied_revisions(make_test_dadbot, monkeypatch, outcome):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    previous = bot.memory_manager._storage._load_projection_recovery_status()["semantic"]
    _append_status_event(bot, {
        "status": outcome, "generation": previous["generation"] + 1,
        "source_revision": previous["source_revision"], "updated_at": previous["updated_at"],
    })
    restored = _reload_projection_status(bot)["semantic"]
    assert restored["applied_revision"] == (previous["source_revision"] if outcome == "sync_succeeded" else None)
    assert restored["dirty"] is (outcome != "sync_succeeded")
    assert restored["rebuild_reason"] == "legacy"


@pytest.mark.parametrize("damage", ["version", "generation", "incomplete", "future", "missing_source", "null_source"])
def test_invalid_projection_metadata_fails_without_rewriting_history(make_test_dadbot, monkeypatch, damage):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    status = bot.memory_manager._storage._load_projection_recovery_status()["semantic"]
    changes = {
        "version": {"schema_version": 2},
        "generation": {"generation": -1},
        "future": {"source_revision": 999, "applied_revision": 999},
        "incomplete": {},
        "missing_source": {},
        "null_source": {"status": "degraded", "source_revision": None},
    }
    status.update(changes[damage])
    if damage == "incomplete":
        del status["applied_revision"]
    elif damage == "missing_source":
        del status["source_revision"]
    _append_status_event(bot, status)
    before = bot._execution_ledger.read()
    with pytest.raises(ValueError):
        _reload_projection_status(bot)
    assert bot._execution_ledger.read() == before


def test_foreign_status_is_not_imported_into_tenant_recovery(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot(tenant_id="t1_" + "a" * 26)
    previous = _reload_projection_status(bot)
    _append_status_event(bot, {"generation": "malformed foreign metadata"}, tenant_id="t1_" + "b" * 26)
    assert _reload_projection_status(bot) == previous


def test_projection_status_reads_and_publication_require_live_authority(make_test_dadbot, monkeypatch, tmp_path):
    from contextvars import copy_context

    from dadbot_system.delegation import SQLiteDelegationStore, runtime_authority_scope
    from dadbot_system.security import AuthorizationError, ServiceTokenManager

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    tenant = "t1_" + "a" * 26
    bot = make_test_dadbot(tenant_id=tenant)
    status = bot.memory_manager._storage._load_projection_recovery_status()["semantic"]
    path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(path)
    store = SQLiteDelegationStore(path, "projection-test-secret")
    principal = ServiceTokenManager("projection-test-secret").verify(
        ServiceTokenManager("projection-test-secret").issue(
            subject="projection-user", tenant_id=tenant, scopes={"write"},
        ),
    )
    store.grant_membership(issuer="dadbot", subject=principal.subject, tenant_id=tenant, scopes={"write"})
    with runtime_authority_scope(store, principal, scope="write", tenant_id=tenant):
        storage = MemoryStorageBackend(bot, bot.memory_manager)
        assert storage._load_projection_recovery_status()["semantic"] == status
        captured = copy_context()
    before = bot._execution_ledger.read()
    with pytest.raises(AuthorizationError):
        storage._load_projection_recovery_status()
    with pytest.raises(AuthorizationError):
        storage._persist_projection_recovery_status("semantic", status)
    store.revoke_membership(issuer="dadbot", subject=principal.subject, tenant_id=tenant)
    with pytest.raises(AuthorizationError):
        captured.run(storage._load_projection_recovery_status)
    with pytest.raises(AuthorizationError):
        captured.run(storage._persist_projection_recovery_status, "semantic", status)
    assert bot._execution_ledger.read() == before
