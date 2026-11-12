import subprocess
import sys
import time
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


def test_overlapping_rebuild_publishes_failure_after_newer_attempts(make_test_dadbot, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    older = bot.memory_manager._storage
    newer = MemoryStorageBackend(bot, bot.memory_manager)
    original = bot.memory_manager.graph_manager.sync_graph_store
    paused, release = Event(), Event()

    def overlap(*, turn_context=None):
        result = original(turn_context=turn_context)
        if current_thread().name.startswith("older-rebuild"):
            paused.set()
            assert release.wait(30)
            raise RuntimeError("Older rebuild failed after overlap")
        return result

    monkeypatch.setattr(bot.memory_manager.graph_manager, "sync_graph_store", overlap)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="older-rebuild") as executor:
        pending = executor.submit(older.recover_local_projections)
        try:
            assert paused.wait(30)
            newer.recover_local_projections()
            newer.recover_local_projections()
            published = _reload_projection_status(bot)["graph"]
        finally:
            release.set()
        pending.result(timeout=30)
    restored = _reload_projection_status(bot)["graph"]
    assert restored["status"] == "degraded"
    assert restored["generation"] > published["generation"]
    assert restored["applied_revision"] == published["applied_revision"]


def test_independent_status_writer_reallocates_generation_after_head_conflict(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    ledger = ExecutionLedger(FileWALLedgerBackend(bot.config.session_log_dir / "execution_ledger.jsonl"))
    ledger.load_from_backend()
    other = MemoryStorageBackend(SimpleNamespace(config=bot.config, _execution_ledger=ledger), bot.memory_manager)
    old = other._load_projection_recovery_status()["semantic"]
    storage = bot.memory_manager._storage
    for _ in range(2):
        current = storage._load_projection_recovery_status()["semantic"]
        storage._persist_projection_recovery_status("semantic", {**current, "generation": current["generation"] + 1})
    published = _reload_projection_status(bot)["semantic"]
    other._persist_projection_recovery_status("semantic", {
        **old, "status": "degraded", "error": "Independent writer failure",
    })
    restored = _reload_projection_status(bot)["semantic"]
    assert restored["status"] == "degraded"
    assert restored["generation"] > published["generation"]


def test_process_writers_reallocate_projection_generation_after_head_conflict(
    make_test_dadbot, monkeypatch,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    session_dir = bot.config.session_log_dir
    tenant_id = bot.config.tenant_id
    baseline_events = len(bot._execution_ledger.read())
    worker = """
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.memory.storage import MemoryStorageBackend
root = Path(sys.argv[1])
tenant_id = sys.argv[2]
writer_id = sys.argv[3]
ledger = ExecutionLedger(FileWALLedgerBackend(root / "execution_ledger.jsonl"))
ledger.load_from_backend()
storage = MemoryStorageBackend(
    SimpleNamespace(config=SimpleNamespace(tenant_id=tenant_id), _execution_ledger=ledger),
    SimpleNamespace(),
)
status = storage._load_projection_recovery_status()["semantic"]
(root / (".projection-writer-" + writer_id + "-ready")).write_text("ready", encoding="utf-8")
while not (root / ".projection-writer-release").exists():
    time.sleep(0.01)
storage._persist_projection_recovery_status(
    "semantic",
    {**status, "status": "degraded", "error": "Concurrent process publication", "updated_at": time.time()},
)
"""
    writers = []
    release = session_dir / ".projection-writer-release"
    try:
        for index in range(2):
            writers.append(subprocess.Popen(
                [sys.executable, "-c", worker, str(session_dir), tenant_id, str(index)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ))
        deadline = time.monotonic() + 30
        ready = [session_dir / f".projection-writer-{index}-ready" for index in range(2)]
        while not all(path.exists() for path in ready):
            exited = [writer for writer in writers if writer.poll() is not None]
            if exited:
                details = [writer.communicate() for writer in exited]
                raise AssertionError(f"Projection writer exited before synchronization: {details}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Projection writer processes did not reach the barrier")
            time.sleep(0.01)
        release.write_text("release", encoding="utf-8")
        for writer in writers:
            stdout, stderr = writer.communicate(timeout=30)
            assert writer.returncode == 0, f"Projection writer failed: {stdout}\n{stderr}"
    finally:
        release.write_text("release", encoding="utf-8")
        for writer in writers:
            if writer.poll() is None:
                writer.kill()
                writer.communicate(timeout=10)

    events = ExecutionLedger(
        FileWALLedgerBackend(session_dir / "execution_ledger.jsonl")
    )
    events.load_from_backend()
    appended = [
        event for event in events.read()[baseline_events:]
        if event.get("type") == "PROJECTION_STATUS_UPDATED"
        and event.get("payload", {}).get("tenant_id") == tenant_id
        and event.get("payload", {}).get("projection") == "semantic"
    ]
    assert len(appended) == 2
    assert len({event["sequence"] for event in appended}) == 2
    restored = _reload_projection_status(bot)["semantic"]
    assert restored["status"] == "degraded"
    assert restored["generation"] == max(event["sequence"] for event in appended)


def test_process_projection_rebuilds_share_local_index_writes(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    bot.mutate_memory_store(memories=[{
        "id": "process-rebuild",
        "summary": "Committed memory for concurrent projection recovery",
    }])
    graph_backend = bot.memory_manager.graph_manager._graph_store_backend
    graph_backend.replace_graph(
        [{"node_key": "stale", "node_type": "topic", "label": "obsolete"}],
        [],
    )
    session_dir = bot.config.session_log_dir
    worker = """
import sys
import time
from pathlib import Path
from dadbot.core.dadbot import DadBot
from dadbot.memory.semantic_manager import SemanticIndexManager

SemanticIndexManager.embed_texts = lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts]
root = Path(sys.argv[1])
worker_id = sys.argv[2]
bot = DadBot()
graph = bot.memory_manager.graph_manager
original = graph.sync_graph_store
(root / (".projection-rebuild-" + worker_id + "-ready")).write_text("ready", encoding="utf-8")
def synchronized_sync(*, turn_context=None):
    (root / (".projection-rebuild-" + worker_id + "-entered")).write_text("entered", encoding="utf-8")
    deadline = time.monotonic() + 60
    while not all((root / (".projection-rebuild-" + peer + "-entered")).exists() for peer in ("0", "1")):
        if time.monotonic() >= deadline:
            raise TimeoutError("Concurrent projection rebuild did not reach the barrier")
        time.sleep(0.01)
    return original(turn_context=turn_context)
graph.sync_graph_store = synchronized_sync
try:
    bot.memory_manager._storage.recover_local_projections()
finally:
    bot.shutdown()
"""
    workers = []
    try:
        for worker_id in ("0", "1"):
            workers.append(subprocess.Popen(
                [sys.executable, "-c", worker, str(session_dir), worker_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ))
        ready = [session_dir / f".projection-rebuild-{worker_id}-ready" for worker_id in ("0", "1")]
        deadline = time.monotonic() + 60
        while not all(path.exists() for path in ready):
            exited = [process for process in workers if process.poll() is not None]
            if exited:
                details = [process.communicate() for process in exited]
                raise AssertionError(f"Projection rebuild process exited before the barrier: {details}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Projection rebuild processes did not reach the barrier")
            time.sleep(0.01)
        for process in workers:
            stdout, stderr = process.communicate(timeout=90)
            assert process.returncode == 0, f"Projection rebuild failed: {stdout}\n{stderr}"
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)

    graph = graph_backend.fetch_graph()
    assert not any(node["node_key"] == "stale" for node in graph["nodes"])
    assert bot.memory_manager._semantic_manager._semantic_index_backend.existing_content_hashes()
    status = _reload_projection_status(bot)
    assert all(status[name]["status"] == "sync_succeeded" for name in ("graph", "semantic"))
    assert all(
        status[name]["applied_revision"] == status[name]["source_revision"]
        for name in ("graph", "semantic")
    )


@pytest.mark.parametrize("same_source", [False, True])
def test_late_failure_preserves_newer_applied_status(make_test_dadbot, monkeypatch, same_source):
    from copy import deepcopy

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    storage = bot.memory_manager._storage
    old = storage._load_projection_recovery_status()["semantic"]
    current = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    state = deepcopy(current.state)
    state["last_mood"] = "positive"
    bot._execution_ledger.commit_tenant_memory(
        tenant_id=bot.config.tenant_id, expected_revision=current.revision,
        operation_id="newer-rebuild-source", state=state,
    )
    storage._tenant_base_revision = current.revision + 1
    storage.recover_local_projections()
    published = _reload_projection_status(bot)["semantic"]
    storage._persist_projection_recovery_status("semantic", {
        **old, "generation": published["generation"] + 1,
        "source_revision": published["source_revision"] if same_source else old["source_revision"],
        "status": "degraded", "error": "Late failure for an older revision",
    })
    restored = _reload_projection_status(bot)["semantic"]
    assert restored["status"] == ("degraded" if same_source else "sync_succeeded")
    assert restored["applied_revision"] == published["applied_revision"]


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
@pytest.mark.parametrize("prior_success", [False, True])
def test_legacy_status_infers_only_known_applied_revisions(make_test_dadbot, monkeypatch, outcome, prior_success):
    from dadbot.memory.semantic_manager import SemanticIndexManager

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    if not prior_success:
        def fail_initial_sync(_self, _memories):
            raise RuntimeError("No previous semantic publication")

        monkeypatch.setattr(SemanticIndexManager, "sync_semantic_memory_index", fail_initial_sync)
    bot = make_test_dadbot()
    previous = bot.memory_manager._storage._load_projection_recovery_status()["semantic"]
    _append_status_event(bot, {
        "status": outcome, "generation": previous["generation"] + 1,
        "source_revision": previous["source_revision"], "updated_at": previous["updated_at"],
    })
    restored = _reload_projection_status(bot)["semantic"]
    expected = previous["source_revision"] if outcome == "sync_succeeded" or prior_success else None
    assert restored["applied_revision"] == expected
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
