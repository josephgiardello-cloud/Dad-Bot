from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from dadbot.core.tenant_memory_ledger import TenantMemoryRevisionConflict
from dadbot_system.state import InMemoryStateStore
from tests.unit.test_local_tenant_establishment_application import _environment


def _owned_document_store(tenant_id, state):
    store = InMemoryStateStore()
    store.owned_sessions.save_session_state(tenant_id, "memory", state)
    return store


def test_mutation_survives_restart(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    first.mutate_memory_store(last_mood="positive")
    assert first._execution_ledger.replay_tenant_memory(first.config.tenant_id).revision == 2
    first.shutdown()
    memory.write_text(json.dumps({"last_mood": "stale-mirror"}), encoding="utf-8")
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    assert second.MEMORY_STORE["last_mood"] == "positive"
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 2
    assert json.loads(memory.read_text(encoding="utf-8"))["last_mood"] == "stale-mirror"


def test_catalog_reads_committed_memory_without_index_writes_or_discarding_staged_changes(
    make_test_dadbot, monkeypatch,
):
    from dadbot.core.execution_context import close_core_state_scope, open_core_state_scope

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    bot.mutate_memory_store(memories=[{"id": "committed", "summary": "Committed preference"}])
    committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    queued = []
    monkeypatch.setattr(bot.memory_manager, "queue_semantic_memory_index", queued.append)
    token = open_core_state_scope(bot.memory_manager.memory_core_state())
    try:
        staged = bot.mutate_memory_store(
            save=False, memories=[{"id": "staged", "summary": "Uncommitted preference"}],
        )
        assert bot.memory_catalog() == committed.state["memories"]
        assert bot.memory_manager.memory_projection() == staged
        assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).state == committed.state
        assert not queued
    finally:
        close_core_state_scope(token)


def test_memory_reads_revalidate_runtime_principal(make_test_dadbot, tmp_path):
    import time

    from dadbot_system.delegation import SQLiteDelegationStore, runtime_authority_scope
    from dadbot_system.security import AuthorizationError, ServicePrincipal

    tenant_a = "t1_" + "a" * 26
    tenant_b = "t1_" + "b" * 26
    bot = make_test_dadbot(tenant_id=tenant_a)
    authority_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(authority_path)
    store = SQLiteDelegationStore(authority_path, "memory-authority-test-secret")
    principal = ServicePrincipal(
        subject="memory-reader", tenant_id=tenant_b, scopes=frozenset({"write"}),
        allowed_tools=frozenset(), issuer="dadbot", expires_at=time.time() + 3600, token_id="memory-reader",
    )
    store.grant_membership(issuer=principal.issuer, subject=principal.subject, tenant_id=tenant_b, scopes={"write"})
    with runtime_authority_scope(store, principal, scope="write", tenant_id=tenant_b):
        with pytest.raises(AuthorizationError, match="tenant mismatch"):
            bot.memory_manager.memory_projection()


@pytest.mark.parametrize("entrypoint", ["projection", "provider"])
def test_memory_reads_reject_missing_required_authority(make_test_dadbot, entrypoint):
    from dadbot.runtime.semantic_memory_bridge import SemanticRetrievalHook
    from dadbot_system.delegation import require_live_effect_authority
    from dadbot_system.security import AuthorizationError

    bot = make_test_dadbot()
    hook = SemanticRetrievalHook(bot)
    with require_live_effect_authority():
        with pytest.raises(AuthorizationError):
            if entrypoint == "projection":
                bot.memory_manager.memory_projection()
            else:
                hook("tea")


def test_recall_uses_committed_insights_and_index_while_memory_changes_are_staged(
    make_test_dadbot, monkeypatch,
):
    from dadbot.core.execution_context import close_core_state_scope, open_core_state_scope
    from dadbot.runtime.semantic_memory_bridge import build_hybrid_memory_recall_provider

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    bot.mutate_memory_store(
        memories=[{"summary": "Committed preference"}],
        consolidated_memories=[{"summary": "Committed insight"}],
    )
    committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    bot.sync_semantic_memory_index(committed.state["memories"])
    observed = []
    def detect(_memories, insights):
        observed.extend(insights)
        return []

    monkeypatch.setattr(bot.memory_coordinator, "detect_memory_contradictions", detect)
    token = open_core_state_scope(bot.memory_manager.memory_core_state())
    try:
        staged = bot.mutate_memory_store(
            save=False,
            memories=[{"summary": "Uncommitted preference"}],
            consolidated_memories=[{"summary": "Uncommitted insight"}],
        )
        build_hybrid_memory_recall_provider(bot)("preference")
        assert observed == committed.state["consolidated_memories"]
        with bot.memory_manager._semantic_manager.verified_semantic_index_read(committed.state["memories"]) as current:
            assert current
        assert bot.memory_manager.memory_projection() == staged
        assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id) == committed
    finally:
        close_core_state_scope(token)


@pytest.mark.parametrize("entrypoint", ["facade", "manager"])
def test_semantic_recall_does_not_schedule_index_writes(make_test_dadbot, monkeypatch, entrypoint):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    bot.mutate_memory_store(memories=[{"id": "committed", "summary": "saving money"}])
    semantic = bot.memory_manager._semantic_manager
    queued = []
    monkeypatch.setattr(semantic, "queue_semantic_memory_index", queued.append)

    target = bot if entrypoint == "facade" else semantic
    target.semantic_memory_matches("saving money", bot.memory_catalog())

    assert not queued


@pytest.mark.parametrize("structured", [False, True])
def test_hybrid_observation_does_not_publish_access_updates(make_test_dadbot, monkeypatch, structured):
    from dadbot.runtime.semantic_memory_bridge import (
        build_hybrid_memory_recall_provider,
        build_hybrid_memory_snippet_provider,
    )

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    bot.mutate_memory_store(memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}])
    assert bot.wait_for_semantic_index_idle(timeout=30)
    before = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    provider = build_hybrid_memory_recall_provider if structured else build_hybrid_memory_snippet_provider
    result = provider(bot)("apple")
    snippets = result.snippets if structured else result
    assert len(snippets) == 1
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id) == before


def test_semantic_observation_reads_real_ranked_results(make_test_dadbot, monkeypatch):
    from dadbot.runtime.semantic_memory_bridge import SemanticRetrievalHook

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()

    def embeddings(texts, **_kwargs):
        return [[1.0, 0.0] for _ in ([texts] if isinstance(texts, str) else texts)]

    monkeypatch.setattr(bot.memory_manager._semantic_manager, "embed_texts", embeddings)
    bot.mutate_memory_store(memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}])
    assert bot.wait_for_semantic_index_idle(timeout=30)
    before = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    snippets = SemanticRetrievalHook(bot)("apple")
    assert len(snippets) == 1
    assert "apple" in snippets[0]
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id) == before


@pytest.mark.parametrize("failure", ["authorization", "authority_unavailable", "corrupt_wal", "corrupt_memory"])
@pytest.mark.parametrize("entrypoint", ["facade", "manager"])
def test_semantic_recall_does_not_hide_authority_errors(make_test_dadbot, monkeypatch, failure, entrypoint):
    from dadbot.core.ledger_backend import WALCorruptionError
    from dadbot.core.tenant_memory_ledger import TenantMemoryLedgerCorruption
    from dadbot_system.delegation import AuthorityUnavailableError
    from dadbot_system.security import AuthorizationError

    errors = {
        "authorization": AuthorizationError,
        "authority_unavailable": AuthorityUnavailableError,
        "corrupt_wal": WALCorruptionError,
        "corrupt_memory": TenantMemoryLedgerCorruption,
    }
    error = errors[failure]
    bot = make_test_dadbot()
    semantic = bot.memory_manager._semantic_manager
    monkeypatch.setattr(semantic, "queue_semantic_memory_index", lambda _memories: None)
    monkeypatch.setattr(semantic, "semantic_query_context", lambda *_args: {"query_embedding": [1.0]})

    def fail(_memories):
        raise error("Recall authority rejected")

    monkeypatch.setattr(semantic, "verified_semantic_index_read", fail)
    target = bot if entrypoint == "facade" else semantic
    with pytest.raises(error, match="Recall authority rejected"):
        target.semantic_memory_matches("saving money", [{"summary": "saving money"}])


def test_save_boundary_mutations_reuse_byte_verified_head_without_full_rehydration(make_test_dadbot, monkeypatch):
    from dadbot.core.execution_ledger import ExecutionLedger

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    ledger = bot._execution_ledger
    backend = ledger._backend
    # Establish a verified append prefix using an ordinary canonical event.
    ledger.write({
        "type": "UI_RECONNECT_AUDIT", "tenant_id": bot.config.tenant_id, "session_id": "audit",
        "trace_id": "verified-save-test", "payload": {}, "kernel_step_id": "audit.test", "timestamp": 1.0,
    })
    loads = []
    rehydrate = ExecutionLedger._rehydrate_loaded_events

    def capture_rehydration(self, *args, **kwargs):
        loads.append(True)
        return rehydrate(self, *args, **kwargs)

    monkeypatch.setattr(ExecutionLedger, "_rehydrate_loaded_events", capture_rehydration)
    for mood in ("positive", "neutral", "positive"):
        bot.mutate_memory_store(last_mood=mood)
    assert loads == []
    assert ledger.replay_tenant_memory(bot.config.tenant_id).revision == 4
    assert ledger.replay_tenant_memory(bot.config.tenant_id).state["last_mood"] == "positive"
    with bot.memory_manager._storage.fenced_graph_index_write(
        4, bot.memory_manager.normalize_memory_store(ledger.replay_tenant_memory(bot.config.tenant_id).state),
    ):
        pass
    assert loads == []
    assert ledger.verify_replay(mode="chain")["ok"]
    restored = ExecutionLedger(type(backend)(backend._path))
    restored.load_from_backend()
    assert restored.read() == ledger.read()


def test_local_commit_keeps_shared_execution_ledger_head_atomic(make_test_dadbot, monkeypatch):
    from dadbot.core import local_tenant_establishment

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    ledger = bot._execution_ledger
    committed = Event()
    release = Event()
    attempted = Event()
    finished = Event()
    original = local_tenant_establishment.commit_local_tenant

    def pause_after_wal_commit(*args, **kwargs):
        revision = original(*args, **kwargs)
        committed.set()
        assert release.wait(10)
        return revision

    def write_audit():
        attempted.set()
        try:
            return ledger.write({
                "type": "UI_RECONNECT_AUDIT", "tenant_id": bot.config.tenant_id,
                "session_id": "audit", "trace_id": "atomic-head-test", "payload": {},
                "kernel_step_id": "audit.test", "timestamp": 1.0,
            })
        finally:
            finished.set()

    monkeypatch.setattr(local_tenant_establishment, "commit_local_tenant", pause_after_wal_commit)
    with ThreadPoolExecutor(max_workers=2) as executor:
        mutation = executor.submit(bot.mutate_memory_store, last_mood="positive")
        try:
            assert committed.wait(10)
            audit = executor.submit(write_audit)
            assert attempted.wait(10)
            if finished.wait(0.2):
                audit.result(timeout=10)
                pytest.fail("Audit append observed an unpublished in-memory WAL head")
        finally:
            release.set()
        mutation.result(timeout=10)
        assert audit.result(timeout=10)["type"] == "UI_RECONNECT_AUDIT"
    ledger.load_from_backend()
    state = ledger.replay_tenant_memory(bot.config.tenant_id).state
    assert isinstance(state, dict)
    assert state["last_mood"] == "positive"
    assert sum(event["type"] == "UI_RECONNECT_AUDIT" for event in ledger.read()) == 1


def test_startup_rebuilds_stale_graph_from_ledger(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    graph_path = tmp_path / "graph.sqlite"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions, graph_store_db_path=graph_path)
    graph = first.memory_manager.graph_manager
    graph._graph_store_backend.replace_graph(
        [{"node_key": "stale", "node_type": "topic", "label": "deleted"}], [],
    )
    first.shutdown()
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions, graph_store_db_path=graph_path)
    persisted = second.memory_manager.graph_manager._graph_store_backend.fetch_graph()
    assert not any(node["node_key"] == "stale" for node in persisted["nodes"])
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 1


def test_startup_recovers_semantic_and_graph_independently(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.memory.semantic_manager import SemanticIndexManager
    from dadbot_system.graph_store import SQLiteGraphStore

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    tenant_id = "tenant-semantic-recovery"
    document_store = _owned_document_store(
        tenant_id,
        {"memories": [{"summary": "Example memory"}]},
    )
    monkeypatch.setattr(
        SemanticIndexManager,
        "embed_texts",
        lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts],
    )
    original = SQLiteGraphStore.replace_graph

    def fail_graph(self, nodes, edges):
        raise OSError("graph unavailable")

    monkeypatch.setattr(SQLiteGraphStore, "replace_graph", fail_graph)
    bot = make_test_dadbot(memory_path=memory, tenant_id=tenant_id, document_store=document_store)
    status = bot.memory_manager._storage._projection_recovery_status
    assert bot.lifecycle_state == "ready"
    assert status["graph"]["status"] == "degraded"
    assert status["semantic"]["status"] == "sync_succeeded"
    assert status["semantic"]["source_revision"] == 1
    assert status["semantic"]["generation"] >= 1
    assert status["semantic"]["updated_at"] > 0
    assert bot.memory_manager._semantic_manager._semantic_index_backend.existing_content_hashes()
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1
    monkeypatch.setattr(SQLiteGraphStore, "replace_graph", original)


def test_semantic_startup_failure_keeps_graph_available_and_reports_degraded(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.memory.semantic_manager import SemanticIndexManager

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    tenant_id = "tenant-semantic-recovery"
    document_store = _owned_document_store(
        tenant_id,
        {"memories": [{"summary": "Example memory"}]},
    )

    def fail_embeddings(self, texts, purpose=None):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(SemanticIndexManager, "embed_texts", fail_embeddings)
    bot = make_test_dadbot(memory_path=memory, tenant_id=tenant_id, document_store=document_store)
    status = bot.memory_manager._storage._projection_recovery_status
    assert bot.lifecycle_state == "ready"
    assert status["semantic"]["status"] == "degraded"
    assert "embedding unavailable" in status["semantic"]["error"]
    assert status["graph"]["status"] == "sync_succeeded"
    assert status["graph"]["source_revision"] == 1
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1


def test_process_exit_during_startup_graph_rebuild_recovers_on_restart(tmp_path):
    env = _environment(tmp_path)
    initial = r"""
from dadbot.core.dadbot import DadBot
bot = DadBot()
bot.memory_manager.graph_manager._graph_store_backend.replace_graph(
    [{"node_key": "stale", "node_type": "topic", "label": "deleted"}], [])
bot.shutdown()
"""
    assert subprocess.run([sys.executable, "-c", initial], env=env, timeout=60, check=False).returncode == 0
    interrupted = r"""
import os
from dadbot_system.graph_store import SQLiteGraphStore
def terminate(self, nodes, edges):
    def transaction(connection):
        connection.execute("DELETE FROM graph_nodes")
        os._exit(73)
    self.with_connection(transaction, write=True)
SQLiteGraphStore.replace_graph = terminate
from dadbot.core.dadbot import DadBot
DadBot()
"""
    assert subprocess.run([sys.executable, "-c", interrupted], env=env, timeout=60, check=False).returncode == 73
    verify = r"""
from dadbot.core.dadbot import DadBot
bot = DadBot()
graph = bot.memory_manager.graph_manager._graph_store_backend.fetch_graph()
assert not any(node["node_key"] == "stale" for node in graph["nodes"]), graph
assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1
bot.shutdown()
"""
    assert subprocess.run([sys.executable, "-c", verify], env=env, timeout=60, check=False).returncode == 0


def test_staged_direct_save_commits_new_revision(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    bot.mutate_memory_store(last_mood="positive", save=False)
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1
    bot.save_memory_store()
    committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    assert committed.revision == 2
    assert committed.state == bot.MEMORY_STORE


def test_clear_mood_only_tenant_commits_and_survives_restart(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    first.mutate_memory_store(last_mood="positive")
    first.clear_memory_store()
    committed = first._execution_ledger.replay_tenant_memory(first.config.tenant_id)
    assert committed.revision == 3
    assert committed.state == first.MEMORY_STORE
    assert committed.state["last_mood"] == "neutral"
    first.shutdown()
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    assert second.MEMORY_STORE == committed.state
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 3


def test_indexed_memory_addition_commits_and_repairs_current_revision(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    bot.mutate_memory_store(memories=[{"id": "test", "summary": "example"}])
    committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    assert committed.revision == 2
    assert committed.state == bot.MEMORY_STORE
    assert bot.memory_manager._semantic_manager._semantic_index_backend.existing_content_hashes()
    status = bot.memory_manager._storage._load_projection_recovery_status()
    assert all(item["status"] == "sync_succeeded" for item in status.values())
    assert all(item["source_revision"] == 2 for item in status.values())


def test_indexed_updates_and_staged_save_remove_obsolete_semantic_rows(make_test_dadbot, monkeypatch):
    import hashlib

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    semantic = bot.memory_manager._semantic_manager
    bot.mutate_memory_store(memories=[{"id": "test", "summary": "first example"}], save=False)
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1
    bot.save_memory_store()
    bot.mutate_memory_store(memories=[{"id": "test", "summary": "updated example"}])
    catalog = bot.MEMORY_STORE["memories"]
    expected = {
        semantic.semantic_memory_key(memory): hashlib.sha256(
            semantic.memory_embedding_text(memory).encode("utf-8")
        ).hexdigest()
        for memory in catalog
    }
    assert semantic._semantic_index_backend.existing_content_hashes() == expected
    bot.mutate_memory_store(memories=[])
    assert semantic._semantic_index_backend.existing_content_hashes() == {}
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 4


def test_tenants_do_not_share_configured_sqlite_projection_files(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.core.local_tenant_establishment import local_projection_path

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    paths = {
        "session_log_dir": tmp_path / "sessions",
        "semantic_memory_db_path": tmp_path / "semantic.sqlite3",
        "graph_store_db_path": tmp_path / "graph.sqlite3",
    }
    first = make_test_dadbot(tenant_id="tenant-a", **paths)
    second = make_test_dadbot(tenant_id="tenant-b", **paths)
    first.mutate_memory_store(memories=[{"id": "shared", "summary": "first tenant example"}])
    second.mutate_memory_store(memories=[{"id": "shared", "summary": "second tenant example"}])
    second_index = second.memory_manager._semantic_manager._semantic_index_backend
    expected = second_index.existing_content_hashes()
    assert expected
    assert local_projection_path(first, paths["semantic_memory_db_path"]) != local_projection_path(
        second, paths["semantic_memory_db_path"],
    )
    assert local_projection_path(first, paths["graph_store_db_path"]) != local_projection_path(
        second, paths["graph_store_db_path"],
    )
    first.clear_memory_store()
    assert second_index.existing_content_hashes() == expected
    assert second._execution_ledger.replay_tenant_memory("tenant-b").revision == 2


def test_unowned_legacy_indexes_are_preserved_and_rebuilt_from_ledger(make_test_dadbot, monkeypatch, tmp_path):
    from pathlib import Path

    from dadbot.core.local_tenant_establishment import local_projection_path

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    semantic = tmp_path / "semantic.sqlite3"
    graph = tmp_path / "graph.sqlite3"
    semantic.write_bytes(b"unowned legacy semantic data")
    graph.write_bytes(b"unowned legacy graph data")
    bot = make_test_dadbot(semantic_memory_db_path=semantic, graph_store_db_path=graph)
    bot.mutate_memory_store(memories=[{"id": "test", "summary": "example"}])
    assert semantic.read_bytes() == b"unowned legacy semantic data"
    assert graph.read_bytes() == b"unowned legacy graph data"
    assert Path(local_projection_path(bot, semantic)).exists()
    assert Path(local_projection_path(bot, graph)).exists()


def test_unsupported_index_backend_is_rejected_before_tenant_commit(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    with monkeypatch.context() as backend_patch:
        backend_patch.setattr(bot.memory_manager._semantic_manager, "_semantic_index_backend", object())
        with pytest.raises(RuntimeError, match="tenant-isolated SQLite"):
            bot.mutate_memory_store(memories=[{"id": "test", "summary": "example"}])
    assert not bot.MEMORY_STORE["memories"]
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1


@pytest.mark.parametrize("projection", ["graph", "semantic"])
def test_unqualified_injected_sqlite_index_is_rejected_before_commit(
    make_test_dadbot, monkeypatch, tmp_path, projection,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    backend = (
        bot.memory_manager._semantic_manager._semantic_index_backend
        if projection == "semantic" else bot.memory_manager.graph_manager._graph_store_backend
    )
    with monkeypatch.context() as backend_patch:
        backend_patch.setattr(backend, "db_path", tmp_path / "unowned.sqlite3")
        with pytest.raises(RuntimeError, match="tenant-qualified"):
            bot.mutate_memory_store(memories=[{"id": "test", "summary": "example"}])
    assert not bot.MEMORY_STORE["memories"]
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1


@pytest.mark.parametrize("point", ["snapshot", "graph", "semantic"])
def test_indexed_mutation_process_termination_repairs_committed_revision(tmp_path, point):
    env = _environment(tmp_path)
    env["DADBOT_GRAPH_ENABLE_LLM_EXTRACTION"] = "0"
    setup = r"""
import os
from pathlib import Path
from dadbot.core.dadbot import DadBot
from dadbot.memory.semantic_manager import SemanticIndexManager
from dadbot.services.vector_memory import build_sovereign_memory
SemanticIndexManager.embed_texts = lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts]
root = Path(os.environ["DADBOT_MEMORY_PATH"]).parent
bot = DadBot(service_overrides={"sovereign_memory": build_sovereign_memory(root / "vector-store")})
"""
    interrupted_code = setup + r"""
import sys
def terminate(*_args, **_kwargs):
    os._exit(73)
point = sys.argv[1]
if point == "snapshot":
    bot.memory_manager._storage._repair_established_snapshot = terminate
elif point == "graph":
    bot.memory_manager.graph_manager._graph_store_backend.replace_graph = terminate
else:
    bot.memory_manager._semantic_manager._semantic_index_backend.apply_changes = terminate
bot.mutate_memory_store(memories=[{"id": "test", "summary": "durable example"}])
"""
    interrupted = subprocess.run(
        [sys.executable, "-c", interrupted_code, point], env=env, timeout=60, check=False,
        capture_output=True, text=True,
    )
    assert interrupted.returncode == 73, interrupted.stderr
    verify = setup + r"""
committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
assert committed.revision == 2
assert bot.MEMORY_STORE == committed.state
expected = bot.memory_manager.normalize_memory_store({"memories": [{"id": "test", "summary": "durable example"}]})
assert bot.MEMORY_STORE["memories"][0]["summary"] == expected["memories"][0]["summary"]
assert bot.memory_manager._semantic_manager._semantic_index_backend.existing_content_hashes()
status = bot.memory_manager._storage._load_projection_recovery_status()
assert all(item["status"] == "sync_succeeded" for item in status.values()), status
assert all(item["source_revision"] == 2 for item in status.values()), status
bot.shutdown()
"""
    resumed = subprocess.run(
        [sys.executable, "-c", verify], env=env, timeout=60, check=False,
        capture_output=True, text=True,
    )
    assert resumed.returncode == 0, resumed.stderr


def test_clear_with_indexed_memories_commits_and_clears_derived_indexes(
    make_test_dadbot, monkeypatch, tmp_path,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    tenant_id = "tenant-semantic-recovery"
    document_store = _owned_document_store(
        tenant_id,
        {"memories": [{"summary": "Example memory"}]},
    )
    bot = make_test_dadbot(memory_path=memory, tenant_id=tenant_id, document_store=document_store)
    bot.clear_memory_store()
    after = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    assert after.revision == 2
    assert after.state == bot.MEMORY_STORE
    assert not after.state["memories"]
    assert not bot.memory_manager._semantic_manager._semantic_index_backend.existing_content_hashes()
    assert not bot.memory_manager.graph_manager._graph_store_backend.fetch_graph()["nodes"]


@pytest.mark.parametrize("failed_projection", ["graph", "semantic"])
def test_index_repair_failure_preserves_commit_and_recovers_after_restart(
    make_test_dadbot, monkeypatch, tmp_path, failed_projection,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    paths = {
        "session_log_dir": tmp_path / "sessions",
        "memory_path": tmp_path / "memory.json",
        "semantic_memory_db_path": tmp_path / "semantic.sqlite3",
        "graph_store_db_path": tmp_path / "graph.sqlite3",
    }
    bot = make_test_dadbot(**paths)
    with monkeypatch.context() as repair_patch:
        def fail(*_args, **_kwargs):
            raise OSError(f"{failed_projection} repair unavailable")

        if failed_projection == "graph":
            repair_patch.setattr(bot.memory_manager.graph_manager, "sync_graph_store", fail)
        else:
            repair_patch.setattr(bot.memory_manager, "sync_semantic_memory_index", fail)
        with pytest.raises(RuntimeError, match=r"committed.*revision 2.*repair"):
            bot.mutate_memory_store(memories=[{"id": "test", "summary": "example"}])
        committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
        assert committed.revision == 2
        assert committed.state == bot.MEMORY_STORE
        status = bot.memory_manager._storage._load_projection_recovery_status()
        assert status[failed_projection]["status"] == "degraded"
        assert status[failed_projection]["source_revision"] == 2
        other = "semantic" if failed_projection == "graph" else "graph"
        assert status[other]["status"] == "sync_succeeded"
        with pytest.raises(RuntimeError, match="restart"):
            bot.mutate_memory_store(last_mood="positive")
    bot.shutdown()
    recovered = make_test_dadbot(**paths)
    assert recovered.MEMORY_STORE == committed.state
    assert recovered._execution_ledger.replay_tenant_memory(recovered.config.tenant_id).revision == 2
    status = recovered.memory_manager._storage._load_projection_recovery_status()
    assert all(item["status"] == "sync_succeeded" for item in status.values())


def test_direct_graph_clear_rejects_established_tenant(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    graph = bot.memory_manager.graph_manager
    with pytest.raises(RuntimeError, match="ledger-authoritative"):
        graph.clear_graph_store()
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1


def test_direct_graph_edge_invalidation_rejects_established_tenant(make_test_dadbot, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    graph = bot.memory_manager.graph_manager
    bot._graph_commit_active = True
    context = SimpleNamespace(state={"_active_graph_stage": "save"})
    snapshot = {
        "nodes": [{"node_key": "item", "node_type": "consolidated_memory", "label": "example"}],
        "edges": [{"source_key": "item"}],
    }
    monkeypatch.setattr(graph, "graph_snapshot", lambda: snapshot)
    with pytest.raises(RuntimeError, match="ledger-authoritative"):
        bot.memory_coordinator._invalidate_superseded_graph_edges(
            {"summary": "example"}, "2026-01-01", turn_context=context,
        )
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1


def test_local_graph_read_does_not_return_unverified_persisted_graph(make_test_dadbot, monkeypatch):
    from dadbot.memory import graph_views

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    graph = bot.memory_manager.graph_manager
    expected = {"nodes": [], "edges": [], "updated_at": None}
    monkeypatch.setattr(graph_views, "build_graph_projection", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        graph._graph_store_backend,
        "fetch_graph",
        lambda: {"nodes": [{"label": "deleted"}], "edges": []},
    )
    assert graph.graph_snapshot() == expected


def test_stale_local_graph_read_fails_closed(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    second.mutate_memory_store(last_mood="positive")
    try:
        with pytest.raises(TenantMemoryRevisionConflict, match="Graph projection"):
            first.memory_manager.graph_manager.graph_snapshot()
    finally:
        first.MEMORY_STORE = second.MEMORY_STORE


@pytest.mark.parametrize("changes", [
    {"last_mood": "positive"},
    {"consolidated_memories": [{"summary": "Uncommitted graph fact", "category": "general"}]},
])
def test_local_graph_read_uses_committed_memory_without_discarding_staged_turn_changes(
    make_test_dadbot, monkeypatch, changes,
):
    from dadbot.core.execution_context import close_core_state_scope, open_core_state_scope

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    graph = bot.memory_manager.graph_manager
    expected = graph.graph_snapshot()
    committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    token = open_core_state_scope(bot.memory_manager.memory_core_state())
    try:
        staged = bot.mutate_memory_store(save=False, **changes)
        assert staged != committed.state
        assert graph.graph_snapshot() == expected
        assert bot.memory_manager.memory_projection() == staged
        assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).state == committed.state
    finally:
        close_core_state_scope(token)


def test_local_graph_refresh_indexes_committed_state_without_publishing_staged_changes(
    make_test_dadbot, monkeypatch,
):
    from dadbot.core.execution_context import close_core_state_scope, open_core_state_scope

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    graph = bot.memory_manager.graph_manager
    expected = graph.graph_snapshot()
    committed = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    token = open_core_state_scope(bot.memory_manager.memory_core_state())
    try:
        staged = bot.mutate_memory_store(
            save=False,
            consolidated_memories=[{"summary": "Uncommitted graph fact", "category": "general"}],
        )
        assert graph.sync_graph_store() == expected
        assert bot.memory_manager.memory_projection() == staged
        assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).state == committed.state
        persisted = graph._graph_store_backend.fetch_graph()
        assert persisted["nodes"] == expected["nodes"]
        assert persisted["edges"] == expected["edges"]
    finally:
        close_core_state_scope(token)


def test_runtime_graph_refresh_does_not_report_revision_degradation(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    graph = bot.refresh_memory_graph(force=True)
    assert isinstance(graph, dict)
    assert bot._memory_graph_dirty is False
    assert not any(issue["purpose"] == "memory graph refresh" for issue in bot.recent_runtime_issues())


def test_semantic_index_rejects_stale_catalog_and_clear(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    tenant_id = "tenant-semantic-recovery"
    document_store = _owned_document_store(
        tenant_id,
        {"memories": [{"summary": "Example memory"}]},
    )
    bot = make_test_dadbot(memory_path=memory, tenant_id=tenant_id, document_store=document_store)
    semantic = bot.memory_manager._semantic_manager
    monkeypatch.setattr(semantic, "embed_texts", lambda texts, purpose=None: [[1.0] * 12 for _ in texts])
    catalog = bot.memory_manager.memory_projection()["memories"]
    semantic.sync_semantic_memory_index(catalog)
    keys = set(semantic._semantic_index_backend.existing_content_hashes())
    assert keys
    with pytest.raises(RuntimeError, match="does not match committed"):
        semantic.sync_semantic_memory_index([])
    with pytest.raises(RuntimeError, match="does not match committed"):
        semantic.clear_semantic_memory_index()
    assert set(semantic._semantic_index_backend.existing_content_hashes()) == keys
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1


def test_stale_semantic_content_is_not_used_for_local_retrieval(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    tenant_id = "tenant-semantic-recovery"
    document_store = _owned_document_store(
        tenant_id,
        {"memories": [{"summary": "Example memory"}]},
    )
    bot = make_test_dadbot(memory_path=memory, tenant_id=tenant_id, document_store=document_store)
    manager = bot.memory_manager
    semantic = manager._semantic_manager
    monkeypatch.setattr(semantic, "embed_texts", lambda texts, purpose=None: [[1.0] * 12 for _ in texts])
    memories = manager.memory_projection()["memories"]
    semantic.sync_semantic_memory_index(memories)
    semantic._semantic_index_backend.with_connection(
        lambda connection: connection.execute("UPDATE semantic_memories SET content_hash = 'stale'"),
        write=True,
    )
    assert manager.semantic_memory_matches("example", memories) == []
    assert semantic._last_retrieval_status["degraded"] is True
    assert semantic._last_retrieval_status["reason"] == "stale_index"


def test_semantic_repair_rejects_catalog_after_external_commit(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.core.execution_ledger import ExecutionLedger
    from dadbot.core.ledger_backend import FileWALLedgerBackend

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    tenant_id = "tenant-semantic-recovery"
    document_store = _owned_document_store(
        tenant_id,
        {"memories": [{"summary": "Example memory"}]},
    )
    bot = make_test_dadbot(
        memory_path=memory,
        tenant_id=tenant_id,
        document_store=document_store,
    )
    semantic = bot.memory_manager._semantic_manager
    monkeypatch.setattr(semantic, "embed_texts", lambda texts, purpose=None: [[1.0] * 12 for _ in texts])
    stale = bot.memory_manager.memory_projection()["memories"]
    semantic.sync_semantic_memory_index(stale)
    other = ExecutionLedger(FileWALLedgerBackend(bot.config.session_log_dir / "execution_ledger.jsonl"))
    other.load_from_backend()
    current = bot.memory_manager.normalize_memory_store({**bot.MEMORY_STORE, "memories": []})
    other.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=1,
        operation_id="external-delete",
        state=current,
    )
    with pytest.raises(RuntimeError, match="does not match committed"):
        semantic.sync_semantic_memory_index(stale)
    semantic.sync_semantic_memory_index([])
    assert semantic._semantic_index_backend.existing_content_hashes() == {}
    other.load_from_backend()
    assert other.replay_tenant_memory(bot.config.tenant_id).revision == 2
    bot.MEMORY_STORE = current


def test_semantic_repair_rejects_candidate_after_concurrent_commit(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    tenant_id = "tenant-semantic-recovery"
    document_store = _owned_document_store(
        tenant_id,
        {"memories": [{"summary": "Example memory"}]},
    )
    sessions = tmp_path / "sessions"
    first = make_test_dadbot(
        memory_path=memory,
        session_log_dir=sessions,
        tenant_id=tenant_id,
        document_store=document_store,
    )
    second = make_test_dadbot(
        memory_path=memory,
        session_log_dir=sessions,
        tenant_id=tenant_id,
        document_store=document_store,
    )
    semantic = first.memory_manager._semantic_manager
    started = Event()
    release = Event()
    commit_started = Event()

    def embed(texts, purpose=None):
        started.set()
        if not release.wait(10):
            raise TimeoutError("semantic repair did not receive release")
        return [[1.0] * 12 for _ in texts]

    def commit():
        commit_started.set()
        second.mutate_memory_store(last_mood="positive")

    semantic._semantic_index_backend.with_connection(
        lambda connection: connection.execute("UPDATE semantic_memories SET content_hash = 'stale'"),
        write=True,
    )
    before = semantic._semantic_index_backend.existing_content_hashes()
    monkeypatch.setattr(semantic, "embed_texts", embed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        repair = pool.submit(semantic.sync_semantic_memory_index, first.MEMORY_STORE["memories"])
        try:
            assert started.wait(10)
            mutation = pool.submit(commit)
            assert commit_started.wait(10)
            mutation.result(timeout=10)
        finally:
            release.set()
        with pytest.raises(TenantMemoryRevisionConflict, match="Semantic candidate"):
            repair.result(timeout=10)
    assert semantic._semantic_index_backend.existing_content_hashes() == before
    second._execution_ledger.load_from_backend()
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 2
    first.MEMORY_STORE = second.MEMORY_STORE


def test_stale_graph_sync_cannot_overwrite_newer_tenant(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    second.mutate_memory_store(last_mood="positive")
    graph = first.memory_manager.graph_manager
    try:
        with pytest.raises(TenantMemoryRevisionConflict, match="Graph projection"):
            graph.sync_graph_store()
    finally:
        first.MEMORY_STORE = second.MEMORY_STORE


def test_graph_rebuild_rechecks_revision_before_replacement(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.memory import graph_views

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    graph = first.memory_manager.graph_manager
    ready = Event()
    release = Event()
    replacement = []
    build = graph_views.build_graph_projection

    def blocked_build(manager, turn_context=None, **kwargs):
        snapshot = build(manager, turn_context=turn_context, **kwargs)
        ready.set()
        if not release.wait(10):
            raise TimeoutError("graph rebuild was not released")
        return snapshot

    monkeypatch.setattr(graph_views, "build_graph_projection", blocked_build)
    monkeypatch.setattr(graph._graph_store_backend, "replace_graph", lambda *args: replacement.append(args))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(graph.sync_graph_store)
        try:
            assert ready.wait(10)
            second.mutate_memory_store(last_mood="positive")
        finally:
            release.set()
        with pytest.raises(TenantMemoryRevisionConflict, match="Graph projection"):
            future.result(timeout=10)
    assert replacement == []
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 2
    first.MEMORY_STORE = second.MEMORY_STORE


def test_staged_save_rejects_external_commit(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    first.mutate_memory_store(last_mood="staged", save=False)
    second.mutate_memory_store(last_mood="positive")
    with pytest.raises(TenantMemoryRevisionConflict):
        first.save_memory_store()
    # Discard the intentionally uncommitted staged state before fixture shutdown.
    first.MEMORY_STORE = second.MEMORY_STORE
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).state["last_mood"] == "positive"


def test_post_append_snapshot_failure_blocks_further_writes_and_recovers(
    make_test_dadbot, monkeypatch, tmp_path,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    original = first.memory_manager._storage._repair_established_snapshot

    def fail(_state):
        raise OSError("snapshot repair failed")

    monkeypatch.setattr(first.memory_manager._storage, "_repair_established_snapshot", fail)
    with pytest.raises(OSError, match="snapshot repair failed"):
        first.mutate_memory_store(last_mood="positive")
    with pytest.raises(RuntimeError, match="uncertain"):
        first._execution_ledger.read()
    assert first.MEMORY_STORE["last_mood"] == "positive"
    with pytest.raises(RuntimeError, match="requires restart"):
        first.mutate_memory_store(last_mood="negative")
    monkeypatch.setattr(first.memory_manager._storage, "_repair_established_snapshot", original)
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    assert second.MEMORY_STORE["last_mood"] == "positive"
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 2
    first._execution_ledger.load_from_backend()
    # Explicit ledger recovery permits a no-op shutdown, not new memory writes.
    first.save_memory_store()
    first.shutdown()


def test_process_exit_after_mutation_append_recovers_on_restart(tmp_path):
    env = _environment(tmp_path)
    script = r"""
import os
from dadbot.core.dadbot import DadBot
bot = DadBot()
def fail_after_append(_state):
    os._exit(73)
bot.memory_manager._storage._repair_established_snapshot = fail_after_append
bot.mutate_memory_store(last_mood="positive")
"""
    interrupted = subprocess.run([sys.executable, "-c", script], env=env, timeout=60, check=False)
    assert interrupted.returncode == 73
    verify = r"""
from dadbot.core.dadbot import DadBot
bot = DadBot()
assert bot.MEMORY_STORE["last_mood"] == "positive"
assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 2
bot.shutdown()
"""
    resumed = subprocess.run([sys.executable, "-c", verify], env=env, timeout=60, check=False)
    assert resumed.returncode == 0
