from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from dadbot.core.tenant_memory_ledger import TenantMemoryRevisionConflict
from tests.unit.test_local_tenant_establishment_application import _environment


def test_mutation_survives_restart(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "sessions"
    memory = tmp_path / "memory.json"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    first.mutate_memory_store(last_mood="positive")
    assert first._execution_ledger.replay_tenant_memory(first.config.tenant_id).revision == 2
    first.shutdown()
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    assert second.MEMORY_STORE["last_mood"] == "positive"
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 2
    assert json.loads(memory.read_text(encoding="utf-8"))["last_mood"] == "positive"


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
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    monkeypatch.setattr(SemanticIndexManager, "embed_texts", lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts])
    original = SQLiteGraphStore.replace_graph

    def fail_graph(self, nodes, edges):
        raise OSError("graph unavailable")

    monkeypatch.setattr(SQLiteGraphStore, "replace_graph", fail_graph)
    bot = make_test_dadbot(memory_path=memory)
    status = bot.memory_manager._storage._projection_recovery_status
    assert bot.lifecycle_state == "ready"
    assert status["graph"]["status"] == "degraded"
    assert status["semantic"] == {"status": "sync_succeeded", "source_revision": 1}
    assert bot.memory_manager._semantic_manager._semantic_index_backend.existing_content_hashes()
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1
    monkeypatch.setattr(SQLiteGraphStore, "replace_graph", original)


def test_semantic_startup_failure_keeps_graph_available_and_reports_degraded(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.memory.semantic_manager import SemanticIndexManager

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")

    def fail_embeddings(self, texts, purpose=None):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(SemanticIndexManager, "embed_texts", fail_embeddings)
    bot = make_test_dadbot(memory_path=memory)
    status = bot.memory_manager._storage._projection_recovery_status
    assert bot.lifecycle_state == "ready"
    assert status["semantic"]["status"] == "degraded"
    assert "embedding unavailable" in status["semantic"]["error"]
    assert status["graph"] == {"status": "sync_succeeded", "source_revision": 1}
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


def test_memory_addition_fails_closed_until_indexes_are_fenced(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    before = bot.MEMORY_STORE
    with pytest.raises(RuntimeError, match="semantic or graph repair"):
        bot.mutate_memory_store(memories=[{"id": "test", "summary": "example"}])
    assert bot.MEMORY_STORE == before
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1


def test_clear_with_indexed_memories_does_not_append_without_fenced_repairs(
    make_test_dadbot, monkeypatch, tmp_path,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    bot = make_test_dadbot(memory_path=memory)
    before = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    with pytest.raises(RuntimeError, match="semantic or graph repair"):
        bot.clear_memory_store()
    after = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id)
    assert after.revision == before.revision == 1
    assert after.state == before.state == bot.MEMORY_STORE


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
    import dadbot.memory.graph_views as graph_views

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


def test_semantic_index_rejects_stale_catalog_and_clear(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    bot = make_test_dadbot(memory_path=memory)
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
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    bot = make_test_dadbot(memory_path=memory)
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
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    bot = make_test_dadbot(memory_path=memory)
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
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    sessions = tmp_path / "sessions"
    first = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
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
    import dadbot.memory.graph_views as graph_views

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

    def blocked_build(manager, turn_context=None):
        snapshot = build(manager, turn_context=turn_context)
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
    assert first.MEMORY_STORE["last_mood"] == "positive"
    with pytest.raises(RuntimeError, match="requires restart"):
        first.mutate_memory_store(last_mood="negative")
    monkeypatch.setattr(first.memory_manager._storage, "_repair_established_snapshot", original)
    second = make_test_dadbot(memory_path=memory, session_log_dir=sessions)
    assert second.MEMORY_STORE["last_mood"] == "positive"
    assert second._execution_ledger.replay_tenant_memory(second.config.tenant_id).revision == 2
    # The committed state is already published; a no-op shutdown performs no write.
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
