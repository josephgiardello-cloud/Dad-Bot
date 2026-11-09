from unittest.mock import MagicMock
import sqlite3
import json
import subprocess
import sys

import pytest

from tests.unit.test_local_tenant_establishment_application import _environment


def test_incomplete_embeddings_do_not_report_success_or_delete_stale_entries(bot, monkeypatch):
    manager = bot.memory_manager._semantic_manager
    backend = MagicMock()
    backend.existing_content_hashes.return_value = {"old": "old-hash"}
    monkeypatch.setattr(manager, "_semantic_index_backend", backend)
    monkeypatch.setattr(manager, "ensure_semantic_memory_db", lambda: None)
    monkeypatch.setattr(manager, "semantic_memory_key", lambda memory: memory["key"])
    monkeypatch.setattr(manager, "memory_embedding_text", lambda memory: memory["text"])
    monkeypatch.setattr(manager, "embed_texts", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="embedding batch mismatch"):
        manager.sync_semantic_memory_index([{"key": "new", "text": "replacement"}])

    backend.delete_keys.assert_not_called()
    backend.upsert_rows.assert_not_called()


def test_semantic_index_updates_replacement_before_deleting_stale_entries(bot, monkeypatch):
    manager = bot.memory_manager._semantic_manager
    backend = MagicMock()
    backend.existing_content_hashes.return_value = {"old": "old-hash"}
    monkeypatch.setattr(manager, "_semantic_index_backend", backend)
    monkeypatch.setattr(manager, "ensure_semantic_memory_db", lambda: None)
    monkeypatch.setattr(manager, "semantic_memory_key", lambda memory: memory["key"])
    monkeypatch.setattr(manager, "memory_embedding_text", lambda memory: memory["text"])
    monkeypatch.setattr(manager, "embed_texts", lambda *args, **kwargs: [[1.0, 0.0]])

    manager.sync_semantic_memory_index([{"key": "new", "text": "replacement"}])

    calls = [call[0] for call in backend.method_calls]
    assert calls.index("upsert_rows") < calls.index("delete_keys")
    backend.delete_keys.assert_called_once_with({"old"})


def test_sqlite_semantic_rebuild_rolls_back_replacement_if_delete_fails(bot, monkeypatch):
    manager = bot.memory_manager._semantic_manager
    backend = manager._semantic_index_backend
    monkeypatch.setattr(manager, "embed_texts", lambda texts, purpose=None: [[1.0] * 12 for _ in texts])
    old = [{"summary": "Old indexed memory", "category": "general"}]
    new = [{"summary": "New indexed memory", "category": "general"}]
    manager.sync_semantic_memory_index(old)
    original = backend.existing_content_hashes()
    backend.with_connection(
        lambda connection: connection.execute(
            "CREATE TRIGGER reject_semantic_delete BEFORE DELETE ON semantic_memories "
            "BEGIN SELECT RAISE(ABORT, 'delete rejected'); END"
        ),
        write=True,
    )
    try:
        with pytest.raises(sqlite3.IntegrityError, match="delete rejected"):
            manager.sync_semantic_memory_index(new)
        assert backend.existing_content_hashes() == original
    finally:
        backend.with_connection(lambda connection: connection.execute("DROP TRIGGER reject_semantic_delete"), write=True)


def test_local_semantic_verification_failure_rolls_back_publication(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.memory.semantic_manager import SemanticIndexManager

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    monkeypatch.setattr(SemanticIndexManager, "embed_texts", lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts])
    bot = make_test_dadbot(memory_path=memory)
    semantic = bot.memory_manager._semantic_manager
    backend = semantic._semantic_index_backend
    backend.with_connection(
        lambda connection: connection.execute("UPDATE semantic_memories SET content_hash = 'stale'"),
        write=True,
    )
    before = backend.existing_content_hashes()
    backend.with_connection(
        lambda connection: connection.execute(
            "CREATE TRIGGER alter_semantic_hash AFTER INSERT ON semantic_memories "
            "BEGIN UPDATE semantic_memories SET content_hash = 'invalid' "
            "WHERE summary_key = NEW.summary_key; END"
        ),
        write=True,
    )
    try:
        with pytest.raises(RuntimeError, match="verification failed"):
            semantic.sync_semantic_memory_index(bot.MEMORY_STORE["memories"])
        assert backend.existing_content_hashes() == before
    finally:
        backend.with_connection(lambda connection: connection.execute("DROP TRIGGER alter_semantic_hash"), write=True)


def test_local_semantic_candidate_rejects_changed_index(make_test_dadbot, monkeypatch, tmp_path):
    from dadbot.memory.semantic_manager import SemanticIndexManager

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"memories": [{"summary": "Example memory"}]}), encoding="utf-8")
    monkeypatch.setattr(SemanticIndexManager, "embed_texts", lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts])
    bot = make_test_dadbot(memory_path=memory)
    semantic = bot.memory_manager._semantic_manager
    backend = semantic._semantic_index_backend
    backend.with_connection(
        lambda connection: connection.execute("UPDATE semantic_memories SET content_hash = 'stale'"),
        write=True,
    )

    def embed_and_change_index(texts, purpose=None):
        backend.with_connection(
            lambda connection: connection.execute("UPDATE semantic_memories SET content_hash = 'other-writer'"),
            write=True,
        )
        return [[1.0] * 12 for _ in texts]

    monkeypatch.setattr(semantic, "embed_texts", embed_and_change_index)
    with pytest.raises(RuntimeError, match="changed while building candidate"):
        semantic.sync_semantic_memory_index(bot.MEMORY_STORE["memories"])
    assert set(backend.existing_content_hashes().values()) == {"other-writer"}


def test_process_exit_during_sqlite_semantic_publication_preserves_previous_index(tmp_path):
    (tmp_path / "memory.json").write_text(
        json.dumps({"memories": [{"summary": "Old indexed memory", "category": "general"}]}),
        encoding="utf-8",
    )
    env = _environment(tmp_path)
    interrupted = r"""
import os
from dadbot.core.dadbot import DadBot
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
bot = DadBot()
semantic = bot.memory_manager._semantic_manager
semantic.embed_texts = lambda texts, purpose=None: [[1.0] * 12 for _ in texts]
old = bot.MEMORY_STORE["memories"]
semantic.sync_semantic_memory_index(old)
state = bot.memory_manager.normalize_memory_store({**bot.MEMORY_STORE, "memories": [
    {"summary": "New indexed memory", "category": "general"}
]})
other = ExecutionLedger(FileWALLedgerBackend(bot.config.session_log_dir / "execution_ledger.jsonl"))
other.load_from_backend()
other.commit_tenant_memory(tenant_id=bot.config.tenant_id, expected_revision=1,
                           operation_id="external-index-change", state=state)
original = semantic._semantic_index_backend._upsert_rows_on_connection
def terminate(connection, rows):
    original(connection, rows)
    os._exit(73)
semantic._semantic_index_backend._upsert_rows_on_connection = terminate
semantic.sync_semantic_memory_index(state["memories"])
"""
    result = subprocess.run([sys.executable, "-c", interrupted], env=env, timeout=60, check=False)
    assert result.returncode == 73
    verify = r"""
import os
import sqlite3
with sqlite3.connect(os.environ["DADBOT_SEMANTIC_DB_PATH"]) as connection:
    keys = [row[0] for row in connection.execute("SELECT summary_key FROM semantic_memories")]
assert len(keys) == 1 and any("old" in key.lower() for key in keys), keys
from dadbot.memory.semantic_manager import SemanticIndexManager
SemanticIndexManager.embed_texts = lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts]
from dadbot.core.dadbot import DadBot
bot = DadBot()
semantic = bot.memory_manager._semantic_manager
keys = semantic._semantic_index_backend.existing_content_hashes()
assert len(keys) == 1 and any("new" in key.lower() for key in keys), keys
assert bot.memory_manager._storage._projection_recovery_status["semantic"]["status"] == "sync_succeeded"
assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 2
bot.shutdown()
"""
    result = subprocess.run([sys.executable, "-c", verify], env=env, timeout=60, check=False)
    assert result.returncode == 0
