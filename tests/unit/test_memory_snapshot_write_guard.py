from __future__ import annotations

import json

import pytest

from dadbot_system.state import InMemoryStateStore


def _owned_document_store(tenant_id, state):
    store = InMemoryStateStore()
    store.owned_sessions.save_session_state(tenant_id, "memory", state)
    return store


@pytest.mark.parametrize("writer", ["_write_memory_store_unlocked", "_write_json_memory_store_unlocked"])
def test_direct_snapshot_write_rejects_established_tenant(
    make_test_dadbot, monkeypatch, tmp_path, writer,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"last_mood": "stale-mirror"}), encoding="utf-8")
    tenant_id = "tenant-snapshot-write-guard"
    document_store = _owned_document_store(tenant_id, {"last_mood": "neutral"})
    bot = make_test_dadbot(
        memory_path=memory,
        tenant_id=tenant_id,
        document_store=document_store,
    )
    authoritative = {**bot.MEMORY_STORE, "last_mood": "positive"}
    bot._execution_ledger.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=1,
        operation_id="direct-write-guard",
        state=authoritative,
    )
    before = memory.read_bytes()
    with pytest.raises(RuntimeError, match="ledger-authoritative"):
        getattr(bot.memory_manager._storage, writer)(bot.MEMORY_STORE)
    assert memory.read_bytes() == before
    bot.MEMORY_STORE = authoritative
