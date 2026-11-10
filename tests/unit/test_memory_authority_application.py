from __future__ import annotations

from dadbot_system.state import InMemoryStateStore


def _owned_document_store(tenant_id, state):
    store = InMemoryStateStore()
    store.owned_sessions.save_session_state(tenant_id, "memory", state)
    return store


def test_durable_bot_migrates_and_records_memory_mutations(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "legacy-memory.json"
    tenant_id = "tenant-memory-authority"
    document_store = _owned_document_store(tenant_id, {"last_mood": "neutral"})
    bot = make_test_dadbot(
        memory_path=memory,
        tenant_id=tenant_id,
        document_store=document_store,
    )

    initial = bot._execution_ledger.replay_tenant_memory(tenant_id)
    assert initial.revision == 1
    assert initial.state == bot.MEMORY_STORE

    bot.mutate_memory_store(last_mood="positive")
    committed = bot._execution_ledger.replay_tenant_memory(tenant_id)
    assert committed.revision == 2
    assert committed.state == bot.MEMORY_STORE
    assert document_store.owned_sessions.load_session_state(tenant_id, "memory") == committed.state
