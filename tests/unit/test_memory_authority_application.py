from __future__ import annotations

import json

def test_durable_bot_migrates_and_records_memory_mutations(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    memory = tmp_path / "legacy-memory.json"
    memory.write_text(json.dumps({"last_mood": "neutral"}), encoding="utf-8")
    bot = make_test_dadbot(memory_path=memory)
    tenant_id = bot.config.tenant_id

    initial = bot._execution_ledger.replay_tenant_memory(tenant_id)
    assert initial.revision == 1
    assert initial.state == bot.MEMORY_STORE

    bot.mutate_memory_store(last_mood="positive")
    committed = bot._execution_ledger.replay_tenant_memory(tenant_id)
    assert committed.revision == 2
    assert committed.state == bot.MEMORY_STORE
    assert json.loads(memory.read_text(encoding="utf-8")) == committed.state
