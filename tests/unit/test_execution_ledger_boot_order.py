from __future__ import annotations

import json

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend, WALCorruptionError
from dadbot.core.tenant_memory_ledger import TenantMemoryLedgerCorruption
from dadbot.memory.storage import MemoryStorageBackend


@pytest.mark.parametrize("durable", [False, True])
def test_ledger_exists_before_memory_hydration_and_is_shared(make_test_dadbot, monkeypatch, durable):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1" if durable else "0")
    original = MemoryStorageBackend.load_memory_store
    observed = []

    def load(backend, **kwargs):
        observed.append(backend.bot._execution_ledger)
        return original(backend, **kwargs)

    monkeypatch.setattr(MemoryStorageBackend, "load_memory_store", load)
    bot = make_test_dadbot()
    assert len(observed) == 1
    assert isinstance(observed[0], ExecutionLedger)
    assert observed[0] is bot.services.turn_orchestrator.control_plane.ledger


def test_corrupt_execution_wal_fails_before_memory_hydration(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "existing-sessions"
    sessions.mkdir()
    (sessions / "execution_ledger.jsonl").write_text('{"truncated":', encoding="utf-8")
    observed = []
    original = MemoryStorageBackend.load_memory_store

    def load(backend, **kwargs):
        observed.append(True)
        return original(backend, **kwargs)

    monkeypatch.setattr(MemoryStorageBackend, "load_memory_store", load)
    with pytest.raises(WALCorruptionError):
        make_test_dadbot(session_log_dir=sessions)
    assert observed == []


def test_invalid_tenant_event_fails_before_memory_hydration(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "existing-sessions"
    sessions.mkdir()
    event = {
        "type": "TENANT_MEMORY_COMMITTED",
        "session_id": "",
        "trace_id": "",
        "kernel_step_id": "tenant-memory",
        "timestamp": "2026-01-01",
        "committed": True,
        "payload": {
            "tenant_id": "another-tenant", "operation_id": "x",
            "expected_revision": 0, "revision": 2, "state": {},
        },
    }
    (sessions / "execution_ledger.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    observed = []
    monkeypatch.setattr(MemoryStorageBackend, "load_memory_store", lambda *args, **kwargs: observed.append(True))
    with pytest.raises(TenantMemoryLedgerCorruption):
        make_test_dadbot(session_log_dir=sessions)
    assert observed == []


def test_established_tenant_authority_cannot_be_overwritten_by_stale_snapshot(
    make_test_dadbot, monkeypatch, tmp_path,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "existing-sessions"
    memory_path = tmp_path / "memory.json"
    snapshot = '{"last_mood": "negative"}'
    memory_path.write_text(snapshot, encoding="utf-8")
    ledger = ExecutionLedger(FileWALLedgerBackend(sessions / "execution_ledger.jsonl"))
    ledger.commit_tenant_memory(
        tenant_id="test-tenant",
        expected_revision=0,
        operation_id="committed",
        state={"last_mood": "positive"},
    )

    bot = make_test_dadbot(
        session_log_dir=sessions,
        memory_path=memory_path,
        tenant_id="test-tenant",
    )
    assert bot.MEMORY_STORE["last_mood"] == "positive"
    assert json.loads(memory_path.read_text(encoding="utf-8")) == {"last_mood": "positive"}
    reloaded = ExecutionLedger(FileWALLedgerBackend(sessions / "execution_ledger.jsonl"))
    reloaded.load_from_backend()
    assert reloaded.replay_tenant_memory("test-tenant").state == {"last_mood": "positive"}


def test_authority_startup_guard_is_tenant_scoped(make_test_dadbot, monkeypatch, tmp_path):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    sessions = tmp_path / "existing-sessions"
    ledger = ExecutionLedger(FileWALLedgerBackend(sessions / "execution_ledger.jsonl"))
    ledger.commit_tenant_memory(
        tenant_id="other-tenant",
        expected_revision=0,
        operation_id="committed",
        state={"last_mood": "happy"},
    )
    bot = make_test_dadbot(session_log_dir=sessions, tenant_id="test-tenant")
    assert bot._execution_ledger.replay_tenant_memory("other-tenant").revision == 1
    assert bot._execution_ledger.replay_tenant_memory("test-tenant").revision == 1


def test_disabling_durable_ledger_cannot_bypass_established_tenant_authority(
    make_test_dadbot, monkeypatch, tmp_path,
):
    sessions = tmp_path / "existing-sessions"
    memory_path = tmp_path / "memory.json"
    snapshot = '{"last_mood": "stale"}'
    memory_path.write_text(snapshot, encoding="utf-8")
    ledger = ExecutionLedger(FileWALLedgerBackend(sessions / "execution_ledger.jsonl"))
    ledger.commit_tenant_memory(
        tenant_id="test-tenant",
        expected_revision=0,
        operation_id="committed",
        state={"last_mood": "committed"},
    )
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "0")

    with pytest.raises(RuntimeError, match="durable execution ledger"):
        make_test_dadbot(
            session_log_dir=sessions,
            memory_path=memory_path,
            tenant_id="test-tenant",
        )

    assert memory_path.read_text(encoding="utf-8") == snapshot


def test_disabled_ledger_does_not_block_unrelated_tenant(make_test_dadbot, monkeypatch, tmp_path):
    sessions = tmp_path / "existing-sessions"
    ledger = ExecutionLedger(FileWALLedgerBackend(sessions / "execution_ledger.jsonl"))
    ledger.commit_tenant_memory(
        tenant_id="other-tenant",
        expected_revision=0,
        operation_id="committed",
        state={"last_mood": "positive"},
    )
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "0")

    bot = make_test_dadbot(session_log_dir=sessions, tenant_id="test-tenant")
    assert bot.MEMORY_STORE["last_mood"] == "neutral"


def test_disabled_ledger_does_not_ignore_corrupt_existing_wal(make_test_dadbot, monkeypatch, tmp_path):
    sessions = tmp_path / "existing-sessions"
    sessions.mkdir()
    (sessions / "execution_ledger.jsonl").write_text('{"truncated":', encoding="utf-8")
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "0")

    with pytest.raises(WALCorruptionError):
        make_test_dadbot(session_log_dir=sessions)


@pytest.mark.parametrize(
    ("backend", "durable", "message"),
    [
        ("invalid", "1", "Unsupported execution ledger backend"),
        ("postgres", "0", "requires DADBOT_DURABLE_EXECUTION_LEDGER"),
        ("postgres", "1", "requires a PostgreSQL durable state store"),
    ],
)
def test_invalid_ledger_configuration_fails_before_memory_hydration(
    make_test_dadbot, monkeypatch, backend, durable, message,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", durable)
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", backend)
    observed = []
    monkeypatch.setattr(MemoryStorageBackend, "load_memory_store", lambda *args, **kwargs: observed.append(True))
    with pytest.raises(ValueError, match=message):
        make_test_dadbot()
    assert observed == []
