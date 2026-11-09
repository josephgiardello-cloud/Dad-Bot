from copy import deepcopy
from types import SimpleNamespace

import pytest

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.tenant_memory_ledger import TenantMemoryRevisionConflict
from dadbot.services.persistence import PersistenceService


def _commit_memory(bot, mood):
    state = deepcopy(bot.MEMORY_STORE)
    state["last_mood"] = mood
    state = bot.memory_manager.normalize_memory_store(state)
    revision = bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision
    bot._execution_ledger.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=revision,
        operation_id=f"restore-test-{revision}",
        state=state,
    )
    bot.MEMORY_STORE = state
    return state


def test_session_restore_does_not_replace_committed_tenant_memory(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    session = bot.snapshot_session_state()
    committed = _commit_memory(bot, "happy")

    bot.load_session_state_snapshot(session)

    assert bot.MEMORY_STORE == committed
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).state == committed


def test_session_restore_reloads_tenant_committed_by_another_writer(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    session = bot.snapshot_session_state()
    state = bot.memory_manager.normalize_memory_store({**bot.MEMORY_STORE, "last_mood": "positive"})
    other = ExecutionLedger(FileWALLedgerBackend(bot.config.session_log_dir / "execution_ledger.jsonl"))
    other.load_from_backend()
    other.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=1,
        operation_id="other-writer",
        state=state,
    )

    bot.load_session_state_snapshot(session)

    assert bot.MEMORY_STORE == state
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).state == state


def test_transaction_rollback_does_not_restore_old_memory_or_graph(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    turn = SimpleNamespace(state={}, metadata={})
    snapshot = PersistenceService._capture_transaction_snapshot(bot, turn)
    committed = _commit_memory(bot, "happy")
    graph = bot.memory_manager.graph_manager
    calls = []
    monkeypatch.setattr(graph, "sync_graph_store", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        graph._graph_store_backend,
        "replace_graph",
        lambda *_args: (_ for _ in ()).throw(AssertionError("old graph restored")),
    )

    PersistenceService._restore_transaction_snapshot(bot, turn, snapshot)

    assert bot.MEMORY_STORE == committed
    assert calls == [{"turn_context": None}]


def test_restore_fails_closed_when_tenant_ledger_is_uncertain(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    session = bot.snapshot_session_state()
    committed = _commit_memory(bot, "happy")
    bot._execution_ledger._backend_uncertain = True

    with pytest.raises(RuntimeError, match="reload"):
        bot.load_session_state_snapshot(session)

    assert bot.MEMORY_STORE == committed
    bot._execution_ledger.load_from_backend()


def test_transaction_rollback_surfaces_graph_repair_failure(make_test_dadbot, monkeypatch):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    turn = SimpleNamespace(state={}, metadata={})
    snapshot = PersistenceService._capture_transaction_snapshot(bot, turn)
    committed = _commit_memory(bot, "happy")

    def fail_repair(**_kwargs):
        raise OSError("graph repair failed")

    monkeypatch.setattr(bot.memory_manager.graph_manager, "sync_graph_store", fail_repair)
    with pytest.raises(OSError, match="graph repair failed"):
        PersistenceService._restore_transaction_snapshot(bot, turn, snapshot)
    assert bot.MEMORY_STORE == committed


@pytest.mark.parametrize("operation", ["mutate", "save", "clear"])
def test_snapshot_write_rejected_after_external_tenant_commit(
    make_test_dadbot, monkeypatch, operation,
):
    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    bot = make_test_dadbot()
    before = deepcopy(bot.MEMORY_STORE)
    state = bot.memory_manager.normalize_memory_store({**before, "last_mood": "positive"})
    other = ExecutionLedger(FileWALLedgerBackend(bot.config.session_log_dir / "execution_ledger.jsonl"))
    other.load_from_backend()
    other.commit_tenant_memory(
        tenant_id=bot.config.tenant_id,
        expected_revision=1,
        operation_id="external-commit",
        state=state,
    )

    with pytest.raises(TenantMemoryRevisionConflict, match="revision changed"):
        if operation == "mutate":
            bot.mutate_memory_store(last_mood="sad")
        elif operation == "save":
            bot.save_memory_store()
        else:
            bot.clear_memory_store()

    assert bot.MEMORY_STORE == before
    bot._execution_ledger.load_from_backend()
    assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).state == state
    bot.MEMORY_STORE = state
