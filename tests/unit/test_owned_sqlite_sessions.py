import multiprocessing

import pytest

from dadbot_system.orchestration import DadBotOrchestrator
from dadbot_system.owned_tasks import CompositeOwnedTasks, InMemoryOwnedTasks
from dadbot_system.state import (
    CompositeStateStore,
    InMemoryStateStore,
    NamespacedStateStore,
    SQLiteStateStore,
    TenantOwnedStateStoreView,
)

TENANT_A = "t1_" + "a" * 26
TENANT_B = "t1_" + "b" * 26


class Broker:
    def enqueue(self, task):
        raise AssertionError("Session persistence must not enqueue tasks")

    def get_result_nowait(self):
        return None


def _write_session_state(path, tenant_id, output):
    orchestrator = DadBotOrchestrator(Broker(), state_store=SQLiteStateStore(path))
    session = orchestrator.session("shared-session", tenant_id)
    session.load_snapshot({"history": [{"role": "assistant", "content": tenant_id}]})
    output.put((tenant_id, "written"))


def _read_session_state(path, tenant_id, output):
    state = SQLiteStateStore(path).owned_sessions.load_session_state(tenant_id, "shared-session")
    output.put((tenant_id, state))


def test_same_session_id_is_owned_and_readable_only_by_each_tenant_across_processes(tmp_path):
    path = str(tmp_path / "state.db")
    SQLiteStateStore(path)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    writers = [
        context.Process(target=_write_session_state, args=(path, tenant_id, output))
        for tenant_id in (TENANT_A, TENANT_B)
    ]
    readers = []
    for process in writers:
        process.start()
    try:
        for process in writers:
            process.join(20)
            assert process.exitcode == 0
        assert {output.get(timeout=5) for _ in writers} == {
            (TENANT_A, "written"),
            (TENANT_B, "written"),
        }

        readers = [
            context.Process(target=_read_session_state, args=(path, tenant_id, output))
            for tenant_id in (TENANT_A, TENANT_B)
        ]
        for process in readers:
            process.start()
        for process in readers:
            process.join(20)
            assert process.exitcode == 0
        read_results = {}
        for _ in readers:
            tenant_id, payload = output.get(timeout=5)
            read_results[tenant_id] = payload
        assert {
            tenant_id: payload["history"]
            for tenant_id, payload in read_results.items()
        } == {
            TENANT_A: [{"role": "assistant", "content": TENANT_A}],
            TENANT_B: [{"role": "assistant", "content": TENANT_B}],
        }
    finally:
        for process in writers + readers:
            if process.is_alive():
                process.terminate()
                process.join()
        output.close()
        output.join_thread()


def test_session_store_rejects_missing_or_mismatched_owner(tmp_path):
    store = SQLiteStateStore(str(tmp_path / "state.db"))
    with pytest.raises(TypeError):
        store.owned_sessions.load_session_state("shared-session")
    with pytest.raises(ValueError, match="ownership identity mismatch"):
        store.owned_sessions.append_event(
            TENANT_A,
            "shared-session",
            {"tenant_id": TENANT_B, "session_id": "shared-session"},
        )
    orchestrator = DadBotOrchestrator(Broker(), state_store=store)
    with pytest.raises(TypeError):
        orchestrator.session("shared-session")
    assert orchestrator.session("shared-session", TENANT_A) is not orchestrator.session(
        "shared-session", TENANT_B
    )


def test_unowned_legacy_session_rows_are_not_adopted(tmp_path):
    store = SQLiteStateStore(str(tmp_path / "state.db"))
    store.save_session_state(f"tenant:{TENANT_A}:shared-session", {"history": ["legacy"]})

    with pytest.raises(RuntimeError, match="Unowned legacy session state"):
        store.owned_sessions.load_session_state(TENANT_A, "shared-session")


def test_tenant_namespaced_runtime_does_not_adopt_legacy_snapshot(tmp_path):
    store = SQLiteStateStore(str(tmp_path / "state.db"))
    store.save_session_state(
        f"tenant:{TENANT_A}:dadbot-runtime",
        {"session_summary": "legacy"},
    )
    runtime_store = NamespacedStateStore(
        store,
        f"tenant:{TENANT_A}",
        tenant_id=TENANT_A,
    )

    with pytest.raises(RuntimeError, match="Unowned legacy session state"):
        runtime_store.load_session_state("dadbot-runtime")


def test_composite_session_store_writes_to_tenant_owned_backends():
    fast = InMemoryStateStore()
    durable = InMemoryStateStore()
    composite = CompositeStateStore(fast_store=fast, durable_store=durable)
    view = TenantOwnedStateStoreView(composite, TENANT_A, InMemoryOwnedTasks({}))

    view.save_session_state("shared-session", {"history": ["owned"]})

    assert fast.owned_sessions.load_session_state(TENANT_A, "shared-session") == {
        "history": ["owned"]
    }
    assert durable.owned_sessions.load_session_state(TENANT_A, "shared-session") == {
        "history": ["owned"]
    }


def test_composite_task_recovery_uses_durable_status_over_stale_fast_entries():
    durable = InMemoryOwnedTasks({})
    fast = InMemoryOwnedTasks({})
    composite = CompositeOwnedTasks([durable, fast])
    task_id = "shared-task"

    def payload(status):
        return {
            "tenant_id": TENANT_A,
            "task_id": task_id,
            "status": status,
            "accepted_at": "2026-01-01T00:00:00Z",
        }

    durable.save_task(TENANT_A, task_id, payload("completed"))
    fast.save_task(TENANT_A, task_id, payload("queued"))
    assert composite.list_incomplete_tasks(TENANT_A) == []

    durable.save_task(TENANT_A, task_id, payload("queued"))
    fast.save_task(TENANT_A, task_id, payload("completed"))
    assert composite.list_incomplete_tasks(TENANT_A) == [payload("queued")]


def test_tenant_scoped_listing_and_termination_do_not_cross_owners(tmp_path):
    store = SQLiteStateStore(str(tmp_path / "state.db"))
    orchestrator = DadBotOrchestrator(Broker(), state_store=store)
    tenant_a = orchestrator.session("shared-session", TENANT_A)
    tenant_b = orchestrator.session("shared-session", TENANT_B)
    tenant_a.load_snapshot({"history": [{"role": "assistant", "content": "a"}]})
    tenant_b.load_snapshot({"history": [{"role": "assistant", "content": "b"}]})

    assert orchestrator.list_sessions(TENANT_A) == ["shared-session"]
    assert orchestrator.list_sessions(TENANT_B) == ["shared-session"]
    assert orchestrator.terminate_session("shared-session", TENANT_A)
    assert orchestrator.session("shared-session", TENANT_B).snapshot()["history"] == [
        {"role": "assistant", "content": "b"}
    ]
    assert orchestrator.list_sessions(TENANT_A) == []


def test_termination_failure_preserves_cached_session_until_store_delete_succeeds(tmp_path, monkeypatch):
    store = SQLiteStateStore(str(tmp_path / "state.db"))
    orchestrator = DadBotOrchestrator(Broker(), state_store=store)
    session = orchestrator.session("shared-session", TENANT_A)
    session.load_snapshot({"history": [{"role": "assistant", "content": "retained"}]})

    def fail_delete(_tenant_id, _session_id):
        raise OSError("session store unavailable")

    monkeypatch.setattr(store.owned_sessions, "delete_session", fail_delete)

    with pytest.raises(OSError, match="session store unavailable"):
        orchestrator.terminate_session("shared-session", TENANT_A)

    assert orchestrator._sessions[(TENANT_A, "shared-session")] is session
    assert orchestrator.session("shared-session", TENANT_A) is session
    assert store.owned_sessions.load_session_state(TENANT_A, "shared-session") is not None
