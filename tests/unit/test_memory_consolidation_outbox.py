import json
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dadbot.core.execution_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.local_tenant_establishment import commit_local_tenant, establish_local_tenant
from dadbot.core.memory_consolidation_outbox import (
    MEMORY_CONSOLIDATION_COMPLETED,
    MEMORY_CONSOLIDATION_TURN_OUTCOME,
    build_memory_consolidation_work,
    pending_memory_consolidation_work,
    record_memory_consolidation_turn_outcome,
)
from dadbot.core.tenant_session import tenant_session_scope
from dadbot.managers.memory_coordination import MemoryCoordinator
from dadbot.memory.manager import MemoryManager
from dadbot.memory.storage import MemoryStorageBackend
from dadbot.services.post_commit_worker import PostCommitWorker, _PostCommitCapability

pytestmark = pytest.mark.unit


def test_idempotent_storage_mutation_does_not_reserve_a_memory_key(monkeypatch):
    storage = MemoryStorageBackend.__new__(MemoryStorageBackend)
    normal: dict[str, Any] = {}
    idempotent: dict[str, Any] = {}
    monkeypatch.setattr(storage, "mutate_memory_store", lambda **kwargs: normal.update(kwargs))
    monkeypatch.setattr(
        storage,
        "mutate_memory_store_idempotently",
        lambda **kwargs: idempotent.update(kwargs),
    )

    manager = MemoryManager.__new__(MemoryManager)
    manager._storage = storage

    manager.store("operation_id", "ordinary memory value")
    state = {}
    normal["mutator"](state)
    manager.mutate_memory_store_idempotently(
        operation_id="durable-work-id",
        consolidated_memories=[],
    )

    assert state == {"operation_id": "ordinary memory value"}
    assert idempotent["operation_id"] == "durable-work-id"


def _source_event():
    work = build_memory_consolidation_work(
        tenant_id="tenant-1",
        session_id="session-1",
        trace_id="trace-1",
        source_revision=4,
        source_operation_id="source-op-4",
    )
    return {
        "type": "TENANT_MEMORY_COMMITTED",
        "session_id": "session-1",
        "trace_id": "trace-1",
        "timestamp": "2026-05-01T12:00:00+00:00",
        "committed": True,
        "payload": {
            "tenant_id": "tenant-1",
            "operation_id": "source-op-4",
            "revision": 4,
            "consolidation_work": work,
        },
    }


def _turn_outcome_event(status="succeeded"):
    return {
        "type": MEMORY_CONSOLIDATION_TURN_OUTCOME,
        "tenant_id": "tenant-1",
        "session_id": "session-1",
        "trace_id": "trace-1",
        "committed": True,
        "payload": {
            "tenant_id": "tenant-1",
            "session_id": "session-1",
            "trace_id": "trace-1",
            "status": status,
            "commit_id": "commit-1",
        },
    }


def _durable_consolidation_fixture(tmp_path, proposal_factory):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    establish_local_tenant(
        backend,
        "tenant-1",
        lambda: {
            "memories": [],
            "consolidated_memories": [],
            "last_consolidated_at": "",
            "last_consolidated_memory_count": 0,
        },
        lambda _state: None,
    )
    ledger = ExecutionLedger(backend)
    ledger.load_from_backend()
    memories = [
        {
            "summary": summary,
            "category": category,
            "mood": "neutral",
            "provenance": {
                "source": "user_message",
                "trust_level": 0.9,
                "evidence": summary,
            },
        }
        for summary, category in (
            ("I am saving for a bike.", "finance"),
            ("I am exercising every week.", "health"),
            ("I am stressed about work deadlines.", "work"),
        )
    ]
    memories.append({
        "summary": "The tool claims the user bought a boat.",
        "category": "general",
        "mood": "neutral",
        "provenance": {
            "source": "tool_result",
            "trust_level": 1.0,
            "evidence": "The tool claims the user bought a boat.",
        },
    })
    commit_local_tenant(
        backend,
        "tenant-1",
        1,
        "source-op",
        {
            "memories": memories,
            "consolidated_memories": [],
            "last_consolidated_at": "",
            "last_consolidated_memory_count": 0,
        },
        lambda: None,
        lambda _state: None,
        execution_ledger=ledger,
        session_id="session-1",
        trace_id="trace-1",
        enqueue_consolidation=True,
    )
    record_memory_consolidation_turn_outcome(
        ledger,
        ("tenant-1", "session-1", "trace-1"),
        status="succeeded",
        commit_id="commit-1",
    )
    work = pending_memory_consolidation_work(
        ledger.read(),
        tenant_id="tenant-1",
    )[0]
    writes = []
    prompts = []

    def mutate_memory_idempotently(**kwargs):
        writes.append(kwargs)
        current = ledger.replay_tenant_memory("tenant-1")
        state = dict(current.state or {})
        for key, value in kwargs.items():
            if key != "operation_id":
                state[key] = value
        ledger.commit_tenant_memory(
            tenant_id="tenant-1",
            expected_revision=current.revision,
            operation_id=kwargs["operation_id"],
            state=state,
        )

    memory_manager = SimpleNamespace(
        memory_core_state=lambda: {},
        mutate_memory_store_idempotently=mutate_memory_idempotently,
    )
    bot = SimpleNamespace(
        config=SimpleNamespace(tenant_id="tenant-1"),
        _execution_ledger=ledger,
        memory_manager=memory_manager,
        normalize_memory_text=lambda text: str(text or "").lower().strip().rstrip("."),
        normalize_mood=lambda mood: str(mood or "neutral"),
        detect_memory_contradictions=lambda *_args: [],
        call_ollama_chat=lambda **kwargs: (
            prompts.append(kwargs["messages"][0]["content"])
            or json.dumps(proposal_factory(work))
        ),
        extract_ollama_message_content=lambda response: response,
        parse_model_json_content=json.loads,
        mark_memory_graph_dirty=lambda: None,
    )
    coordinator = MemoryCoordinator.__new__(MemoryCoordinator)
    coordinator.bot = bot
    coordinator.detect_memory_contradictions = lambda memories, existing_insights=None: []
    coordinator._normalize_merge_inputs = lambda entries: entries
    coordinator._enforce_merge_constraints = lambda merged_values: merged_values
    coordinator._resolve_conflicts = lambda merged_values, *, now_stamp: (merged_values, [])
    source = ledger.replay_tenant_memory_at_revision(
        "tenant-1",
        work["source_revision"],
    )
    assert source.state is not None
    return coordinator, work, source.state, writes, prompts, ledger


def test_pending_work_is_bound_to_exact_source_and_completion_ack():
    source = _source_event()

    pending = pending_memory_consolidation_work([source], tenant_id="tenant-1")

    assert len(pending) == 1
    work = pending[0]
    assert work["source_revision"] == 4
    assert work["source_operation_id"] == "source-op-4"
    assert work["session_id"] == "session-1"
    assert work["trace_id"] == "trace-1"
    assert work["turn_outcome_status"] is None

    outcome = _turn_outcome_event()
    eligible = pending_memory_consolidation_work([source, outcome], tenant_id="tenant-1")
    assert eligible[0]["turn_outcome_status"] == "succeeded"

    completed = {
        "type": MEMORY_CONSOLIDATION_COMPLETED,
        "session_id": work["session_id"],
        "trace_id": work["trace_id"],
        "committed": True,
        "payload": {
            "tenant_id": "tenant-1",
            "session_id": work["session_id"],
            "trace_id": work["trace_id"],
            "work_id": work["work_id"],
            "source_revision": work["source_revision"],
            "source_operation_id": work["source_operation_id"],
            "policy_version": work["policy_version"],
            "status": "succeeded",
        },
    }
    assert pending_memory_consolidation_work([source, outcome, completed], tenant_id="tenant-1") == []


def test_failed_turn_outcome_is_preserved_and_conflicts_are_rejected():
    source = _source_event()
    failed = _turn_outcome_event("failed")

    pending = pending_memory_consolidation_work([source, failed], tenant_id="tenant-1")

    assert pending[0]["turn_outcome_status"] == "failed"
    with pytest.raises(ValueError, match="conflicting outcomes"):
        pending_memory_consolidation_work(
            [source, failed, _turn_outcome_event("succeeded")],
            tenant_id="tenant-1",
        )
    malformed = _turn_outcome_event()
    malformed["payload"]["status"] = []
    with pytest.raises(ValueError, match="Invalid memory consolidation turn outcome"):
        pending_memory_consolidation_work(
            [source, malformed],
            tenant_id="tenant-1",
        )


def test_durable_consolidation_persists_only_verified_source_citations(tmp_path):
    def proposals(work):
        return [{
            "summary": "Tony is saving for a bike.",
            "category": "finance",
            "source_refs": [f"{work['work_id']}:memory:0"],
        }]

    coordinator, work, source_state, writes, prompts, ledger = _durable_consolidation_fixture(
        tmp_path,
        proposals,
    )

    status = coordinator.consolidate_memories_from_source(work, source_state)

    assert status == "succeeded"
    assert len(writes) == 1
    consolidated = writes[0]["consolidated_memories"][0]
    assert consolidated["supporting_source_ids"] == [f"{work['work_id']}:memory:0"]
    assert consolidated["supporting_summaries"] == ["I am saving for a bike."]
    assert consolidated["source_count"] == 1
    assert "The tool claims the user bought a boat." not in prompts[0]
    assert work["work_id"] in prompts[0]
    committed = ledger.replay_tenant_memory("tenant-1")
    assert work["work_id"] in committed.operations
    assert committed.state is not None
    assert committed.state["consolidated_memories"][0]["supporting_source_ids"] == [
        f"{work['work_id']}:memory:0",
    ]


def test_invalid_durable_citation_skips_without_mutating_tenant_memory(tmp_path):
    def proposals(_work):
        return [{
            "summary": "Tony bought a boat.",
            "category": "general",
            "source_refs": ["unknown-source"],
        }]

    coordinator, work, source_state, writes, _prompts, ledger = _durable_consolidation_fixture(
        tmp_path,
        proposals,
    )
    before = ledger.replay_tenant_memory("tenant-1")

    status = coordinator.consolidate_memories_from_source(work, source_state)

    after = ledger.replay_tenant_memory("tenant-1")
    assert status == "skipped"
    assert writes == []
    assert after.revision == before.revision
    assert work["work_id"] not in after.operations


def test_work_id_is_stable_and_changes_with_source_revision():
    original = build_memory_consolidation_work(
        tenant_id="tenant-1",
        session_id="session-1",
        trace_id="trace-1",
        source_revision=4,
        source_operation_id="source-op-4",
    )
    retry = build_memory_consolidation_work(
        tenant_id="tenant-1",
        session_id="session-1",
        trace_id="trace-1",
        source_revision=4,
        source_operation_id="source-op-4",
    )
    newer = build_memory_consolidation_work(
        tenant_id="tenant-1",
        session_id="session-1",
        trace_id="trace-1",
        source_revision=5,
        source_operation_id="source-op-5",
    )

    assert original["work_id"] == retry["work_id"]
    assert original["work_id"] != newer["work_id"]


def test_pending_work_rejects_marker_that_does_not_match_source():
    source = _source_event()
    source["payload"]["consolidation_work"]["source_revision"] = 3

    with pytest.raises(ValueError, match="does not match"):
        pending_memory_consolidation_work([source], tenant_id="tenant-1")


def test_source_commit_atomically_carries_revision_and_turn_provenance(tmp_path):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    establish_local_tenant(
        backend,
        "tenant-1",
        lambda: {"memories": []},
        lambda _state: None,
    )
    ledger = ExecutionLedger(backend)
    ledger.load_from_backend()

    revision = commit_local_tenant(
        backend,
        "tenant-1",
        1,
        "source-op",
        {"memories": ["new memory"]},
        lambda: None,
        lambda _state: None,
        execution_ledger=ledger,
        session_id="session-1",
        trace_id="trace-1",
        enqueue_consolidation=True,
    )

    pending = pending_memory_consolidation_work(ledger.read(), tenant_id="tenant-1")

    assert revision == 2
    assert len(pending) == 1
    assert pending[0]["turn_outcome_status"] is None
    assert pending[0]["source_revision"] == 2
    assert pending[0]["source_operation_id"] == "source-op"
    assert pending[0]["session_id"] == "session-1"
    assert pending[0]["trace_id"] == "trace-1"


def test_memory_storage_captures_active_turn_provenance_for_catalog_changes(tmp_path):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    establish_local_tenant(
        backend,
        "tenant-1",
        lambda: {"memories": []},
        lambda _state: None,
    )
    ledger = ExecutionLedger(backend)
    ledger.load_from_backend()
    bot = SimpleNamespace(config=SimpleNamespace(tenant_id="tenant-1"), _execution_ledger=ledger)
    storage = SimpleNamespace(
        bot=bot,
        _authenticated_authority_required=False,
        _tenant_recovery_required=False,
        _tenant_base_revision=1,
        _authorize_tenant_memory=lambda: None,
        _local_authority_backend=lambda: backend,
        _require_tenant_isolated_indexes=lambda: None,
        _repair_established_snapshot=lambda _state: None,
        _repair_committed_indexes=lambda _revision: None,
    )

    with (
        tenant_session_scope("tenant-1", "session-1"),
        bind_execution_trace(
            ExecutionTraceRecorder(trace_token="trace-1", prompt="[test]"),
            required=True,
        ),
    ):
        MemoryStorageBackend._commit_local_projection_fenced(
            cast(MemoryStorageBackend, storage),
            {"memories": ["new memory"]},
        )

    pending = pending_memory_consolidation_work(ledger.read(), tenant_id="tenant-1")

    assert len(pending) == 1
    assert pending[0]["turn_outcome_status"] is None
    assert pending[0]["session_id"] == "session-1"
    assert pending[0]["trace_id"] == "trace-1"


def test_worker_recovers_commit_before_completion_without_repeating_consolidation(tmp_path, monkeypatch):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    establish_local_tenant(
        backend,
        "tenant-1",
        lambda: {"memories": []},
        lambda _state: None,
    )
    ledger = ExecutionLedger(backend)
    ledger.load_from_backend()
    commit_local_tenant(
        backend,
        "tenant-1",
        1,
        "source-op",
        {"memories": ["new memory"]},
        lambda: None,
        lambda _state: None,
        execution_ledger=ledger,
        session_id="session-1",
        trace_id="trace-1",
        enqueue_consolidation=True,
    )
    record_memory_consolidation_turn_outcome(
        ledger,
        ("tenant-1", "session-1", "trace-1"),
        status="succeeded",
        commit_id="commit-1",
    )

    class Coordinator:
        calls = 0

        def consolidate_memories_from_source(self, work, _source_state):
            self.calls += 1
            current = ledger.replay_tenant_memory(work["tenant_id"])
            assert current.state is not None
            ledger.commit_tenant_memory(
                tenant_id=work["tenant_id"],
                expected_revision=current.revision,
                operation_id=work["work_id"],
                state=current.state,
            )
            return "succeeded"

    coordinator = Coordinator()
    bot = SimpleNamespace(
        config=SimpleNamespace(tenant_id="tenant-1"),
        _execution_ledger=ledger,
        memory_coordinator=coordinator,
    )
    worker = PostCommitWorker.__new__(PostCommitWorker)
    worker._capability = _PostCommitCapability(bot)
    worker._initialize_retry_tracking()
    original_write = ledger.write
    fail_completion = True

    def fail_first_completion(event):
        nonlocal fail_completion
        if event.get("type") == MEMORY_CONSOLIDATION_COMPLETED and fail_completion:
            fail_completion = False
            raise OSError("completion append interrupted")
        return original_write(event)

    monkeypatch.setattr(ledger, "write", fail_first_completion)

    assert worker._drain_pending_consolidation_work() == "failed"
    assert coordinator.calls == 1
    work_id = next(iter(worker._retry_deadlines))
    monkeypatch.setattr(
        "dadbot.services.post_commit_worker.time.monotonic",
        lambda: worker._retry_deadlines[work_id],
    )
    assert worker._drain_pending_consolidation_work() == "succeeded"
    assert coordinator.calls == 1
    assert pending_memory_consolidation_work(ledger.read(), tenant_id="tenant-1") == []


def test_worker_scans_durable_outbox_at_startup_without_event_bus(tmp_path):
    backend = FileWALLedgerBackend(tmp_path / "ledger")
    establish_local_tenant(
        backend,
        "tenant-1",
        lambda: {"memories": []},
        lambda _state: None,
    )
    ledger = ExecutionLedger(backend)
    ledger.load_from_backend()
    commit_local_tenant(
        backend,
        "tenant-1",
        1,
        "source-op",
        {"memories": ["new memory"]},
        lambda: None,
        lambda _state: None,
        execution_ledger=ledger,
        session_id="session-1",
        trace_id="trace-1",
        enqueue_consolidation=True,
    )
    record_memory_consolidation_turn_outcome(
        ledger,
        ("tenant-1", "session-1", "trace-1"),
        status="succeeded",
        commit_id="commit-1",
    )

    class Coordinator:
        calls = 0

        def consolidate_memories_from_source(self, work, _source_state):
            self.calls += 1
            current = ledger.replay_tenant_memory(work["tenant_id"])
            assert current.state is not None
            ledger.commit_tenant_memory(
                tenant_id=work["tenant_id"],
                expected_revision=current.revision,
                operation_id=work["work_id"],
                state=current.state,
            )
            return "succeeded"

    coordinator = Coordinator()
    bot = SimpleNamespace(
        config=SimpleNamespace(tenant_id="tenant-1"),
        _execution_ledger=ledger,
        memory_coordinator=coordinator,
    )

    worker = PostCommitWorker(bot)
    thread = worker._thread
    assert thread is not None
    worker.stop()
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert coordinator.calls == 1
    assert pending_memory_consolidation_work(ledger.read(), tenant_id="tenant-1") == []


def test_worker_defers_unresolved_sources_until_outcome_is_recorded(monkeypatch):
    now = 100.0
    work = {"work_id": "work-1", "turn_outcome_status": None}

    class Capability:
        completed = False
        calls = 0

        def pending_consolidation_work(self):
            return [] if self.completed else [work]

        def consolidate_source(self, _work):
            self.calls += 1
            return "succeeded"

        def complete_consolidation(self, _work, *, status):
            assert status == "succeeded"
            self.completed = True

    worker = PostCommitWorker.__new__(PostCommitWorker)
    capability = Capability()
    worker._capability = cast(_PostCommitCapability, capability)
    worker._initialize_retry_tracking()
    monkeypatch.setattr("dadbot.services.post_commit_worker.time.monotonic", lambda: now)

    assert worker._drain_pending_consolidation_work() == "deferred"
    assert capability.calls == 0
    assert worker._retry_deadlines["work-1"] == 105.0

    work["turn_outcome_status"] = "succeeded"
    now = 105.0
    assert worker._drain_pending_consolidation_work() == "succeeded"
    assert capability.calls == 1


def test_worker_acknowledges_failed_turn_without_consolidating():
    work = {"work_id": "work-1", "turn_outcome_status": "failed"}

    class Capability:
        completed = False
        calls = 0
        completion_status = ""

        def pending_consolidation_work(self):
            return [] if self.completed else [work]

        def consolidate_source(self, _work):
            self.calls += 1
            return "succeeded"

        def complete_consolidation(self, _work, *, status):
            self.completion_status = status
            self.completed = True

    worker = PostCommitWorker.__new__(PostCommitWorker)
    capability = Capability()
    worker._capability = cast(_PostCommitCapability, capability)
    worker._initialize_retry_tracking()

    assert worker._drain_pending_consolidation_work() == "succeeded"
    assert capability.calls == 0
    assert capability.completion_status == "skipped"


def test_worker_retries_failed_outbox_work_after_backoff(monkeypatch):
    now = 100.0
    work = {"work_id": "work-1", "turn_outcome_status": "succeeded"}

    class Capability:
        completed = False
        calls = 0

        def pending_consolidation_work(self):
            return [] if self.completed else [work]

        def consolidate_source(self, _work):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary consolidation failure")
            return "succeeded"

        def complete_consolidation(self, _work, *, status):
            assert status == "succeeded"
            self.completed = True

    worker = PostCommitWorker.__new__(PostCommitWorker)
    capability = Capability()
    worker._capability = cast(_PostCommitCapability, capability)
    worker._initialize_retry_tracking()
    monkeypatch.setattr("dadbot.services.post_commit_worker.time.monotonic", lambda: now)
    monkeypatch.setattr(
        worker,
        "_retry_delay_seconds",
        float,
    )

    assert worker._drain_pending_consolidation_work() == "failed"
    assert worker._retry_deadlines["work-1"] == 101.0
    assert worker._drain_pending_consolidation_work() == "deferred"
    assert capability.calls == 1

    now = 101.0
    assert worker._drain_pending_consolidation_work() == "succeeded"
    assert capability.calls == 2
    assert worker._retry_deadlines == {}


def test_worker_loop_wakes_for_pending_retry_without_event_bus(monkeypatch):
    work = {"work_id": "work-1", "turn_outcome_status": "succeeded"}

    class Capability:
        completed = False
        calls = 0

        def pending_consolidation_work(self):
            return [] if self.completed else [work]

        def consolidate_source(self, _work):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary consolidation failure")
            return "succeeded"

        def complete_consolidation(self, _work, *, status):
            assert status == "succeeded"
            self.completed = True
            worker.stop()

    worker = PostCommitWorker.__new__(PostCommitWorker)
    worker._stop_event = Event()
    worker._subscriber = None
    worker._initialize_retry_tracking()
    capability = Capability()
    worker._capability = cast(_PostCommitCapability, capability)
    monkeypatch.setattr(PostCommitWorker, "_RETRY_BASE_DELAY_MS", 10.0)
    monkeypatch.setattr(PostCommitWorker, "_RETRY_MAX_DELAY_SECONDS", 0.05)

    thread = Thread(target=worker._run)
    thread.start()
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert capability.calls == 2
