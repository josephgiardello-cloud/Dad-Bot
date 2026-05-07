from __future__ import annotations

from types import SimpleNamespace

import pytest

from dadbot.core.post_commit_events import POST_COMMIT_READY, PostCommitEvent
from dadbot.services.persistence import PersistenceService, StateDivergenceError
from dadbot.services.post_commit_worker import PostCommitWorker
from dadbot_system.events import InMemoryEventBus

pytestmark = pytest.mark.unit


class _PersistenceManagerStub:
    def __init__(
        self,
        *,
        tamper_session_state: dict | None = None,
        tamper_checkpoint_state: dict | None = None,
    ):
        self._latest_checkpoint: dict | None = None
        self._tamper_session_state = dict(tamper_session_state or {}) if tamper_session_state is not None else None
        self._tamper_checkpoint_state = dict(tamper_checkpoint_state or {}) if tamper_checkpoint_state is not None else None

    def persist_turn_event(self, event):
        return None

    def persist_graph_checkpoint(self, checkpoint, _skip_turn_event: bool = False):
        _ = _skip_turn_event
        self._latest_checkpoint = dict(checkpoint or {})

    def load_latest_graph_checkpoint(self, trace_id: str = ""):
        _ = trace_id
        if not isinstance(self._latest_checkpoint, dict):
            return {}
        payload = dict(self._latest_checkpoint)
        if self._tamper_checkpoint_state is not None:
            payload["state"] = dict(self._tamper_checkpoint_state)
        if self._tamper_session_state is not None:
            payload["session_state"] = dict(self._tamper_session_state)
        return payload


class _RelationshipManagerStub:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[object] = []

    def materialize_projection(self, *, turn_context):
        self.calls.append(turn_context)
        if self.should_fail:
            raise RuntimeError("projection failed")


class _GraphManagerStub:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[object] = []

    def sync_graph_store(self, *, turn_context):
        self.calls.append(turn_context)
        if self.should_fail:
            raise RuntimeError("graph sync failed")


class _MemoryCoordinatorStub:
    def __init__(self) -> None:
        self.consolidate_calls: list[object] = []
        self.forgetting_calls: list[object] = []

    def consolidate_memories(self, *, turn_context=None):
        self.consolidate_calls.append(turn_context)

    def apply_controlled_forgetting(self, *, turn_context=None):
        self.forgetting_calls.append(turn_context)


class _MemoryRuntimeStub:
    def __init__(self) -> None:
        self.saved_moods: list[str] = []

    def save_mood_state(self, mood: str) -> None:
        self.saved_moods.append(str(mood or "neutral"))


class _TurnServiceStub:
    def __init__(self, bot) -> None:
        self.bot = bot
        self.finalize_calls: list[tuple[str, str, str, object, object]] = []

    def finalize_user_turn(
        self,
        turn_text,
        mood,
        reply,
        norm_attachments,
        *,
        turn_context,
    ):
        self.finalize_calls.append((turn_text, mood, reply, norm_attachments, turn_context))
        return reply, False


def _make_runtime(*, tenant_id: str = "tenant-1", graph_sync_fails: bool = False, projection_fails: bool = False):
    event_bus = InMemoryEventBus()
    graph_manager = _GraphManagerStub(should_fail=graph_sync_fails)
    relationship_manager = _RelationshipManagerStub(should_fail=projection_fails)
    memory_coordinator = _MemoryCoordinatorStub()
    memory_runtime = _MemoryRuntimeStub()
    runtime = SimpleNamespace(
        config=SimpleNamespace(tenant_id=tenant_id, merkle_anchor_enabled=False),
        _runtime_event_bus=event_bus,
        relationship_manager=relationship_manager,
        memory=memory_runtime,
        memory_manager=SimpleNamespace(graph_manager=graph_manager),
        memory_coordinator=memory_coordinator,
        _background_memory_store_patch_queue=[],
        _graph_commit_active=False,
        _current_turn_time_base=None,
        MEMORY_STORE={},
        _last_turn_pipeline={},
        snapshot_session_state=lambda: {},
        load_session_state_snapshot=lambda snapshot: None,
    )
    return runtime, event_bus, memory_coordinator, relationship_manager, graph_manager, memory_runtime


def _make_turn_context(*, trace_id: str = "trace-123", session_id: str = "session-123"):
    return SimpleNamespace(
        user_input="need help",
        attachments=[],
        trace_id=trace_id,
        temporal=SimpleNamespace(wall_time="2026-05-01T00:00:00Z"),
        state={
            "turn_text": "need help",
            "mood": "neutral",
            "norm_attachments": [],
        },
        metadata={"control_plane": {"session_id": session_id}},
        mutation_queue=None,
        checkpoint_snapshot=None,
    )


def _checkpoint_snapshot_factory(*, trace_id: str) -> object:
    def _snapshot(*, stage: str, status: str, error: str | None = None):
        return {
            "trace_id": trace_id,
            "stage": str(stage or "save"),
            "status": str(status or "atomic_finalize"),
            "error": str(error or ""),
            "state": {"safe_result": ("done", False)},
            "metadata": {"determinism": {"lock_hash": "lock-xyz"}},
        }

    return _snapshot


def test_finalize_turn_emits_post_commit_event_once_on_success():
    runtime, event_bus, _memory_coordinator, relationship_manager, graph_manager, _memory_runtime = _make_runtime()
    service = PersistenceService(_PersistenceManagerStub(), turn_service=_TurnServiceStub(runtime))
    turn_context = _make_turn_context()

    result = service.finalize_turn(turn_context, ("done", False))

    assert result == ("done", False)
    events = event_bus.events()
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PostCommitEvent)
    assert event.event_type == POST_COMMIT_READY
    assert event.session_id == "session-123"
    assert event.trace_id == "trace-123"
    assert event.tenant_id == "tenant-1"
    assert event.payload["turn_context"] is turn_context
    assert relationship_manager.calls == [turn_context]
    assert graph_manager.calls == [turn_context]


def test_finalize_turn_does_not_emit_post_commit_event_on_failed_commit():
    runtime, event_bus, _memory_coordinator, relationship_manager, graph_manager, _memory_runtime = _make_runtime(
        graph_sync_fails=True,
    )
    service = PersistenceService(_PersistenceManagerStub(), turn_service=_TurnServiceStub(runtime))
    turn_context = _make_turn_context()

    with pytest.raises(RuntimeError, match="graph sync failed"):
        service.finalize_turn(turn_context, ("done", False))

    assert event_bus.events() == []
    assert relationship_manager.calls == [turn_context]
    assert graph_manager.calls == [turn_context]


def test_emitted_post_commit_event_can_be_replayed_from_retained_bus_history():
    runtime, event_bus, memory_coordinator, _relationship_manager, _graph_manager, _memory_runtime = _make_runtime()
    service = PersistenceService(_PersistenceManagerStub(), turn_service=_TurnServiceStub(runtime))
    turn_context = _make_turn_context(trace_id="trace-replay", session_id="session-replay")

    service.finalize_turn(turn_context, ("done", False))

    retained_event = event_bus.events()[0]
    worker = PostCommitWorker.__new__(PostCommitWorker)
    worker._capability = PostCommitWorker.__init__.__class__  # satisfy __slots__ check
    from dadbot.services.post_commit_worker import _PostCommitCapability
    worker._capability = _PostCommitCapability(runtime)
    worker._handle_post_commit_event(retained_event)

    assert len(memory_coordinator.consolidate_calls) == 1
    assert len(memory_coordinator.forgetting_calls) == 1
    assert memory_coordinator.consolidate_calls[0] is turn_context
    assert memory_coordinator.forgetting_calls[0] is turn_context
    intents = list(turn_context.state.get("memory_write_intents") or [])
    assert len(intents) >= 4
    assert any(str(item.get("op") or "") == "consolidate_memories" for item in intents)
    assert any(str(item.get("op") or "") == "apply_controlled_forgetting" for item in intents)
    summary = dict(turn_context.state.get("memory_delta_summary") or {})
    assert str(summary.get("version") or "") == "1.0"
    assert int(summary.get("intent_count") or 0) == len(intents)


def test_finalize_turn_allows_commit_when_memory_authority_matches():
    runtime, event_bus, _memory_coordinator, _relationship_manager, _graph_manager, _memory_runtime = _make_runtime()
    runtime.snapshot_session_state = lambda: {}
    persistence = _PersistenceManagerStub()
    service = PersistenceService(persistence, turn_service=_TurnServiceStub(runtime))
    turn_context = _make_turn_context(trace_id="trace-authority-ok")
    turn_context.checkpoint_snapshot = _checkpoint_snapshot_factory(trace_id="trace-authority-ok")

    result = service.finalize_turn(turn_context, ("done", False))

    assert result == ("done", False)
    check = dict(turn_context.state.get("memory_authority_check") or {})
    assert check.get("consistent") is True
    assert str(check.get("projected_hash") or "")
    assert str(check.get("event_sourced_hash") or "")
    assert len(event_bus.events()) == 1


def test_finalize_turn_blocks_commit_on_memory_authority_divergence():
    runtime, event_bus, _memory_coordinator, _relationship_manager, _graph_manager, _memory_runtime = _make_runtime()
    runtime.snapshot_session_state = lambda: {}
    persistence = _PersistenceManagerStub(
        tamper_checkpoint_state={"safe_result": ("tampered", False)},
    )
    service = PersistenceService(persistence, turn_service=_TurnServiceStub(runtime))
    turn_context = _make_turn_context(trace_id="trace-authority-bad")
    turn_context.checkpoint_snapshot = _checkpoint_snapshot_factory(trace_id="trace-authority-bad")

    with pytest.raises(StateDivergenceError, match="commit blocked") as exc_info:
        service.finalize_turn(turn_context, ("done", False))

    report = dict(getattr(exc_info.value, "report", {}) or {})
    assert report.get("consistent") is False
    assert int(report.get("difference_count") or 0) >= 1
    assert any("checkpoint.state" in str(item.get("path") or "") for item in list(report.get("differences") or []))
    assert event_bus.events() == []
