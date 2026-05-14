from __future__ import annotations

import pytest

from dadbot.core.core_state import CoreState, InputEvent, deterministic_id, project_views, transition

pytestmark = pytest.mark.unit


def test_core_state_transition_is_pure_and_immutable() -> None:
    state0 = CoreState()
    event = InputEvent(
        event_type="turn_committed",
        payload={
            "trace_id": "tr-1",
            "response": "ok",
            "should_end": False,
            "memory_retrieval_set": [
                {
                    "summary": "alpha",
                    "category": "prefs",
                    "mood": "neutral",
                    "updated_at": "2026-05-10",
                    "created_at": "2026-05-10",
                },
            ],
        },
    )

    state1 = transition(state0, event)

    assert state0.version == 0
    assert state1.version == 1
    assert len(state0.events) == 0
    assert len(state1.events) == 1
    assert state1.execution.trace_id == "tr-1"


def test_core_state_memory_is_sorted_vector_on_insert() -> None:
    state = CoreState()
    event = InputEvent(
        event_type="turn_committed",
        payload={
            "trace_id": "tr-sort",
            "memory_retrieval_set": [
                {
                    "summary": "same",
                    "category": "zeta",
                    "mood": "happy",
                    "updated_at": "2026-05-10",
                    "created_at": "2026-05-10",
                },
                {
                    "summary": "same",
                    "category": "alpha",
                    "mood": "calm",
                    "updated_at": "2026-05-10",
                    "created_at": "2026-05-10",
                },
            ],
        },
    )
    state2 = transition(state, event)
    categories = [item.category for item in state2.memory.entries]
    assert categories == ["alpha", "zeta"]


def test_deterministic_id_uses_parent_state_and_event_payload() -> None:
    input_event = InputEvent(event_type="x", payload={"a": 1})
    id_a = deterministic_id(parent_state_hash="p1", input_event=input_event)
    id_b = deterministic_id(parent_state_hash="p1", input_event=input_event)
    id_c = deterministic_id(parent_state_hash="p2", input_event=input_event)

    assert id_a == id_b
    assert id_a != id_c
    assert len(id_a) == 64


def test_projections_are_recomputed_from_single_core_state() -> None:
    state = CoreState()
    event1 = InputEvent(event_type="turn_committed", payload={"trace_id": "tr-1", "response": "r1"})
    event2 = InputEvent(event_type="turn_committed", payload={"trace_id": "tr-1", "response": "r2"})
    state = transition(state, event1)
    state = transition(state, event2)

    views = project_views(state)
    assert len(views.canonical.events) == 2
    assert views.execution.state.last_response == "r2"
    assert views.facade.as_payload()["event_count"] == 2
