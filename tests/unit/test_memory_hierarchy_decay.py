from __future__ import annotations

import pytest

from dadbot.core.memory_hierarchy_manager import MemoryHierarchyManager


@pytest.mark.unit
def test_episodic_decay_uses_elapsed_time_once_and_preserves_history(monkeypatch):
    now = [100_000.0]
    monkeypatch.setattr("dadbot.core.memory_hierarchy_manager.time.time", lambda: now[0])
    state = {
        "memory_hierarchy": {
            "episodic": [
                {"summary": f"entry {index}", "timestamp": 96_400.0, "weight": 0.6}
                for index in range(80)
            ],
        },
    }
    manager = MemoryHierarchyManager()

    first = manager.lifecycle_maintenance(state=state)["episodic"]
    assert len(first) == 80
    assert first[0]["weight"] == pytest.approx(0.6 * 0.995)
    second = manager.lifecycle_maintenance(state=state)["episodic"]
    assert second[0]["weight"] == pytest.approx(first[0]["weight"])
    now[0] += 3600
    third = manager.lifecycle_maintenance(state=state)["episodic"]
    assert third[0]["weight"] == pytest.approx(0.6 * 0.995**2)


@pytest.mark.unit
def test_episodic_decay_does_not_replace_zero_timestamp_or_weight(monkeypatch):
    monkeypatch.setattr("dadbot.core.memory_hierarchy_manager.time.time", lambda: 3_600.0)
    state = {"memory_hierarchy": {"episodic": [{"timestamp": 0.0, "weight": 0.0}]}}
    row = MemoryHierarchyManager().lifecycle_maintenance(state=state)["episodic"][0]
    assert row["weight"] == 0.0


@pytest.mark.unit
def test_promoting_turn_preserves_existing_episodic_history(monkeypatch):
    monkeypatch.setattr("dadbot.core.memory_hierarchy_manager.time.time", lambda: 3_600.0)
    state = {"memory_hierarchy": {
        "episodic": [{"summary": f"entry {index}", "timestamp": 1.0} for index in range(256)],
    }}
    hierarchy = MemoryHierarchyManager().promote_turn(
        state=state,
        trace_id="next",
        user_input="A sufficiently long new turn",
        response_text="A response",
        semantic_items=[],
    )
    assert len(hierarchy["episodic"]) == 257
    assert hierarchy["episodic"][0]["summary"] == "entry 0"
    assert hierarchy["episodic"][-1]["trace_id"] == "next"
