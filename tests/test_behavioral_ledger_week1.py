from __future__ import annotations

import json
from types import SimpleNamespace

from dadbot.services.persistence import PersistenceService


class _PersistenceManagerStub:
    pass


class _RuntimeStub:
    def __init__(self) -> None:
        self._turn_index = 0
        self._topic_drift_streak = 0
        self._relational_trust_credit = 0.5
        self.active_thread_id = "thread-week1"
        self.config = SimpleNamespace(session_log_dir=None)

    def trust_level(self) -> int:
        return 72

    def recent_memory_topics(self, limit: int = 4) -> list[str]:
        return ["python", "testing", "refactor"][:limit]

    def session_turn_count(self) -> int:
        return self._turn_index

    @staticmethod
    def significant_tokens(text: str) -> set[str]:
        return {
            token.strip().lower()
            for token in str(text or "").replace("-", " ").split()
            if len(token.strip()) >= 3
        }


def test_week1_behavioral_ledger_injects_relational_and_temporal_state() -> None:
    service = PersistenceService(persistence_manager=_PersistenceManagerStub())
    runtime = _RuntimeStub()
    context = SimpleNamespace(state={}, metadata={})

    service._inject_behavioral_ledger_state(runtime, context, "python refactor plan")

    relational = dict(context.state.get("relational_state") or {})
    temporal = dict(context.state.get("temporal_budget") or {})
    meta = dict(context.metadata.get("behavioral_ledger") or {})

    assert relational
    assert temporal
    assert relational.get("topic_drift_detected") is False
    assert relational.get("dominant_topic") == "python"
    assert isinstance(meta.get("relational_state"), dict)
    assert isinstance(meta.get("temporal_budget"), dict)


def test_week1_topic_drift_gate_passes_20_turn_sequence() -> None:
    service = PersistenceService(persistence_manager=_PersistenceManagerStub())
    runtime = _RuntimeStub()

    on_topic_turns = [
        "python refactor tests",
        "pytest fixtures for integration",
        "refactor this persistence function",
        "python typing cleanup",
        "testing deterministic replay",
        "write unit tests for refactor",
        "python lint pass",
        "fix flaky test retry",
        "improve refactor readability",
        "testing coverage expansion",
    ]
    drift_turns = [
        "tomato garden watering schedule",
        "best compost mix for soil",
        "weekend hiking trail plan",
        "grilling steaks and marinades",
        "replace bike chain and tires",
        "kitchen paint color samples",
        "vacation hotel packing list",
        "baseball lineup and batting",
        "home espresso grinder settings",
        "dog leash training routine",
    ]

    drift_flags: list[bool] = []
    streak_values: list[int] = []
    for text in on_topic_turns + drift_turns:
        context = SimpleNamespace(state={}, metadata={})
        service._inject_behavioral_ledger_state(runtime, context, text)
        relational = dict(context.state.get("relational_state") or {})
        temporal = dict(context.state.get("temporal_budget") or {})
        drift_flags.append(bool(relational.get("topic_drift_detected")))
        streak_values.append(int(temporal.get("topic_drift_streak") or 0))
        runtime._turn_index += 1

    first_half_drift = sum(1 for flag in drift_flags[:10] if flag)
    second_half_drift = sum(1 for flag in drift_flags[10:] if flag)

    # Gate target: over 20 turns, topic drift must be recognized reliably.
    assert first_half_drift <= 2
    assert second_half_drift >= 8
    assert max(streak_values) >= 8


def test_week1_relational_ledger_persists_jsonl_and_updates_credit(tmp_path) -> None:
    service = PersistenceService(persistence_manager=_PersistenceManagerStub())
    runtime = _RuntimeStub()
    runtime.config.session_log_dir = tmp_path

    aligned_context = SimpleNamespace(
        trace_id="trace-aligned",
        state={
            "session_goals": [
                {"id": "goal-1", "description": "finish python unit tests for persistence"},
            ],
        },
        metadata={"control_plane": {"session_id": "session-a"}},
    )
    service._record_relational_ledger(runtime, aligned_context, "finish python tests now")

    drift_context = SimpleNamespace(
        trace_id="trace-drift",
        state={
            "session_goals": [
                {"id": "goal-1", "description": "finish python unit tests for persistence"},
            ],
        },
        metadata={"control_plane": {"session_id": "session-a"}},
    )
    service._record_relational_ledger(runtime, drift_context, "plan weekend fishing trip")

    ledger_path = tmp_path / "relational_ledger.jsonl"
    assert ledger_path.exists()
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first_entry = json.loads(lines[0])
    second_entry = json.loads(lines[1])
    assert first_entry["decision"] == "followed_intent"
    assert second_entry["decision"] == "diverted_from_intent"
    assert float(first_entry["trust_credit_after"]) > float(first_entry["trust_credit_before"])
    assert float(second_entry["trust_credit_after"]) < float(second_entry["trust_credit_before"])


def test_personality_policy_marks_disappointed_dad_on_second_diversion(tmp_path) -> None:
    service = PersistenceService(persistence_manager=_PersistenceManagerStub())
    runtime = _RuntimeStub()
    runtime.config.session_log_dir = tmp_path

    base_state = {
        "session_goals": [
            {"id": "goal-1", "description": "finish python unit tests for persistence"},
        ],
    }

    turn1 = SimpleNamespace(
        trace_id="trace-diversion-1",
        state=dict(base_state),
        metadata={"control_plane": {"session_id": "session-p"}},
    )
    service._record_relational_ledger(runtime, turn1, "plan a hiking route")

    turn2 = SimpleNamespace(
        trace_id="trace-diversion-2",
        state=dict(base_state),
        metadata={"control_plane": {"session_id": "session-p"}},
    )
    service._record_relational_ledger(runtime, turn2, "compare grilling marinades")

    ledger_path = tmp_path / "relational_ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    second_entry = json.loads(lines[1])
    assert second_entry["decision"] == "diverted_from_intent"
    assert second_entry["dad_mode"] == "disappointed_dad"
