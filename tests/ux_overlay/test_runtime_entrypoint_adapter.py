from __future__ import annotations

from datetime import datetime, timezone

from dadbot.uril.truth_binding import ClaimEvidenceValidator, build_synthetic_state
from dadbot.ux_overlay import UxOverlayRuntimeAdapter


def test_adapter_can_toggle_ux_per_session() -> None:
    adapter = UxOverlayRuntimeAdapter()

    adapter.set_session_enabled("s1", False)
    adapter.set_session_enabled("s2", True)

    assert adapter.get_session_enabled("s1") is False
    assert adapter.get_session_enabled("s2") is True


def test_disabled_session_returns_original_response() -> None:
    adapter = UxOverlayRuntimeAdapter()
    adapter.set_session_enabled("s1", False)

    base = "Deterministic core output."
    out = adapter.process_turn(
        session_id="s1",
        base_response=base,
        user_text="hello",
    )

    assert out["ux_enabled"] is False
    assert out["rendered_response"] == base
    assert out["original_response"] == base


def test_enabled_session_shapes_response_and_tracks_state() -> None:
    adapter = UxOverlayRuntimeAdapter()
    api = adapter.control_api("s1")
    api.set_warmth(0.9)
    api.set_verbosity(0.8)

    out = adapter.process_turn(
        session_id="s1",
        base_response="We can do this in three steps.",
        user_text="Can you help me plan my week around family and work?",
        user_sentiment="happy",
        positive_signal=0.7,
        conversation_break=False,
        unresolved_intents=["weekly_plan"],
        emotional_label="encouraging",
        raw_memory_events=[
            {
                "text": "Remember this important family planning concern from this week.",
                "created_at": datetime.now(timezone.utc),
                "emotional_intensity": 0.7,
            }
        ],
    )

    assert out["ux_enabled"] is True
    assert out["rendered_response"] != out["original_response"]
    assert "weekly_plan" in out["conversation_state"].unresolved_intents
    assert len(out["curated_memories"]) >= 1


def test_sessions_are_isolated() -> None:
    adapter = UxOverlayRuntimeAdapter()

    adapter.process_turn(
        session_id="alpha",
        base_response="A",
        user_text="Discuss career growth and long term planning",
        unresolved_intents=["career_followup"],
        emotional_label="focused",
    )

    adapter.process_turn(
        session_id="beta",
        base_response="B",
        user_text="Let's talk hobbies",
        unresolved_intents=[],
        emotional_label="light",
    )

    alpha = adapter.snapshot("alpha")
    beta = adapter.snapshot("beta")

    assert "career_followup" in alpha["continuity"].unresolved_intents
    assert "career_followup" not in beta["continuity"].unresolved_intents


def test_control_api_edits_curated_memory_store() -> None:
    adapter = UxOverlayRuntimeAdapter()

    adapter.process_turn(
        session_id="s1",
        base_response="x",
        user_text="Remember this important topic for next time",
        raw_memory_events=[
            {
                "text": "Remember this important topic for next time and family planning.",
                "created_at": datetime.now(timezone.utc),
                "emotional_intensity": 0.8,
            }
        ],
    )

    state = adapter.ensure_session("s1")
    assert state.memory_store

    key = next(iter(state.memory_store.keys()))
    api = adapter.control_api("s1")
    updated = api.edit_memory(key, summary="edited summary", emotional_weight=0.4)

    assert updated["summary"] == "edited summary"
    assert updated["emotional_weight"] == 0.4


def test_adapter_does_not_mutate_truth_binding_state() -> None:
    adapter = UxOverlayRuntimeAdapter()

    state = build_synthetic_state(
        turn_id="adapter-isolation",
        stages=["plan", "execute", "respond"],
        tools=["search_web"],
        memory_keys=["family_topic"],
    )
    baseline = {
        "receipts": list(state["_execution_receipts"]),
        "plan": dict(state["plan"]),
        "tool_ir": {"executions": list(state["tool_ir"]["executions"])},
        "memory_structured": dict(state["memory_structured"]),
    }

    _ = adapter.process_turn(
        session_id="s1",
        base_response="core result",
        user_text="hello",
        user_sentiment="neutral",
    )

    validator = ClaimEvidenceValidator()
    claim = validator.extract_claim_from_state(state, "adapter-isolation")
    evidence = validator.extract_evidence_from_state(state, "adapter-isolation")
    result = validator.validate(claim, evidence)

    assert result.valid, result.to_dict()
    assert state["_execution_receipts"] == baseline["receipts"]
    assert state["plan"] == baseline["plan"]
    assert state["tool_ir"] == baseline["tool_ir"]
    assert state["memory_structured"] == baseline["memory_structured"]
