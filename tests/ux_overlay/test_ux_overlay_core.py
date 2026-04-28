from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dadbot.uril.truth_binding import ClaimEvidenceValidator, build_synthetic_state
from dadbot.ux_overlay import (
    ConversationContinuityEngine,
    InteractionState,
    InteractionStateEngine,
    MemoryCurator,
    ModalAdapter,
    ResponseProfile,
    ResponseShapingEngine,
    UXControlAPI,
)


def test_interaction_state_engine_updates_social_state_only() -> None:
    engine = InteractionStateEngine()
    before = engine.state
    after = engine.apply_turn_feedback(
        positive_signal=0.8,
        user_sentiment="happy",
        conversation_break=False,
    )

    assert after.user_affinity > before.user_affinity
    assert after.emotional_tone in {"friendly", "playful", "calm"}
    assert 0.0 <= after.continuity_score <= 1.0


def test_memory_curator_filters_scores_and_compresses() -> None:
    now = datetime.now(timezone.utc)
    curator = MemoryCurator(minimum_length=10, meaningful_terms=("remember", "important"))

    raw = [
        {
            "text": "Remember this important milestone from our family planning chat.",
            "created_at": now - timedelta(hours=2),
            "emotional_intensity": 0.7,
        },
        {
            "text": "This is short",
            "created_at": now,
            "emotional_intensity": 0.9,
        },
        {
            "text": "Remember this important milestone from our family planning chat.",
            "created_at": now - timedelta(hours=1),
            "emotional_intensity": 0.8,
        },
    ]

    ingested = curator.ingestion_filter(raw)
    assert len(ingested) == 2

    curated = curator.compress(ingested)
    assert len(curated) >= 1
    assert 0.0 <= curated[0].emotional_weight <= 1.0
    assert "related" in curated[0].summary or len(curated[0].summary) > 0


def test_response_shaper_preserves_original_and_changes_presentation() -> None:
    shaper = ResponseShapingEngine()
    profile = ResponseProfile(verbosity=0.8, warmth=0.9, assertiveness=0.8, curiosity_level=0.8)
    interaction = InteractionState(emotional_tone="friendly")

    original = "We can tackle this by splitting the task into three deterministic steps."
    shaped = shaper.shape(content=original, profile=profile, interaction=interaction)

    assert shaped.original == original
    assert shaped.rendered != original
    assert "warmth" in shaped.metadata


def test_conversation_continuity_tracks_topics_intents_and_arc() -> None:
    engine = ConversationContinuityEngine()
    state = engine.ingest_turn(
        topics=["career", "health"],
        unresolved_intents=["follow_up_checkin"],
        emotional_label="supportive",
    )

    assert "career" in state.active_topics
    assert "follow_up_checkin" in state.unresolved_intents
    assert state.emotional_arc[-1] == "supportive"

    state = engine.resolve_intent("follow_up_checkin")
    assert "follow_up_checkin" not in state.unresolved_intents


def test_ux_control_api_allows_tuning_and_memory_edit() -> None:
    interaction = InteractionState()
    profile = ResponseProfile()
    memory_store = {"m1": {"summary": "old", "emotional_weight": 0.2}}

    api = UXControlAPI(
        interaction_state=interaction,
        response_profile=profile,
        memory_store=memory_store,
    )

    api.set_tone("calm")
    api.adjust_emotion(0.9)
    api.set_verbosity(0.3)
    api.set_warmth(0.75)
    updated = api.edit_memory("m1", summary="updated", emotional_weight=0.85)

    assert interaction.emotional_tone == "calm"
    assert interaction.engagement_level == 0.9
    assert profile.verbosity == 0.3
    assert profile.warmth == 0.75
    assert updated["summary"] == "updated"
    assert updated["emotional_weight"] == 0.85


def test_modal_adapter_defaults_future_proof_flags() -> None:
    adapter = ModalAdapter()
    assert adapter.text_enabled is True
    assert adapter.voice_enabled is False
    assert adapter.avatar_enabled is False
    assert adapter.streaming_enabled is False


def test_ux_overlay_does_not_mutate_truth_binding_state() -> None:
    state = build_synthetic_state(
        turn_id="ux-isolation-001",
        stages=["plan", "execute", "respond"],
        tools=["search_web"],
        memory_keys=["family_topic"],
    )
    baseline_receipts = list(state["_execution_receipts"])

    shaper = ResponseShapingEngine()
    shaped = shaper.shape(
        content="Deterministic answer body.",
        profile=ResponseProfile(warmth=0.9, curiosity_level=0.8),
        interaction=InteractionState(emotional_tone="friendly"),
    )
    assert shaped.rendered

    validator = ClaimEvidenceValidator()
    claim = validator.extract_claim_from_state(state, "ux-isolation-001")
    evidence = validator.extract_evidence_from_state(state, "ux-isolation-001")
    result = validator.validate(claim, evidence)

    assert result.valid, result.to_dict()
    assert state["_execution_receipts"] == baseline_receipts
