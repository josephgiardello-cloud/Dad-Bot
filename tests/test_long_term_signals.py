def test_should_evolve_persona_respects_session_gap(bot):
    bot.CADENCE = {
        "persona_evolution_min_sessions": 4,
        "persona_evolution_session_gap": 3,
    }
    bot.MEMORY_STORE["session_archive"] = [{"id": str(index), "summary": f"s{index}", "created_at": f"2026-04-{index + 1:02d}T10:00:00", "topics": ["general"], "dominant_mood": "neutral", "turn_count": 2} for index in range(5)]
    bot.MEMORY_STORE["persona_evolution"] = [{"trait": "more direct", "session_count": 3, "applied_at": "2026-04-01T10:00:00"}]

    assert bot.long_term_signals.should_evolve_persona() is False

    bot.MEMORY_STORE["session_archive"].append({"id": "6", "summary": "s6", "created_at": "2026-04-07T10:00:00", "topics": ["general"], "dominant_mood": "neutral", "turn_count": 2})

    assert bot.long_term_signals.should_evolve_persona() is True


def test_evolve_persona_persists_entry_and_queues_announcement(bot, mocker):
    bot.MEMORY_STORE["session_archive"] = [
        {"id": str(index), "summary": f"Session {index}", "created_at": f"2026-04-{index + 1:02d}T10:00:00", "topics": ["work"], "dominant_mood": "neutral", "turn_count": 3}
        for index in range(6)
    ]
    mocker.patch.object(bot.long_term_signals, "should_evolve_persona", return_value=True)
    mocker.patch.object(bot, "call_ollama_chat", side_effect=[
        {"message": {"content": '{"new_trait": "gentler when you are hard on yourself", "reason": "You trust me more when you are self-critical."}'}},
        {"message": {"content": '{"score": 9, "approved": true, "feedback": "Specific and steady.", "suggested_refinement": null}'}}
    ])

    entry = bot.long_term_signals.evolve_persona()

    assert entry is not None
    assert entry["trait"] == "gentler when you are hard on yourself"
    assert entry["critique_score"] == 9
    assert bot.persona_evolution_history()[-1]["trait"] == entry["trait"]
    assert any("little more gentler when you are hard on yourself" in message["message"] for message in bot.pending_proactive_messages())


def test_evolve_persona_rejects_low_quality_trait_after_critique(bot, mocker):
    bot.MEMORY_STORE["session_archive"] = [
        {"id": str(index), "summary": f"Session {index}", "created_at": f"2026-04-{index + 1:02d}T10:00:00", "topics": ["work"], "dominant_mood": "neutral", "turn_count": 3}
        for index in range(6)
    ]
    mocker.patch.object(bot.long_term_signals, "should_evolve_persona", return_value=True)
    mocker.patch.object(bot, "call_ollama_chat", side_effect=[
        {"message": {"content": '{"new_trait": "more nice", "reason": "It sounds good."}'}},
        {"message": {"content": '{"score": 4, "approved": false, "feedback": "Too generic.", "suggested_refinement": null}'}}
    ])

    entry = bot.long_term_signals.evolve_persona()

    assert entry is None
    assert bot.persona_evolution_history() == []


def test_update_trait_impact_from_relationship_feedback_reinforces_latest_trait(bot):
    bot.MEMORY_STORE["persona_evolution"] = [
        {
            "trait": "more coach-like",
            "reason": "Tony responds well to structure",
            "announcement": "",
            "session_count": 8,
            "applied_at": "2026-04-10T20:00:00",
            "last_reinforced_at": "2026-04-10T20:00:00",
            "strength": 1.0,
            "impact_score": 0.0,
        }
    ]

    updated = bot.long_term_signals.update_trait_impact_from_relationship_feedback(2, 1)

    assert updated[-1]["impact_score"] == 3.0
    assert updated[-1]["strength"] > 1.0


def test_reject_persona_trait_removes_latest_entry(bot):
    bot.MEMORY_STORE["persona_evolution"] = [
        {"trait": "more reflective", "reason": "", "announcement": "", "session_count": 5, "applied_at": "2026-04-08T10:00:00"},
        {"trait": "gentler when you are hard on yourself", "reason": "", "announcement": "", "session_count": 7, "applied_at": "2026-04-10T10:00:00"},
    ]

    removed = bot.long_term_signals.reject_persona_trait("trait")

    assert removed is not None
    assert removed["trait"] == "gentler when you are hard on yourself"
    assert [entry["trait"] for entry in bot.persona_evolution_history()] == ["more reflective"]


def test_consolidate_persona_evolution_history_keeps_strongest_ranked_traits(bot):
    history = []
    for index in range(10):
        history.append({
            "trait": f"trait {index}",
            "reason": f"reason {index}",
            "announcement": "",
            "session_count": 10 + index,
            "applied_at": f"2026-04-{index + 1:02d}T10:00:00",
            "last_reinforced_at": f"2026-04-{index + 1:02d}T10:00:00",
            "strength": 0.5 + index * 0.2,
            "impact_score": float(index),
            "critique_score": min(10, 2 + index),
        })

    consolidated = bot.long_term_signals.consolidate_persona_evolution_history(history)

    assert len(consolidated) == 8
    kept_traits = [entry["trait"] for entry in consolidated]
    assert "trait 0" not in kept_traits
    assert "trait 1" not in kept_traits
    assert kept_traits[-1] == "trait 9"


def test_generate_wisdom_insight_uses_mocked_ollama_and_deduplicates_recent_entries(bot, mocker):
    bot.CADENCE = {
        "wisdom_min_archived_sessions": 2,
        "wisdom_turn_interval": 3,
    }
    bot.MEMORY_STORE["session_archive"] = [
        {"id": "a", "summary": "Work has been heavy.", "created_at": "2026-04-01T10:00:00", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4},
        {"id": "b", "summary": "Budgeting is helping.", "created_at": "2026-04-02T10:00:00", "topics": ["finance"], "dominant_mood": "positive", "turn_count": 4},
    ]
    bot.MEMORY_STORE["consolidated_memories"] = [{"summary": "Tony has been carrying work stress while trying to stay steady.", "category": "work", "source_count": 2, "updated_at": "2026-04-02"}]
    bot.MEMORY_STORE["memory_graph"] = {
        "nodes": [{"id": "category:work", "label": "work", "type": "category", "weight": 3}],
        "edges": [{"source": "work", "target": "stressed", "weight": 2}],
        "updated_at": "2026-04-02",
    }
    bot.session_turn_count = lambda: 3
    bot.top_relationship_topics = lambda *_args, **_kwargs: ["work"]
    mocker.patch.object(bot, "call_ollama_chat", return_value={
        "message": {"content": '{"summary": "Tony does better when he gives himself a smaller next step instead of carrying the whole week at once.", "topic": "work"}'}
    })

    first = bot.long_term_signals.generate_wisdom_insight("Work still feels heavy.")
    second = bot.long_term_signals.generate_wisdom_insight("Work still feels heavy.", force=True)

    assert first is not None
    assert first["topic"] == "work"
    assert second == first
    assert len(bot.wisdom_catalog()) == 1


def test_detect_life_patterns_skips_existing_pattern_identity(bot):
    bot.CADENCE = {
        "life_pattern_min_archived_sessions": 4,
        "life_pattern_window": 12,
        "life_pattern_min_occurrences": 3,
        "life_pattern_confidence_threshold": 70,
        "life_pattern_queue_limit": 2,
    }
    bot.MEMORY_STORE["life_patterns"] = [{
        "summary": "Tony often carries work stressed on Saturdays.",
        "topic": "work",
        "mood": "stressed",
        "day_hint": "Saturday",
        "confidence": 85,
        "last_seen_at": "2026-04-11T20:00:00",
        "proactive_message": "Saturdays seem heavy for work lately.",
    }]
    bot.MEMORY_STORE["session_archive"] = [
        {"summary": "Work felt heavy again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-04T20:00:00", "id": "a"},
        {"summary": "Saturday work dread again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-11T20:00:00", "id": "b"},
        {"summary": "Another tough Saturday night about work.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-18T20:00:00", "id": "c"},
        {"summary": "Still talking about work stress.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-25T20:00:00", "id": "d"},
    ]

    detected = bot.long_term_signals.detect_life_patterns(force=True)

    assert detected == []
    assert len(bot.life_patterns()) == 1


def test_detect_life_patterns_queues_proactive_message_for_new_pattern(bot):
    bot.CADENCE = {
        "life_pattern_min_archived_sessions": 4,
        "life_pattern_window": 12,
        "life_pattern_min_occurrences": 3,
        "life_pattern_confidence_threshold": 70,
        "life_pattern_queue_limit": 2,
    }
    bot.MEMORY_STORE["session_archive"] = [
        {"summary": "Sunday work dread again.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-05T20:00:00", "id": "a"},
        {"summary": "Another tough Sunday night about work.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-12T20:00:00", "id": "b"},
        {"summary": "Still talking about work stress.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-19T20:00:00", "id": "c"},
        {"summary": "Work feels heavy every Sunday.", "topics": ["work"], "dominant_mood": "stressed", "turn_count": 4, "created_at": "2026-04-26T20:00:00", "id": "d"},
    ]

    detected = bot.detect_life_patterns(force=True)

    assert len(detected) == 1
    assert detected[0]["day_hint"] == "Sunday"
    assert any(message["source"] == "life-pattern" for message in bot.pending_proactive_messages())
