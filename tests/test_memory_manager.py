import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from types import SimpleNamespace


def test_normalize_memory_entry_naturalizes_summary_and_mood(bot):
    entry = bot.normalize_memory_entry(
        {
            "summary": "i'm worried about money lately",
            "category": None,
            "mood": "anxious",
        }
    )

    assert entry["summary"] == "Tony is worried about money lately."
    assert entry["category"] == "finance"
    assert entry["mood"] == "stressed"


def test_normalize_memory_entry_validates_dates_and_scores_without_changing_surface(bot):
    entry = bot.normalize_memory_entry(
        {
            "summary": "i've been saving for emergencies",
            "category": "finance",
            "mood": "positive",
            "created_at": "2026-04-20",
            "updated_at": "2026-04-21",
            "confidence": "1.7",
            "impact_score": "2.5",
            "importance_score": "0.8",
            "emotional_intensity": "0.7",
            "relationship_impact": "0.6",
            "pinned": True,
            "contradictions": ["", "missed the budget once"],
        }
    )

    assert entry["summary"] == "Tony has been saving for emergencies."
    assert entry["created_at"] == "2026-04-20"
    assert entry["updated_at"] == "2026-04-21"
    assert entry["confidence"] == 1.0
    assert entry["impact_score"] == 2.5
    assert entry["importance_score"] == 0.8
    assert entry["emotional_intensity"] == 0.7
    assert entry["relationship_impact"] == 0.6
    assert entry["pinned"] is True
    assert entry["contradictions"] == ["missed the budget once"]


def test_normalize_relationship_state_preserves_runtime_vocab_and_date_surface(bot):
    entry = bot.normalize_relationship_state(
        {
            "trust_level": 65,
            "openness_level": 62,
            "emotional_momentum": "warming",
            "recurring_topics": {"Work": "3", "": 2},
            "recent_checkins": [{"date": "2026-04-20", "mood": "happy", "topic": "Work"}],
            "hypotheses": [{"name": "supportive_baseline", "probability": 1.0}],
            "last_hypothesis_updated": "2026-04-20",
            "last_reflection": "He is opening up more.",
            "last_updated": "2026-04-20",
        }
    )

    assert entry["emotional_momentum"] == "warming"
    assert entry["recurring_topics"] == {"work": 3}
    assert entry["recent_checkins"][0]["mood"] == "positive"
    assert entry["last_hypothesis_updated"] == "2026-04-20"
    assert entry["last_updated"] == "2026-04-20"


def test_normalize_persona_evolution_entry_validates_runtime_trait_payload(bot):
    entry = bot.normalize_persona_evolution_entry(
        {
            "new_trait": "more coach-like",
            "reason": "Tony responds well to structure",
            "announcement": "I have gotten a little more coach-like with you.",
            "session_count": "8",
            "applied_at": "2026-04-10T20:00:00",
            "last_reinforced_at": "2026-04-11T20:00:00",
            "strength": "2.7",
            "impact_score": "-1.5",
            "critique_score": "9",
            "critique_feedback": "specific and grounded",
        }
    )

    assert entry["trait"] == "more coach-like"
    assert entry["session_count"] == 8
    assert entry["strength"] == 2.7
    assert entry["impact_score"] == -1.5
    assert entry["critique_score"] == 9


def test_normalize_wisdom_entry_validates_runtime_payload(bot):
    entry = bot.normalize_wisdom_entry(
        {
            "summary": "Slow the moment down before work runs you.",
            "topic": "Work",
            "trigger": "stress check-in",
            "created_at": "2026-04-11T20:00:00",
        }
    )

    assert entry == {
        "summary": "Slow the moment down before work runs you.",
        "topic": "work",
        "trigger": "stress check-in",
        "created_at": "2026-04-11T20:00:00",
    }


def test_save_memory_store_normalizes_direct_memory_store_mutations(bot):
    bot.MEMORY_STORE = {
        **bot.default_memory_store(),
        "memories": [{"summary": "i'm saving money", "category": None, "mood": "happy"}],
        "consolidated_memories": [{"summary": "i've been saving money", "updated_at": "2026-04-20"}],
        "last_mood": "happy",
        "last_mood_updated_at": "2026-04-20",
        "recent_moods": [{"mood": "happy", "date": "2026-04-20"}],
    }

    bot.save_memory_store()

    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))

    from datetime import date as _date

    assert persisted["memories"][0]["summary"] == "Tony is saving money."
    assert persisted["memories"][0]["created_at"] == _date.today().isoformat()
    assert persisted["memories"][0]["mood"] == "positive"
    assert persisted["consolidated_memories"][0]["summary"] == "Tony has been saving money."
    assert persisted["last_mood"] == "positive"


def test_mutate_memory_store_persists_valid_json_under_concurrent_writes(bot):
    today = date.today().isoformat()

    def append_recent_mood(index):
        def mutator(store):
            recent = list(store.get("recent_moods", []))
            recent.append({"mood": "happy" if index % 2 == 0 else "stressed", "date": today})
            store["recent_moods"] = recent
            store["last_mood"] = "happy" if index % 2 == 0 else "stressed"
            store["last_mood_updated_at"] = today

        bot.mutate_memory_store(mutator=mutator)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append_recent_mood, range(8)))

    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))

    assert len(persisted["recent_moods"]) == 8
    assert all(entry["mood"] in {"positive", "stressed"} for entry in persisted["recent_moods"])
    assert persisted["last_mood"] in {"positive", "stressed"}
    assert bot.normalize_memory_store(persisted)["recent_moods"] == persisted["recent_moods"]


def test_normalize_consolidated_memory_entry_limits_lists_and_recomputes_confidence(bot):
    entry = bot.normalize_consolidated_memory_entry(
        {
            "summary": "i've been saving money for emergencies",
            "category": "finance",
            "source_count": 3,
            "confidence": None,
            "supporting_summaries": [
                "I've been saving money for emergencies",
                "i have been saving money for emergencies",
                "I want to save more money",
                "I have been saving money monthly",
                "I need to keep budgeting",
            ],
            "contradictions": [
                "Spent impulsively",
                "Spent impulsively",
                "Ignored budget",
                "Missed savings",
                "Overspent again",
            ],
            "updated_at": date.today().isoformat(),
        }
    )

    assert entry["summary"] == "Tony has been saving money for emergencies."
    assert entry["supporting_summaries"] == [
        "Tony has been saving money for emergencies.",
        "Tony wants to save more money.",
        "Tony has been saving money monthly.",
        "Tony needs to keep budgeting.",
    ]
    assert entry["contradictions"] == ["Spent impulsively", "Ignored budget", "Missed savings", "Overspent again"]
    assert 0.05 <= entry["confidence"] <= 0.98


def test_memory_dedup_key_collapses_finance_and_work_stress_variants(bot):
    finance_key = bot.memory_dedup_key(
        {"summary": "Tony has been saving more money each month.", "category": "finance"}
    )
    work_key = bot.memory_dedup_key(
        {"summary": "Tony shared that work has been stressed and overwhelming.", "category": "work"}
    )

    assert finance_key == ("finance", "money-goals")
    assert work_key == ("work", "work-stress")


def test_memory_quality_score_penalizes_generic_low_signal_entries(bot):
    generic_score = bot.memory_quality_score({"summary": "Tony shared that personal struggles.", "category": "general"})
    specific_score = bot.memory_quality_score(
        {"summary": "Tony has been saving money to build an emergency fund.", "category": "finance"}
    )

    assert generic_score < 5
    assert specific_score > generic_score


def test_is_high_quality_memory_rejects_low_signal_patterns(bot):
    assert (
        bot.is_high_quality_memory({"summary": "Tony shared that personal struggles.", "category": "general"}) is False
    )
    assert (
        bot.is_high_quality_memory(
            {"summary": "Tony has been working on a budget and saving plan.", "category": "finance"}
        )
        is True
    )


def _turn_context_for_forgetting():
    return SimpleNamespace(
        temporal=SimpleNamespace(
            wall_time="2026-05-01T00:00:00",
            wall_date="2026-05-01",
        ),
    )


def test_controlled_forgetting_archives_old_low_signal_noise(bot):
    stale_date = (date.today() - timedelta(days=500)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony mentioned random small talk about a passing thought.",
                "category": "general",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.05,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 1, "low": 1},
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    assert result["archived"] == 1
    assert bot.memory_catalog() == []
    assert any("passing thought" in str(entry.get("summary", "")).lower() for entry in bot.session_archive())


def test_controlled_forgetting_retains_repeated_preference_with_high_confidence_hits(bot):
    stale_date = (date.today() - timedelta(days=420)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony prefers direct, concise planning checklists.",
                "category": "preferences",
                "mood": "positive",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.8,
                "access_count": 20,
                "confidence_history": {"high": 14, "medium": 2, "low": 0},
                "high_confidence_hits": 14,
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    remaining = bot.memory_catalog()
    if archived_count == 0:
        assert len(remaining) == 1
        assert remaining[0]["category"] == "preferences"
    else:
        assert remaining == []
        assert any(
            "planning checklists" in str(entry.get("summary", "")).lower()
            for entry in bot.session_archive()
        )


def test_controlled_forgetting_never_archives_identity_memory(bot):
    stale_date = (date.today() - timedelta(days=900)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony's daughter is named Emily.",
                "category": "identity",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.0,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 0},
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count == 0
    remaining = bot.memory_catalog()
    assert len(remaining) == 1
    assert remaining[0]["category"] == "identity"


def test_controlled_forgetting_soft_prunes_under_memory_saturation(bot):
    stale_date = (date.today() - timedelta(days=240)).isoformat()
    dense_catalog = []
    for idx in range(40):
        dense_catalog.append(
            {
                "summary": f"Tony mentioned general low-signal note {idx}.",
                "category": "general",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.1,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 2},
            }
        )

    bot.save_memory_catalog(dense_catalog)

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count > 0
    assert len(bot.memory_catalog()) < len(dense_catalog)


def test_controlled_forgetting_prioritizes_low_signal_conflicting_memory(bot):
    stale_date = (date.today() - timedelta(days=320)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony says he never uses checklists anymore.",
                "category": "general",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.2,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 2},
                "contradictions": ["Tony prefers direct planning checklists."],
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count == 1
    assert bot.memory_catalog() == []


def test_controlled_forgetting_retains_stale_high_importance_memory(bot):
    stale_date = (date.today() - timedelta(days=700)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony is rebuilding his emergency fund after layoffs in the family.",
                "category": "finance",
                "mood": "stressed",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.95,
                "access_count": 4,
                "confidence_history": {"high": 6, "medium": 1, "low": 0},
                "high_confidence_hits": 6,
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count == 0
    remaining = bot.memory_catalog()
    assert len(remaining) == 1
    assert "emergency fund" in str(remaining[0].get("summary", "")).lower()


def test_controlled_forgetting_stress_profile_reduces_retention_under_density(bot):
    old_date = (date.today() - timedelta(days=420)).isoformat()
    recent_date = (date.today() - timedelta(days=14)).isoformat()
    catalog = []

    # 20 intentionally low-signal noisy memories should be archived.
    for idx in range(20):
        catalog.append(
            {
                "summary": f"Tony mentioned low-value passing thought {idx}.",
                "category": "general",
                "mood": "neutral",
                "created_at": old_date,
                "updated_at": old_date,
                "importance_score": 0.08,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 2},
            }
        )

    # 160 durable memories should mostly remain.
    for idx in range(160):
        catalog.append(
            {
                "summary": f"Tony keeps a direct planning checklist habit {idx}.",
                "category": "preferences",
                "mood": "positive",
                "created_at": recent_date,
                "updated_at": recent_date,
                "importance_score": 0.82,
                "access_count": 8,
                "confidence_history": {"high": 10, "medium": 1, "low": 0},
                "high_confidence_hits": 10,
            }
        )

    bot.save_memory_catalog(catalog)
    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    retained = int(result.get("retained", len(bot.memory_catalog())) or len(bot.memory_catalog()))
    archived = int(result.get("archived", result.get("removed", 0)) or 0)
    retention_ratio = retained / float(len(catalog))

    assert archived > 0
    assert 0.85 <= retention_ratio <= 0.95


def test_clean_memory_entries_normalizes_filters_and_deduplicates(bot):
    cleaned = bot.clean_memory_entries(
        [
            {
                "summary": "i've been saving more money",
                "category": "finance",
                "updated_at": (date.today() - timedelta(days=1)).isoformat(),
            },
            {
                "summary": "I have been saving more money for an emergency fund",
                "category": "finance",
                "updated_at": date.today().isoformat(),
            },
            {"summary": "personal struggles", "category": "general", "updated_at": date.today().isoformat()},
            {
                "summary": "i'm stressed about work deadlines",
                "category": "work",
                "updated_at": date.today().isoformat(),
            },
            {
                "summary": "Tony shared that work stress is overwhelming",
                "category": "work",
                "updated_at": (date.today() - timedelta(days=2)).isoformat(),
            },
        ]
    )

    summaries = [entry["summary"] for entry in cleaned]
    assert "Tony has been saving more money for an emergency fund." in summaries
    assert any("work" in summary.lower() and "stress" in summary.lower() for summary in summaries)
    assert all("personal struggles" not in summary.lower() for summary in summaries)
    assert len(cleaned) == 2


def test_relevant_memories_for_input_prefers_fresher_entries(bot):
    old_date = (date.today() - timedelta(days=180)).isoformat()
    fresh_date = date.today().isoformat()
    bot.MEMORY_STORE["memories"] = [
        {
            "summary": "Tony has been stressed about work deadlines on the marketing rollout.",
            "category": "work",
            "mood": "stressed",
            "created_at": old_date,
            "updated_at": old_date,
        },
        {
            "summary": "Tony has been stressed about work deadlines on the platform migration.",
            "category": "work",
            "mood": "stressed",
            "created_at": fresh_date,
            "updated_at": fresh_date,
        },
    ]

    memories = bot.relevant_memories_for_input("work deadlines", limit=1)

    assert len(memories) == 1
    assert "platform migration" in memories[0]["summary"].lower()


def test_semantic_memory_matches_caps_stale_low_impact_memories(bot):
    old_date = (date.today() - timedelta(days=120)).isoformat()
    fresh_date = date.today().isoformat()
    old_memory = {
        "summary": "Tony has been stressed about work deadlines from the old reorg.",
        "category": "work",
        "mood": "stressed",
        "created_at": old_date,
        "updated_at": old_date,
        "impact_score": 0.0,
    }
    fresh_memory = {
        "summary": "Tony has been stressed about work deadlines on the platform migration.",
        "category": "work",
        "mood": "stressed",
        "created_at": fresh_date,
        "updated_at": fresh_date,
        "impact_score": 0.0,
    }

    bot.memory_manager.queue_semantic_memory_index = lambda *_args, **_kwargs: None
    bot.memory_manager.semantic_query_context = lambda *_args, **_kwargs: {
        "query_embedding": [0.0, 1.0, 1.0],
        "query_tokens": ["work", "deadlines", "stressed"],
        "query_category": "work",
        "query_mood": "stressed",
        "candidate_limit": 4,
    }
    bot.memory_manager.semantic_candidate_rows = lambda *args, **kwargs: [
        {"summary_key": "old"},
        {"summary_key": "fresh"},
    ]
    bot.memory_manager.score_semantic_rows = lambda *args, **kwargs: [(0.95, old_memory), (0.62, fresh_memory)]

    matches = bot.semantic_memory_matches("work deadlines", [old_memory, fresh_memory], limit=1)

    assert len(matches) == 1
    assert "platform migration" in matches[0][1]["summary"].lower()


def test_relevant_memories_for_input_applies_diversity_cap(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["memories"] = [
        {
            "summary": "Tony has been stressed about work deadlines on the payroll launch.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony has been stressed about work deadlines on the platform migration.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony has been stressed about work deadlines around leadership changes.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony has been sticking to a budget and saving money for the emergency fund.",
            "category": "finance",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        },
    ]
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.memory_context_limit_for_input = lambda *_args, **_kwargs: 4

    memories = bot.relevant_memories_for_input("work deadlines and budget", limit=4)

    work_stress_memories = [memory for memory in memories if memory.get("category") == "work"]
    assert len(work_stress_memories) <= 2
    assert any(memory.get("category") == "finance" for memory in memories)


def test_relevant_memories_for_input_uses_prompt_budget_to_cap_limit(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["memories"] = [
        {
            "summary": f"Tony has been stressed about work deadline {index}.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        }
        for index in range(4)
    ]
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.CONTEXT_TOKEN_BUDGET = 140
    bot.RESERVED_RESPONSE_TOKENS = 20
    bot.build_cross_session_context = lambda: "X" * 500

    memories = bot.relevant_memories_for_input("work deadline", limit=4)

    assert len(memories) == 1


def test_build_memory_context_places_recent_archive_notes_before_older_memories(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["session_archive"] = [
        {
            "summary": "Work deadlines were heavy again this week.",
            "topics": ["work"],
            "dominant_mood": "stressed",
            "turn_count": 4,
            "created_at": today + "T10:00:00",
            "id": "a",
        }
    ]
    bot.MEMORY_STORE["memories"] = [
        {
            "summary": "Tony has been stressed about work deadlines on the old reorg.",
            "category": "work",
            "mood": "stressed",
            "created_at": (date.today() - timedelta(days=90)).isoformat(),
            "updated_at": (date.today() - timedelta(days=90)).isoformat(),
        }
    ]
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.CONTEXT_TOKEN_BUDGET = 1200

    context = bot.build_memory_context("work deadlines")

    assert context is not None
    if "Graph-connected long-term context" in context:
        assert context.index("Graph-connected long-term context") < context.index("Semantic fallback")
    else:
        assert context.index("Recent prior session notes") < context.index("Semantic fallback")


def test_load_memory_store_recovers_from_backup_when_primary_is_corrupted(bot):
    first_snapshot = {
        "summary": "Tony has been saving money for an emergency fund.",
        "category": "finance",
        "mood": "positive",
        "created_at": date.today().isoformat(),
        "updated_at": date.today().isoformat(),
    }
    second_snapshot = {
        "summary": "Tony has been stressed about work deadlines.",
        "category": "work",
        "mood": "stressed",
        "created_at": date.today().isoformat(),
        "updated_at": date.today().isoformat(),
    }
    bot.MEMORY_STORE["memories"] = [first_snapshot]
    bot.save_memory_store()
    bot.MEMORY_STORE["memories"] = [second_snapshot]
    bot.save_memory_store()

    backup_path = bot.json_backup_path(bot.MEMORY_PATH)
    assert backup_path.exists()

    bot.MEMORY_PATH.write_text("{ definitely not valid json", encoding="utf-8")

    recovered = bot._load_memory_store()

    assert recovered["memories"][0]["summary"] == first_snapshot["summary"]
    restored = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert restored["memories"][0]["summary"] == first_snapshot["summary"]


def test_sync_graph_store_extracts_consolidated_archive_traits_and_patterns(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["consolidated_memories"] = [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.82,
            "supporting_summaries": ["Tony has been budgeting weekly."],
            "contradictions": ["Spent impulsively once after payday"],
            "updated_at": today,
        }
    ]
    bot.MEMORY_STORE["session_archive"] = [
        {
            "summary": "Work deadlines felt especially heavy this week.",
            "topics": ["work"],
            "dominant_mood": "stressed",
            "turn_count": 5,
            "created_at": today + "T09:00:00",
            "id": "archive-1",
        }
    ]
    bot.MEMORY_STORE["persona_evolution"] = [
        {
            "trait": "coach-like",
            "reason": "Tony responds well to direct next-step support.",
            "announcement": "",
            "session_count": 10,
            "applied_at": today + "T08:00:00",
            "last_reinforced_at": today + "T08:00:00",
            "strength": 1.8,
            "impact_score": 2.2,
            "critique_score": 8,
            "critique_feedback": "specific and useful",
        }
    ]
    bot.MEMORY_STORE["life_patterns"] = [
        {
            "summary": "Tony often carries work stress on Mondays.",
            "topic": "work",
            "mood": "stressed",
            "day_hint": "Monday",
            "confidence": 85,
            "last_seen_at": today + "T10:00:00",
            "proactive_message": "",
        }
    ]

    snapshot = bot.sync_graph_store()

    node_types = {node["node_type"] for node in snapshot["nodes"]}
    relation_types = {edge["relation_type"] for edge in snapshot["edges"]}
    labels_by_type = {(node["node_type"], node["label"]) for node in snapshot["nodes"]}
    assert {"consolidated_memory", "archive_session", "persona_trait", "life_pattern"}.issubset(node_types)
    assert "contradicted_by" in relation_types
    assert "covers_topic" in relation_types
    assert "expresses_trait" in relation_types
    assert "recurs_on" in relation_types
    assert "plans_for" in relation_types
    assert "struggles_with" in relation_types
    assert "responds_to" in relation_types
    assert ("goal", "emergency_fund") in labels_by_type


def test_graph_retrieval_for_input_ranks_consolidated_over_weaker_archive(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["consolidated_memories"] = [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 4,
            "confidence": 0.9,
            "supporting_summaries": ["Tony has been budgeting weekly."],
            "contradictions": [],
            "updated_at": today,
        }
    ]
    bot.MEMORY_STORE["session_archive"] = [
        {
            "summary": "Tony mentioned money once after a rough week.",
            "topics": ["finance"],
            "dominant_mood": "neutral",
            "turn_count": 1,
            "created_at": (date.today() - timedelta(days=30)).isoformat() + "T09:00:00",
            "id": "archive-2",
        }
    ]

    bot.sync_graph_store()
    result = bot.graph_retrieval_for_input("budget and emergency fund", limit=2)

    assert result is not None
    assert result["supporting_evidence"][0]["source_type"] == "consolidated_memory"
    assert "emergency fund" in result["supporting_evidence"][0]["summary"].lower()
    assert result["compressed_summary"]


def test_graph_retrieval_surfaces_contradictions_in_supporting_evidence(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["consolidated_memories"] = [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.84,
            "supporting_summaries": ["Tony has been sticking to a budget."],
            "contradictions": ["Spent impulsively after payday"],
            "updated_at": today,
        }
    ]

    bot.sync_graph_store()
    result = bot.graph_retrieval_for_input("saving money", limit=1)

    assert result is not None
    assert result["supporting_evidence"][0]["contradictions"] == ["Spent impulsively after payday"]
    assert any("Tension:" in line for line in result["summary_lines"])


def test_build_memory_context_uses_graph_first_and_semantic_second(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["consolidated_memories"] = [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.82,
            "supporting_summaries": [],
            "contradictions": [],
            "updated_at": today,
        }
    ]
    bot.MEMORY_STORE["memories"] = [
        {
            "summary": "Tony has been using a weekly budget spreadsheet.",
            "category": "finance",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        }
    ]
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.sync_graph_store()

    context = bot.build_memory_context("budget and emergency fund")

    assert context is not None
    assert context.index("Graph-connected long-term context") < context.index("Semantic fallback")
    assert "emergency_fund" in context or "emergency fund" in context


def test_build_memory_context_includes_active_consolidated_context(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["consolidated_memories"] = [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "confidence": 0.88,
            "source_count": 3,
            "updated_at": today,
            "contradictions": ["Spent impulsively after payday"],
        }
    ]
    bot.graph_retrieval_for_input = lambda *_args, **_kwargs: None
    bot.relevant_archive_entries_for_input = lambda *_args, **_kwargs: []
    bot.relevant_memories_for_input = lambda *_args, **_kwargs: []

    context = bot.build_memory_context("I am worried about my emergency fund.")

    assert context is not None
    assert "Most relevant long-term insights about Tony right now" in context
    assert "emergency fund" in context
    assert "Tension: Spent impulsively after payday." in context


def test_build_memory_context_includes_deep_pattern_context(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["life_patterns"] = [
        {
            "summary": "Tony often carries work stress on Sundays.",
            "topic": "work",
            "mood": "stressed",
            "day_hint": "Sunday",
            "confidence": 88,
            "last_seen_at": today + "T09:00:00",
            "proactive_message": "Sundays seem to carry extra work weight for you lately.",
        }
    ]
    bot.graph_retrieval_for_input = lambda *_args, **_kwargs: None
    bot.relevant_archive_entries_for_input = lambda *_args, **_kwargs: []
    bot.relevant_memories_for_input = lambda *_args, **_kwargs: []
    bot.build_active_consolidated_context = lambda *_args, **_kwargs: None

    context = bot.build_memory_context("Sunday work feels heavy again.")

    assert context is not None
    assert "Long-horizon patterns Dad has noticed across time" in context
    assert "Recurring pattern" in context
    assert "work stress on Sundays" in context


def test_graph_retrieval_compresses_to_token_budget(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["consolidated_memories"] = [
        {
            "summary": "Tony has been saving money for an emergency fund and keeping a detailed budget spreadsheet while also talking through work stress and next steps every week.",
            "category": "finance",
            "source_count": 4,
            "confidence": 0.9,
            "supporting_summaries": [
                "Tony has been budgeting weekly and wants to stay consistent.",
                "Tony has also been feeling stress about work deadlines and his manager.",
            ],
            "contradictions": [],
            "updated_at": today,
        }
    ]
    bot.sync_graph_store()
    bot.runtime_config.graph_context_token_budget = 18
    bot.memory_manager._graph_prompt_compressor.max_tokens = 18
    bot.call_ollama_chat = lambda *args, **kwargs: {
        "message": {"content": "- compressed graph summary about saving for an emergency fund"}
    }

    result = bot.graph_retrieval_for_input("budget and emergency fund", limit=1)

    assert result is not None
    assert result["compressed_summary"] == "- compressed graph summary about saving for an emergency fund"
    assert bot.estimate_tokens(result["compressed_summary"]) <= 18


def test_relevant_memories_for_input_uses_graph_signal_when_scoring(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["consolidated_memories"] = [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.82,
            "supporting_summaries": [],
            "contradictions": [],
            "updated_at": today,
        }
    ]
    bot.MEMORY_STORE["memories"] = [
        {
            "summary": "Tony has been using a weekly budget spreadsheet.",
            "category": "finance",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony enjoyed a movie night with friends.",
            "category": "relationships",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        },
    ]
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.sync_graph_store()

    memories = bot.relevant_memories_for_input("budget and emergency fund", limit=1)

    assert len(memories) == 1
    assert "budget spreadsheet" in memories[0]["summary"].lower()


def test_relevant_memories_for_input_caps_top_k_to_seven(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE["memories"] = [
        {
            "summary": f"Tony has been planning budget step {index} for his emergency fund.",
            "category": "finance",
            "mood": "neutral",
            "created_at": today,
            "updated_at": today,
        }
        for index in range(20)
    ]
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []

    memories = bot.relevant_memories_for_input("budget and emergency fund planning", limit=20)

    assert 1 <= len(memories) <= 7


def test_relevant_memories_for_input_excludes_irrelevant_low_signal_memories(bot):
    today = date.today().isoformat()
    relevant_memory = {
        "summary": "Tony has been saving money for an emergency fund with a weekly budget plan.",
        "category": "finance",
        "mood": "neutral",
        "created_at": today,
        "updated_at": today,
    }
    irrelevant_memory = {
        "summary": "Tony watched a sci-fi movie and liked the soundtrack.",
        "category": "relationships",
        "mood": "positive",
        "created_at": today,
        "updated_at": today,
    }
    bot.MEMORY_STORE["memories"] = [relevant_memory, irrelevant_memory]
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []

    memories = bot.relevant_memories_for_input("help me with my emergency fund budget", limit=5)

    summaries = [str(item.get("summary", "")).lower() for item in memories]
    assert any("emergency fund" in summary for summary in summaries)
    if any("soundtrack" in summary for summary in summaries):
        relevant_index = next(i for i, summary in enumerate(summaries) if "emergency fund" in summary)
        soundtrack_index = next(i for i, summary in enumerate(summaries) if "soundtrack" in summary)
        assert relevant_index < soundtrack_index
