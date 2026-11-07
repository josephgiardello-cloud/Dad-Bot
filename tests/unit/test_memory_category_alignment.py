from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("impact", [0.0, 2.5])
@pytest.mark.parametrize("category", ["work", "routines", "travel", "general"])
def test_inferred_category_does_not_penalize_neutral_memory(bot, category, impact):
    memory = {"summary": "Synthetic independent fact", "category": category, "mood": "neutral", "impact_score": impact}
    assert bot.memory_query.memory_alignment_weight(memory, query_category="work") == 1.0


def test_category_removal_preserves_recent_topic_and_mood_alignment(bot):
    memory = {"summary": "Synthetic independent fact", "category": "travel", "mood": "sad", "impact_score": 0.0}
    assert bot.memory_query.memory_alignment_weight(memory, query_category="general", recent_topics=["work"]) == 0.75
    assert bot.memory_query.memory_alignment_weight(memory, query_category="travel", query_mood="happy") == 0.6


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Tell me about work", "I commute by train and work on billing."),
        ("What do you know about my family?", "My daughter is Maya and we eat together on Thursday."),
        ("What are my routines?", "My weekday wake-up routine starts at six."),
    ],
)
def test_broad_query_topical_memories_remain_retrievable(bot, monkeypatch, query, expected):
    memories = [
        {"summary": "I commute by train and work on billing.", "category": "work"},
        {"summary": "My daughter is Maya and we eat together on Thursday.", "category": "family"},
        {"summary": "My weekday wake-up routine starts at six.", "category": "routines"},
    ]
    monkeypatch.setattr(bot, "memory_catalog", lambda: memories)
    monkeypatch.setattr(
        bot,
        "semantic_memory_matches",
        lambda *_args, **_kwargs: [(0.9 if memory["summary"] == expected else 0.3, memory) for memory in memories],
    )
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    assert bot.memory_query.relevant_memories_for_input(query, limit=1)[0]["summary"] == expected
