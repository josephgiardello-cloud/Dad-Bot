from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_diversification_reorders_without_replacing_top_k(bot):
    ranked = [
        (0.99, {"summary": "Top work memory", "category": "work"}),
        (0.98, {"summary": "Second work memory", "category": "work"}),
        (0.97, {"summary": "Top family memory", "category": "family"}),
        (0.96, {"summary": "Weaker home memory", "category": "home"}),
    ]
    selected = bot.memory_query.select_diverse_ranked_memories(ranked, 3)
    assert selected == [ranked[0], ranked[2], ranked[1]]
    assert {item[1]["summary"] for item in selected} == {item[1]["summary"] for item in ranked[:3]}


def test_diversification_deduplicates_before_top_k_selection(bot):
    memory = {"summary": "Same work memory", "category": "work"}
    ranked = [
        (0.99, memory),
        (0.98, dict(memory)),
        (0.97, {"summary": "Other work memory", "category": "work"}),
        (0.96, {"summary": "Family memory", "category": "family"}),
    ]
    selected = bot.memory_query.select_diverse_ranked_memories(ranked, 3)
    assert selected == [ranked[0], ranked[3], ranked[2]]
    assert len(selected) == 3


@pytest.mark.parametrize("limit", [1, 2, 3, 8])
def test_diversification_same_category_preserves_score_order(bot, limit):
    ranked = [(1 - number * 0.1, {"summary": f"Memory {number}", "category": "work"}) for number in range(5)]
    assert bot.memory_query.select_diverse_ranked_memories(ranked, limit) == ranked[:limit]
    assert bot.memory_query.select_diverse_ranked_memories([], limit) == []


def test_diversification_supports_alternative_memory_index(bot):
    ranked = [
        ({"summary": "Work first", "category": "work"}, 0.99),
        ({"summary": "Work second", "category": "work"}, 0.98),
        ({"summary": "Family third", "category": "family"}, 0.97),
    ]
    assert bot.memory_query.select_diverse_ranked_memories(ranked, 3, memory_index=0) == [
        ranked[0],
        ranked[2],
        ranked[1],
    ]


def test_reordered_display_does_not_inflate_top_score_gap(bot, monkeypatch):
    memories = [
        {"summary": "Strong work fact", "category": "work"},
        {"summary": "Close work fact", "category": "work"},
        {"summary": "Weaker family fact", "category": "family"},
    ]
    scores = dict(zip((memory["summary"] for memory in memories), (0.9, 0.89, 0.7), strict=True))
    monkeypatch.setattr(bot, "memory_catalog", lambda: memories)
    monkeypatch.setattr(bot, "semantic_memory_matches", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(bot.memory_query, "score", lambda memory, _context: scores[memory["summary"]])
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    selected = bot.memory_query.relevant_memories_for_input("work", limit=3)
    assert [memory["summary"] for memory in selected] == [
        "Strong work fact",
        "Weaker family fact",
        "Close work fact",
    ]
    assert bot.memory_query.get_retrieval_diagnostics()["top_score_gap"] == 0.01
