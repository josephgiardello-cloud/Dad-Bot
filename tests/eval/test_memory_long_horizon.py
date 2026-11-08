from __future__ import annotations

import json

import pytest


@pytest.mark.unit
def test_time_scoped_lexical_recall_and_no_answer_across_months(bot, monkeypatch):
    memories = [
        {"summary": "My preferred fruit is an apple.", "category": "preferences", "event_time": "2025-01-15"},
        {"summary": "My preferred fruit is a pear.", "category": "preferences", "event_time": "2025-03-15"},
        {"summary": "My preferred fruit is a peach.", "category": "preferences", "event_time": "2025-06-15"},
        {"summary": "My hobby is pottery.", "category": "hobbies", "event_time": "2025-02-20"},
        {"summary": "My pet is a rabbit.", "category": "pets", "event_time": "2025-05-10"},
    ]
    bot.mutate_memory_store(memories=memories)
    monkeypatch.setattr(bot, "semantic_memory_matches", lambda *_args, **_kwargs: [])
    cases = [
        ("preferred fruit", "2025-01-01", "2025-01-31", "apple"),
        ("preferred fruit", "2025-03-01", "2025-03-31", "pear"),
        ("preferred fruit", "2025-06-01", "2025-06-30", "peach"),
        ("pottery", "2025-01-01", "2025-06-30", "pottery"),
        ("rabbit", "2025-01-01", "2025-06-30", "rabbit"),
        ("constellation", "2025-01-01", "2025-06-30", None),
    ]

    correct = 0
    answer_cases = 0
    hits = 0
    returned = 0
    for query, start, end, expected in cases:
        result = bot.memory_query.relevant_memories_for_input(
            query, limit=1, graph_result={}, from_date=start, to_date=end,
        )
        returned += len(result)
        if expected is None:
            assert result == []
            continue
        answer_cases += 1
        if result and expected in result[0]["summary"].lower():
            hits += 1
            correct += 1
    assert hits / answer_cases == 1.0
    assert correct / returned == 1.0

    saved = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))["memories"]
    assert len(saved) == len(memories)
    assert {item["event_time"] for item in saved} == {item["event_time"] for item in memories}
    assert all(item["access_count"] == 1 for item in saved)
