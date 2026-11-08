from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="Current memory ranking can return importance-based results for unrelated queries.",
)
def test_unrelated_question_does_not_inject_a_memory(bot, monkeypatch):
    bot.mutate_memory_store(memories=[
        {"summary": "My preferred fruit is an apple.", "category": "preferences", "importance_score": 0.9},
        {"summary": "My weekend hobby is pottery.", "category": "hobbies", "importance_score": 0.8},
    ])
    monkeypatch.setattr(bot, "semantic_memory_matches", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    assert bot.memory_query.relevant_memories_for_input(
        "What is my favorite constellation?", graph_result={},
    ) == []
