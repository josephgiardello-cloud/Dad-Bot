from __future__ import annotations

import pytest


@pytest.mark.unit
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


@pytest.mark.unit
def test_zero_semantic_scores_and_unrelated_graph_do_not_inject_memory(bot, monkeypatch):
    bot.mutate_memory_store(memories=[
        {"summary": "My preferred fruit is an apple.", "category": "preferences", "importance_score": 0.9},
    ])
    monkeypatch.setattr(
        bot, "semantic_memory_matches",
        lambda _query, memories, **_kwargs: [(0.0, memory) for memory in memories],
    )
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args: None)
    graph = {"supporting_evidence": [{"summary": "unrelated kayaking", "matched_nodes": []}]}
    assert bot.memory_query.relevant_memories_for_input(
        "What is my favorite constellation?", graph_result=graph,
    ) == []
