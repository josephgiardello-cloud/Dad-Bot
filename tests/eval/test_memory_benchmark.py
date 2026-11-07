from __future__ import annotations

from datetime import date

import pytest

from tests.eval.memory_benchmark import MEMORY_BENCHMARK_FACTS, memory_benchmark_cases

pytestmark = pytest.mark.unit


def test_memory_retrieval_benchmark_meets_recall_threshold(bot, monkeypatch):
    cases = memory_benchmark_cases()
    assert len(cases) >= 200
    assert len({case.question for case in cases}) == len(cases)

    today = date.today().isoformat()
    bot.MEMORY_STORE = {
        **bot.default_memory_store(),
        "memories": [
            {
                "summary": fact.summary,
                "category": fact.category,
                "confidence": 0.9,
                "importance_score": 0.9,
                "created_at": today,
                "updated_at": today,
            }
            for fact in MEMORY_BENCHMARK_FACTS
        ],
    }
    bot.sync_semantic_memory_index(bot.MEMORY_STORE["memories"])
    bot.sync_graph_store()
    monkeypatch.setattr(bot, "queue_semantic_memory_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args, **_kwargs: None)

    hits_at_1 = 0
    hits_at_3 = 0
    misses: list[tuple[str, str, list[str]]] = []
    for case in cases:
        retrieved = bot.memory_query.relevant_memories_for_input(
            case.question,
            limit=3,
        )
        summaries = [str(memory.get("summary") or "") for memory in retrieved]
        expected = case.expected_summary.casefold()
        top_3_hit = any(expected in summary.casefold() for summary in summaries[:3])
        hits_at_1 += bool(summaries and expected in summaries[0].casefold())
        hits_at_3 += top_3_hit
        if not top_3_hit and len(misses) < 5:
            misses.append((case.question, case.expected_summary, summaries))

    assert hits_at_1 / len(cases) >= 0.90, f"top-1 misses: {misses}"
    assert hits_at_3 / len(cases) >= 0.98, f"top-3 misses: {misses}"
