from __future__ import annotations

import os
from datetime import date

import pytest

from tests.eval.memory_benchmark import MEMORY_BENCHMARK_FACTS, memory_benchmark_cases

pytestmark = pytest.mark.unit

# The fixed, isolated 50-fact vault must meet this target with both embedding modes.
RECALL_AT_3_THRESHOLD = 0.98


def run_seeded_benchmark(bot, monkeypatch):
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
    # Index the same normalized catalog that retrieval will query, not raw seed
    # text whose keys change during normalization.
    catalog = bot.memory_catalog()
    assert len(catalog) == len(MEMORY_BENCHMARK_FACTS)
    assert all(
        any(fact.summary.casefold() in memory["summary"].casefold() for memory in catalog)
        for fact in MEMORY_BENCHMARK_FACTS
    )
    assert bot.wait_for_semantic_index_idle(timeout=30)
    bot.sync_semantic_memory_index(catalog)
    bot.sync_graph_store()
    monkeypatch.setattr(bot, "queue_semantic_memory_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot.memory_query, "_record_memory_access_signals", lambda *_args, **_kwargs: None)

    hits_at_1 = 0
    hits_at_3 = 0
    misses_at_1: list[tuple[str, str, list[str]]] = []
    misses_at_3: list[tuple[str, str, list[str]]] = []
    for case in cases:
        retrieved = bot.memory_query.relevant_memories_for_input(
            case.question,
            limit=3,
        )
        summaries = [str(memory.get("summary") or "") for memory in retrieved]
        expected = case.expected_summary.casefold()
        top_3_hit = any(expected in summary.casefold() for summary in summaries[:3])
        top_1_hit = bool(summaries and expected in summaries[0].casefold())
        hits_at_1 += top_1_hit
        hits_at_3 += top_3_hit
        if not top_1_hit:
            misses_at_1.append((case.question, case.expected_summary, summaries))
        if not top_3_hit:
            misses_at_3.append((case.question, case.expected_summary, summaries))

    print(f"Seeded benchmark: top-1 {hits_at_1}/{len(cases)}, top-3 {hits_at_3}/{len(cases)}")
    return len(cases), hits_at_1, hits_at_3, misses_at_1, misses_at_3


def test_memory_retrieval_benchmark_meets_recall_threshold(bot, monkeypatch):
    total, hits_at_1, hits_at_3, misses_at_1, misses_at_3 = run_seeded_benchmark(bot, monkeypatch)
    assert hits_at_1 / total >= 0.90, f"top-1 misses: {misses_at_1}"
    assert hits_at_3 / total >= RECALL_AT_3_THRESHOLD, f"top-3 misses: {misses_at_3}"


@pytest.mark.skipif(
    os.environ.get("DADBOT_RUN_REAL_EMBEDDING_BENCHMARK") != "1",
    reason="requires local Ollama and nomic-embed-text",
)
def test_memory_retrieval_with_real_ollama_embeddings(bot, monkeypatch):
    del bot.embed_texts
    bot.PREFERRED_EMBEDDING_MODELS = ("nomic-embed-text",)
    bot.ACTIVE_EMBEDDING_MODEL = "nomic-embed-text"
    vectors = bot.embed_texts(["isolated benchmark embedding probe"])
    assert len(vectors) == 1 and len(vectors[0]) == 768, "Ollama Nomic embedding unavailable"
    assert bot.ACTIVE_EMBEDDING_MODEL == "nomic-embed-text"
    semantic_manager = bot.memory_manager._semantic_manager
    original_embed = semantic_manager.embed_texts
    embedding_calls = []

    def checked_embed(texts, purpose="semantic retrieval"):
        vectors = original_embed(texts, purpose=purpose)
        expected_count = 1 if isinstance(texts, str) else len(texts)
        assert len(vectors) == expected_count and all(len(vector) == 768 for vector in vectors), (
            f"Nomic embeddings unavailable during {purpose}"
        )
        assert bot.ACTIVE_EMBEDDING_MODEL == "nomic-embed-text"
        embedding_calls.append(purpose)
        return vectors

    monkeypatch.setattr(semantic_manager, "embed_texts", checked_embed)
    total, hits_at_1, hits_at_3, misses_at_1, misses_at_3 = run_seeded_benchmark(bot, monkeypatch)
    assert embedding_calls.count("semantic memory search") == total
    assert "semantic memory indexing" in embedding_calls
    print(f"Top-1 misses: {misses_at_1}")
    print(f"Top-3 misses: {misses_at_3}")
    assert hits_at_1 / total >= 0.90, f"top-1 misses: {misses_at_1}"
    assert hits_at_3 / total >= RECALL_AT_3_THRESHOLD, f"top-3 misses: {misses_at_3}"
