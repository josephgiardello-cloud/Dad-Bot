from __future__ import annotations

import hashlib
import json
import os
from datetime import date

import pytest

from dadbot.managers.memory_query import MEMORY_DENSE_WEIGHT
from tests.eval.memory_benchmark import MEMORY_BENCHMARK_FACTS, memory_benchmark_cases, memory_paraphrase_cases

pytestmark = pytest.mark.unit

# The fixed, isolated 50-fact vault must meet this target with both embedding modes.
RECALL_AT_3_THRESHOLD = 0.98


def install_fusion_capture(bot, monkeypatch):
    fusion_rows = []
    components = {}
    semantic_candidates = set()
    candidate_filter = {}
    cosine_scores = {}
    semantic_manager = bot.memory_manager._semantic_manager
    original_candidates = semantic_manager.catalog_semantic_rows
    original_context = semantic_manager.semantic_query_context
    original_score = bot.memory_query.score
    original_cosine = semantic_manager.score_semantic_rows

    def capture_cosine(rows, current_memories, query_embedding):
        scored = original_cosine(rows, current_memories, query_embedding)
        cosine_scores.clear()
        cosine_scores.update((bot.semantic_memory_key(memory), score) for score, memory in scored)
        return scored

    def capture_context(query, limit):
        context = original_context(query, limit)
        candidate_filter.update(context)
        return context

    def capture_candidates(current_keys):
        rows = original_candidates(current_keys)
        assert {row["summary_key"] for row in rows} == set(current_keys)
        semantic_candidates.clear()
        semantic_candidates.update(row["summary_key"] for row in rows)
        return rows

    def capture_score(memory, context):
        key = bot.semantic_memory_key(memory)
        components[memory["summary"]] = {
            "candidate_present": key in semantic_candidates,
            "semantic": context["semantic_scores"].get(key, 0.0),
            "positive_cosine": cosine_scores.get(key, 0.0),
            "bm25": context["lexical_scores"].get(key, 0.0),
            "legacy_filter": {
                "tokens": candidate_filter.get("query_tokens"),
                "category": candidate_filter.get("query_category"),
                "mood": candidate_filter.get("query_mood"),
                "limit": candidate_filter.get("candidate_limit"),
                "token_match": any(
                    token.lower() in memory["summary"].lower() for token in candidate_filter.get("query_tokens", [])
                ),
                "category_match": memory.get("category") == candidate_filter.get("query_category"),
                "mood_match": memory.get("mood") == candidate_filter.get("query_mood"),
            },
        }
        fusion_rows.append(
            (
                original_score(memory, {**context, "user_message": ""}),
                context["lexical_scores"].get(key, 0.0),
                memory,
            )
        )
        return original_score(memory, context)

    monkeypatch.setattr(bot.memory_query, "score", capture_score)
    monkeypatch.setattr(semantic_manager, "catalog_semantic_rows", capture_candidates)
    monkeypatch.setattr(semantic_manager, "semantic_query_context", capture_context)
    monkeypatch.setattr(semantic_manager, "score_semantic_rows", capture_cosine)
    return fusion_rows, components


def run_seeded_benchmark(bot, monkeypatch, cases=None):
    cases = memory_benchmark_cases() if cases is None else cases
    assert len(cases) >= 25
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
    ranked_summaries = []
    measure_fusion = os.environ.get("DADBOT_MEASURE_FUSION") == "1"
    fusion_rows = []
    expected_components = {}
    if measure_fusion:
        fusion_rows, expected_components = install_fusion_capture(bot, monkeypatch)
    fusion_cases = []
    for case in cases:
        fusion_rows.clear()
        expected_components.clear()
        retrieved = bot.memory_query.relevant_memories_for_input(
            case.question,
            limit=3,
        )
        summaries = [str(memory.get("summary") or "") for memory in retrieved]
        ranked_summaries.append(summaries)
        if measure_fusion:
            fusion_cases.append((case, list(fusion_rows)))
        expected = case.expected_summary.casefold()
        top_3_hit = any(expected in summary.casefold() for summary in summaries[:3])
        top_1_hit = bool(summaries and expected in summaries[0].casefold())
        hits_at_1 += top_1_hit
        hits_at_3 += top_3_hit
        if not top_1_hit:
            misses_at_1.append((case.question, case.expected_summary, summaries))
        if not top_3_hit:
            misses_at_3.append((case.question, case.expected_summary, summaries))
            if measure_fusion:
                components = [
                    values for summary, values in expected_components.items() if expected in summary.casefold()
                ]
                print(f"Miss components for {case.question!r}: {components}")

    print(f"Seeded benchmark: top-1 {hits_at_1}/{len(cases)}, top-3 {hits_at_3}/{len(cases)}")
    fingerprint = hashlib.sha256(json.dumps(ranked_summaries, ensure_ascii=True).encode("ascii")).hexdigest()
    print(f"Ranking fingerprint: {fingerprint}")
    if measure_fusion:
        measured = measure_sparse_dense_balance(bot, fusion_cases)[MEMORY_DENSE_WEIGHT]
        assert measured["train"][0] + measured["held_out"][0] == hits_at_1
        assert measured["train"][1] + measured["held_out"][1] == hits_at_3
    return len(cases), hits_at_1, hits_at_3, misses_at_1, misses_at_3


def measure_sparse_dense_balance(bot, cases):
    # Hold out whole topics, not paraphrases of training questions.
    topic_indices = {fact.summary: index for index, fact in enumerate(MEMORY_BENCHMARK_FACTS)}
    results = {}
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0):
        counts = {"train": [0, 0, 0], "held_out": [0, 0, 0]}
        for case, rows in cases:
            split = "held_out" if topic_indices[case.expected_summary] % 5 == 0 else "train"
            ranked = sorted(
                ((alpha * dense + (1 - alpha) * sparse, memory) for dense, sparse, memory in rows),
                key=lambda item: (item[0], item[1].get("updated_at", ""), item[1].get("summary", "")),
                reverse=True,
            )
            filtered = [item for item in ranked if item[0] >= 0.18] or ranked[:1]
            selected = bot.memory_query.select_diverse_ranked_memories(filtered, 3)
            summaries = [memory["summary"].casefold() for _, memory in selected]
            expected = case.expected_summary.casefold()
            counts[split][0] += bool(summaries and expected in summaries[0])
            counts[split][1] += any(expected in summary for summary in summaries)
            counts[split][2] += 1
        print(
            f"Fusion dense_weight={alpha:.2f} train/top1,top3,total={counts['train']} "
            f"held_out/top1,top3,total={counts['held_out']}"
        )
        results[alpha] = counts
    # Select only on training data; ties prefer the current default.
    selected = max(
        results,
        key=lambda alpha: (
            results[alpha]["train"][0],
            results[alpha]["train"][1],
            -abs(alpha - MEMORY_DENSE_WEIGHT),
        ),
    )
    print(f"Training-selected dense_weight={selected:.2f}; held_out={results[selected]['held_out']}")
    return results


def test_memory_retrieval_benchmark_meets_recall_threshold(bot, monkeypatch):
    assert len(memory_benchmark_cases()) >= 200
    total, hits_at_1, hits_at_3, misses_at_1, misses_at_3 = run_seeded_benchmark(bot, monkeypatch)
    assert hits_at_1 / total >= 0.90, f"top-1 misses: {misses_at_1}"
    assert hits_at_3 / total >= RECALL_AT_3_THRESHOLD, f"top-3 misses: {misses_at_3}"


def test_paraphrase_questions_have_distinct_seeded_targets():
    cases = memory_paraphrase_cases()
    assert len(cases) == 25
    assert len({case.question for case in cases}) == 25
    assert len({case.expected_summary for case in cases}) == 25
    assert {case.expected_summary for case in cases} <= {fact.summary for fact in MEMORY_BENCHMARK_FACTS}


@pytest.mark.skipif(
    os.environ.get("DADBOT_RUN_REAL_EMBEDDING_BENCHMARK") != "1",
    reason="requires local Ollama and nomic-embed-text",
)
@pytest.mark.parametrize("question_set", ["templated", "paraphrase"])
def test_memory_retrieval_with_real_ollama_embeddings(bot, monkeypatch, question_set):
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
    cases = memory_paraphrase_cases() if question_set == "paraphrase" else memory_benchmark_cases()
    total, hits_at_1, hits_at_3, misses_at_1, misses_at_3 = run_seeded_benchmark(bot, monkeypatch, cases)
    assert embedding_calls.count("semantic memory search") == total
    assert "semantic memory indexing" in embedding_calls
    print(f"Top-1 misses: {misses_at_1}")
    print(f"Top-3 misses: {misses_at_3}")
    if question_set == "templated":
        assert total >= 200
        assert hits_at_1 / total >= 0.90, f"top-1 misses: {misses_at_1}"
        assert hits_at_3 / total >= RECALL_AT_3_THRESHOLD, f"top-3 misses: {misses_at_3}"
    else:
        # Diagnostic set: report weaknesses, not an invented passing baseline.
        assert total == 25
