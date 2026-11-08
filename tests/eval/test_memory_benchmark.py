from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from dadbot.managers.memory_query import MEMORY_DENSE_WEIGHT
from dadbot_system.semantic_index import PGVectorSemanticIndex, SQLiteSemanticIndex
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
        current_keys = set(current_keys)
        rows = original_candidates(current_keys)
        assert {row["summary_key"] for row in rows} == set(current_keys)
        semantic_candidates.clear()
        semantic_candidates.update(row["summary_key"] for row in rows)
        return rows

    def capture_score(memory, context):
        key = bot.semantic_memory_key(memory)
        dense = original_score(memory, {**context, "user_message": "", "lexical_query_tokens": set()})
        fused = original_score(memory, context)
        components[memory["summary"]] = {
            "candidate_present": key in semantic_candidates,
            "semantic": context["semantic_scores"].get(key, 0.0),
            "positive_cosine": cosine_scores.get(key, 0.0),
            "alignment": bot.memory_alignment_weight(
                memory,
                query_tokens=candidate_filter.get("query_tokens"),
                query_category=candidate_filter.get("query_category", "general"),
                query_mood=candidate_filter.get("query_mood", "neutral"),
                recent_topics=bot.recent_memory_topics(limit=4),
                mood_trend=bot.current_memory_mood_trend(),
            ),
            "bm25": context["lexical_scores"].get(key, 0.0),
            "dense": dense,
            "fused": fused,
            "category": memory.get("category"),
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
                dense,
                context["lexical_scores"].get(key, 0.0),
                memory,
                cosine_scores.get(key, 0.0),
            )
        )
        return fused

    monkeypatch.setattr(bot.memory_query, "score", capture_score)
    monkeypatch.setattr(semantic_manager, "catalog_semantic_rows", capture_candidates)
    monkeypatch.setattr(semantic_manager, "semantic_query_context", capture_context)
    monkeypatch.setattr(semantic_manager, "score_semantic_rows", capture_cosine)
    return fusion_rows, components


def report_ranking_miss(bot, case, components):
    expected = case.expected_summary.casefold()
    summary = next(summary for summary in components if expected in summary.casefold())
    metrics = components[summary]
    print(f"Miss components for {case.question!r}: {metrics}")
    for label, key in (("cosine", "positive_cosine"), ("contextual_dense", "dense"), ("fusion", "fused")):
        ranked = sorted(components, key=lambda text: (components[text][key], text), reverse=True)
        print(
            f"Expected {label} rank: {ranked.index(summary) + 1}; "
            f"top-3: {[(text, components[text][key]) for text in ranked[:3]]}"
        )
    ranked = sorted(
        ((values["fused"], {"summary": text, "category": values["category"]}) for text, values in components.items()),
        key=lambda item: (item[0], item[1]["summary"]),
        reverse=True,
    )
    eligible = [item for item in ranked if item[0] >= 0.18] or ranked[:1]
    print(f"Pre-diversification fused top-1/top-2 gap: {ranked[0][0] - ranked[1][0]:.6f}")
    for count in (3, 5, 10):
        selected = bot.memory_query.select_diverse_ranked_memories(eligible, count)
        rank = next((index + 1 for index, (_, memory) in enumerate(selected) if memory["summary"] == summary), None)
        print(f"Expected diversified candidate rank at k={count}: {rank}")


def run_seeded_benchmark(bot, monkeypatch, cases=None, *, rankings=None):
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
    ranked_summaries = [] if rankings is None else rankings
    measure_fusion = os.environ.get("DADBOT_MEASURE_FUSION") == "1" or os.environ.get("DADBOT_RUN_QWEN_RERANKER") == "1"
    fusion_rows = []
    expected_components = {}
    if measure_fusion:
        fusion_rows, expected_components = install_fusion_capture(bot, monkeypatch)
    fusion_cases = []
    latencies = []
    reranker_fired = 0
    reranker_times = []
    for case in cases:
        fusion_rows.clear()
        expected_components.clear()
        retrieved = bot.memory_query.relevant_memories_for_input(
            case.question,
            limit=3,
        )
        diagnostics = bot.memory_query.get_retrieval_diagnostics()
        latencies.append(float(diagnostics["total_seconds"]))
        reranker_fired += bool(diagnostics.get("reranker_fired"))
        if diagnostics.get("reranker_fired"):
            reranker_times.append(float(diagnostics["reranker_seconds"]))
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
                report_ranking_miss(bot, case, expected_components)

    print(f"Seeded benchmark: top-1 {hits_at_1}/{len(cases)}, top-3 {hits_at_3}/{len(cases)}")
    ordered_latencies = sorted(latencies)
    print(
        f"Retrieval p50={ordered_latencies[math.ceil(0.50 * len(latencies)) - 1]:.3f}s "
        f"p95={ordered_latencies[math.ceil(0.95 * len(latencies)) - 1]:.3f}s; "
        f"reranker fired={reranker_fired}/{len(cases)} "
        f"reranker total={sum(reranker_times):.3f}s"
    )
    fingerprint = hashlib.sha256(json.dumps(ranked_summaries, ensure_ascii=True).encode("ascii")).hexdigest()
    print(f"Ranking fingerprint: {fingerprint}")
    if measure_fusion:
        measure_retrieval_experiments(bot, fusion_cases, ranked_summaries, (hits_at_1, hits_at_3))
    return len(cases), hits_at_1, hits_at_3, misses_at_1, misses_at_3


def measure_retrieval_experiments(bot, fusion_cases, ranked_summaries, hits):
    measured = measure_sparse_dense_balance(bot, fusion_cases)[MEMORY_DENSE_WEIGHT]
    if not bot.runtime_config.memory_reranker_enabled:
        assert measured["train"][0] + measured["held_out"][0] == hits[0]
        assert measured["train"][1] + measured["held_out"][1] == hits[1]
    if os.environ.get("DADBOT_RUN_QWEN_RERANKER") == "1" and len(fusion_cases) == 50:
        from tests.eval.reranker_experiment import run_reranker_experiment

        assert not bot.runtime_config.memory_reranker_enabled, "Experiment requires an explicit unreranked baseline."
        run_reranker_experiment(bot, fusion_cases, ranked_summaries)


def measure_sparse_dense_balance(bot, cases):
    # Hold out whole topics, not paraphrases of training questions.
    topic_indices = {fact.summary: index for index, fact in enumerate(MEMORY_BENCHMARK_FACTS)}
    results = {}
    settings = [(alpha, alpha, None) for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)]
    settings.extend((f"adaptive raw_cosine>0.85 dense_weight={alpha}", alpha, 0.85) for alpha in (0.85, 0.9))
    for label, alpha, threshold in settings:
        counts = {"train": [0, 0, 0], "held_out": [0, 0, 0]}
        gated = 0
        for case, rows in cases:
            split = "held_out" if topic_indices[case.expected_summary] % 5 == 0 else "train"
            confidence = max((cosine for _, _, _, cosine in rows), default=0.0)
            effective_alpha = alpha if threshold is None or confidence > threshold else MEMORY_DENSE_WEIGHT
            gated += threshold is not None and confidence > threshold
            ranked = sorted(
                (
                    (effective_alpha * dense + (1 - effective_alpha) * sparse, memory)
                    for dense, sparse, memory, _ in rows
                ),
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
            f"Fusion setting={label} train/top1,top3,total={counts['train']} "
            f"held_out/top1,top3,total={counts['held_out']} gated={gated}"
        )
        if threshold is None:
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
    assert len(cases) == 50
    assert len({case.question for case in cases}) == 50
    assert {case.expected_summary for case in cases} == {fact.summary for fact in MEMORY_BENCHMARK_FACTS}


@pytest.mark.skipif(
    os.environ.get("DADBOT_RUN_REAL_EMBEDDING_BENCHMARK") != "1",
    reason="requires local Ollama and nomic-embed-text",
)
@pytest.mark.parametrize(
    "question_set,backend",
    [
        ("templated", "sqlite"),
        ("paraphrase", "sqlite"),
        ("paraphrase", "pgvector"),
    ],
)
@pytest.mark.usefixtures("real_semantic_backend")
def test_memory_retrieval_with_real_ollama_embeddings(bot, monkeypatch, question_set, backend):
    semantic_manager = bot.memory_manager._semantic_manager
    assert semantic_manager._semantic_index_backend.name == backend
    del bot.embed_texts
    bot.PREFERRED_EMBEDDING_MODELS = ("nomic-embed-text",)
    bot.ACTIVE_EMBEDDING_MODEL = "nomic-embed-text"
    vectors = bot.embed_texts(["isolated benchmark embedding probe"])
    assert len(vectors) == 1 and len(vectors[0]) == 768, "Ollama Nomic embedding unavailable"
    assert bot.ACTIVE_EMBEDDING_MODEL == "nomic-embed-text"
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
    rankings = []
    total, hits_at_1, hits_at_3, misses_at_1, misses_at_3 = run_seeded_benchmark(
        bot,
        monkeypatch,
        cases,
        rankings=rankings,
    )
    assert embedding_calls.count("semantic memory search") == total
    assert "semantic memory indexing" in embedding_calls
    print(f"Top-1 misses: {misses_at_1}")
    print(f"Top-3 misses: {misses_at_3}")
    if question_set == "templated":
        assert total >= 200
        assert hits_at_1 / total >= 0.90, f"top-1 misses: {misses_at_1}"
        assert hits_at_3 / total >= RECALL_AT_3_THRESHOLD, f"top-3 misses: {misses_at_3}"
    else:
        # Enforce the measured baseline without pretending recall is perfect.
        assert total == 50
        top1_floor = 48 if bot.runtime_config.memory_reranker_enabled else 43
        top3_floor = 50 if bot.runtime_config.memory_reranker_enabled else 49
        assert hits_at_1 >= top1_floor, f"top-1 regression: {misses_at_1}"
        assert hits_at_3 >= top3_floor, f"top-3 regression: {misses_at_3}"
        report_original_paraphrases(cases[:25], rankings[:25])
    if backend == "pgvector":
        verify_sqlite_ranking_parity(bot, monkeypatch, cases, rankings)
        assert embedding_calls.count("semantic memory search") == total * 2


def report_original_paraphrases(cases, rankings):
    matches = [
        [case.expected_summary.casefold() in text.casefold() for text in rows]
        for case, rows in zip(cases, rankings, strict=True)
    ]
    print(
        f"Original 25 paraphrases: top-1 {sum(row[0] for row in matches)}/25, "
        f"top-3 {sum(any(row) for row in matches)}/25"
    )


def verify_sqlite_ranking_parity(bot, monkeypatch, cases, expected_rankings):
    manager = bot.memory_manager._semantic_manager
    catalog = bot.memory_catalog()
    rows = manager.catalog_semantic_rows(bot.semantic_memory_key(memory) for memory in catalog)
    mirror = SQLiteSemanticIndex(bot, Path(bot.semantic_memory_db_path).with_name("paraphrase_parity.sqlite3"))
    mirror.upsert_rows([{**row, "content_hash": row["summary_key"]} for row in rows])
    with monkeypatch.context() as patch:
        patch.setattr(manager, "_semantic_index_backend", mirror)
        actual = [
            [memory["summary"] for memory in bot.memory_query.relevant_memories_for_input(case.question, limit=3)]
            for case in cases
        ]
    assert actual == expected_rankings, "Postgres and SQLite must return identical full top-3 rankings."
    print(f"Verified live pgvector/SQLite full-ranking parity on {len(cases)} questions.")


@pytest.fixture
def real_semantic_backend(bot, monkeypatch, backend):
    if backend == "sqlite":
        yield
        return
    dsn = os.environ.get("DADBOT_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Requires dedicated DADBOT_TEST_POSTGRES_DSN; never uses the production table.")
    import psycopg

    table = f"semantic_paraphrase_{uuid4().hex}"
    manager = bot.memory_manager._semantic_manager
    assert bot.wait_for_semantic_index_idle(timeout=30)
    index = PGVectorSemanticIndex(dsn, table=table)
    with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as connection:
        try:
            index.ensure_storage()
            monkeypatch.setattr(manager, "_semantic_index_backend", index)
            yield
        finally:
            try:
                assert bot.wait_for_semantic_index_idle(timeout=30)
            finally:
                connection.execute(psycopg.sql.SQL("DROP TABLE IF EXISTS {}").format(psycopg.sql.Identifier(table)))
