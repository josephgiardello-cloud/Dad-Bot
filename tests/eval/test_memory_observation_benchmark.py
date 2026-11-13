from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import date
from threading import Barrier
from time import perf_counter
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dadbot.runtime.context_pruner import MemoryRecall
from dadbot.runtime.semantic_memory_bridge import (
    MemoryIndexer,
    _memory_signals,
    build_hybrid_memory_recall_provider,
)
from tests.eval.memory_benchmark import (
    MEMORY_BENCHMARK_FACTS,
    OBSERVATION_ARCHIVE_FACT,
    OBSERVATION_CONSOLIDATED_FACTS,
    OBSERVATION_CORPUS_DATE,
    OBSERVATION_CORPUS_SHA256,
    OBSERVATION_DATASET_VERSION,
    OBSERVATION_SUPERSEDED_FACT,
    observation_benchmark_cases,
)

pytestmark = pytest.mark.unit
TENANT = "t1_" + "a" * 26


@pytest.fixture
def observation_bot(make_test_dadbot, monkeypatch):
    from dadbot.core.execution_receipt import ReceiptSigner

    monkeypatch.setenv("DADBOT_DURABLE_EXECUTION_LEDGER", "1")
    monkeypatch.setenv("DADBOT_EXECUTION_LEDGER_BACKEND", "file")
    monkeypatch.setenv("DADBOT_RECEIPT_KEY", os.urandom(32).hex())
    bot = make_test_dadbot(tenant_id=TENANT)
    bot.services.turn_orchestrator.control_plane.graph.configure_receipt_signer(ReceiptSigner())
    monkeypatch.setattr(
        bot, "days_since_iso_date",
        lambda value: (date.fromisoformat(OBSERVATION_CORPUS_DATE) - date.fromisoformat(str(value)[:10])).days
        if value else None,
    )
    memories = [
        {
            "summary": fact.summary, "category": fact.category,
            "confidence": 0.9, "importance_score": 0.9,
            "created_at": OBSERVATION_CORPUS_DATE, "updated_at": OBSERVATION_CORPUS_DATE,
            "provenance": {
                "source": "synthetic_fixture", "trust_level": 1.0, "agent": OBSERVATION_DATASET_VERSION,
                "evidence": fact.summary, "trace_id": f"synthetic-session-{index % 3}",
            },
        }
        for index, fact in enumerate(MEMORY_BENCHMARK_FACTS)
    ]
    consolidated = [
        {
            "summary": summary, "category": "general", "source_count": 3,
            "confidence": 0.9, "importance_score": 0.9,
            "updated_at": OBSERVATION_CORPUS_DATE,
        }
        for summary in OBSERVATION_CONSOLIDATED_FACTS
    ]
    consolidated.append({
        "summary": OBSERVATION_SUPERSEDED_FACT, "category": "general",
        "source_count": 3, "confidence": 1.0, "importance_score": 1.0,
        "updated_at": OBSERVATION_CORPUS_DATE, "superseded": True,
        "superseded_by": OBSERVATION_CONSOLIDATED_FACTS[2],
        "superseded_at": OBSERVATION_CORPUS_DATE,
    })
    bot.mutate_memory_store(
        memories=memories, consolidated_memories=consolidated,
        session_archive=[{
            "id": "synthetic-observatory-session", "summary": OBSERVATION_ARCHIVE_FACT,
            "topics": ["observatory"], "dominant_mood": "neutral", "turn_count": 4,
            "created_at": f"{OBSERVATION_CORPUS_DATE}T09:00:00",
        }],
    )
    assert bot.wait_for_semantic_index_idle(timeout=30)
    bot.sync_semantic_memory_index(bot.memory_catalog())
    bot.sync_graph_store()
    return bot


def _baseline_provider(bot):
    indexer = MemoryIndexer(bot, session_id="observation-benchmark")

    def recall(query):
        indexer._authorize()
        memories = bot.memory_query.relevant_memories_for_input(query, limit=5, record_access=False)
        return MemoryRecall(
            snippets=[f"[{memory.get('category', 'general')}] {memory['summary']}" for memory in memories],
            signals=_memory_signals(bot, memories),
        )

    return recall


def _measure(provider):
    cases = observation_benchmark_cases()
    results = []
    latencies = []
    for case in cases:
        start = perf_counter()
        recall = provider(case.question)
        latencies.append(perf_counter() - start)
        results.append(recall)
    scores = []
    for case, recall in zip(cases, results, strict=True):
        if not case.relevant_summaries:
            continue
        seen = set()
        gains = []
        for snippet in recall.snippets[:5]:
            text = snippet.split("\nEvidence:", 1)[0]
            matched = {
                summary for summary in case.relevant_summaries
                if summary.casefold() in text.casefold()
            }
            gains.append(len(matched - seen))
            seen.update(matched)
        hits = sum(
            any(summary.casefold() in snippet.casefold() for snippet in recall.snippets[:5])
            for summary in case.relevant_summaries
        )
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(5, len(case.relevant_summaries))))
        scores.append((
            case.group, hits / len(case.relevant_summaries),
            sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains)) / ideal,
        ))
    return {
        "recall_at_5": sum(score[1] for score in scores) / len(scores),
        "ndcg_at_5": sum(score[2] for score in scores) / len(scores),
        "p95_seconds": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
        "groups": {
            group: {
                "recall_at_5": sum(score[1] for score in scores if score[0] == group)
                / sum(score[0] == group for score in scores),
                "ndcg_at_5": sum(score[2] for score in scores if score[0] == group)
                / sum(score[0] == group for score in scores),
            }
            for group in sorted({score[0] for score in scores})
        },
    }, results


def test_local_observation_quality_and_latency(observation_bot, record_testsuite_property):
    bot = observation_bot
    baseline = _baseline_provider(bot)
    candidate = build_hybrid_memory_recall_provider(bot, top_k=5, session_id="observation-benchmark")
    # Warm both paths before measuring the same immutable corpus and query order.
    baseline("What is my coffee order?")
    candidate("What is my coffee order?")
    ledger = bot.services.turn_orchestrator.control_plane.ledger
    assert bot.memory_manager._storage._local_authority_backend() is not None
    before = ledger.read()
    baseline_metrics, baseline_results = _measure(baseline)
    candidate_metrics, candidate_results = _measure(candidate)
    projection = bot.memory_manager.memory_projection()
    corpus = {key: projection[key] for key in ("memories", "consolidated_memories", "session_archive")}
    evidence = {
        "dataset": OBSERVATION_DATASET_VERSION,
        "corpus_sha256": hashlib.sha256(json.dumps(
            corpus, sort_keys=True, default=str,
        ).encode()).hexdigest(),
        "queries": len(observation_benchmark_cases()),
        "query_clock": OBSERVATION_CORPUS_DATE,
        "backend": "File-WAL/SQLite",
        "embeddings": "tests.conftest.fake_embed_texts:12-dimensional-offline",
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "top_k": 5, "max_context_chars": 12000, "absolute_p95_budget_seconds": 0.2,
        "baseline": baseline_metrics, "candidate": candidate_metrics,
    }
    print(json.dumps(evidence, sort_keys=True))
    record_testsuite_property("observation_quality", json.dumps(evidence, sort_keys=True))
    assert evidence["corpus_sha256"] == OBSERVATION_CORPUS_SHA256
    assert ledger.read() == before, "Observation must not publish authority or access updates"
    assert candidate_metrics["recall_at_5"] >= 0.85
    assert candidate_metrics["ndcg_at_5"] >= 0.80
    for group in ("graph", "cross_session", "supersession", "archive"):
        assert candidate_metrics["groups"][group]["recall_at_5"] == 1.0, f"Missing {group} evidence"
    assert candidate_metrics["p95_seconds"] <= min(0.2, baseline_metrics["p95_seconds"] * 1.20)
    for case, previous, current in zip(
        observation_benchmark_cases(), baseline_results, candidate_results, strict=True,
    ):
        assert not any(OBSERVATION_SUPERSEDED_FACT in snippet for snippet in current.snippets)
        if not case.relevant_summaries:
            assert current.snippets == [], f"Unrelated evidence for {case.question!r}"
        if case.group == "exact":
            for summary in case.relevant_summaries:
                if any(summary.casefold() in snippet.casefold() for snippet in previous.snippets[:5]):
                    assert any(summary.casefold() in snippet.casefold() for snippet in current.snippets[:5])


def test_native_evidence_preserves_provenance_and_graph_identity(observation_bot):
    provider = build_hybrid_memory_recall_provider(observation_bot, top_k=5)
    recalled = provider("my coffee order")
    assert any(
        item["tenant_id"] == TENANT and item["authority_revision"] is not None
        and item["provenance"]["evidence"] == "My coffee order is a double espresso."
        and item["provenance"]["trace_id"].startswith("synthetic-session-")
        for item in recalled.signals["evidence"] if item["provenance"]
    )
    graph = provider("woodworking club plans")
    evidence = next(item for item in graph.signals["evidence"] if item["source_type"] == "consolidated_memory")
    assert evidence["source_id"]
    assert evidence["validity"] and all(edge["edge_key"] for edge in evidence["validity"])
    assert evidence["source_metadata"]["source_count"] == 3


def test_observation_rejects_mixed_authority_revisions(observation_bot, monkeypatch):
    bot = observation_bot
    original = bot.graph_retrieval_for_input

    def commit_after_graph(query, limit):
        graph = original(query, limit=limit)
        bot.mutate_memory_store(last_mood="positive")
        return graph

    monkeypatch.setattr(bot, "graph_retrieval_for_input", commit_after_graph)
    with pytest.raises(RuntimeError, match="authority revisions differ"):
        build_hybrid_memory_recall_provider(bot)("woodworking club plans")


def test_observation_keeps_colliding_tenants_separate(observation_bot, make_test_dadbot):
    other = make_test_dadbot(tenant_id="t1_" + "b" * 26)
    other.mutate_memory_store(memories=[{
        "summary": "My coffee order is a single latte.", "category": "preferences",
        "created_at": OBSERVATION_CORPUS_DATE, "updated_at": OBSERVATION_CORPUS_DATE,
    }])
    own = build_hybrid_memory_recall_provider(observation_bot)("my coffee order")
    foreign = build_hybrid_memory_recall_provider(other)("my coffee order")
    assert any("double espresso" in snippet for snippet in own.snippets)
    assert not any("single latte" in snippet for snippet in own.snippets)
    assert any("single latte" in snippet for snippet in foreign.snippets)
    assert all(item["tenant_id"] == TENANT for item in own.signals["evidence"])


def test_concurrent_observations_keep_query_scores_and_revisions_scoped(observation_bot, monkeypatch):
    manager = observation_bot.memory_query
    queries = ["my coffee order", "woodworking club plans"]
    expected = [manager.retrieve_context(query, limit=5, record_access=False) for query in queries]
    barrier = Barrier(2)
    original = manager._read_only_context

    def overlap(*args):
        barrier.wait(timeout=30)
        return original(*args)

    monkeypatch.setattr(manager, "_read_only_context", overlap)
    before = observation_bot._execution_ledger.read()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda query: manager.retrieve_context(query, limit=5, record_access=False), queries,
        ))
    assert results == expected
    assert observation_bot._execution_ledger.read() == before


def test_native_observation_degradation_is_visible_and_read_only(observation_bot, monkeypatch):
    bot = observation_bot
    monkeypatch.setattr(bot, "embed_texts", lambda *_args, **_kwargs: [])
    before = bot.services.turn_orchestrator.control_plane.ledger.read()
    recall = build_hybrid_memory_recall_provider(bot)("Recall my coffee order using precise detail")
    assert any("double espresso" in snippet for snippet in recall.snippets)
    assert recall.signals["retrieval"]["status"] == "degraded"
    assert recall.signals["retrieval"]["reason"] == "embedding_unavailable"
    assert bot.services.turn_orchestrator.control_plane.ledger.read() == before


def test_native_observation_revalidates_authority_without_empty_fallback(observation_bot, tmp_path):
    from dadbot_system.delegation import SQLiteDelegationStore, runtime_authority_scope
    from dadbot_system.security import AuthorizationError, ServicePrincipal

    path = str(tmp_path / "observation-authority.sqlite")
    SQLiteDelegationStore.initialize(path)
    store = SQLiteDelegationStore(path, "synthetic-observation-secret")
    principal = ServicePrincipal(
        subject="synthetic-observation-reader", tenant_id=TENANT, scopes=frozenset({"write"}),
        allowed_tools=frozenset(), issuer="dadbot", expires_at=time.time() + 3600,
        token_id="synthetic-observation-reader",
    )
    store.grant_membership(issuer=principal.issuer, subject=principal.subject, tenant_id=TENANT, scopes={"write"})
    with runtime_authority_scope(store, principal, scope="write", tenant_id=TENANT):
        provider = build_hybrid_memory_recall_provider(observation_bot)
        captured = copy_context()
        assert provider("my coffee order").snippets
    with pytest.raises(AuthorizationError):
        provider("my coffee order")
    store.revoke_membership(issuer=principal.issuer, subject=principal.subject, tenant_id=TENANT)
    with pytest.raises(AuthorizationError):
        captured.run(provider, "my coffee order")


def test_corrupt_authority_is_not_an_empty_success(observation_bot):
    from dadbot.core.ledger_backend import WALCorruptionError

    path = observation_bot.memory_manager._storage._local_authority_backend().path()
    original = path.read_bytes()
    try:
        path.write_bytes(b"corrupt observation authority\n")
        with pytest.raises(WALCorruptionError):
            build_hybrid_memory_recall_provider(observation_bot)("my coffee order")
    finally:
        path.write_bytes(original)
        observation_bot._execution_ledger.load_from_backend()


@pytest.mark.parametrize("entry", ["interactive", "durable", "resume", "proactive", "heartbeat"])
def test_default_ingresses_use_native_memory_evidence(observation_bot, monkeypatch, tmp_path, entry):
    from dadbot.assistant_runtime import AssistantRuntime
    from dadbot.runtime.agent_driver_loop import DriverLoopPolicy

    bot = observation_bot
    monkeypatch.setattr("dadbot_system.local_mcp_server._project_root", lambda: tmp_path)
    monkeypatch.setattr(bot.model_port, "generate", lambda *_args, **_kwargs: "Step completed.")
    monkeypatch.setattr(bot.model_port, "generate_async", AsyncMock(return_value="Step completed."))
    runtime = AssistantRuntime(bot)
    contexts = []
    kernel_evidence = []
    orchestrator = bot.services.turn_orchestrator
    retrieve = orchestrator._run_canonical_loop_memory_retrieval

    def capture_kernel_retrieval(context, query):
        retrieve(context, query)
        kernel_evidence.extend(context.state["subconscious_memory_fragments"])

    monkeypatch.setattr(orchestrator, "_run_canonical_loop_memory_retrieval", capture_kernel_retrieval)

    def reflect(context):
        contexts.append(context)
        return {"action_input": "my coffee order"}

    monkeypatch.setattr(runtime, "_build_default_reflection_hook", lambda _tools: reflect)
    if entry in {"proactive", "heartbeat"}:
        bot.queue_proactive_message("Recall my coffee order.", source="synthetic-benchmark")
        if entry == "heartbeat":
            monkeypatch.setattr(bot.maintenance_scheduler, "run_proactive_heartbeat", lambda **_kwargs: {})
            result = runtime.run_heartbeat()["agent_loop"]
        else:
            result = runtime.run_pending_proactive_trigger()
        assert result["status"] == "processed"
    else:
        policy = DriverLoopPolicy(max_turns=2)
        options: dict[str, Any] = {"durable": True} if entry in {"durable", "resume"} else {}
        if entry == "resume":
            def interrupt_second(context):
                if context["turn_index"] == 2:
                    raise SystemExit("interrupted before second dispatch")
                return reflect(context)
            with pytest.raises(SystemExit, match="interrupted"):
                runtime.run_agent_loop(
                    "my coffee order", session_id="benchmark", policy=policy,
                    reflection_hook=interrupt_second, **options,
                )
            contexts.clear()
            options = {"resume": True}
        result = runtime.run_agent_loop("my coffee order", session_id="benchmark", policy=policy, **options)
        assert result.completed_turns == 2
    assert contexts
    assert all(context["memory_signals"]["evidence"] for context in contexts)
    assert all("double espresso" in context["observation"] for context in contexts)
    assert all(len(context["observation"]) <= 12000 for context in contexts)
    assert all("my coffee order" in context["initial_observation"].lower() for context in contexts)
    assert kernel_evidence
    assert all(
        item["memory_id"] and item["authority_revision"] > 0 and item["tenant_id"] == TENANT
        and item["source"] == "canonical_tenant_memory"
        for item in kernel_evidence
    )
    assert any(
        "double espresso" in item["summary"] and item["provenance"]
        and item["provenance"]["evidence"] == "My coffee order is a double espresso."
        and item["provenance"]["trace_id"]
        for item in kernel_evidence
    )


@pytest.mark.parametrize("opt_out", ["disabled", "custom"])
def test_explicit_memory_opt_out_does_not_query(observation_bot, monkeypatch, tmp_path, opt_out):
    from dadbot.assistant_runtime import AssistantRuntime
    from dadbot.runtime.agent_driver_loop import DriverLoopPolicy

    bot = observation_bot
    monkeypatch.setattr("dadbot_system.local_mcp_server._project_root", lambda: tmp_path)
    runtime = AssistantRuntime(bot)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Explicit observation opt-out called the default memory provider")

    monkeypatch.setattr(bot.memory_query, "retrieve_context", forbidden)
    options: dict[str, Any] = (
        {"enable_semantic_memory": False} if opt_out == "disabled"
        else {"observation_hook": lambda _ctx: "custom"}
    )
    result = runtime.run_agent_loop(
        "my coffee order", policy=DriverLoopPolicy(max_turns=1),
        reflection_hook=lambda _ctx: {"should_continue": False}, **options,
    )
    assert result.stop_reason == "reflection_stop"
