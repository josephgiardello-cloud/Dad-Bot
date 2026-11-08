from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from dadbot.config import DadRuntimeConfig
from dadbot.memory.reranker import get_local_reranker, select_candidates
from dadbot.services.source_grounding import SOURCE_REFUSAL, GroundingViolation, SourceGroundingService

pytestmark = pytest.mark.unit


def test_model_is_loaded_once_across_concurrent_services(tmp_path, monkeypatch):
    loads = []
    model = object()
    monkeypatch.setattr("dadbot.memory.reranker._MODELS", {})

    def load(path):
        loads.append(path)
        return model

    monkeypatch.setattr("dadbot.memory.reranker.LocalQwenReranker", load)
    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(get_local_reranker, [str(tmp_path)] * 8))
    assert all(item is model for item in loaded)
    assert len(loads) == 1
    with pytest.raises(ValueError, match="requires"):
        get_local_reranker("")


def test_enabled_chat_model_failure_is_not_a_silent_fallback(bot, monkeypatch):
    bot.runtime_config.memory_reranker_enabled = True

    def fail(_path):
        raise FileNotFoundError("Missing local model")

    monkeypatch.setattr("dadbot.managers.memory_query.get_local_reranker", fail)
    with pytest.raises(FileNotFoundError, match="Missing local model"):
        bot.memory_query.rerank_memories(
            "query",
            [(0.9, {"summary": "First"}), (0.89, {"summary": "Second"})],
            3,
        )


def test_confidence_gate_boundary():
    ranked = [(1.0, "first"), (0.99, "second")]
    assert select_candidates(ranked, str)[0]
    assert not select_candidates(ranked, str, top_score_threshold=1.0)[0]
    assert select_candidates([(0.999, "first"), (0.99, "second")], str, top_score_threshold=1.0)[0]
    with pytest.raises(ValueError):
        select_candidates(ranked, str, top_score_threshold=float("nan"))


def test_chat_reranker_runs_before_category_order_and_retains_limit(bot, monkeypatch):
    bot.runtime_config.memory_reranker_enabled = True
    bot.runtime_config.memory_reranker_path = "test-model"
    ranked = [(0.9 - index * 0.01, {"summary": f"Fact {index}", "category": "work"}) for index in range(7)]
    received = []

    def score(_query, documents):
        received.extend(documents)
        return [0.1, 0.2, 0.3, 0.4, 0.9]

    monkeypatch.setattr("dadbot.managers.memory_query.get_local_reranker", lambda _path: SimpleNamespace(score=score))
    selected, diagnostics = bot.memory_query.rerank_memories("query", ranked, 7)
    assert received == [f"Fact {index}" for index in range(5)]
    assert selected[0][1]["summary"] == "Fact 4"
    assert len(selected) == 7
    assert diagnostics["reranker_fired"]


def test_chat_closed_gate_does_not_load_model(bot, monkeypatch):
    bot.runtime_config.memory_reranker_enabled = True
    monkeypatch.setattr("dadbot.managers.memory_query.get_local_reranker", lambda _path: pytest.fail("Closed gate"))
    ranked = [(0.9, {"summary": "First"}), (0.5, {"summary": "Second"})]
    selected, diagnostics = bot.memory_query.rerank_memories("query", ranked, 3)
    assert selected == ranked
    assert not diagnostics["reranker_fired"]


def test_source_reranker_keeps_approval_refusal_and_revalidation(tmp_path, monkeypatch):
    records = [{"id": str(index), "summary": f"Telescope item {index}", "category": "home"} for index in range(6)]
    bot = SimpleNamespace(
        memory_path=str(tmp_path / "memory.json"),
        PROFILE={},
        config=SimpleNamespace(tenant_id="test"),
        runtime_config=SimpleNamespace(memory_reranker_enabled=True, memory_reranker_path="test-model"),
        memory_manager=SimpleNamespace(memory_projection=lambda: {"memories": records}),
    )
    service = SourceGroundingService(bot)
    for source in service.memory_sources()[:5]:
        service.set_memory_approval(source, approved=True)
    received = []
    queries = []

    def score(_query, documents):
        received.append(documents)
        queries.append(_query)
        return [index / len(documents) for index in range(len(documents))]

    monkeypatch.setattr(
        "dadbot.services.source_grounding.get_local_reranker", lambda _path: SimpleNamespace(score=score)
    )
    reply = service.answer("telescope")
    assert len(received[0]) == 5
    assert "Telescope item 5" not in str(received)
    service.validate_reply("telescope", reply)
    assert len(received) == 2  # Independent validation bypasses answer cache.
    assert service.answer("  TELESCOPE  ") == reply
    service.validate_reply("  TELESCOPE  ", reply)
    assert queries == ["telescope"] * 3
    assert service.answer("unrelated subject") == SOURCE_REFUSAL
    assert len(received) == 3
    cited = next(record for record in records if record["summary"] in reply)
    cited["summary"] = "Telescope item changed"
    with pytest.raises(GroundingViolation):
        service.validate_reply("telescope", reply)


@pytest.mark.skipif(
    os.environ.get("DADBOT_RUN_REAL_EMBEDDING_BENCHMARK") != "1",
    reason="requires explicitly installed local Qwen model",
)
def test_real_source_reranking_is_extractive_and_edit_bound(tmp_path, monkeypatch):
    config = DadRuntimeConfig(memory_reranker_enabled=True)
    model = get_local_reranker(config.memory_reranker_path)
    records = [
        {"summary": text}
        for text in (
            "The telescope is kept in the blue cupboard.",
            "The telescope was purchased during a garden sale.",
            "The telescope was bought by an unrelated tenant.",
        )
    ]
    bot = SimpleNamespace(
        memory_path=str(tmp_path / "memory.json"),
        PROFILE={},
        config=SimpleNamespace(tenant_id="real-source-test"),
        runtime_config=config,
        memory_manager=SimpleNamespace(memory_projection=lambda: {"memories": records}),
    )
    service = SourceGroundingService(bot)
    for source in service.memory_sources()[:2]:
        service.set_memory_approval(source, approved=True)
    calls = []
    original_score = model.score

    def checked_score(query, documents):
        calls.append(documents)
        assert all("unrelated tenant" not in document for document in documents)
        return original_score(query, documents)

    monkeypatch.setattr(model, "score", checked_score)
    reply = service.answer("telescope")
    service.validate_reply("telescope", reply)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert all(record["summary"] in reply for record in records[:2])
    records[0]["summary"] = "The telescope has moved."
    with pytest.raises(GroundingViolation):
        service.validate_reply("telescope", reply)
