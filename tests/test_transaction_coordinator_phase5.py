from __future__ import annotations

import json
from copy import deepcopy

import pytest
from dadbot.core.state_lineage import canonical_state_hash

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("transaction_coordinator")]


def _stable_embedding_batch(texts, purpose=""):
    del purpose
    if isinstance(texts, str):
        texts = [texts]
    return [[0.001 * (index + 1) for index in range(16)] for _ in (texts or [])]


def _sample_memory(summary: str) -> dict:
    return {
        "summary": summary,
        "category": "work",
        "mood": "stressed",
    }


def test_cross_store_partial_failure_rolls_back_all(bot, monkeypatch):
    before_store = deepcopy(dict(bot.MEMORY_STORE or {}))
    before_graph = deepcopy(bot.graph_snapshot())

    semantic_manager = bot.memory_manager.semantic

    def _boom(memories):
        raise RuntimeError("semantic commit failed")

    monkeypatch.setattr(semantic_manager, "sync_semantic_memory_index", _boom)

    with pytest.raises(RuntimeError, match="Cross-store transaction failed"):
        bot.mutate_memory_store(memories=[_sample_memory("I am overwhelmed with deadlines")])

    after_store = deepcopy(dict(bot.MEMORY_STORE or {}))
    after_graph = deepcopy(bot.graph_snapshot())

    assert after_store == before_store
    assert after_graph == before_graph


@pytest.mark.slow
def test_cross_store_commit_prevents_representation_drift(bot, monkeypatch):
    semantic_manager = bot.memory_manager.semantic
    monkeypatch.setattr(semantic_manager, "embed_texts", _stable_embedding_batch)

    bot.mutate_memory_store(
        memories=[_sample_memory("I am stressed about this migration")],
        consolidated_memories=[{"summary": "I am stressed about this migration", "updated_at": "2026-04-27"}],
    )

    semantic_rows = int(bot.semantic_index_row_count() or 0)
    graph_snapshot = bot.graph_snapshot()

    assert semantic_rows >= 1
    assert len(list(graph_snapshot.get("nodes") or [])) >= 1


def test_crash_mid_commit_recovers_previous_state(bot, monkeypatch):
    semantic_manager = bot.memory_manager.semantic
    graph_manager = bot.memory_manager.graph_manager
    monkeypatch.setattr(semantic_manager, "embed_texts", _stable_embedding_batch)

    before_store = deepcopy(dict(bot.MEMORY_STORE or {}))

    def _graph_crash(turn_context=None):
        del turn_context
        raise RuntimeError("graph commit crash")

    monkeypatch.setattr(graph_manager, "sync_graph_store", _graph_crash)

    with pytest.raises(RuntimeError, match="Cross-store transaction failed"):
        bot.mutate_memory_store(memories=[_sample_memory("I need help with budgeting")])

    after_store = deepcopy(dict(bot.MEMORY_STORE or {}))
    expected_rows = len(list(after_store.get("memories") or []))
    after_semantic_rows = int(bot.semantic_index_row_count() or 0)

    assert after_store == before_store
    assert after_semantic_rows == expected_rows

    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert persisted == before_store


def test_full_replay_equivalence(bot):
    bot.mutate_memory_store(
        memories=[_sample_memory("I need a stable savings plan")],
        consolidated_memories=[
            {
                "summary": "We agreed to focus on monthly savings discipline",
                "updated_at": "2026-05-10",
                "category": "finance",
            },
        ],
    )

    state1 = deepcopy(dict(bot.MEMORY_STORE or {}))
    hash1 = canonical_state_hash(state1)
    bot.save_memory_store()

    replay_bot = bot.__class__()
    try:
        replay_bot.MEMORY_PATH = bot.MEMORY_PATH
        replay_bot.SEMANTIC_MEMORY_DB_PATH = bot.SEMANTIC_MEMORY_DB_PATH
        replay_bot.GRAPH_STORE_DB_PATH = bot.GRAPH_STORE_DB_PATH
        replay_bot.SESSION_LOG_DIR = bot.SESSION_LOG_DIR
        replay_bot.memory._load_memory_store()

        state2 = deepcopy(dict(replay_bot.MEMORY_STORE or {}))
        hash2 = canonical_state_hash(state2)

        assert hash2 == hash1
        assert state2 == state1
    finally:
        try:
            replay_bot.shutdown()
        except Exception:
            pass
        try:
            replay_bot.wait_for_semantic_index_idle(5)
        except Exception:
            pass
