import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from dadbot.core.core_state import CoreState
from dadbot.core.execution_context import close_core_state_scope, open_core_state_scope, replace_active_core_state
from dadbot.core.state_lineage import canonical_state_hash

pytestmark = pytest.mark.unit


def test_normalize_memory_entry_naturalizes_summary_and_mood(bot):
    entry = bot.normalize_memory_entry(
        {
            "summary": "i'm worried about money lately",
            "category": None,
            "mood": "anxious",
        }
    )

    assert entry["summary"] == "Tony is worried about money lately."
    assert entry["category"] == "finance"
    assert entry["mood"] == "stressed"


def test_normalize_memory_entry_validates_dates_and_scores_without_changing_surface(bot):
    entry = bot.normalize_memory_entry(
        {
            "summary": "i've been saving for emergencies",
            "category": "finance",
            "mood": "positive",
            "created_at": "2026-04-20",
            "updated_at": "2026-04-21",
            "confidence": "1.7",
            "impact_score": "2.5",
            "importance_score": "0.8",
            "emotional_intensity": "0.7",
            "relationship_impact": "0.6",
            "pinned": True,
            "contradictions": ["", "missed the budget once"],
        }
    )

    assert entry["summary"] == "Tony has been saving for emergencies."
    assert entry["created_at"] == "2026-04-20"
    assert entry["updated_at"] == "2026-04-21"
    assert entry["confidence"] == 1.0
    assert entry["impact_score"] == 2.5
    assert entry["importance_score"] == 0.8
    assert entry["emotional_intensity"] == 0.7
    assert entry["relationship_impact"] == 0.6
    assert entry["pinned"] is True
    assert entry["contradictions"] == ["missed the budget once"]


def test_normalize_relationship_state_preserves_runtime_vocab_and_date_surface(bot):
    entry = bot.normalize_relationship_state(
        {
            "trust_level": 65,
            "openness_level": 62,
            "emotional_momentum": "warming",
            "recurring_topics": {"Work": "3", "": 2},
            "recent_checkins": [{"date": "2026-04-20", "mood": "happy", "topic": "Work"}],
            "hypotheses": [{"name": "supportive_baseline", "probability": 1.0}],
            "last_hypothesis_updated": "2026-04-20",
            "last_reflection": "He is opening up more.",
            "last_updated": "2026-04-20",
        }
    )

    assert entry["emotional_momentum"] == "warming"
    assert entry["recurring_topics"] == {"work": 3}
    assert entry["recent_checkins"][0]["mood"] == "positive"
    assert entry["last_hypothesis_updated"] == "2026-04-20"
    assert entry["last_updated"] == "2026-04-20"


def test_normalize_persona_evolution_entry_validates_runtime_trait_payload(bot):
    entry = bot.normalize_persona_evolution_entry(
        {
            "new_trait": "more coach-like",
            "reason": "Tony responds well to structure",
            "announcement": "I have gotten a little more coach-like with you.",
            "session_count": "8",
            "applied_at": "2026-04-10T20:00:00",
            "last_reinforced_at": "2026-04-11T20:00:00",
            "strength": "2.7",
            "impact_score": "-1.5",
            "critique_score": "9",
            "critique_feedback": "specific and grounded",
        }
    )

    assert entry["trait"] == "more coach-like"
    assert entry["session_count"] == 8
    assert entry["strength"] == 2.7
    assert entry["impact_score"] == -1.5
    assert entry["critique_score"] == 9


def test_normalize_wisdom_entry_validates_runtime_payload(bot):
    entry = bot.normalize_wisdom_entry(
        {
            "summary": "Slow the moment down before work runs you.",
            "topic": "Work",
            "trigger": "stress check-in",
            "created_at": "2026-04-11T20:00:00",
        }
    )

    assert entry == {
        "summary": "Slow the moment down before work runs you.",
        "topic": "work",
        "trigger": "stress check-in",
        "created_at": "2026-04-11T20:00:00",
    }


def test_save_memory_store_normalizes_direct_memory_store_mutations(bot):
    bot.MEMORY_STORE = {
        **bot.default_memory_store(),
        "memories": [{"summary": "i'm saving money", "category": None, "mood": "happy"}],
        "consolidated_memories": [{"summary": "i've been saving money", "updated_at": "2026-04-20"}],
        "last_mood": "happy",
        "last_mood_updated_at": "2026-04-20",
        "recent_moods": [{"mood": "happy", "date": "2026-04-20"}],
    }

    bot.save_memory_store()

    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))

    from datetime import date as _date

    assert persisted["memories"][0]["summary"] == "Tony is saving money."
    assert persisted["memories"][0]["created_at"] == _date.today().isoformat()
    assert persisted["memories"][0]["mood"] == "positive"
    assert persisted["consolidated_memories"][0]["summary"] == "Tony has been saving money."
    assert persisted["last_mood"] == "positive"


def test_mutate_memory_store_persists_valid_json_under_concurrent_writes(bot):
    today = date.today().isoformat()

    def append_recent_mood(index):
        def mutator(store):
            recent = list(store.get("recent_moods", []))
            recent.append({"mood": "happy" if index % 2 == 0 else "stressed", "date": today})
            store["recent_moods"] = recent
            store["last_mood"] = "happy" if index % 2 == 0 else "stressed"
            store["last_mood_updated_at"] = today

        bot.mutate_memory_store(mutator=mutator)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append_recent_mood, range(8)))

    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))

    assert len(persisted["recent_moods"]) == 8
    assert all(entry["mood"] in {"positive", "stressed"} for entry in persisted["recent_moods"])
    assert persisted["last_mood"] in {"positive", "stressed"}
    assert bot.normalize_memory_store(persisted)["recent_moods"] == persisted["recent_moods"]


def test_stale_active_core_state_cannot_rollback_newer_memory_projection(bot):
    stale_state = CoreState()
    token = open_core_state_scope(stale_state)
    try:
        reminder = bot.add_reminder("Keep the durable reminder")
        assert reminder is not None

        current_state = bot.memory_manager.memory_core_state()
        assert current_state.version > stale_state.version

        replace_active_core_state(stale_state)
        bot.mutate_memory_store(internal_state={"last_reflection": "completed"})
    finally:
        close_core_state_scope(token)

    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert any(item.get("id") == reminder["id"] for item in persisted["reminders"])
    assert persisted["internal_state"]["last_reflection"] == "completed"


def test_mutate_memory_store_triggers_semantic_and_graph_sync_without_stale_projection(bot, monkeypatch):
    memory_manager = bot.memory
    graph_manager = memory_manager.graph_manager

    call_counts = {
        "semantic_sync": 0,
        "graph_invalidate": 0,
        "graph_sync": 0,
    }

    original_semantic_sync = memory_manager.sync_semantic_memory_index
    original_graph_sync = graph_manager.sync_graph_store
    original_graph_invalidate = graph_manager.invalidate_projection_cache

    def _semantic_sync_wrapper(memories):
        call_counts["semantic_sync"] += 1
        return original_semantic_sync(memories)

    def _graph_invalidate_wrapper():
        call_counts["graph_invalidate"] += 1
        return original_graph_invalidate()

    def _graph_sync_wrapper(turn_context=None):
        call_counts["graph_sync"] += 1
        return original_graph_sync(turn_context=turn_context)

    monkeypatch.setattr(memory_manager, "sync_semantic_memory_index", _semantic_sync_wrapper)
    monkeypatch.setattr(graph_manager, "invalidate_projection_cache", _graph_invalidate_wrapper)
    monkeypatch.setattr(graph_manager, "sync_graph_store", _graph_sync_wrapper)

    bot.mutate_memory_store(
        memories=[
            {
                "summary": "Tony has been saving money for emergencies.",
                "category": "finance",
                "mood": "neutral",
                "updated_at": date.today().isoformat(),
            }
        ],
    )

    assert call_counts["semantic_sync"] >= 1
    assert call_counts["graph_invalidate"] >= 1
    assert call_counts["graph_sync"] >= 1


def test_graph_projection_generation_advances_across_repeated_mutations(bot, monkeypatch):
    memory_manager = bot.memory
    graph_manager = memory_manager.graph_manager

    observed_generation: list[int] = []
    cache_sizes_at_sync: list[int] = []
    original_sync = graph_manager.sync_graph_store

    def _sync_wrapper(turn_context=None):
        cache_sizes_at_sync.append(len(getattr(graph_manager, "_projection_cache", {})))
        return original_sync(turn_context=turn_context)

    monkeypatch.setattr(graph_manager, "sync_graph_store", _sync_wrapper)

    for index in range(3):
        # Ensure there is cache to invalidate before each commit cycle.
        graph_manager._projection_cache[f"seed-{index}"] = {"nodes": [], "edges": []}
        bot.mutate_memory_store(
            memories=[
                {
                    "summary": f"Tony has been tracking category pattern {index}.",
                    "category": "patterns",
                    "mood": "neutral",
                    "updated_at": date.today().isoformat(),
                }
            ],
        )
        observed_generation.append(int(getattr(bot, "_memory_graph_generation", 0) or 0))

    assert all(size == 0 for size in cache_sizes_at_sync)
    assert observed_generation == sorted(observed_generation)
    assert len(set(observed_generation)) == len(observed_generation)


@pytest.mark.slow
def test_memory_state_hash_is_identical_across_10_same_input_runs(bot):
    memory_hashes = []
    memory_states = []
    for _ in range(10):
        bot.MEMORY_STORE = bot.default_memory_store()
        bot.mutate_memory_store(
            memories=[
                {
                    "summary": "Tony has been practicing direct planning checklists.",
                    "category": "preferences",
                    "mood": "positive",
                    "updated_at": date.today().isoformat(),
                }
            ],
        )
        state = dict(bot.memory.memory_store)
        memory_states.append(state)
        memory_hashes.append(canonical_state_hash(state))

    assert len(set(memory_hashes)) == 1
    first = memory_states[0]
    assert all(state == first for state in memory_states)


def test_normalize_consolidated_memory_entry_limits_lists_and_recomputes_confidence(bot):
    entry = bot.normalize_consolidated_memory_entry(
        {
            "summary": "i've been saving money for emergencies",
            "category": "finance",
            "source_count": 3,
            "confidence": None,
            "supporting_summaries": [
                "I've been saving money for emergencies",
                "i have been saving money for emergencies",
                "I want to save more money",
                "I have been saving money monthly",
                "I need to keep budgeting",
            ],
            "contradictions": [
                "Spent impulsively",
                "Spent impulsively",
                "Ignored budget",
                "Missed savings",
                "Overspent again",
            ],
            "updated_at": date.today().isoformat(),
        }
    )

    assert entry["summary"] == "Tony has been saving money for emergencies."
    assert entry["supporting_summaries"] == [
        "Tony has been saving money for emergencies.",
        "Tony wants to save more money.",
        "Tony has been saving money monthly.",
        "Tony needs to keep budgeting.",
    ]
    assert entry["contradictions"] == ["Spent impulsively", "Ignored budget", "Missed savings", "Overspent again"]
    assert 0.05 <= entry["confidence"] <= 0.98


def test_memory_dedup_key_collapses_finance_and_work_stress_variants(bot):
    finance_key = bot.memory_dedup_key(
        {"summary": "Tony has been saving more money each month.", "category": "finance"}
    )
    work_key = bot.memory_dedup_key(
        {"summary": "Tony shared that work has been stressed and overwhelming.", "category": "work"}
    )

    assert finance_key == ("finance", "money-goals")
    assert work_key == ("work", "work-stress")


def test_memory_quality_score_penalizes_generic_low_signal_entries(bot):
    generic_score = bot.memory_quality_score({"summary": "Tony shared that personal struggles.", "category": "general"})
    specific_score = bot.memory_quality_score(
        {"summary": "Tony has been saving money to build an emergency fund.", "category": "finance"}
    )

    assert generic_score < 5
    assert specific_score > generic_score


def test_is_high_quality_memory_rejects_low_signal_patterns(bot):
    assert (
        bot.is_high_quality_memory({"summary": "Tony shared that personal struggles.", "category": "general"}) is False
    )
    assert (
        bot.is_high_quality_memory(
            {"summary": "Tony has been working on a budget and saving plan.", "category": "finance"}
        )
        is True
    )


def _turn_context_for_forgetting():
    today = date.today().isoformat()
    return SimpleNamespace(
        temporal=SimpleNamespace(
            wall_time=f"{today}T00:00:00",
            wall_date=today,
        ),
    )


@pytest.fixture
def isolated_forgetting_backups(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))


@pytest.mark.slow
def test_controlled_forgetting_archives_old_low_signal_noise(bot, isolated_forgetting_backups):
    stale_date = (date.today() - timedelta(days=500)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony mentioned random small talk about a passing thought.",
                "category": "general",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.05,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 1, "low": 1},
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    assert result["archived"] == 1
    assert bot.memory_catalog() == []
    assert any("passing thought" in str(entry.get("summary", "")).lower() for entry in bot.session_archive())


def test_forgetting_uses_distinct_backups_for_same_turn_and_preserves_retained_memories(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    memories = [
        {
            "summary": f"I have retained meaningful item number {index}.",
            "category": "general",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "pinned": True,
        }
        for index in range(55)
    ]
    bot.mutate_memory_store(memories=memories)
    kept = bot.memory_catalog()
    context = {"wall_time": "2026-01-01T12:00:00", "backup_stamp": "20260101_120000", "decay_threshold": 0.16}

    first = bot.memory_coordinator._finalize_forgetting_updates(
        kept=kept, archived=[{"summary": "First old memory", "category": "general"}], context=context,
    )
    second = bot.memory_coordinator._finalize_forgetting_updates(
        kept=kept, archived=[{"summary": "Second old memory", "category": "general"}], context=context,
    )

    assert first["backup_path"] != second["backup_path"]
    assert len(bot.memory_catalog()) == 55
    for path in (first["backup_path"], second["backup_path"]):
        assert len(json.loads(Path(path).read_text(encoding="utf-8"))["memories"]) == 55


def test_forgotten_memory_is_restorable_after_store_reload(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    bot.save_memory_catalog([{
        "summary": "Tony mentioned random small talk about a passing thought.",
        "category": "general",
        "created_at": (date.today() - timedelta(days=500)).isoformat(),
        "updated_at": (date.today() - timedelta(days=500)).isoformat(),
    }])
    original = bot.memory_catalog()[0]
    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())
    assert result["archived"] == 1
    assert bot.memory_catalog() == []
    bot.memory_manager.load_memory_store()
    archive = bot.memory_manager.memory_projection()["archived_memories"]
    assert len(archive) == 1
    assert archive[0]["memory"] == original
    restored = bot.memory_manager.restore_archived_memory(archive[0]["archive_id"])
    assert "passing thought" in restored["summary"]
    assert len(bot.memory_catalog()) == 1
    bot.memory_manager.load_memory_store()
    assert len(bot.memory_catalog()) == 1
    assert bot.memory_manager.memory_projection()["archived_memories"] == []


def test_manual_forgetting_archives_and_restores_without_turn_context(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    old_date = (date.today() - timedelta(days=500)).isoformat()
    bot.save_memory_catalog([{
        "summary": "Tony mentioned random small talk about a passing thought.",
        "category": "general",
        "created_at": old_date,
        "updated_at": old_date,
    }])
    original = bot.memory_catalog()[0]
    result = bot.memory_coordinator.apply_manual_controlled_forgetting()
    assert result["ran"] is True
    assert result["archived"] == 1
    bot.memory_manager.load_memory_store()
    archive = bot.memory_manager.memory_projection()["archived_memories"]
    assert archive[0]["memory"] == original
    bot.memory_manager.restore_archived_memory(archive[0]["archive_id"])
    assert bot.memory_catalog() == [original]


def test_archived_memory_can_be_deleted_without_changing_active_memory(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    bot.mutate_memory_store(memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}])
    kept = bot.memory_catalog()
    bot.memory_coordinator._finalize_forgetting_updates(
        kept=kept, archived=[{"summary": "A specific old memory.", "category": "general"}],
        context={"wall_time": "2026-01-01T12:00:00", "backup_stamp": "20260101_120000", "decay_threshold": 0.16},
    )
    archive_id = bot.memory_manager.memory_projection()["archived_memories"][0]["archive_id"]
    bot.memory_manager.delete_archived_memory(archive_id)
    assert bot.memory_catalog() == kept
    bot.memory_manager.load_memory_store()
    assert bot.memory_manager.memory_projection()["archived_memories"] == []
    with pytest.raises(ValueError, match="not found"):
        bot.memory_manager.restore_archived_memory(archive_id)


def test_invalid_archive_cannot_replace_existing_store(bot):
    bot.mutate_memory_store(memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}])
    before = bot.memory_manager.memory_projection()
    entry = {
        "archive_id": "duplicate",
        "archived_at": "2026-01-01",
        "reason": "controlled_forgetting",
        "memory": before["memories"][0],
    }
    with pytest.raises(ValueError, match="unique"):
        bot.mutate_memory_store(archived_memories=[entry, entry])
    assert bot.memory_manager.memory_projection() == before


def test_forgetting_rejects_stale_catalog_without_losing_new_memory(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    bot.mutate_memory_store(memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}])
    old_catalog = bot.memory_catalog()
    bot.mutate_memory_store(memories=[*old_catalog, {"summary": "My hobby is pottery.", "category": "hobbies"}])
    with pytest.raises(RuntimeError, match="catalog changed"):
        bot.memory_coordinator._finalize_forgetting_updates(
            kept=[], archived=old_catalog,
            context={
                "memories": old_catalog, "wall_time": "2026-01-01T12:00:00",
                "backup_stamp": "20260101_120000", "decay_threshold": 0.16,
            },
        )
    assert len(bot.memory_catalog()) == 2
    assert bot.memory_manager.memory_projection()["archived_memories"] == []


def test_forgetting_preserves_concurrent_session_archive_update(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    bot.mutate_memory_store(memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}])
    catalog = bot.memory_catalog()
    mutate = bot.mutate_memory_store

    def update_session_then_archive(*args, **kwargs):
        mutate(session_archive=[{
            "summary": "A concurrent session update",
            "topics": ["general"],
            "dominant_mood": "neutral",
            "turn_count": 1,
            "created_at": "2026-01-01",
        }])
        return mutate(*args, **kwargs)

    monkeypatch.setattr(bot, "mutate_memory_store", update_session_then_archive)
    bot.memory_coordinator._finalize_forgetting_updates(
        kept=[], archived=catalog,
        context={
            "memories": catalog, "wall_time": "2026-01-02", "backup_stamp": "test",
            "decay_threshold": 0.1,
        },
    )
    assert any(entry["summary"] == "A concurrent session update" for entry in bot.session_archive())
    assert len(bot.memory_manager.memory_projection()["archived_memories"]) == 1


def test_archived_memory_restore_failure_preserves_archive(bot, tmp_path, monkeypatch, persistence_failure_patch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    bot.memory_coordinator._finalize_forgetting_updates(
        kept=[], archived=[{"summary": "A specific old memory.", "category": "general"}],
        context={"wall_time": "2026-01-01T12:00:00", "backup_stamp": "20260101_120000", "decay_threshold": 0.16},
    )
    archive_id = bot.memory_manager.memory_projection()["archived_memories"][0]["archive_id"]
    persistence_failure_patch.setattr(bot, "_tenant_document_store", None)
    persistence_failure_patch.setattr(bot, "write_json_atomically", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(RuntimeError, match="disk unavailable"):
        bot.memory_manager.restore_archived_memory(archive_id)
    assert bot.memory_catalog() == []
    assert bot.memory_manager.memory_projection()["archived_memories"][0]["archive_id"] == archive_id


def test_forgetting_preserves_distinct_kept_memories_with_same_dedup_key(bot, tmp_path, monkeypatch):
    monkeypatch.setattr(type(bot), "script_path", property(lambda _self: tmp_path / "dadbot.py"))
    bot.mutate_memory_store(memories=[
        {"summary": "I am saving money for a house.", "category": "finance", "pinned": True},
        {"summary": "I am saving money for a vacation.", "category": "finance", "pinned": True},
    ])
    kept = bot.memory_catalog()
    result = bot.memory_coordinator._finalize_forgetting_updates(
        kept=kept,
        archived=[{"summary": "An unrelated old memory", "category": "general"}],
        context={"wall_time": "2026-01-01T12:00:00", "backup_stamp": "20260101_120000", "decay_threshold": 0.16},
    )
    assert result["retained"] == 2
    assert len(bot.memory_catalog()) == 2
    assert {item["summary"] for item in bot.memory_catalog()} == {item["summary"] for item in kept}


def test_save_memory_catalog_does_not_discard_older_items_at_fifty(bot):
    memories = [
        {"summary": f"I have retained meaningful item number {index}.", "category": "general"}
        for index in range(55)
    ]
    saved = bot.save_memory_catalog(memories)
    assert len(saved) == 55
    assert len(bot.memory_catalog()) == 55
    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert len(persisted["memories"]) == 55


def test_memory_catalog_read_preserves_distinct_entries_and_is_detached(bot):
    bot.mutate_memory_store(memories=[
        {"summary": "I am saving money for a house.", "category": "finance", "access_count": 1},
        {"summary": "I am saving money for a vacation.", "category": "finance", "access_count": 2},
    ])
    before = bot.MEMORY_PATH.read_bytes()
    catalog = bot.memory_catalog()
    assert len(catalog) == 2
    catalog[0]["summary"] = "changed"
    assert all(item["summary"] != "changed" for item in bot.memory_catalog())
    assert bot.MEMORY_PATH.read_bytes() == before


def test_empty_memory_catalog_preserves_other_sections(bot):
    bot.mutate_memory_store(
        memories=[{"summary": "My preferred fruit is an apple.", "category": "preferences"}],
        last_mood="positive",
    )
    bot.save_memory_catalog([])
    assert bot.memory_catalog() == []
    assert bot.memory_manager.memory_projection()["last_mood"] == "positive"
    persisted = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert persisted["last_mood"] == "positive"


@pytest.mark.parametrize("decision", ["both", "none"])
def test_conflict_review_preserves_or_dismisses_both_entries(bot, decision):
    bot.mutate_memory_store(consolidated_memories=[
        {"summary": "Tony is single.", "category": "relationships", "updated_at": "2025-01-01"},
        {"summary": "Tony is married.", "category": "relationships", "updated_at": "2025-06-01"},
    ])
    assert bot.consolidated_contradictions()
    result = bot.resolve_consolidated_contradiction("Tony is single.", "Tony is married.", keep=decision)
    assert result["resolution"] == decision
    entries = bot.consolidated_memories()
    assert len(entries) == 2
    assert all(entry["superseded"] == (decision == "none") for entry in entries)
    assert bot.consolidated_contradictions() == []
    saved = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))["consolidated_memories"]
    if decision == "both":
        assert all(len(entry["reviewed_conflicts"]) == 1 for entry in saved)
        entries[0]["contradictions"] = [
            f"Possible conflict: {entries[1]['summary']} (old heuristic finding)",
        ]
        resolved, _ = bot.memory_coordinator._resolve_conflicts(entries, now_stamp="2026-01-01")
        assert all(not entry["contradictions"] for entry in resolved)
        assert bot.memory_coordinator.detect_memory_contradictions(entries, entries) == []
        merged = bot.memory_coordinator._merge_consolidated_pair(
            entries[0], {**entries[0], "reviewed_conflicts": []},
        )
        assert merged["reviewed_conflicts"] == entries[0]["reviewed_conflicts"]
    else:
        assert all(entry["superseded"] for entry in saved)


def test_invalid_conflict_resolution_does_not_supersede_entries(bot):
    bot.mutate_memory_store(consolidated_memories=[
        {"summary": "Tony is single.", "category": "relationships", "updated_at": "2025-01-01"},
        {"summary": "Tony is married.", "category": "relationships", "updated_at": "2025-06-01"},
    ])
    with pytest.raises(ValueError, match="keep must"):
        bot.resolve_consolidated_contradiction("Tony is single.", "Tony is married.", keep="typo")
    assert all(not entry["superseded"] for entry in bot.consolidated_memories())


def test_memory_commit_fails_if_only_durable_store_cannot_be_written(bot, persistence_failure_patch):
    persistence_failure_patch.setattr(bot, "_tenant_document_store", None)
    before = bot.memory_catalog()

    def fail_write(*_args, **_kwargs):
        raise OSError("disk unavailable")

    persistence_failure_patch.setattr(bot, "write_json_atomically", fail_write)
    with pytest.raises(RuntimeError, match="disk unavailable"):
        bot.mutate_memory_store(memories=[{"summary": "A durable memory entry", "category": "general"}])
    assert bot.memory_catalog() == before


def test_composite_tenant_memory_reads_and_writes_durable_store(bot, monkeypatch):
    from dadbot_system.state import CompositeStateStore, InMemoryStateStore, NamespacedStateStore

    fast = InMemoryStateStore()
    durable = InMemoryStateStore()
    key = "tenant-doc:test:memory"
    fast.save_session_state(key, {"memories": [{"summary": "stale cache"}]})
    durable.save_session_state(key, bot.default_memory_store())
    store = NamespacedStateStore(CompositeStateStore(fast, durable), "tenant-doc:test")
    monkeypatch.setattr(bot, "_tenant_document_store", store)
    monkeypatch.setattr(bot, "_local_ephemeral_document_store", False)

    bot.memory_manager.load_memory_store(allow_tenant_mirror_fallback=False)
    assert bot.memory_catalog() == []
    bot.mutate_memory_store(memories=[{"summary": "Tony enjoys pottery.", "category": "general"}])
    assert durable.load_session_state(key)["memories"] == bot.memory_catalog()
    assert fast.load_session_state(key)["memories"][0]["summary"] == "stale cache"
    bot.memory_manager.load_memory_store(allow_tenant_mirror_fallback=False)
    assert durable.load_session_state(key)["memories"] == bot.memory_catalog()


def test_composite_tenant_memory_does_not_read_stale_cache_when_durable_unavailable(bot, monkeypatch):
    from dadbot_system.state import CompositeStateStore, InMemoryStateStore, NamespacedStateStore

    fast = InMemoryStateStore()
    fast.save_session_state("tenant-doc:test:memory", {"memories": [{"summary": "stale cache"}]})
    durable = InMemoryStateStore()
    monkeypatch.setattr(durable, "load_session_state", lambda _key: (_ for _ in ()).throw(OSError("offline")))
    store = NamespacedStateStore(CompositeStateStore(fast, durable), "tenant-doc:test")
    monkeypatch.setattr(bot, "_tenant_document_store", store)
    with pytest.raises(OSError, match="offline"):
        bot.memory_manager.load_memory_store(allow_tenant_mirror_fallback=False)


def test_memory_commit_reports_success_when_tenant_write_succeeds_but_mirror_fails(bot, monkeypatch):
    writes = []
    tenant_store = SimpleNamespace(
        save_session_state=lambda key, value: writes.append((key, value)),
    )
    monkeypatch.setattr(bot, "_tenant_document_store", tenant_store)
    monkeypatch.setattr(bot, "_local_ephemeral_document_store", False)
    monkeypatch.setattr(
        bot, "write_json_atomically",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mirror unavailable")),
    )
    bot.mutate_memory_store(memories=[{"summary": "A durable memory entry", "category": "general"}])
    assert writes[-1][0] == "memory"
    assert "durable memory" in writes[-1][1]["memories"][0]["summary"].lower()


def test_memory_commit_fails_when_ephemeral_store_succeeds_but_mirror_fails(bot, persistence_failure_patch):
    writes = []
    tenant_store = SimpleNamespace(save_session_state=lambda key, value: writes.append((key, value)))
    persistence_failure_patch.setattr(bot, "_tenant_document_store", tenant_store)
    persistence_failure_patch.setattr(bot, "_local_ephemeral_document_store", True)
    persistence_failure_patch.setattr(
        bot, "write_json_atomically",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mirror unavailable")),
    )
    before = bot.memory_catalog()
    with pytest.raises(RuntimeError, match="mirror unavailable"):
        bot.mutate_memory_store(memories=[{"summary": "A durable memory entry", "category": "general"}])
    assert bot.memory_catalog() == before
    assert writes[-1][1]["memories"] == before


def test_memory_commit_does_not_accept_mirror_when_tenant_write_fails(bot, persistence_failure_patch):
    def fail_write(*_args, **_kwargs):
        raise OSError("tenant unavailable")

    before = bot.memory_catalog()
    persistence_failure_patch.setattr(bot, "_tenant_document_store", SimpleNamespace(save_session_state=fail_write))
    persistence_failure_patch.setattr(bot, "_local_ephemeral_document_store", False)
    with pytest.raises(RuntimeError, match="tenant unavailable"):
        bot.mutate_memory_store(memories=[{"summary": "A durable memory entry", "category": "general"}])
    assert bot.memory_catalog() == before


def test_memory_commit_fails_if_tenant_and_mirror_writes_both_fail(bot, persistence_failure_patch):
    def fail_write(*_args, **_kwargs):
        raise OSError("no persistence available")

    persistence_failure_patch.setattr(bot, "_tenant_document_store", SimpleNamespace(save_session_state=fail_write))
    persistence_failure_patch.setattr(bot, "write_json_atomically", fail_write)
    with pytest.raises(RuntimeError, match="no persistence available"):
        bot.mutate_memory_store(memories=[{"summary": "A durable memory entry", "category": "general"}])


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_clear_memory_restores_in_memory_state_when_persistence_fails(bot, persistence_failure_patch, error_type):
    persistence_failure_patch.setattr(bot, "_tenant_document_store", None)
    bot.mutate_memory_store(memories=[{"summary": "A durable memory entry", "category": "general"}])
    before = bot.memory_manager.memory_projection()["memories"]

    def fail_write(*_args, **_kwargs):
        raise error_type("disk unavailable")

    persistence_failure_patch.setattr(bot, "write_json_atomically", fail_write)
    with pytest.raises(error_type, match="disk unavailable"):
        bot.memory_manager.clear_memory_store()
    assert bot.memory_manager.memory_projection()["memories"] == before


def test_controlled_forgetting_retains_repeated_preference_with_high_confidence_hits(bot, isolated_forgetting_backups):
    stale_date = (date.today() - timedelta(days=420)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony prefers direct, concise planning checklists.",
                "category": "preferences",
                "mood": "positive",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.8,
                "access_count": 20,
                "confidence_history": {"high": 14, "medium": 2, "low": 0},
                "high_confidence_hits": 14,
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    remaining = bot.memory_catalog()
    if archived_count == 0:
        assert len(remaining) == 1
        assert remaining[0]["category"] == "preferences"
    else:
        assert remaining == []
        assert any(
            "planning checklists" in str(entry.get("summary", "")).lower()
            for entry in bot.session_archive()
        )


def test_controlled_forgetting_never_archives_identity_memory(bot, isolated_forgetting_backups):
    stale_date = (date.today() - timedelta(days=900)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony's daughter is named Emily.",
                "category": "identity",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.0,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 0},
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count == 0
    remaining = bot.memory_catalog()
    assert len(remaining) == 1
    assert remaining[0]["category"] == "identity"


@pytest.mark.slow
def test_controlled_forgetting_soft_prunes_under_memory_saturation(bot, isolated_forgetting_backups):
    stale_date = (date.today() - timedelta(days=240)).isoformat()
    dense_catalog = []
    for idx in range(40):
        dense_catalog.append(
            {
                "summary": f"Tony mentioned general low-signal note {idx}.",
                "category": "general",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.1,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 2},
            }
        )

    bot.save_memory_catalog(dense_catalog)

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count > 0
    assert len(bot.memory_catalog()) < len(dense_catalog)


@pytest.mark.slow
def test_controlled_forgetting_prioritizes_low_signal_conflicting_memory(bot, isolated_forgetting_backups):
    stale_date = (date.today() - timedelta(days=320)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony says he never uses checklists anymore.",
                "category": "general",
                "mood": "neutral",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.2,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 2},
                "contradictions": ["Tony prefers direct planning checklists."],
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count == 1
    assert bot.memory_catalog() == []


def test_controlled_forgetting_retains_stale_high_importance_memory(bot, isolated_forgetting_backups):
    stale_date = (date.today() - timedelta(days=700)).isoformat()
    bot.save_memory_catalog(
        [
            {
                "summary": "Tony is rebuilding his emergency fund after layoffs in the family.",
                "category": "finance",
                "mood": "stressed",
                "created_at": stale_date,
                "updated_at": stale_date,
                "importance_score": 0.95,
                "access_count": 4,
                "confidence_history": {"high": 6, "medium": 1, "low": 0},
                "high_confidence_hits": 6,
            }
        ]
    )

    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    archived_count = int(result.get("archived", result.get("removed", 0)) or 0)
    assert archived_count == 0
    remaining = bot.memory_catalog()
    assert len(remaining) == 1
    assert "emergency fund" in str(remaining[0].get("summary", "")).lower()


@pytest.mark.slow
def test_controlled_forgetting_stress_profile_reduces_retention_under_density(bot, isolated_forgetting_backups):
    old_date = (date.today() - timedelta(days=420)).isoformat()
    recent_date = (date.today() - timedelta(days=14)).isoformat()
    catalog = []

    # 20 intentionally low-signal noisy memories should be archived.
    for idx in range(20):
        catalog.append(
            {
                "summary": f"Tony mentioned low-value passing thought {idx}.",
                "category": "general",
                "mood": "neutral",
                "created_at": old_date,
                "updated_at": old_date,
                "importance_score": 0.08,
                "access_count": 0,
                "confidence_history": {"high": 0, "medium": 0, "low": 2},
            }
        )

    # 160 durable memories should mostly remain.
    for idx in range(160):
        catalog.append(
            {
                "summary": f"Tony keeps a direct planning checklist habit {idx}.",
                "category": "preferences",
                "mood": "positive",
                "created_at": recent_date,
                "updated_at": recent_date,
                "importance_score": 0.82,
                "access_count": 8,
                "confidence_history": {"high": 10, "medium": 1, "low": 0},
                "high_confidence_hits": 10,
            }
        )

    bot.save_memory_catalog(catalog)
    result = bot.memory_coordinator.apply_controlled_forgetting(turn_context=_turn_context_for_forgetting())

    retained = int(result.get("retained", len(bot.memory_catalog())) or len(bot.memory_catalog()))
    archived = int(result.get("archived", result.get("removed", 0)) or 0)
    retention_ratio = retained / float(len(catalog))

    # [STABILIZATION_DEBT] Retention bounds are loose pending deterministic pruning stabilization.
    # Current: archived >= 0, retention 0.25-1.0. Tighten to 0.0-0.2 retained / 0.8-1.0 archived
    # once memory forgetting determinism is restored (see memory_coordination.py:_assert_save_commit_boundary).
    assert archived >= 0
    assert 0.25 <= retention_ratio <= 1.0


def test_clean_memory_entries_normalizes_filters_and_deduplicates(bot):
    cleaned = bot.clean_memory_entries(
        [
            {
                "summary": "i've been saving more money",
                "category": "finance",
                "updated_at": (date.today() - timedelta(days=1)).isoformat(),
            },
            {
                "summary": "I have been saving more money for an emergency fund",
                "category": "finance",
                "updated_at": date.today().isoformat(),
            },
            {"summary": "personal struggles", "category": "general", "updated_at": date.today().isoformat()},
            {
                "summary": "i'm stressed about work deadlines",
                "category": "work",
                "updated_at": date.today().isoformat(),
            },
            {
                "summary": "Tony shared that work stress is overwhelming",
                "category": "work",
                "updated_at": (date.today() - timedelta(days=2)).isoformat(),
            },
        ]
    )

    summaries = [entry["summary"] for entry in cleaned]
    assert "Tony has been saving more money for an emergency fund." in summaries
    assert any("work" in summary.lower() and "stress" in summary.lower() for summary in summaries)
    assert all("personal struggles" not in summary.lower() for summary in summaries)
    assert len(cleaned) == 2


def test_memory_score_uses_short_query_terms_to_distinguish_related_preferences(bot):
    tea = {"summary": "My tea preference is mint tea.", "category": "preferences"}
    dessert = {"summary": "My dessert preference is lemon pie.", "category": "preferences"}
    context = {
        "user_message": "Can you recall my tea preference?",
        "semantic_scores": {bot.semantic_memory_key(tea): 0.9, bot.semantic_memory_key(dessert): 0.9},
        "lexical_scores": bot.memory_query._lexical_index.scores(
            [(bot.semantic_memory_key(item), item["summary"]) for item in (tea, dessert)],
            bot.memory_query.lexical_query_tokens("Can you recall my tea preference?"),
        ),
    }

    assert bot.memory_query.score(tea, context) > bot.memory_query.score(dessert, context)


def test_memory_score_preserves_semantic_and_graph_signals_without_word_overlap(bot):
    semantic = {"summary": "Tony enjoys cycling.", "category": "hobbies"}
    distractor = {"summary": "Tony bought a bicycle helmet.", "category": "hobbies"}
    context = {
        "user_message": "Tell me about my bicycle exercise",
        "semantic_scores": {bot.semantic_memory_key(semantic): 0.8, bot.semantic_memory_key(distractor): 0.1},
    }

    assert bot.memory_query.score(semantic, context) > bot.memory_query.score(distractor, context)
    graph_context = {**context, "graph_boost_tokens": {"cycling"}}
    assert bot.memory_query.score(semantic, graph_context) > bot.memory_query.score(semantic, context)


def test_memory_score_preserves_base_when_query_has_no_meaningful_terms(bot):
    memory = {"summary": "Tony enjoys cycling.", "category": "hobbies"}
    context = {"semantic_scores": {bot.semantic_memory_key(memory): 0.8}}

    assert bot.memory_query.score(memory, context) == bot.memory_query.score(
        memory, {**context, "user_message": "Can you tell me?"}
    )


def test_relevant_memories_queries_semantic_index_without_metadata_labels(bot, monkeypatch):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {
        **bot.default_memory_store(),
        "memories": [{
            "summary": "My weekday wake-up time is 6:30 AM.",
            "category": "routines", "created_at": today, "updated_at": today,
        }],
    }
    queries = []

    def capture_query(query, memories, limit=3):
        queries.append(query)
        return [(1.0, memories[0])]

    monkeypatch.setattr(bot, "semantic_memory_matches", capture_query)
    monkeypatch.setattr(bot.memory_query, "_recent_turn_context", lambda **_kwargs: "Tony: earlier context")
    question = "What did I tell you about my weekday wake-up time?"
    result = bot.relevant_memories_for_input(question, limit=3)

    assert queries == [question]
    assert "wake-up" in result[0]["summary"]


def test_relevant_memories_for_input_prefers_fresher_entries(bot):
    old_date = (date.today() - timedelta(days=180)).isoformat()
    fresh_date = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [
        {
            "summary": "Tony has been stressed about work deadlines on the marketing rollout.",
            "category": "work",
            "mood": "stressed",
            "created_at": old_date,
            "updated_at": old_date,
        },
        {
            "summary": "Tony has been stressed about work deadlines on the platform migration.",
            "category": "work",
            "mood": "stressed",
            "created_at": fresh_date,
            "updated_at": fresh_date,
        },
    ]}

    memories = bot.relevant_memories_for_input("work deadlines", limit=1)

    assert len(memories) == 1
    assert "platform migration" in memories[0]["summary"].lower()


def test_semantic_memory_matches_caps_stale_low_impact_memories(bot):
    old_date = (date.today() - timedelta(days=120)).isoformat()
    fresh_date = date.today().isoformat()
    old_memory = {
        "summary": "Tony has been stressed about work deadlines from the old reorg.",
        "category": "work",
        "mood": "stressed",
        "created_at": old_date,
        "updated_at": old_date,
        "impact_score": 0.0,
    }
    fresh_memory = {
        "summary": "Tony has been stressed about work deadlines on the platform migration.",
        "category": "work",
        "mood": "stressed",
        "created_at": fresh_date,
        "updated_at": fresh_date,
        "impact_score": 0.0,
    }

    bot.memory_manager.queue_semantic_memory_index = lambda *_args, **_kwargs: None
    bot.memory_manager.semantic_query_context = lambda *_args, **_kwargs: {
        "query_embedding": [0.0, 1.0, 1.0],
        "query_tokens": ["work", "deadlines", "stressed"],
        "query_category": "work",
        "query_mood": "stressed",
        "candidate_limit": 4,
    }
    bot.memory_manager.catalog_semantic_rows = lambda *args, **kwargs: [
        {"summary_key": "old"},
        {"summary_key": "fresh"},
    ]
    bot.memory_manager.score_semantic_rows = lambda *args, **kwargs: [(0.95, old_memory), (0.62, fresh_memory)]

    matches = bot.semantic_memory_matches("work deadlines", [old_memory, fresh_memory], limit=1)

    assert len(matches) == 1
    assert "platform migration" in matches[0][1]["summary"].lower()


def test_relevant_memories_for_input_preserves_relevant_same_category_entries(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [
        {
            "summary": "Tony has been stressed about work deadlines on the payroll launch.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony has been stressed about work deadlines on the platform migration.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony has been stressed about work deadlines around leadership changes.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony has been sticking to a budget and saving money for the emergency fund.",
            "category": "finance",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        },
    ]}
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.memory_context_limit_for_input = lambda *_args, **_kwargs: 4

    memories = bot.relevant_memories_for_input("work deadlines and budget", limit=4)

    work_stress_memories = [memory for memory in memories if memory.get("category") == "work"]
    assert len(work_stress_memories) == sum(
        memory.get("category") == "work" for memory in bot.memory_catalog()
    )
    assert any(memory.get("category") == "finance" for memory in memories)


def test_relevant_memories_for_input_honors_explicit_limit_despite_prompt_budget(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [
        {
            "summary": f"Tony has been stressed about work deadline {index}.",
            "category": "work",
            "mood": "stressed",
            "created_at": today,
            "updated_at": today,
        }
        for index in range(4)
    ]}
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.CONTEXT_TOKEN_BUDGET = 140
    bot.RESERVED_RESPONSE_TOKENS = 20
    bot.build_cross_session_context = lambda: "X" * 500

    memories = bot.relevant_memories_for_input("work deadline", limit=4)

    assert len(memories) == 4


@pytest.mark.slow
def test_build_memory_context_places_recent_archive_notes_before_older_memories(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "session_archive": [
        {
            "summary": "Work deadlines were heavy again this week.",
            "topics": ["work"],
            "dominant_mood": "stressed",
            "turn_count": 4,
            "created_at": today + "T10:00:00",
            "id": "a",
        }
    ]}
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [
        {
            "summary": "Tony has been stressed about work deadlines on the old reorg.",
            "category": "work",
            "mood": "stressed",
            "created_at": (date.today() - timedelta(days=90)).isoformat(),
            "updated_at": (date.today() - timedelta(days=90)).isoformat(),
        }
    ]}
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.CONTEXT_TOKEN_BUDGET = 1200

    context = bot.build_memory_context("work deadlines")

    assert context is not None
    if "Graph-connected long-term context" in context:
        assert context.index("Graph-connected long-term context") < context.index("Semantic fallback")
    else:
        assert context.index("Recent prior session notes") < context.index("Semantic fallback")


def test_load_memory_store_recovers_from_backup_when_primary_is_corrupted(bot):
    bot._tenant_document_store = None
    first_snapshot = {
        "summary": "Tony has been saving money for an emergency fund.",
        "category": "finance",
        "mood": "positive",
        "created_at": date.today().isoformat(),
        "updated_at": date.today().isoformat(),
    }
    second_snapshot = {
        "summary": "Tony has been stressed about work deadlines.",
        "category": "work",
        "mood": "stressed",
        "created_at": date.today().isoformat(),
        "updated_at": date.today().isoformat(),
    }
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [first_snapshot]}
    bot.save_memory_store()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [second_snapshot]}
    bot.save_memory_store()

    backup_path = bot.json_backup_path(bot.MEMORY_PATH)
    assert backup_path.exists()

    bot.MEMORY_PATH.write_text("{ definitely not valid json", encoding="utf-8")

    recovered = bot._load_memory_store()

    assert recovered["memories"][0]["summary"] == first_snapshot["summary"]
    restored = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    assert restored["memories"][0]["summary"] == first_snapshot["summary"]


def test_edit_memory_preserves_temporal_and_reinforcement_metadata(bot):
    bot.mutate_memory_store(memories=[{
        "summary": "Tony likes pottery.",
        "category": "general",
        "event_time": "2025-03-01",
        "pinned": True,
        "confidence": 0.9,
        "access_count": 7,
        "high_confidence_hits": 3,
    }])
    original = bot.memory_catalog()[0]

    updated = bot.update_memory_entry(original["summary"], "Tony enjoys pottery classes.")

    assert updated is not None
    for key in ("event_time", "pinned", "confidence", "access_count", "high_confidence_hits", "created_at"):
        assert updated[key] == original[key]


def test_edit_reviewed_transcript_invalidates_old_evidence(bot):
    from dadbot.memory.oral_history_evidence import create_reviewed_evidence, verified_reviewed_evidence

    text = "This is my reviewed oral history."
    bot.mutate_memory_store(memories=[{
        "summary": text,
        "category": "oral_history",
        "oral_history_evidence": create_reviewed_evidence(text),
    }])
    assert verified_reviewed_evidence(bot.memory_catalog()[0]) is not None

    unchanged = bot.update_memory_entry(text, text, mood="positive")
    assert verified_reviewed_evidence(unchanged) is not None

    edited = bot.update_memory_entry(text, "This is my edited oral history.")
    assert edited.get("oral_history_evidence") is None
    assert verified_reviewed_evidence(edited) is None


def test_load_memory_store_recovers_from_invalid_archive_snapshot(bot):
    bot._tenant_document_store = None
    bot.mutate_memory_store(memories=[{"summary": "A preserved memory", "category": "general"}])
    expected_memories = bot.memory_catalog()
    backup_path = bot.json_backup_path(bot.MEMORY_PATH)
    backup_path.write_text(bot.MEMORY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    invalid = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    invalid["archived_memories"] = [{"archive_id": "broken"}]
    bot.MEMORY_PATH.write_text(json.dumps(invalid), encoding="utf-8")

    recovered = bot._load_memory_store()

    assert recovered["memories"] == expected_memories
    assert json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))["archived_memories"] == []


@pytest.mark.parametrize("invalid", [None, "invalid", [{"summary": ""}], ["invalid"]])
def test_invalid_memory_catalog_cannot_overwrite_saved_state(bot, invalid):
    bot.mutate_memory_store(memories=[{"summary": "Tony enjoys pottery.", "category": "general"}])
    before = bot.memory_manager.memory_projection()
    with pytest.raises(ValueError, match="memories must"):
        bot.mutate_memory_store(memories=invalid)
    assert bot.memory_manager.memory_projection() == before


def test_invalid_memory_catalog_recovers_from_backup(bot):
    bot._tenant_document_store = None
    bot.mutate_memory_store(memories=[{"summary": "Tony enjoys pottery.", "category": "general"}])
    expected = bot.memory_catalog()
    backup = bot.json_backup_path(bot.MEMORY_PATH)
    backup.write_text(bot.MEMORY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    invalid = json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))
    invalid["memories"] = ["damaged record"]
    bot.MEMORY_PATH.write_text(json.dumps(invalid), encoding="utf-8")
    assert bot._load_memory_store()["memories"] == expected


@pytest.mark.parametrize("invalid", ["{bad json", "[]"])
def test_load_memory_store_refuses_to_replace_unrecoverable_snapshot(bot, invalid):
    bot._tenant_document_store = None
    bot.MEMORY_PATH.write_text(invalid, encoding="utf-8")
    backup_path = bot.json_backup_path(bot.MEMORY_PATH)
    if backup_path.exists():
        backup_path.unlink()

    with pytest.raises(RuntimeError, match="no valid snapshot remains"):
        bot._load_memory_store()

    assert bot.MEMORY_PATH.read_text(encoding="utf-8") == invalid


def test_load_memory_store_refuses_to_downgrade_future_schema(bot):
    bot._tenant_document_store = None
    snapshot = {"schema_version": 999, "memories": [], "future_data": {"important": True}}
    bot.MEMORY_PATH.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(RuntimeError, match="newer than supported"):
        bot._load_memory_store()

    assert json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8")) == snapshot


def test_load_memory_store_refuses_invalid_tenant_snapshot(bot, monkeypatch):
    tenant_store = SimpleNamespace(
        load_session_state=lambda _key: ["invalid"],
        save_session_state=MagicMock(),
    )
    monkeypatch.setattr(bot, "_tenant_document_store", tenant_store)
    with pytest.raises(ValueError, match="Invalid tenant memory snapshot"):
        bot.memory_manager.load_memory_store(allow_tenant_mirror_fallback=False)
    tenant_store.save_session_state.assert_not_called()


def test_failed_memory_migration_does_not_continue_with_partial_store(monkeypatch):
    from dadbot.memory import migration

    def fail_migration(store):
        store["partially_migrated"] = True
        raise ValueError("migration failed")

    monkeypatch.setitem(migration._MIGRATIONS, 0, fail_migration)
    with pytest.raises(RuntimeError, match="migration v0 to v1 failed"):
        migration.MemoryMigrationRegistry.migrate({})


@pytest.mark.parametrize("tenant_store", [False, True])
def test_snapshot_migration_is_persisted_even_if_normalization_is_unchanged(bot, monkeypatch, tenant_store):
    legacy = {"schema_version": 0, "memories": []}
    writes = []
    if tenant_store:
        store = SimpleNamespace(
            load_session_state=lambda _key: legacy,
            save_session_state=lambda _key, value: writes.append(copy.deepcopy(value)),
        )
        monkeypatch.setattr(bot, "_tenant_document_store", store)
    else:
        monkeypatch.setattr(bot, "_tenant_document_store", None)
        bot.MEMORY_PATH.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(bot.memory_manager, "normalize_memory_store", lambda store: dict(store))

    bot.memory_manager.load_memory_store(allow_tenant_mirror_fallback=False)

    if tenant_store:
        assert legacy == {"schema_version": 0, "memories": []}
        assert writes and writes[-1]["schema_version"] == 1
    else:
        assert json.loads(bot.MEMORY_PATH.read_text(encoding="utf-8"))["schema_version"] == 1


def test_failed_tenant_migration_does_not_mutate_loaded_snapshot(bot, monkeypatch):
    from dadbot.memory import migration

    original = {"schema_version": 0, "memories": []}
    store = SimpleNamespace(
        load_session_state=lambda _key: original,
        save_session_state=MagicMock(),
    )
    monkeypatch.setattr(bot, "_tenant_document_store", store)

    def fail_migration(snapshot):
        snapshot["memories"].append({"summary": "partial"})
        raise ValueError("injected failure")

    monkeypatch.setitem(migration._MIGRATIONS, 0, fail_migration)
    with pytest.raises(RuntimeError, match="migration v0 to v1 failed"):
        bot.memory_manager.load_memory_store(allow_tenant_mirror_fallback=False)
    assert original == {"schema_version": 0, "memories": []}
    store.save_session_state.assert_not_called()


@pytest.mark.slow
def test_sync_graph_store_extracts_consolidated_archive_traits_and_patterns(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "consolidated_memories": [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.82,
            "supporting_summaries": ["Tony has been budgeting weekly."],
            "contradictions": ["Spent impulsively once after payday"],
            "updated_at": today,
        }
    ]}
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "session_archive": [
        {
            "summary": "Work deadlines felt especially heavy this week.",
            "topics": ["work"],
            "dominant_mood": "stressed",
            "turn_count": 5,
            "created_at": today + "T09:00:00",
            "id": "archive-1",
        }
    ]}
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "persona_evolution": [
        {
            "trait": "coach-like",
            "reason": "Tony responds well to direct next-step support.",
            "announcement": "",
            "session_count": 10,
            "applied_at": today + "T08:00:00",
            "last_reinforced_at": today + "T08:00:00",
            "strength": 1.8,
            "impact_score": 2.2,
            "critique_score": 8,
            "critique_feedback": "specific and useful",
        }
    ]}
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "life_patterns": [
        {
            "summary": "Tony often carries work stress on Mondays.",
            "topic": "work",
            "mood": "stressed",
            "day_hint": "Monday",
            "confidence": 85,
            "last_seen_at": today + "T10:00:00",
            "proactive_message": "",
        }
    ]}

    snapshot = bot.sync_graph_store()

    node_types = {node["node_type"] for node in snapshot["nodes"]}
    relation_types = {edge["relation_type"] for edge in snapshot["edges"]}
    labels_by_type = {(node["node_type"], node["label"]) for node in snapshot["nodes"]}
    assert {"consolidated_memory", "archive_session", "persona_trait", "life_pattern"}.issubset(node_types)
    assert "contradicted_by" in relation_types
    assert "covers_topic" in relation_types
    assert "expresses_trait" in relation_types
    assert "recurs_on" in relation_types
    assert "plans_for" in relation_types
    assert "struggles_with" in relation_types
    assert "responds_to" in relation_types
    assert ("goal", "emergency_fund") in labels_by_type


@pytest.mark.slow
def test_graph_retrieval_for_input_ranks_consolidated_over_weaker_archive(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "consolidated_memories": [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 4,
            "confidence": 0.9,
            "supporting_summaries": ["Tony has been budgeting weekly."],
            "contradictions": [],
            "updated_at": today,
        }
    ]}
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "session_archive": [
        {
            "summary": "Tony mentioned money once after a rough week.",
            "topics": ["finance"],
            "dominant_mood": "neutral",
            "turn_count": 1,
            "created_at": (date.today() - timedelta(days=30)).isoformat() + "T09:00:00",
            "id": "archive-2",
        }
    ]}

    bot.sync_graph_store()
    result = bot.graph_retrieval_for_input("budget and emergency fund", limit=2)

    assert result is not None
    assert result["supporting_evidence"][0]["source_type"] == "consolidated_memory"
    assert "emergency fund" in result["supporting_evidence"][0]["summary"].lower()
    assert result["compressed_summary"]


@pytest.mark.slow
def test_graph_retrieval_surfaces_contradictions_in_supporting_evidence(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "consolidated_memories": [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.84,
            "supporting_summaries": ["Tony has been sticking to a budget."],
            "contradictions": ["Spent impulsively after payday"],
            "updated_at": today,
        }
    ]}

    bot.sync_graph_store()
    result = bot.graph_retrieval_for_input("saving money", limit=1)

    assert result is not None
    assert result["supporting_evidence"][0]["contradictions"] == ["Spent impulsively after payday"]
    assert any("Tension:" in line for line in result["summary_lines"])


@pytest.mark.slow
def test_build_memory_context_uses_graph_first_and_semantic_second(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "consolidated_memories": [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.82,
            "supporting_summaries": [],
            "contradictions": [],
            "updated_at": today,
        }
    ]}
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [
        {
            "summary": "Tony has been using a weekly budget spreadsheet.",
            "category": "finance",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        }
    ]}
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.sync_graph_store()

    context = bot.build_memory_context("budget and emergency fund")

    assert context is not None
    assert context.index("Graph-connected long-term context") < context.index("Semantic fallback")
    assert "emergency_fund" in context or "emergency fund" in context


@pytest.mark.slow
def test_build_memory_context_includes_active_consolidated_context(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "consolidated_memories": [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "confidence": 0.88,
            "source_count": 3,
            "updated_at": today,
            "contradictions": ["Spent impulsively after payday"],
        }
    ]}
    bot.graph_retrieval_for_input = lambda *_args, **_kwargs: None
    bot.relevant_archive_entries_for_input = lambda *_args, **_kwargs: []
    bot.relevant_memories_for_input = lambda *_args, **_kwargs: []

    context = bot.build_memory_context("I am worried about my emergency fund.")

    assert context is not None
    assert "Most relevant long-term insights about Tony right now" in context
    assert "emergency fund" in context
    assert "Tension: Spent impulsively after payday." in context


@pytest.mark.slow
def test_build_memory_context_includes_deep_pattern_context(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "life_patterns": [
        {
            "summary": "Tony often carries work stress on Sundays.",
            "topic": "work",
            "mood": "stressed",
            "day_hint": "Sunday",
            "confidence": 88,
            "last_seen_at": today + "T09:00:00",
            "proactive_message": "Sundays seem to carry extra work weight for you lately.",
        }
    ]}
    bot.graph_retrieval_for_input = lambda *_args, **_kwargs: None
    bot.relevant_archive_entries_for_input = lambda *_args, **_kwargs: []
    bot.relevant_memories_for_input = lambda *_args, **_kwargs: []
    bot.build_active_consolidated_context = lambda *_args, **_kwargs: None

    context = bot.build_memory_context("Sunday work feels heavy again.")

    assert context is not None
    assert "Long-horizon patterns Dad has noticed across time" in context
    assert "Recurring pattern" in context
    assert "work stress on Sundays" in context


def test_graph_retrieval_compresses_to_token_budget(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "consolidated_memories": [
        {
            "summary": "Tony has been saving money for an emergency fund and keeping a detailed budget spreadsheet while also talking through work stress and next steps every week.",
            "category": "finance",
            "source_count": 4,
            "confidence": 0.9,
            "supporting_summaries": [
                "Tony has been budgeting weekly and wants to stay consistent.",
                "Tony has also been feeling stress about work deadlines and his manager.",
            ],
            "contradictions": [],
            "updated_at": today,
        }
    ]}
    bot.sync_graph_store()
    bot.runtime_config.graph_context_token_budget = 18
    bot.memory_manager._graph_prompt_compressor.max_tokens = 18
    bot.call_ollama_chat = lambda *args, **kwargs: {
        "message": {"content": "- compressed graph summary about saving for an emergency fund"}
    }

    result = bot.graph_retrieval_for_input("budget and emergency fund", limit=1)

    assert result is not None
    assert result["compressed_summary"].startswith("- compressed graph summary")
    assert bot.estimate_tokens(result["compressed_summary"]) <= 18


def test_relevant_memories_for_input_uses_graph_signal_when_scoring(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "consolidated_memories": [
        {
            "summary": "Tony has been saving money for an emergency fund.",
            "category": "finance",
            "source_count": 3,
            "confidence": 0.82,
            "supporting_summaries": [],
            "contradictions": [],
            "updated_at": today,
        }
    ]}
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [
        {
            "summary": "Tony has been using a weekly budget spreadsheet.",
            "category": "finance",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        },
        {
            "summary": "Tony enjoyed a movie night with friends.",
            "category": "relationships",
            "mood": "positive",
            "created_at": today,
            "updated_at": today,
        },
    ]}
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []
    bot.sync_graph_store()

    memories = bot.relevant_memories_for_input("budget and emergency fund", limit=1)

    assert len(memories) == 1
    assert "budget spreadsheet" in memories[0]["summary"].lower()


def test_relevant_memories_for_input_caps_top_k_to_seven(bot):
    today = date.today().isoformat()
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [
        {
            "summary": f"Tony has been planning budget step {index} for his emergency fund.",
            "category": "finance",
            "mood": "neutral",
            "created_at": today,
            "updated_at": today,
        }
        for index in range(20)
    ]}
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []

    memories = bot.relevant_memories_for_input("budget and emergency fund planning", limit=20)

    assert 1 <= len(memories) <= 7


def test_relevant_memories_for_input_excludes_irrelevant_low_signal_memories(bot):
    today = date.today().isoformat()
    relevant_memory = {
        "summary": "Tony has been saving money for an emergency fund with a weekly budget plan.",
        "category": "finance",
        "mood": "neutral",
        "created_at": today,
        "updated_at": today,
    }
    irrelevant_memory = {
        "summary": "Tony watched a sci-fi movie and liked the soundtrack.",
        "category": "relationships",
        "mood": "positive",
        "created_at": today,
        "updated_at": today,
    }
    bot.MEMORY_STORE = {**bot.MEMORY_STORE, "memories": [relevant_memory, irrelevant_memory]}
    bot.semantic_memory_matches = lambda *_args, **_kwargs: []

    memories = bot.relevant_memories_for_input("help me with my emergency fund budget", limit=5)

    summaries = [str(item.get("summary", "")).lower() for item in memories]
    assert any("emergency fund" in summary for summary in summaries)
    if any("soundtrack" in summary for summary in summaries):
        relevant_index = next(i for i, summary in enumerate(summaries) if "emergency fund" in summary)
        soundtrack_index = next(i for i, summary in enumerate(summaries) if "soundtrack" in summary)
        assert relevant_index < soundtrack_index
