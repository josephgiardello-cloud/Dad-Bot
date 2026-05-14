from __future__ import annotations

import pytest

from dadbot.core.world_model import (
    InMemoryAsyncWorldModelPersistence,
    SQLiteAsyncWorldModelPersistence,
    WorldModelStore,
)


@pytest.mark.asyncio
async def test_world_model_store_evolves_and_merges_lists() -> None:
    store = WorldModelStore(InMemoryAsyncWorldModelPersistence())

    first = await store.evolve_from_turn(
        {
            "session_id": "s1",
            "policy_version": "dad_v3.0",
            "trust_level": 61,
            "openness_level": 54,
            "emotional_momentum": "steady",
            "key_facts": ["Tony has two sisters"],
            "active_goals": ["sleep better"],
            "family_map": {"mom": "supportive"},
        }
    )

    second = await store.evolve_from_turn(
        {
            "session_id": "s1",
            "trust_level": 66,
            "openness_level": 58,
            "emotional_momentum": "warming",
            "key_facts": ["Tony has two sisters", "Tony likes evening walks"],
            "active_goals": ["sleep better", "exercise"],
            "contradictions": ["sleep schedule changed"],
            "family_map": {"dad": "mentor"},
        }
    )

    assert first.session_id == "s1"
    assert second.policy_version == "dad_v3.0"
    assert second.trust_level == 66
    assert second.openness_level == 58
    assert second.emotional_momentum == "warming"
    assert "Tony has two sisters" in second.key_facts
    assert "Tony likes evening walks" in second.key_facts
    assert "exercise" in second.active_goals
    assert "sleep schedule changed" in second.contradictions
    assert second.family_map["mom"] == "supportive"
    assert second.family_map["dad"] == "mentor"
    assert len(second.emotional_timeline) == 2


@pytest.mark.asyncio
async def test_world_model_store_sqlite_persists_across_instances(tmp_path) -> None:
    db_path = tmp_path / "world_model.sqlite3"
    store_one = WorldModelStore(SQLiteAsyncWorldModelPersistence(str(db_path)))
    store_two = WorldModelStore(SQLiteAsyncWorldModelPersistence(str(db_path)))

    await store_one.evolve_from_turn(
        {
            "session_id": "persisted-session",
            "policy_version": "dad_v3.1",
            "trust_level": 72,
            "openness_level": 64,
            "emotional_momentum": "calm",
        }
    )

    loaded = await store_two.get_model("persisted-session")
    assert loaded is not None
    assert loaded.session_id == "persisted-session"
    assert loaded.policy_version == "dad_v3.1"
    assert loaded.trust_level == 72
    assert loaded.openness_level == 64