from __future__ import annotations

import pytest

from dadbot.core.policy_store import DadPolicy, DadPolicyStore, InMemoryAsyncPolicyPersistence


@pytest.mark.asyncio
async def test_policy_store_seeds_default_policy_and_persists_history() -> None:
    store = DadPolicyStore(InMemoryAsyncPolicyPersistence())

    policy = await store.get_current_policy()

    assert policy.version == "dad_v1.0"
    assert policy.persona_style["warmth"] == 0.9

    history = await store.list_history()
    assert len(history) == 1
    assert history[0].version == "dad_v1.0"


@pytest.mark.asyncio
async def test_policy_store_supports_rollback_by_version() -> None:
    store = DadPolicyStore(InMemoryAsyncPolicyPersistence())
    base = await store.get_current_policy()

    updated = DadPolicy(
        version="dad_v1.1",
        persona_style={**base.persona_style, "humor": 0.9},
        relationship_rules=dict(base.relationship_rules),
        safety_boundaries=dict(base.safety_boundaries),
        memory_preferences=dict(base.memory_preferences),
        created_at=base.created_at,
        comment="Bump humor",
    )
    await store.save_policy(updated, comment="new version")

    rolled = await store.rollback_to_version("dad_v1.0", comment="rollback")

    assert rolled.version == "dad_v1.0"
    current = await store.get_current_policy()
    assert current.version == "dad_v1.0"
    assert current.comment == "rollback"


@pytest.mark.asyncio
async def test_policy_store_rejects_invalid_harmful_keywords_schema() -> None:
    store = DadPolicyStore(InMemoryAsyncPolicyPersistence())
    base = await store.get_current_policy()

    with pytest.raises(ValueError, match="harmful_keywords"):
        invalid = DadPolicy(
            version="dad_v1.2",
            persona_style=dict(base.persona_style),
            relationship_rules=dict(base.relationship_rules),
            safety_boundaries={**dict(base.safety_boundaries), "harmful_keywords": "hack"},
            memory_preferences=dict(base.memory_preferences),
            created_at=base.created_at,
        )
        await store.save_policy(invalid)


@pytest.mark.asyncio
async def test_policy_store_normalizes_harmful_keywords_to_lowercase() -> None:
    store = DadPolicyStore(InMemoryAsyncPolicyPersistence())
    base = await store.get_current_policy()

    policy = DadPolicy(
        version="dad_v1.3",
        persona_style=dict(base.persona_style),
        relationship_rules=dict(base.relationship_rules),
        safety_boundaries={**dict(base.safety_boundaries), "harmful_keywords": ["Hack", "Bomb"]},
        memory_preferences=dict(base.memory_preferences),
        created_at=base.created_at,
    )
    await store.save_policy(policy)

    current = await store.get_current_policy()
    assert current.safety_boundaries["harmful_keywords"] == ["hack", "bomb"]
