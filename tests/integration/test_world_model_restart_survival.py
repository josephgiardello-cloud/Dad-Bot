from __future__ import annotations

import pytest

from dadbot.core.turn_handler import TurnContext, TurnHandler
from dadbot.core.world_model import SQLiteAsyncWorldModelPersistence, WorldModelStore


pytestmark = [pytest.mark.integration, pytest.mark.durability]


@pytest.mark.asyncio
async def test_world_model_survives_store_restart(tmp_path) -> None:
    db_path = tmp_path / "world_model_restart.sqlite3"

    first_store = WorldModelStore(SQLiteAsyncWorldModelPersistence(str(db_path)))
    second_store = WorldModelStore(SQLiteAsyncWorldModelPersistence(str(db_path)))

    metadata_refs: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            metadata_refs.append(metadata)
        return ("ok", False)

    handler_one = TurnHandler(
        submit_turn=_submit_turn,
        world_model_store=first_store,
        relationship_snapshotter=lambda: {
            "trust_level": 52,
            "openness_level": 47,
            "emotional_momentum": "guarded",
        },
    )
    await handler_one.process_turn(
        TurnContext(
            user_input="I want to rebuild trust with my daughter. My mom is very supportive.",
            session_id="restart-session",
        ),
    )

    handler_two = TurnHandler(
        submit_turn=_submit_turn,
        world_model_store=second_store,
        relationship_snapshotter=lambda: {
            "trust_level": 60,
            "openness_level": 55,
            "emotional_momentum": "warming",
        },
    )
    await handler_two.process_turn(
        TurnContext(
            user_input="Actually I changed my mind, I need to sleep earlier.",
            session_id="restart-session",
        ),
    )

    # We keep references to outbound metadata to ensure post-submit world-model evolution
    # is reflected back into the same object after persistence merge.
    latest_metadata = metadata_refs[-1]
    evolved = dict(latest_metadata.get("user_world_model") or {})

    goals = [str(goal) for goal in list(evolved.get("active_goals") or [])]
    contradictions = [str(item) for item in list(evolved.get("contradictions") or [])]
    family_map = dict(evolved.get("family_map") or {})

    assert any("rebuild trust with my daughter" in goal for goal in goals)
    assert any("sleep earlier" in goal for goal in goals)
    assert any("changed my mind" in entry.lower() for entry in contradictions)
    assert family_map.get("mom") == "very supportive"
    assert int(evolved.get("trust_level", 0) or 0) == 60
    assert int(evolved.get("openness_level", 0) or 0) == 55

    persisted = await second_store.get_model("restart-session")
    assert persisted is not None
    assert len(persisted.emotional_timeline) == 2
