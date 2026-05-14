from __future__ import annotations

import pytest

from dadbot.core.policy_store import DadPolicy
from dadbot.core.turn_handler import TurnContext, TurnHandler, thin_turn_handler_enabled


pytestmark = pytest.mark.dev


@pytest.mark.asyncio
async def test_turn_handler_process_turn_delegates_with_enforce_confluence() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("ok", False)

    handler = TurnHandler(submit_turn=_submit_turn)
    result = await handler.process_turn(
        TurnContext(
            user_input="hello",
            session_id="s1",
            attachments=[{"kind": "text", "value": "a"}],
            confluence_key="turn:key",
            metadata={"x": 1},
            timeout_seconds=3.0,
        ),
    )

    assert result == ("ok", False)
    assert len(calls) == 1
    call = calls[0]
    assert call["user_input"] == "hello"
    assert call["session_id"] == "s1"
    assert call["confluence_key"] == "turn:key"
    metadata = dict(call["metadata"] or {})
    assert metadata["confluence_mode"] == "enforce"
    assert metadata["confluence_key"] == "turn:key"
    assert metadata["x"] == 1


def test_thin_turn_handler_enabled_is_always_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_USE_THIN_TURN_HANDLER", "1")
    assert thin_turn_handler_enabled() is True

    monkeypatch.setenv("DADBOT_USE_THIN_TURN_HANDLER", "off")
    assert thin_turn_handler_enabled() is True


@pytest.mark.asyncio
async def test_turn_handler_injects_policy_template_when_store_is_provided() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("ok", False)

    class _PolicyStore:
        async def get_current_policy(self) -> DadPolicy:
            return DadPolicy(
                version="dad_v2.0",
                persona_style={"warmth": 1.0},
                relationship_rules={"build_trust": True},
                safety_boundaries={"block_harmful_advice": True},
                memory_preferences={"keep_family_memories": True},
                created_at="2026-01-01T00:00:00Z",
            )

    handler = TurnHandler(submit_turn=_submit_turn, policy_store=_PolicyStore())
    await handler.process_turn(TurnContext(user_input="hello", session_id="s1"))

    metadata = dict(calls[0]["metadata"] or {})
    assert metadata["dad_policy_version"] == "dad_v2.0"
    assert metadata["policy_version"] == "dad_v2.0"
    template = dict(metadata["dad_policy_template"] or {})
    assert template["persona_style"]["warmth"] == 1.0


@pytest.mark.asyncio
async def test_turn_handler_injects_prompt_context_into_metadata() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("ok", False)

    handler = TurnHandler(
        submit_turn=_submit_turn,
        prompt_builder=lambda: "Dad prompt context",
    )
    await handler.process_turn(TurnContext(user_input="hello", session_id="s1"))

    metadata = dict(calls[0]["metadata"] or {})
    assert metadata["prompt_context"] == "Dad prompt context"
    rich_context = dict(metadata["rich_context"] or {})
    assert rich_context["prompt_context"] == "Dad prompt context"


@pytest.mark.asyncio
async def test_turn_handler_appends_memory_ledger_event_on_success() -> None:
    class _MemoryLedger:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def append_memory_event(self, event: dict[str, object]) -> None:
            self.events.append(dict(event))

    ledger = _MemoryLedger()

    async def _submit_turn(user_input: str, **kwargs: object):
        return ("reply text", False)

    handler = TurnHandler(submit_turn=_submit_turn, memory_ledger=ledger)
    await handler.process_turn(
        TurnContext(
            user_input="hello",
            session_id="s1",
            confluence_key="turn:key",
            metadata={"policy_version": "dad_v2.0"},
        ),
    )

    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event["event_type"] == "turn.finalized"
    assert event["session_id"] == "s1"
    assert event["confluence_key"] == "turn:key"
    assert event["policy_version"] == "dad_v2.0"
    assert event["reply_preview"] == "reply text"


@pytest.mark.asyncio
async def test_turn_handler_injects_relationship_projection_after_submit() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("reply", False)

    handler = TurnHandler(
        submit_turn=_submit_turn,
        relationship_snapshotter=lambda: {
            "trust_level": 70,
            "openness_level": 64,
            "emotional_momentum": "steady",
        },
    )
    await handler.process_turn(TurnContext(user_input="hello", session_id="s1"))

    metadata = dict(calls[0]["metadata"] or {})
    projection = dict(metadata["relationship_projection"] or {})
    assert projection["trust_level"] == 70
    assert projection["openness_level"] == 64
    assert projection["emotional_momentum"] == "steady"


@pytest.mark.asyncio
async def test_turn_handler_writes_relationship_projection_to_ledger_event() -> None:
    class _MemoryLedger:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def append_memory_event(self, event: dict[str, object]) -> None:
            self.events.append(dict(event))

    ledger = _MemoryLedger()

    async def _submit_turn(user_input: str, **kwargs: object):
        return ("reply text", False)

    handler = TurnHandler(
        submit_turn=_submit_turn,
        memory_ledger=ledger,
        relationship_snapshotter=lambda: {
            "trust_level": 71,
            "openness_level": 52,
            "emotional_momentum": "warming",
        },
    )
    await handler.process_turn(TurnContext(user_input="hello", session_id="s1"))

    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event["relationship_trust_level"] == 71
    assert event["relationship_openness_level"] == 52
    assert event["relationship_emotional_momentum"] == "warming"


@pytest.mark.asyncio
async def test_turn_handler_injects_user_world_model_snapshot() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("reply", False)

    class _PolicyStore:
        async def get_current_policy(self) -> DadPolicy:
            return DadPolicy(
                version="dad_v3.0",
                persona_style={"warmth": 1.0},
                relationship_rules={"build_trust": True},
                safety_boundaries={"block_harmful_advice": True},
                memory_preferences={"keep_family_memories": True},
                created_at="2026-01-01T00:00:00Z",
            )

    handler = TurnHandler(
        submit_turn=_submit_turn,
        policy_store=_PolicyStore(),
        prompt_builder=lambda: "dad world prompt",
        relationship_snapshotter=lambda: {
            "trust_level": 88,
            "openness_level": 73,
            "emotional_momentum": "steady",
        },
    )
    await handler.process_turn(TurnContext(user_input="hello", session_id="world-session"))

    metadata = dict(calls[0]["metadata"] or {})
    world_model = dict(metadata["user_world_model"] or {})
    assert world_model["session_id"] == "world-session"
    assert world_model["policy_version"] == "dad_v3.0"
    assert world_model["prompt_context"] == "dad world prompt"
    assert world_model["trust_level"] == 88
    assert world_model["openness_level"] == 73
    assert world_model["emotional_momentum"] == "steady"


@pytest.mark.asyncio
async def test_turn_handler_writes_world_model_projection_to_ledger_event() -> None:
    class _MemoryLedger:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        async def append_memory_event(self, event: dict[str, object]) -> None:
            self.events.append(dict(event))

    ledger = _MemoryLedger()

    async def _submit_turn(user_input: str, **kwargs: object):
        return ("reply text", False)

    handler = TurnHandler(
        submit_turn=_submit_turn,
        memory_ledger=ledger,
        relationship_snapshotter=lambda: {
            "trust_level": 64,
            "openness_level": 59,
            "emotional_momentum": "warming",
        },
    )
    await handler.process_turn(
        TurnContext(
            user_input="hello",
            session_id="s1",
            metadata={"policy_version": "dad_v3.1", "prompt_context": "cached prompt"},
        ),
    )

    event = ledger.events[0]
    assert event["world_model_trust_level"] == 64
    assert event["world_model_openness_level"] == 59
    assert event["world_model_policy_version"] == "dad_v3.1"


@pytest.mark.asyncio
async def test_turn_handler_extracts_active_goals_from_user_input() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("reply", False)

    handler = TurnHandler(submit_turn=_submit_turn)
    await handler.process_turn(
        TurnContext(
            user_input="I want to rebuild trust with my daughter and I need to sleep earlier.",
            session_id="s-goals",
        ),
    )

    metadata = calls[0]["metadata"]
    world_model = dict(getattr(metadata, "get", lambda *_: {})("user_world_model") or {})
    goals = [str(goal) for goal in list(world_model.get("active_goals") or [])]
    assert any("rebuild trust with my daughter" in goal for goal in goals)
    assert any("sleep earlier" in goal for goal in goals)


@pytest.mark.asyncio
async def test_turn_handler_extracts_contradictions_and_family_map() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("reply", False)

    handler = TurnHandler(submit_turn=_submit_turn)
    await handler.process_turn(
        TurnContext(
            user_input="Actually I changed my mind. My dad is skeptical about therapy.",
            session_id="s-family",
        ),
    )

    metadata = calls[0]["metadata"]
    world_model = dict(getattr(metadata, "get", lambda *_: {})("user_world_model") or {})
    contradictions = [str(item) for item in list(world_model.get("contradictions") or [])]
    family_map = dict(world_model.get("family_map") or {})
    assert contradictions
    assert any("changed my mind" in entry.lower() for entry in contradictions)
    assert family_map.get("dad") == "skeptical about therapy"
