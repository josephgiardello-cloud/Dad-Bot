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


def test_thin_turn_handler_enabled_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_USE_THIN_TURN_HANDLER", "1")
    assert thin_turn_handler_enabled() is True

    monkeypatch.setenv("DADBOT_USE_THIN_TURN_HANDLER", "off")
    assert thin_turn_handler_enabled() is False


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
