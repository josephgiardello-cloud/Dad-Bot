from __future__ import annotations

import pytest

from dadbot.core.turn_handler import TurnContext, TurnHandler


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_thin_spine_matches_direct_submit_contract() -> None:
    calls: list[dict[str, object]] = []

    async def _submit_turn(user_input: str, **kwargs: object):
        calls.append({"user_input": user_input, **kwargs})
        return ("ok", False)

    handler = TurnHandler(
        submit_turn=_submit_turn,
        prompt_builder=lambda: "dad baseline context",
    )

    turn = TurnContext(
        user_input="hello dad",
        session_id="session-a",
        confluence_key="turn:abc",
        metadata={"policy_version": "dad_v2.0"},
        timeout_seconds=5.0,
    )

    thin_result = await handler.process_turn(turn)
    thin_call = dict(calls[0])

    # Legacy baseline: direct orchestrator submit with the same normalized payload.
    direct_result = await _submit_turn(
        turn.user_input,
        attachments=turn.attachments,
        session_id=turn.session_id,
        confluence_key=turn.confluence_key,
        metadata=dict(thin_call["metadata"] or {}),
        timeout_seconds=turn.timeout_seconds,
    )
    direct_call = dict(calls[1])

    assert thin_result == direct_result
    assert thin_call["user_input"] == direct_call["user_input"]
    assert thin_call["session_id"] == direct_call["session_id"]
    assert thin_call["confluence_key"] == direct_call["confluence_key"]
    assert dict(thin_call["metadata"] or {}) == dict(direct_call["metadata"] or {})
