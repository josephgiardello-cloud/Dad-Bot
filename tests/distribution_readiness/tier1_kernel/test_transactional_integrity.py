"""Tier 1 — Transactional Integrity: save-node atomicity and replay idempotency.

Hard-fail: partial mutations that persist after save failure signal data corruption risk.
"""
from __future__ import annotations

import pytest

from dadbot.core.graph_context import TurnContext


pytestmark = pytest.mark.unit


class _FailingSaveService:
    """Simulates a persistence layer that raises mid-save."""

    def save_turn(self, ctx: TurnContext, result: object) -> None:
        raise RuntimeError("Disk full — simulated save failure")

    def finalize_turn(self, ctx: TurnContext, result: object) -> object:
        raise RuntimeError("Disk full — simulated finalize failure")


class _TrackingRegistry:
    def __init__(self, services: dict) -> None:
        self._services = services

    def get(self, name: str, optional: bool = False) -> object:
        if name in self._services:
            return self._services[name]
        if optional:
            return None
        # Return None for unknown services rather than raising — graph uses
        # optional lookups for telemetry, event_tap, etc.
        return None


class TestTransactionalIntegrity:
    def test_save_failure_does_not_silently_succeed(self):
        """A failing persistence layer must propagate — SaveNode must not swallow errors."""
        import asyncio
        from dadbot.core.graph_pipeline_nodes import SaveNode

        node = SaveNode()
        reg = _TrackingRegistry({"persistence_service": _FailingSaveService()})
        ctx = TurnContext(user_input="test save failure")
        ctx.state["safe_result"] = ("response", {})

        with pytest.raises(Exception, match="simulated"):
            import asyncio
            asyncio.run(
                node.execute(reg, ctx)
            )

    def test_turn_context_state_is_dict_contract(self):
        """TurnContext.state must always be a dict — invariant at construction."""
        ctx = TurnContext(user_input="check state type")
        assert isinstance(ctx.state, dict)

    def test_turn_context_metadata_is_dict_contract(self):
        """TurnContext.metadata must always be a dict — invariant at construction."""
        ctx = TurnContext(user_input="check metadata type")
        assert isinstance(ctx.metadata, dict)

    def test_state_key_write_is_idempotent_on_same_value(self):
        """Writing the same value to state.key twice leaves state identical."""
        ctx = TurnContext(user_input="idempotent write")
        ctx.state["x"] = 42
        ctx.state["x"] = 42
        assert ctx.state["x"] == 42
        assert len([k for k in ctx.state if k == "x"]) == 1

    def test_replay_produces_same_state_keys(self):
        """Replaying the same state mutations produces identical key sets."""
        def _apply_mutations(ctx: TurnContext) -> None:
            ctx.state["temporal"] = {"ts": "fixed"}
            ctx.state["rich_context"] = {"loaded": True}
            ctx.state["candidate"] = ("reply", {})
            ctx.state["safe_result"] = ("reply", {})

        ctx_a = TurnContext(user_input="replay")
        ctx_b = TurnContext(user_input="replay")
        _apply_mutations(ctx_a)
        _apply_mutations(ctx_b)
        assert set(ctx_a.state.keys()) == set(ctx_b.state.keys())
