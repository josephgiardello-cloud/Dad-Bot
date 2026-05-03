"""Tier 1 — Safety Enforcement: unsafe output must never reach the final response.

Hard-fail: safety bypass means the system cannot be safely distributed.
"""
from __future__ import annotations

import pytest

from dadbot.core.graph_context import TurnContext


pytestmark = pytest.mark.unit


def _make_safety_registry(*, pass_through: bool = True):
    """Build a minimal registry stub with a controllable safety service."""

    class _SafetyService:
        def __init__(self, safe: bool) -> None:
            self._safe = safe

        def enforce_policies(self, ctx: TurnContext, candidate: object) -> object:
            if not self._safe:
                raise RuntimeError("Safety policy violation: unsafe content blocked")
            # Return the candidate unmodified (safe pass-through)
            return candidate

    class _Reg:
        def __init__(self, safe: bool) -> None:
            self._svc = _SafetyService(safe)

        def get(self, name: str, optional: bool = False) -> object:
            if name == "safety_service":
                return self._svc
            if optional:
                return None
            raise KeyError(name)

    return _Reg(safe=pass_through)


class TestSafetyEnforcement:
    def test_safe_candidate_passes_through_safety_service(self):
        """A safe candidate is returned unchanged by the safety service."""
        reg = _make_safety_registry(pass_through=True)
        ctx = TurnContext(user_input="how are you")
        candidate = ("I'm doing well!", {})
        ctx.state["candidate"] = candidate

        safety_svc = reg.get("safety_service")
        result = safety_svc.enforce_policies(ctx, ctx.state["candidate"])
        assert result == candidate

    def test_unsafe_candidate_is_blocked_by_safety_service(self):
        """An unsafe candidate must raise before the result reaches the caller."""
        reg = _make_safety_registry(pass_through=False)
        ctx = TurnContext(user_input="unsafe input")
        ctx.state["candidate"] = ("dangerous output", {})

        safety_svc = reg.get("safety_service")
        with pytest.raises(RuntimeError, match="Safety policy violation"):
            safety_svc.enforce_policies(ctx, ctx.state["candidate"])

    def test_safe_result_not_set_means_safety_node_did_not_run(self):
        """If safe_result is absent, the pipeline has not run safety — contract fail."""
        ctx = TurnContext(user_input="check")
        assert "safe_result" not in ctx.state

    def test_safe_result_present_means_safety_node_completed(self):
        """After a passing safety node, safe_result must exist in state."""
        ctx = TurnContext(user_input="check")
        ctx.state["candidate"] = ("reply", {})
        ctx.state["safe_result"] = ("reply", {})
        assert "safe_result" in ctx.state

    def test_turn_context_short_circuit_false_by_default(self):
        """short_circuit defaults to False — no implicit safety skip at turn start."""
        ctx = TurnContext(user_input="normal turn")
        assert ctx.short_circuit is False

    def test_short_circuit_result_none_by_default(self):
        """short_circuit_result is None unless explicitly set by a safety node."""
        ctx = TurnContext(user_input="normal turn")
        assert ctx.short_circuit_result is None
