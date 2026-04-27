"""Adversarial — phase regression injection tests.

Confirms that directly attempting backward phase transitions raises,
while valid forward-only sequences succeed.
"""
from __future__ import annotations

import pytest

from dadbot.core.graph import TurnContext, TurnPhase


def _ctx() -> TurnContext:
    return TurnContext(user_input="phase test")


class TestPhaseRegressionRejection:
    """Valid forward transitions succeed; backward transitions raise."""

    def test_plan_to_act_allowed(self):
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.PLAN, reason="test")
        ctx.transition_phase(TurnPhase.ACT, reason="test")
        assert ctx.phase == TurnPhase.ACT

    def test_act_to_observe_allowed(self):
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.PLAN, reason="test")
        ctx.transition_phase(TurnPhase.ACT, reason="test")
        ctx.transition_phase(TurnPhase.OBSERVE, reason="test")
        assert ctx.phase == TurnPhase.OBSERVE

    def test_observe_to_respond_allowed(self):
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.PLAN, reason="test")
        ctx.transition_phase(TurnPhase.ACT, reason="test")
        ctx.transition_phase(TurnPhase.OBSERVE, reason="test")
        ctx.transition_phase(TurnPhase.RESPOND, reason="test")
        assert ctx.phase == TurnPhase.RESPOND

    def test_full_forward_sequence_succeeds(self):
        ctx = _ctx()
        for phase in (TurnPhase.PLAN, TurnPhase.ACT, TurnPhase.OBSERVE, TurnPhase.RESPOND):
            ctx.transition_phase(phase, reason="test")
        assert ctx.phase == TurnPhase.RESPOND

    @pytest.mark.parametrize("from_phase,to_phase", [
        (TurnPhase.RESPOND, TurnPhase.PLAN),
        (TurnPhase.RESPOND, TurnPhase.ACT),
        (TurnPhase.RESPOND, TurnPhase.OBSERVE),
        (TurnPhase.ACT, TurnPhase.PLAN),
        (TurnPhase.OBSERVE, TurnPhase.PLAN),
        (TurnPhase.OBSERVE, TurnPhase.ACT),
    ])
    def test_backward_transition_raises(self, from_phase: TurnPhase, to_phase: TurnPhase):
        ctx = _ctx()
        # Advance to the starting phase
        _ADVANCE = {
            TurnPhase.PLAN: [TurnPhase.PLAN],
            TurnPhase.ACT: [TurnPhase.PLAN, TurnPhase.ACT],
            TurnPhase.OBSERVE: [TurnPhase.PLAN, TurnPhase.ACT, TurnPhase.OBSERVE],
            TurnPhase.RESPOND: [TurnPhase.PLAN, TurnPhase.ACT, TurnPhase.OBSERVE, TurnPhase.RESPOND],
        }
        for p in _ADVANCE[from_phase]:
            ctx.transition_phase(p, reason="test")
        assert ctx.phase == from_phase

        with pytest.raises(RuntimeError):
            ctx.transition_phase(to_phase, reason="test")


class TestPhaseHistoryIntegrity:
    def test_phase_history_grows_per_transition(self):
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.PLAN, reason="test")
        ctx.transition_phase(TurnPhase.ACT, reason="test")
        # PLAN->PLAN is no-op; only PLAN->ACT emits one transition
        assert len(ctx.phase_history) == 1

    def test_phase_history_has_from_and_to_fields(self):
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.ACT, reason="test")
        entry = ctx.phase_history[-1]
        assert "from" in entry
        assert "to" in entry

    def test_phase_history_to_matches_current_phase(self):
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.ACT, reason="test")
        assert ctx.phase_history[-1]["to"] == TurnPhase.ACT.value

    def test_failed_transition_does_not_corrupt_history(self):
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.PLAN, reason="test")
        ctx.transition_phase(TurnPhase.ACT, reason="test")
        original_len = len(ctx.phase_history)
        try:
            ctx.transition_phase(TurnPhase.PLAN, reason="test")  # regression
        except RuntimeError:
            pass
        assert len(ctx.phase_history) == original_len
        assert ctx.phase == TurnPhase.ACT

    def test_same_phase_repeated_transition_is_idempotent_or_raises(self):
        """Transitioning to same phase twice must either be idempotent or raise — not corrupt."""
        ctx = _ctx()
        ctx.transition_phase(TurnPhase.PLAN, reason="test")
        try:
            ctx.transition_phase(TurnPhase.PLAN, reason="test")
        except RuntimeError:
            pass  # also valid
        # Context phase must still be PLAN
        assert ctx.phase == TurnPhase.PLAN
