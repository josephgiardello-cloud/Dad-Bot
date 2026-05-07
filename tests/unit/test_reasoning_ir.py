"""Tests for Phase E reasoning dispatch."""

from __future__ import annotations

import pytest

from dadbot.core.reasoning_ir import (
    ReasoningActionType,
    ReasoningContext,
    ReasoningEngine,
)
from dadbot.core.recovery_ir import RecoveryDecision, RecoveryStrategy

pytestmark = pytest.mark.unit


class TestReasoningEngine:
    def test_retry_strategy_maps_to_retry_action(self) -> None:
        engine = ReasoningEngine()
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY_SAME,
            reason="transient",
            bounded_attempts=2,
        )

        action = engine.select_action(ReasoningContext(recovery_decision=decision, attempt_index=1))

        assert action.action_type == ReasoningActionType.RETRY_TOOL
        assert action.payload == {"remaining_attempts": 2}
        assert action.terminal is False

    def test_degrade_strategy_maps_to_return_degraded(self) -> None:
        engine = ReasoningEngine()
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.DEGRADE_GRACEFULLY,
            reason="fallback",
            degraded_output={"degraded": True},
        )

        action = engine.select_action(ReasoningContext(recovery_decision=decision))

        assert action.action_type == ReasoningActionType.RETURN_DEGRADED
        assert action.payload == {"degraded": True}

    def test_require_approval_maps_to_request_approval(self) -> None:
        engine = ReasoningEngine()
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.REQUIRE_APPROVAL,
            reason="approval required",
            matched_rules=("rule_1",),
        )

        action = engine.select_action(ReasoningContext(recovery_decision=decision))

        assert action.action_type == ReasoningActionType.REQUEST_APPROVAL
        assert action.requires_user_input is True
        assert action.payload == {"matched_rules": ["rule_1"]}

    def test_replay_strategy_maps_to_replay_action(self) -> None:
        engine = ReasoningEngine()
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.REPLAY_CHECKPOINT,
            reason="replay",
            checkpoint_id="cp-1",
        )

        action = engine.select_action(ReasoningContext(recovery_decision=decision))

        assert action.action_type == ReasoningActionType.REPLAY_FROM_CHECKPOINT
        assert action.payload == {"checkpoint_id": "cp-1"}

    def test_halt_strategy_maps_to_terminal_halt(self) -> None:
        engine = ReasoningEngine()
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.HALT_SAFE,
            reason="fatal",
        )

        action = engine.select_action(ReasoningContext(recovery_decision=decision))

        assert action.action_type == ReasoningActionType.HALT_EXECUTION
        assert action.terminal is True
