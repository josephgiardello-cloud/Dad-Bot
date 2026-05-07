"""Phase E: Agent Reasoning Layer.

Maps Phase D recovery decisions to explicit executable actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dadbot.core.recovery_ir import RecoveryDecision, RecoveryStrategy


class ReasoningActionType(str, Enum):
    """Concrete action types that the runtime can execute."""

    RETRY_TOOL = "retry_tool"
    RETURN_DEGRADED = "return_degraded"
    REQUEST_APPROVAL = "request_approval"
    HALT_EXECUTION = "halt_execution"
    REPLAY_FROM_CHECKPOINT = "replay_from_checkpoint"


@dataclass(frozen=True)
class ReasoningContext:
    """Execution context for selecting an action from a recovery decision."""

    recovery_decision: RecoveryDecision
    attempt_index: int = 0
    turn_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningAction:
    """Selected action with enough metadata for direct execution."""

    action_type: ReasoningActionType
    reason: str
    payload: Any = None
    terminal: bool = False
    requires_user_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "reason": self.reason,
            "payload": self.payload,
            "terminal": bool(self.terminal),
            "requires_user_input": bool(self.requires_user_input),
        }


class ReasoningEngine:
    """Phase E dispatcher: RecoveryDecision -> ReasoningAction."""

    def select_action(self, context: ReasoningContext) -> ReasoningAction:
        decision = context.recovery_decision

        if decision.strategy == RecoveryStrategy.RETRY_SAME:
            return ReasoningAction(
                action_type=ReasoningActionType.RETRY_TOOL,
                reason=decision.reason or "Retry requested",
                payload={"remaining_attempts": int(decision.bounded_attempts)},
            )

        if decision.strategy == RecoveryStrategy.REPLAY_CHECKPOINT:
            return ReasoningAction(
                action_type=ReasoningActionType.REPLAY_FROM_CHECKPOINT,
                reason=decision.reason or "Replay from checkpoint requested",
                payload={"checkpoint_id": str(decision.checkpoint_id or "")},
            )

        if decision.strategy == RecoveryStrategy.DEGRADE_GRACEFULLY:
            return ReasoningAction(
                action_type=ReasoningActionType.RETURN_DEGRADED,
                reason=decision.reason or "Returning degraded output",
                payload=decision.degraded_output,
            )

        if decision.strategy == RecoveryStrategy.REQUIRE_APPROVAL:
            return ReasoningAction(
                action_type=ReasoningActionType.REQUEST_APPROVAL,
                reason=decision.reason or "User approval required",
                payload={"matched_rules": list(decision.matched_rules)},
                requires_user_input=True,
            )

        return ReasoningAction(
            action_type=ReasoningActionType.HALT_EXECUTION,
            reason=decision.reason or "Safe halt requested",
            payload={"status": "halted"},
            terminal=True,
        )


__all__ = [
    "ReasoningActionType",
    "ReasoningContext",
    "ReasoningAction",
    "ReasoningEngine",
]
