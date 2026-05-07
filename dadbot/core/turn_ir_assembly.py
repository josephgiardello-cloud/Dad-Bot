"""TurnIR — Integrated IR assembly for turn orchestration.

Provides the complete IR routing pipeline:
User Input → TurnIntent IR → PlannerDecision IR → PolicyInput IR → Canonical Events

This module consolidates the IR assembly surface and demonstrates how to route
the full turn execution through typed IR instead of raw dicts/tuples.

Key role:
- Assemble TurnIntent from user input and turn context
- Route planner output to PlannerDecision IR
- Route policy evaluation to PolicyInput IR
- Coordinate IR flow through orchestration pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dadbot.core.planner_ir import PlannerDecision, build_planner_ir
from dadbot.core.policy_compiler import PolicyCompiler
from dadbot.core.turn_ir import ExecutionContext, PolicyInput, TurnIntent, build_policy_input


@dataclass(frozen=True)
class TurnIRAssembly:
    """Complete IR representation of a single turn orchestration."""

    # Turn intent from user input and execution context
    turn_intent: TurnIntent

    # Planner output (candidates, metadata)
    planner_decision: PlannerDecision | None = None

    # Selected candidate from planner
    selected_candidate_ir: Any = None  # CandidateIR

    # Policy evaluation IR (built from policy_compiler)
    policy_input: PolicyInput | None = None

    # Policy decision outcome
    policy_decision: Any = None  # PolicyDecision

    # Opaque runtime context for backward compatibility
    _runtime_context: Any = field(default=None, repr=False)

    def has_planner_output(self) -> bool:
        """Check if planner has produced decision."""
        return self.planner_decision is not None

    def has_policy_output(self) -> bool:
        """Check if policy has evaluated."""
        return self.policy_decision is not None


def assemble_turn_ir(
    user_input: str,
    turn_context: Any,
    plan_result: Any = None,
    candidate: Any = None,
    policy_plan: Any = None,
) -> TurnIRAssembly:
    """Assemble complete turn IR from execution phases.

    Coordinates IR assembly across the full turn pipeline:
    1. Extract TurnIntent from turn_context
    2. Convert PlanResult to PlannerDecision
    3. Build PolicyInput from candidate + turn_context
    4. Optionally evaluate policy (if policy_plan provided)

    Args:
        user_input: Raw user input text
        turn_context: Raw turn context dict/object
        plan_result: Optional PlanResult from planner.build()
        candidate: Optional candidate from inference
        policy_plan: Optional policy plan from PolicyCompiler.compile_safety()

    Returns:
        TurnIRAssembly: Complete IR representation of turn state
    """
    # Phase 1: Extract TurnIntent from turn_context
    turn_intent = _extract_turn_intent(user_input, turn_context)

    # Phase 2: Convert PlanResult to PlannerDecision if available
    planner_decision = None
    selected_candidate_ir = None
    if plan_result is not None:
        planner_decision = build_planner_ir(plan_result, turn_context)
        selected_candidate_ir = planner_decision.get_primary_candidate()

    # Phase 3: Build PolicyInput from candidate
    policy_input = None
    if candidate is not None:
        policy_input = build_policy_input("default_policy", turn_context, candidate)

    # Phase 4: Optional policy evaluation (if plan provided)
    policy_decision = None
    if policy_plan is not None and policy_input is not None:
        policy_decision = PolicyCompiler.evaluate_safety_input(policy_plan, policy_input)

    return TurnIRAssembly(
        turn_intent=turn_intent,
        planner_decision=planner_decision,
        selected_candidate_ir=selected_candidate_ir,
        policy_input=policy_input,
        policy_decision=policy_decision,
        _runtime_context=turn_context,
    )


def _extract_turn_intent(user_input: str, turn_context: Any) -> TurnIntent:
    """Extract TurnIntent from user input and turn context.

    Reconstructs intent type and strategy from execution result or turn state.
    """
    # Try to extract from execution result
    execution_result = getattr(turn_context, "execution_result", None) or {}
    initial_result = execution_result.get("initial_result", {})

    intent_type = str(initial_result.get("intent_type", "unspecified") or "unspecified")
    strategy = str(initial_result.get("strategy", "default") or "default")

    # Fallback: inspect turn_context dict directly
    if isinstance(turn_context, dict):
        exec_result = turn_context.get("execution_result") or {}
        initial = exec_result.get("initial_result") or {}
        intent_type = str(initial.get("intent_type", intent_type))
        strategy = str(initial.get("strategy", strategy))

    return TurnIntent(
        intent_type=intent_type,
        strategy=strategy,
        tool_request_count=_extract_tool_request_count(turn_context),
    )


def _extract_tool_request_count(turn_context: Any) -> int:
    """Extract tool_request_count from turn_context."""
    if isinstance(turn_context, dict):
        count = turn_context.get("tool_request_count") or turn_context.get("tool_calls") or 0
        return int(count or 0)
    count = getattr(turn_context, "tool_request_count", None) or getattr(turn_context, "tool_calls", None)
    return int(count or 0)


def _extract_execution_context(turn_context: Any) -> ExecutionContext:
    """Build ExecutionContext from turn_context."""
    return ExecutionContext(
        session_id=_extract_session_id(turn_context),
        tenant_id=_extract_tenant_id(turn_context),
        trace_id=_extract_trace_id(turn_context),
        mode="live",
    )


def _extract_session_id(turn_context: Any) -> str:
    """Extract session_id from turn_context."""
    if isinstance(turn_context, dict):
        return str(turn_context.get("session_id") or turn_context.get("thread_id") or "default")
    session_id = getattr(turn_context, "session_id", None) or getattr(turn_context, "thread_id", None)
    return str(session_id or "default")


def _extract_tenant_id(turn_context: Any) -> str:
    """Extract tenant_id from turn_context."""
    if isinstance(turn_context, dict):
        return str(turn_context.get("tenant_id") or "default")
    tenant_id = getattr(turn_context, "tenant_id", None)
    return str(tenant_id or "default")


def _extract_trace_id(turn_context: Any) -> str:
    """Extract trace_id from turn_context."""
    if isinstance(turn_context, dict):
        return str(turn_context.get("trace_id") or turn_context.get("turn_id") or "")
    trace_id = getattr(turn_context, "trace_id", None) or getattr(turn_context, "turn_id", None)
    return str(trace_id or "")


# ============================================================================
# Convenience routing functions
# ============================================================================


def route_planner_through_ir(planner_class: Any, turn_state: dict[str, Any], execution_result: dict[str, Any]) -> PlannerDecision:
    """Convenience: Route planner output through IR.

    Args:
        planner_class: Planner class instance (e.g., from Planner())
        turn_state: Turn state dict from orchestrator
        execution_result: Execution result dict

    Returns:
        PlannerDecision: Immutable planner IR
    """
    return planner_class.build_ir(turn_state, execution_result)


def route_policy_through_ir(
    policy_plan: Any,
    turn_context: Any,
    candidate: Any,
) -> Any:  # PolicyDecision
    """Convenience: Route policy evaluation through IR.

    Args:
        policy_plan: PolicyPlan from PolicyCompiler.compile_safety()
        turn_context: Raw turn context
        candidate: Candidate from inference

    Returns:
        PolicyDecision: Immutable policy evaluation result
    """
    policy_input = build_policy_input("default_policy", turn_context, candidate)
    return PolicyCompiler.evaluate_safety_input(policy_plan, policy_input)
