"""graph_pipeline_nodes — canonical pipeline node stubs for TurnGraph.

These are the minimal service-delegation node implementations used as defaults
when TurnGraph is constructed without custom nodes.  The full production
implementations live in dadbot.core.nodes.

Extracted from graph.py to reduce TurnGraph god-class surface area.
All names are re-exported from dadbot.core.graph for backward compatibility.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Protocol

from dadbot.core.critic import CritiqueEngine
from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_types import NodeType
from dadbot.core.invariant_gate import InvariantGate
from dadbot.core.policy_ir import PolicyDecisionIR
from dadbot.core.policy_compiler import PolicyCompiler
from dadbot.core.planner import PlannerNode
from dadbot.core.reasoning_ir import (
    ReasoningAction,
    ReasoningActionType,
    ReasoningContext,
    ReasoningEngine,
)
from dadbot.core.recovery_ir import (
    RecoveryContext,
    RecoveryDecision,
    RecoverySelector,
    RecoveryStrategy,
)
from dadbot.core.runtime_types import CanonicalPayload, ToolExecutionStatus, ToolResult
from dadbot.core.cognition_event import CognitionEnvelope, emit_cognition

logger = logging.getLogger(__name__)


class GraphNode(Protocol):
    @property
    def name(self) -> str: ...

    def dependencies(self) -> tuple[str, ...]: ...

    async def run(self, registry: Any, ctx: TurnContext) -> None: ...

    async def execute(self, registry: Any, turn_context: TurnContext) -> None: ...


class _NodeContractMixin:
    def dependencies(self) -> tuple[str, ...]:
        return ()

    async def run(self, registry: Any, ctx: TurnContext) -> None:
        execute_method = getattr(self, "execute")
        execute_params = inspect.signature(execute_method).parameters
        result = execute_method(registry, ctx) if len(execute_params) >= 2 else execute_method(ctx)
        if inspect.isawaitable(result):
            await result


async def _invoke_node_run_compat(run_method: Any, registry: Any, turn_context: TurnContext) -> Any:
    run_params = inspect.signature(run_method).parameters
    result = run_method(registry, turn_context) if len(run_params) >= 2 else run_method(turn_context)
    if inspect.isawaitable(result):
        result = await result
    return result


class HealthNode(_NodeContractMixin):
    name = "health"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("maintenance_service")
        turn_context.state["health"] = service.tick(turn_context)


class ContextBuilderNode(_NodeContractMixin):
    name = "context_builder"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("context_service")
        rich_context = dict(service.build_context(turn_context) or {})
        rich_context.setdefault("temporal", dict(turn_context.state.get("temporal") or {}))
        turn_context.state["rich_context"] = rich_context
        # Keep semantic cognition contract: planner always runs after context build.
        await PlannerNode().run(turn_context)


MemoryNode = ContextBuilderNode


class ValidationGateNode(_NodeContractMixin):
    """Middleware between ContextBuilderNode/Planner and InferenceNode.

    Validates every tool request in ``context.state["tool_ir"]["requests"]``
    against the declared required-arg schema before inference runs.

    Valid path   → pass-through; inference proceeds normally.
    Invalid path → emit a CONTRACT_VIOLATION CognitionEnvelope, write a
                   ``_validation_gate_repair`` context, strip violating
                   requests from tool_ir, and re-run PlannerNode once so the
                   planner can adjust its strategy with the error in context.
    """

    name = "validation_gate"
    _MAX_REPAIR_ITERATIONS: int = 1

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        for _attempt in range(self._MAX_REPAIR_ITERATIONS + 1):
            violations = self._collect_violations(turn_context)
            if not violations:
                return  # All requests valid — pass through
            decision = InvariantGate.decide_remediation(
                "validation_contract_violation",
                reason="ValidationGate detected missing required tool arguments.",
                attempt=_attempt,
                max_attempts=self._MAX_REPAIR_ITERATIONS,
                details={"violations": violations},
            )
            self._handle_violations(turn_context, violations, decision)
            # Re-run PlannerNode with the repair context in state so it can
            # adjust its tool selection strategy.
            if decision.action.value == "replan":
                await PlannerNode().run(turn_context)
        # After the repair attempt, remove any still-violating requests to
        # prevent them from reaching the execution layer.
        remaining = self._collect_violations(turn_context)
        if remaining:
            decision = InvariantGate.decide_remediation(
                "validation_contract_violation",
                reason="ValidationGate downgraded invalid tool requests after repair exhaustion.",
                attempt=self._MAX_REPAIR_ITERATIONS,
                max_attempts=self._MAX_REPAIR_ITERATIONS,
                details={"violations": remaining},
            )
            self._strip_violations(turn_context, remaining, decision)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_violations(self, turn_context: TurnContext) -> list[dict[str, Any]]:
        tool_ir = turn_context.state.get("tool_ir") or {}
        requests = list(tool_ir.get("requests") or [])
        required_args_by_tool: dict[str, frozenset[str]] = {}
        try:
            from dadbot.core.nodes import get_tool_required_args  # noqa: PLC0415

            required_args_by_tool = dict(get_tool_required_args())
        except Exception:
            # ValidationGate stays resilient if the tool registry is unavailable.
            pass
        violations: list[dict[str, Any]] = []
        for req in requests:
            tool_name = str(req.get("tool_name") or "").strip().lower()
            args = dict(req.get("args") or {})
            required = required_args_by_tool.get(tool_name)
            if required is None:
                # Unknown tool — not our schema, skip validation
                continue
            missing = sorted(required - set(args.keys()))
            if missing:
                violations.append({
                    "tool_name": tool_name,
                    "missing_args": missing,
                    "received_args": sorted(args.keys()),
                    "repair_hint": (
                        f"Tool '{tool_name}' requires args {sorted(required)}; "
                        f"received {sorted(args.keys())}. Missing: {missing}."
                    ),
                })
        return violations

    def _handle_violations(
        self,
        turn_context: TurnContext,
        violations: list[dict[str, Any]],
        decision: Any,
    ) -> None:
        repair_context = {
            "violations": violations,
            "repair_requested": True,
            "remediation": {
                "action": decision.action.value,
                "failure_class": decision.failure_class,
                "attempt": decision.attempt,
                "max_attempts": decision.max_attempts,
                "reason": decision.reason,
            },
        }
        turn_context.state["_validation_gate_repair"] = repair_context
        for v in violations:
            emit_cognition(
                turn_context,
                CognitionEnvelope(
                    step_id=f"{turn_context.trace_id}:validation_gate:violation:{v['tool_name']}",
                    thought_trace=(
                        f"ValidationGate: CONTRACT_VIOLATION for tool '{v['tool_name']}' — "
                        f"missing args {v['missing_args']}. action={decision.action.value}. {v['repair_hint']}"
                    ),
                    target_node="validation_gate",
                    confidence_score=0.0,
                ),
            )

    def _strip_violations(
        self,
        turn_context: TurnContext,
        violations: list[dict[str, Any]],
        decision: Any,
    ) -> None:
        violating_names = {v["tool_name"] for v in violations}
        tool_ir = dict(turn_context.state.get("tool_ir") or {})
        requests = [
            r for r in list(tool_ir.get("requests") or [])
            if str(r.get("tool_name") or "").strip().lower() not in violating_names
        ]
        tool_ir["requests"] = requests
        tool_ir["validation_gate"] = {
            "stripped_tools": sorted(violating_names),
            "violations": violations,
            "remediation": {
                "action": decision.action.value,
                "failure_class": decision.failure_class,
                "attempt": decision.attempt,
                "max_attempts": decision.max_attempts,
                "reason": decision.reason,
            },
        }
        turn_context.state["tool_ir"] = tool_ir


class InferenceNode(_NodeContractMixin):
    name = "inference"
    _GOAL_ALIGNMENT_OVERLAP_THRESHOLD = 0.12
    _MANDATORY_HALT_AFTER_DIVERSIONS = 3
    _REALIGN_CONFIRMATION_MARKERS = (
        "realign",
        "back on goal",
        "return to goal",
        "focus goal",
        "goal confirmed",
    )

    def __init__(self) -> None:
        self._critique_engine = CritiqueEngine()

    def _run_critique_check(self, turn_context: TurnContext, candidate: Any, iteration: int) -> bool:
        plan = dict(turn_context.state.get("turn_plan") or {})
        reply = candidate[0] if isinstance(candidate, tuple) else str(candidate or "")
        critique = self._critique_engine.critique(reply, turn_context.user_input, plan, iteration)
        passed = bool(getattr(critique, "passed", False))
        hint = str(getattr(critique, "revision_hint", "") or "")
        score = float(getattr(critique, "score", 0.0))
        turn_context.state["critique_record"] = {
            "iteration": iteration,
            "score": score,
            "passed": passed,
            "issues": list(getattr(critique, "issues", []) or []),
            "revision_hint": hint,
            "tool_necessity_score": getattr(critique, "tool_necessity_score", 0.0),
            "tool_correctness_score": getattr(critique, "tool_correctness_score", 0.0),
        }
        if not passed:
            turn_context.state["_critique_revision_context"] = hint
        emit_cognition(turn_context, CognitionEnvelope(
            step_id=f"{turn_context.trace_id}:critique:{iteration}",
            thought_trace=(
                f"Critique (iter={iteration}): passed={passed}, score={score:.2f}"
                + (f", hint={hint[:80]!r}" if hint else "")
            ),
            target_node="critique",
            confidence_score=score,
        ))
        return passed

    @staticmethod
    def _significant_tokens(text: str) -> set[str]:
        return {
            chunk.strip().lower()
            for chunk in str(text or "").replace("-", " ").split()
            if len(chunk.strip()) >= 4
        }

    @staticmethod
    def _is_recalibrate_request(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        markers = (
            "recalibrate",
            "recalibration",
            "resynthesize",
            "re-synthesize",
            "adjust goals",
            "adjust goal",
        )
        return any(marker in lowered for marker in markers)

    @classmethod
    def _goal_resynthesis_reply(cls, turn_context: TurnContext) -> str | None:
        state = turn_context.state if isinstance(turn_context.state, dict) else {}
        resynthesis = dict(state.get("goal_resynthesis") or {})
        if not bool(resynthesis.get("should_re_synthesize", False)):
            return None

        user_text = str(turn_context.user_input or "")
        mandatory_halt = bool(state.get("goal_alignment_mandatory_halt", False))
        if not mandatory_halt and not cls._is_recalibrate_request(user_text):
            return None

        proposal = dict(resynthesis.get("proposal") or {})
        message = str(resynthesis.get("message") or "Sustained friction detected.")
        revised_goal = str(proposal.get("revised_goal") or "")
        rationale = str(proposal.get("rationale") or "")
        constraints = [
            str(item) for item in list(proposal.get("suggested_constraints") or []) if str(item).strip()
        ]
        next_steps = [
            str(item) for item in list(proposal.get("next_steps") or []) if str(item).strip()
        ]

        lines = [message]
        if revised_goal:
            lines.append(f"Revised goal: {revised_goal}")
        if rationale:
            lines.append(f"Why now: {rationale}")
        if constraints:
            lines.append("Constraints:")
            lines.extend(f"- {item}" for item in constraints[:3])
        if next_steps:
            lines.append("Next steps:")
            lines.extend(f"- {item}" for item in next_steps[:3])
        lines.append("Reply 'realign' to proceed with this plan.")
        return "\n".join(lines)

    @classmethod
    def _goal_alignment_interrupt_reply(cls, turn_context: TurnContext) -> str | None:
        state = turn_context.state if isinstance(turn_context.state, dict) else {}
        if not bool(state.get("goal_alignment_guard_enabled", False)):
            return None

        goals_raw = state.get("session_goals")
        if not isinstance(goals_raw, list):
            goals_raw = state.get("goals")
        goals = [dict(item) for item in list(goals_raw or []) if isinstance(item, dict)]
        if not goals:
            return None

        user_tokens = cls._significant_tokens(turn_context.user_input)
        if len(user_tokens) < 4:
            return None

        user_text = str(turn_context.user_input or "").strip().lower()
        confirms_realign = any(marker in user_text for marker in cls._REALIGN_CONFIRMATION_MARKERS)

        best_overlap = 0.0
        best_goal = ""
        for goal in goals[:6]:
            description = str(goal.get("description") or goal.get("goal") or "").strip()
            if not description:
                continue
            goal_tokens = cls._significant_tokens(description)
            if not goal_tokens:
                continue
            overlap = len(user_tokens & goal_tokens) / float(len(user_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_goal = description

        mandatory_halt = bool(state.get("goal_alignment_mandatory_halt", False))
        if mandatory_halt and confirms_realign:
            state["goal_alignment_mandatory_halt"] = False
            state["goal_alignment_diversion_streak"] = 0
            mandatory_halt = False

        if best_overlap >= cls._GOAL_ALIGNMENT_OVERLAP_THRESHOLD:
            state["goal_alignment_diversion_streak"] = 0
            state["goal_alignment_mandatory_halt"] = False
            return None

        diversion_streak = int(state.get("goal_alignment_diversion_streak") or 0) + 1
        state["goal_alignment_diversion_streak"] = diversion_streak
        if diversion_streak >= cls._MANDATORY_HALT_AFTER_DIVERSIONS:
            state["goal_alignment_mandatory_halt"] = True
            mandatory_halt = True

        if mandatory_halt:
            return (
                "MANDATORY_HALT: We are drifting from your declared objective. "
                "Say 'realign' and restate your goal before we continue."
            )

        short_goal = best_goal[:96] if best_goal else "the goal we already set"
        return (
            "Quick check before we keep going: this sounds off our agreed direction. "
            f"Are we solving this now, or returning to {short_goal}?"
        )

    @staticmethod
    def _blend_daily_checkin_reply(service: Any, turn_context: TurnContext, candidate: Any) -> Any:
        bot = getattr(service, "bot", None)
        tone_context = getattr(bot, "tone_context", None)
        blend = getattr(tone_context, "blend_daily_checkin_reply", None)
        if not callable(blend):
            return candidate
        current_mood = str(turn_context.state.get("mood") or "neutral")
        if isinstance(candidate, tuple):
            reply = str(candidate[0] or "")
            return blend(reply, current_mood), bool(candidate[1])
        return blend(str(candidate or ""), current_mood)

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("agent_service")
        rich_context = turn_context.state.get("rich_context", {})
        plan = dict(turn_context.state.get("turn_plan") or {})
        resynthesis_reply = self._goal_resynthesis_reply(turn_context)
        if resynthesis_reply is not None:
            turn_context.state["goal_alignment_guard"] = {
                "triggered": True,
                "reason": "goal_resynthesis",
                "mandatory_halt": bool(turn_context.state.get("goal_alignment_mandatory_halt", False)),
            }
            candidate = (resynthesis_reply, False)
            self._run_critique_check(turn_context, candidate, 0)
            turn_context.state.pop("_critique_revision_context", None)
            turn_context.state["candidate"] = candidate
            emit_cognition(turn_context, CognitionEnvelope(
                step_id=f"{turn_context.trace_id}:goal_guard:resynthesis",
                thought_trace="Goal re-synthesis guidance issued due to sustained friction.",
                target_node="inference",
                confidence_score=0.82,
            ))
            return

        interrupt_reply = self._goal_alignment_interrupt_reply(turn_context)
        if interrupt_reply is not None:
            mandatory_halt = bool(turn_context.state.get("goal_alignment_mandatory_halt", False))
            diversion_streak = int(turn_context.state.get("goal_alignment_diversion_streak") or 0)
            turn_context.state["goal_alignment_guard"] = {
                "triggered": True,
                "overlap_threshold": self._GOAL_ALIGNMENT_OVERLAP_THRESHOLD,
                "reason": "mandatory_halt" if mandatory_halt else "goal_diversion_detected",
                "diversion_streak": diversion_streak,
                "mandatory_halt": mandatory_halt,
            }
            candidate = (interrupt_reply, False)
            self._run_critique_check(turn_context, candidate, 0)
            turn_context.state.pop("_critique_revision_context", None)
            turn_context.state["candidate"] = candidate
            emit_cognition(turn_context, CognitionEnvelope(
                step_id=f"{turn_context.trace_id}:goal_guard:interrupt",
                thought_trace="Goal guard interrupt issued due to low goal overlap.",
                target_node="inference",
                confidence_score=0.8,
            ))
            return

        emit_cognition(turn_context, CognitionEnvelope(
            step_id=f"{turn_context.trace_id}:inference:start",
            thought_trace=(
                f"Inference: running agent, strategy={plan.get('strategy', '?')!r}, "
                f"intent={plan.get('intent_type', '?')!r}"
            ),
            target_node="inference",
            confidence_score=0.5,
        ))
        candidate = await service.run_agent(turn_context, rich_context)
        candidate = self._blend_daily_checkin_reply(service, turn_context, candidate)
        self._run_critique_check(turn_context, candidate, 0)
        turn_context.state.pop("_critique_revision_context", None)
        turn_context.state["candidate"] = candidate


class SafetyNode(_NodeContractMixin):
    name = "safety"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("safety_service")
        candidate = turn_context.state.get("candidate")
        plan = PolicyCompiler.compile_safety(service)
        decision = PolicyCompiler.evaluate_safety(plan, turn_context, candidate)
        turn_context.state["safe_result"] = decision.output
        turn_context.state["safety_policy_decision"] = {
            "action": decision.action,
            "step_name": decision.step_name,
            "details": dict(decision.details or {}),
            "trace": dict(decision.trace or {}),
        }
        policy_events = list(turn_context.state.get("policy_trace_events") or [])
        policy_events.append(
            {
                "event_type": "policy_decision",
                "policy": "safety",
                "node": self.name,
                "sequence": len(policy_events) + 1,
                "trace": dict(decision.trace or {}),
            },
        )
        turn_context.state["policy_trace_events"] = policy_events
        if decision.action == "passthrough":
            turn_context.state["safety_passthrough"] = {
                "reason": "no_safety_manager",
                "failure_mode": "passthrough",
            }


class RecoveryNode(_NodeContractMixin):
    """Phase E: execute recovery strategy chosen from policy + runtime state."""

    name = "recovery"

    def __init__(
        self,
        selector: RecoverySelector | None = None,
        reasoner: ReasoningEngine | None = None,
    ) -> None:
        self._selector = selector or RecoverySelector()
        self._reasoner = reasoner or ReasoningEngine()

    @staticmethod
    def _coerce_policy_decision(turn_context: TurnContext) -> PolicyDecisionIR:
        policy_decision = turn_context.state.get("policy_decision_ir")
        if isinstance(policy_decision, PolicyDecisionIR):
            return policy_decision

        safe_result = turn_context.state.get("safe_result")
        safety_decision = dict(turn_context.state.get("safety_policy_decision") or {})
        action = str(safety_decision.get("action") or "").strip().lower()
        status = ToolExecutionStatus.DENIED if action in {"denied", "deny", "blocked", "rejected"} else ToolExecutionStatus.OK
        tool_result = ToolResult(
            tool_name="safety",
            invocation_id=str(turn_context.trace_id or "turn"),
            status=status,
            payload=CanonicalPayload(safe_result, payload_type="safe_result"),
            error="Policy denied output" if status == ToolExecutionStatus.DENIED else "",
            metadata={"source": "safety_policy_decision", "action": action},
        )
        return PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=(),
            emitted_effects=(),
            final_output=safe_result,
            output_was_modified=False,
        )

    @staticmethod
    def _decision_to_dict(decision: RecoveryDecision) -> dict[str, Any]:
        return {
            "strategy": decision.strategy.value,
            "reason": decision.reason,
            "bounded_attempts": int(decision.bounded_attempts),
            "checkpoint_id": str(decision.checkpoint_id or ""),
            "requires_user_approval": bool(decision.requires_user_approval),
            "matched_rules": list(decision.matched_rules),
        }

    @staticmethod
    def _apply_action(
        turn_context: TurnContext,
        decision: RecoveryDecision,
        action: ReasoningAction,
    ) -> None:
        if action.action_type == ReasoningActionType.RETRY_TOOL:
            turn_context.state["recovery_retry_requested"] = True
            return

        if action.action_type == ReasoningActionType.RETURN_DEGRADED:
            if decision.degraded_output is not None:
                turn_context.state["safe_result"] = decision.degraded_output
            return

        if action.action_type == ReasoningActionType.REQUEST_APPROVAL:
            turn_context.state["safe_result"] = (
                "I need your approval before I continue this action.",
                False,
            )
            turn_context.state["recovery_requires_approval"] = True
            return

        if action.action_type == ReasoningActionType.REPLAY_FROM_CHECKPOINT:
            checkpoint_id = str(decision.checkpoint_id or "")
            turn_context.state["recovery_replay_checkpoint"] = checkpoint_id
            return

        turn_context.state["safe_result"] = (
            "I paused for safety. Please try again with a safer request.",
            False,
        )
        turn_context.state["recovery_halted"] = True

    async def execute(self, _registry: Any, turn_context: TurnContext) -> None:
        policy_decision = self._coerce_policy_decision(turn_context)
        retry_count = int(turn_context.state.get("recovery_retry_count") or 0)
        previous_strategies_raw = list(turn_context.state.get("recovery_previous_strategies") or [])
        previous_strategies = [
            RecoveryStrategy(item)
            for item in previous_strategies_raw
            if str(item) in {strategy.value for strategy in RecoveryStrategy}
        ]

        recovery_context = RecoveryContext(
            tool_result=policy_decision.tool_result,
            policy_decision=policy_decision,
            retry_count=retry_count,
            previous_strategies=previous_strategies,
            user_context={"trace_id": str(turn_context.trace_id or "")},
        )
        decision = self._selector.select(recovery_context)
        action = self._reasoner.select_action(
            ReasoningContext(
                recovery_decision=decision,
                attempt_index=retry_count,
                turn_state=dict(turn_context.state or {}),
            ),
        )

        turn_context.state["recovery_decision"] = self._decision_to_dict(decision)
        turn_context.state["recovery_action"] = action.to_dict()
        turn_context.state["recovery_strategy"] = decision.strategy.value

        if decision.strategy == RecoveryStrategy.RETRY_SAME:
            turn_context.state["recovery_retry_count"] = retry_count + 1
        else:
            turn_context.state["recovery_retry_count"] = retry_count

        history = list(previous_strategies_raw)
        history.append(decision.strategy.value)
        turn_context.state["recovery_previous_strategies"] = history

        self._apply_action(turn_context, decision, action)


class SaveNode(_NodeContractMixin):
    name = "save"
    node_type = NodeType.COMMIT

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("persistence_service")
        result = turn_context.state.get("safe_result")
        finalize = getattr(service, "finalize_turn", None)
        if callable(finalize):
            try:
                turn_context.state["safe_result"] = finalize(turn_context, result)
                save_checkpoint = getattr(service, "save_graph_checkpoint", None)
                if callable(save_checkpoint):
                    checkpoint = turn_context.checkpoint_snapshot(
                        stage="save",
                        status="atomic_commit",
                        error=None,
                    )
                    save_checkpoint(checkpoint, _skip_turn_event=True)
                turn_context.fidelity.save = True
                return
            except Exception as exc:  # noqa: BLE001 — SaveNode optimistic path; non-fatal
                logger.debug("SaveNode finalize_turn failed: %s", exc)
                raise
        service.save_turn(turn_context, result)
        turn_context.fidelity.save = True


class TemporalNode(_NodeContractMixin):
    name = "temporal"
    node_type = NodeType.STANDARD

    async def execute(self, _registry: Any, turn_context: TurnContext) -> None:
        if getattr(turn_context, "temporal", None) is None:
            raise RuntimeError(
                "TemporalNode missing — deterministic execution violated",
            )
        temporal_payload = turn_context.temporal_snapshot()
        turn_context.state.setdefault("temporal", temporal_payload)
        turn_context.metadata.setdefault("temporal", temporal_payload)


class ReflectionNode(_NodeContractMixin):
    name = "reflection"
    node_type = NodeType.STANDARD

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        try:
            service = registry.get("reflection")
        except (KeyError, AttributeError):
            return
        result = turn_context.state.get("safe_result") or turn_context.state.get(
            "candidate",
        )
        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        current_mood = turn_context.state.get("mood") or "neutral"
        reply_text = result[0] if isinstance(result, tuple) else str(result or "")

        reflect_after_turn = getattr(service, "reflect_after_turn", None)
        if callable(reflect_after_turn):
            turn_context.state["reflection"] = reflect_after_turn(
                turn_text,
                current_mood,
                reply_text,
            )
            return

        reflect = getattr(service, "reflect", None)
        if callable(reflect):
            turn_context.state["reflection"] = reflect(turn_context, result)
