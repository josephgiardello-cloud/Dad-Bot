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
from dadbot.core.planner import PlannerNode
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


# ---------------------------------------------------------------------------
# Required-arg schema per built-in tool (single source of truth for the Gate)
# ---------------------------------------------------------------------------

_TOOL_REQUIRED_ARGS: dict[str, frozenset[str]] = {
    "memory_lookup": frozenset({"query"}),
    "echo": frozenset({"message"}),
    "current_time": frozenset(),
}


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
            self._handle_violations(turn_context, violations)
            # Re-run PlannerNode with the repair context in state so it can
            # adjust its tool selection strategy.
            await PlannerNode().run(turn_context)
        # After the repair attempt, remove any still-violating requests to
        # prevent them from reaching the execution layer.
        remaining = self._collect_violations(turn_context)
        if remaining:
            self._strip_violations(turn_context, remaining)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_violations(self, turn_context: TurnContext) -> list[dict[str, Any]]:
        tool_ir = turn_context.state.get("tool_ir") or {}
        requests = list(tool_ir.get("requests") or [])
        violations: list[dict[str, Any]] = []
        for req in requests:
            tool_name = str(req.get("tool_name") or "").strip().lower()
            args = dict(req.get("args") or {})
            required = _TOOL_REQUIRED_ARGS.get(tool_name)
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
    ) -> None:
        repair_context = {
            "violations": violations,
            "repair_requested": True,
        }
        turn_context.state["_validation_gate_repair"] = repair_context
        for v in violations:
            emit_cognition(
                turn_context,
                CognitionEnvelope(
                    step_id=f"{turn_context.trace_id}:validation_gate:violation:{v['tool_name']}",
                    thought_trace=(
                        f"ValidationGate: CONTRACT_VIOLATION for tool '{v['tool_name']}' — "
                        f"missing args {v['missing_args']}. {v['repair_hint']}"
                    ),
                    target_node="validation_gate",
                    confidence_score=0.0,
                ),
            )

    def _strip_violations(
        self,
        turn_context: TurnContext,
        violations: list[dict[str, Any]],
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
        }
        turn_context.state["tool_ir"] = tool_ir


class InferenceNode(_NodeContractMixin):
    name = "inference"

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
        enforce = getattr(service, "enforce_policies", None)
        if callable(enforce):
            turn_context.state["safe_result"] = enforce(turn_context, candidate)
            return
        validate = getattr(service, "validate", None)
        if callable(validate):
            turn_context.state["safe_result"] = validate(candidate)
            return
        turn_context.state["safe_result"] = candidate
        turn_context.state["safety_passthrough"] = {
            "reason": "no_safety_manager",
            "failure_mode": "passthrough",
        }


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
                logger.debug("SaveNode optimistic checkpoint skipped: %s", exc)
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
