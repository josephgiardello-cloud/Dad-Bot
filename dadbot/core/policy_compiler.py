from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from dadbot.core.turn_ir import PolicyInput, build_policy_input

_KIND_BINARY = "binary"
_KIND_UNARY = "unary"
_KIND_PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class PolicyRuleEvaluation:
    index: int
    name: str
    kind: str
    applicable: bool
    reason: str


@dataclass(frozen=True)
class PolicyStep:
    name: str
    kind: str
    handler: Callable[..., Any]


@dataclass(frozen=True)
class PolicyPlan:
    policy_name: str
    steps: tuple[PolicyStep, ...]


@dataclass(frozen=True)
class PolicyIntentGraph:
    policy_name: str
    policy_input: PolicyInput
    rules: tuple[PolicyStep, ...]


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    output: Any
    step_name: str
    details: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


class PolicyCompilationError(RuntimeError):
    """Raised when policy compilation/evaluation cannot proceed safely."""


class PolicyCompiler:
    """Compiles runtime policy handlers into an explicit, testable plan."""

    @staticmethod
    def _strict_mode() -> bool:
        raw = str(os.environ.get("DADBOT_STRICT_SAFETY_POLICY", "")).strip().lower()
        return raw in {"1", "true", "on", "yes"}

    @staticmethod
    def _fallback_step() -> PolicyStep:
        return PolicyStep(
            name="passthrough",
            kind=_KIND_PASSTHROUGH,
            handler=lambda _candidate: _candidate,
        )

    @staticmethod
    def _summarize_policy_input(policy_input: PolicyInput) -> dict[str, Any]:
        return policy_input.summary()

    @staticmethod
    def _build_trace(
        *,
        intent_graph: PolicyIntentGraph,
        considered_rules: tuple[PolicyStep, ...],
        rule_evaluations: tuple[PolicyRuleEvaluation, ...],
        selected_step: PolicyStep,
        selected_index: int,
        action: str,
        output_mutated: bool,
        candidate_hash: str,
        output_hash: str,
    ) -> dict[str, Any]:
        considered_rule_entries = [
            {
                "name": step.name,
                "kind": step.kind,
            }
            for step in considered_rules
        ]
        selected_rule = {
            "name": selected_step.name,
            "kind": selected_step.kind,
            "index": selected_index,
            "reason": "first_applicable_rule_in_compiled_order",
            "evidence": {
                "applicable": True,
                "compiled_order": selected_index,
            },
        }
        rejected_alternatives = [
            {
                "name": step.name,
                "kind": step.kind,
                "reason": "higher_priority_rule_selected_earlier",
            }
            for idx, step in enumerate(intent_graph.rules)
            if idx > selected_index
        ]
        evaluation_entries = [
            {
                "name": evaluation.name,
                "kind": evaluation.kind,
                "index": evaluation.index,
                "applicable": evaluation.applicable,
                "reason": evaluation.reason,
            }
            for evaluation in rule_evaluations
        ]
        return {
            "policy": intent_graph.policy_name,
            "intent_graph_summary": PolicyCompiler._summarize_policy_input(intent_graph.policy_input),
            "considered_rules": considered_rule_entries,
            "selected_rule": selected_rule,
            "rejected_alternatives": rejected_alternatives,
            "final_action": {
                "action": action,
                "step_name": selected_step.name,
                "kind": selected_step.kind,
                "output_mutated": bool(output_mutated),
                "candidate_hash": str(candidate_hash or ""),
                "output_hash": str(output_hash or ""),
            },
            "metadata": {
                "trace_schema": "policy-decision-trace-v1",
                "rule_count": len(considered_rule_entries),
                "compile_phases": [
                    "match_rules",
                    "resolve_rule",
                    "transform_output",
                    "emit_decision",
                ],
                "rule_evaluations": evaluation_entries,
            },
        }

    @staticmethod
    def compile_safety(service: Any, *, strict: bool = False) -> PolicyPlan:
        steps: list[PolicyStep] = []
        enforce = getattr(service, "enforce_policies", None)
        if callable(enforce):
            steps.append(PolicyStep(name="enforce_policies", kind=_KIND_BINARY, handler=enforce))

        validate = getattr(service, "validate", None)
        if callable(validate):
            steps.append(PolicyStep(name="validate", kind=_KIND_UNARY, handler=validate))

        if not steps:
            if strict or PolicyCompiler._strict_mode():
                raise PolicyCompilationError(
                    "Safety policy missing enforce_policies/validate handlers in strict mode",
                )
            steps.append(PolicyCompiler._fallback_step())

        return PolicyPlan(policy_name="safety", steps=tuple(steps))

    @staticmethod
    def build_intent_graph(
        plan: PolicyPlan,
        policy_input: PolicyInput,
    ) -> PolicyIntentGraph:
        return PolicyIntentGraph(
            policy_name=plan.policy_name,
            policy_input=policy_input,
            rules=tuple(plan.steps),
        )

    @staticmethod
    def match_rules(intent_graph: PolicyIntentGraph) -> tuple[PolicyStep, ...]:
        """Match applicable rules with bounds safety check.
        
        FIX: Added index < len(intent_graph.rules) to prevent IndexError
        when rules list is empty. See compile() for the same pattern.
        """
        evaluations = PolicyCompiler.evaluate_rules(intent_graph)
        return tuple(
            intent_graph.rules[evaluation.index]
            for evaluation in evaluations
            if evaluation.applicable and evaluation.index < len(intent_graph.rules)
        )

    @staticmethod
    def evaluate_rules(intent_graph: PolicyIntentGraph) -> tuple[PolicyRuleEvaluation, ...]:
        evaluations: list[PolicyRuleEvaluation] = []
        for index, step in enumerate(intent_graph.rules):
            supported_kind = step.kind in {_KIND_BINARY, _KIND_UNARY, _KIND_PASSTHROUGH}
            if not supported_kind:
                evaluations.append(
                    PolicyRuleEvaluation(
                        index=index,
                        name=step.name,
                        kind=step.kind,
                        applicable=False,
                        reason="unsupported_step_kind",
                    ),
                )
                continue
            evaluations.append(
                PolicyRuleEvaluation(
                    index=index,
                    name=step.name,
                    kind=step.kind,
                    applicable=True,
                    reason="handler_available",
                ),
            )
        if not evaluations:
            evaluations.append(
                PolicyRuleEvaluation(
                    index=0,
                    name="passthrough",
                    kind=_KIND_PASSTHROUGH,
                    applicable=True,
                    reason="fallback_passthrough",
                ),
            )
        return tuple(evaluations)

    @staticmethod
    def resolve_rule(
        intent_graph: PolicyIntentGraph,
        considered_rules: tuple[PolicyStep, ...],
    ) -> tuple[PolicyStep, int]:
        if considered_rules:
            selected_step = considered_rules[0]
            for index, step in enumerate(intent_graph.rules):
                if step.name == selected_step.name and step.kind == selected_step.kind:
                    return selected_step, index
            return selected_step, 0
        if PolicyCompiler._strict_mode():
            raise PolicyCompilationError(
                "No applicable policy rules in strict mode",
            )
        fallback_step = PolicyCompiler._fallback_step()
        _ = intent_graph
        return fallback_step, 0

    @staticmethod
    def _execute_binary(intent_graph: PolicyIntentGraph, selected_step: PolicyStep) -> Any:
        runtime_context = intent_graph.policy_input.runtime_turn_context
        return selected_step.handler(
            runtime_context if runtime_context is not None else intent_graph.policy_input,
            intent_graph.policy_input.candidate,
        )

    @staticmethod
    def _execute_unary(intent_graph: PolicyIntentGraph, selected_step: PolicyStep) -> Any:
        return selected_step.handler(intent_graph.policy_input.candidate)

    @staticmethod
    def _execute_passthrough(intent_graph: PolicyIntentGraph, selected_step: PolicyStep) -> Any:
        _ = selected_step
        return intent_graph.policy_input.candidate

    @staticmethod
    def transform_output(
        intent_graph: PolicyIntentGraph,
        selected_step: PolicyStep,
    ) -> tuple[str, Any, dict[str, Any], bool, str, str]:
        executors = {
            _KIND_BINARY: PolicyCompiler._execute_binary,
            _KIND_UNARY: PolicyCompiler._execute_unary,
            _KIND_PASSTHROUGH: PolicyCompiler._execute_passthrough,
        }
        executor = executors.get(selected_step.kind)
        if executor is None:
            if PolicyCompiler._strict_mode():
                raise PolicyCompilationError(
                    f"Unsupported policy step kind in strict mode: {selected_step.kind}",
                )
            candidate = intent_graph.policy_input.candidate
            candidate_hash = PolicyCompiler._stable_hash(candidate)
            output_hash = candidate_hash
            return "passthrough", candidate, {
                "policy": intent_graph.policy_name,
                "kind": _KIND_PASSTHROUGH,
                "fallback_from": selected_step.kind,
                "output_mutated": False,
                "candidate_hash": candidate_hash,
                "output_hash": output_hash,
            }, False, candidate_hash, output_hash
        output = executor(intent_graph, selected_step)
        action = "handled" if selected_step.kind in {_KIND_BINARY, _KIND_UNARY} else "passthrough"
        candidate = intent_graph.policy_input.candidate
        candidate_hash = PolicyCompiler._stable_hash(candidate)
        output_hash = PolicyCompiler._stable_hash(output)
        output_mutated = output_hash != candidate_hash
        return action, output, {
            "policy": intent_graph.policy_name,
            "kind": selected_step.kind,
            "output_mutated": output_mutated,
            "candidate_hash": candidate_hash,
            "output_hash": output_hash,
        }, output_mutated, candidate_hash, output_hash

    @staticmethod
    def _stable_hash(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8"),
        ).hexdigest()

    @staticmethod
    def compile(intent_graph: PolicyIntentGraph) -> PolicyDecision:
        rule_evaluations = PolicyCompiler.evaluate_rules(intent_graph)
        considered = tuple(
            intent_graph.rules[evaluation.index]
            for evaluation in rule_evaluations
            if evaluation.applicable and evaluation.index < len(intent_graph.rules)
        )
        selected_step, selected_index = PolicyCompiler.resolve_rule(intent_graph, considered)
        action, output, details, output_mutated, candidate_hash, output_hash = PolicyCompiler.transform_output(intent_graph, selected_step)
        trace = PolicyCompiler._build_trace(
            intent_graph=intent_graph,
            considered_rules=considered,
            rule_evaluations=rule_evaluations,
            selected_step=selected_step,
            selected_index=selected_index,
            action=action,
            output_mutated=output_mutated,
            candidate_hash=candidate_hash,
            output_hash=output_hash,
        )
        return PolicyDecision(
            action=action,
            output=output,
            step_name=selected_step.name,
            details=details,
            trace=trace,
        )

    @staticmethod
    def evaluate_safety(plan: PolicyPlan, turn_context: Any, candidate: Any) -> PolicyDecision:
        policy_input = build_policy_input(plan.policy_name, turn_context, candidate)
        intent_graph = PolicyCompiler.build_intent_graph(plan, policy_input)
        return PolicyCompiler.compile(intent_graph)

    @staticmethod
    def evaluate_safety_input(plan: PolicyPlan, policy_input: PolicyInput) -> PolicyDecision:
        intent_graph = PolicyCompiler.build_intent_graph(plan, policy_input)
        return PolicyCompiler.compile(intent_graph)


__all__ = [
    "PolicyCompiler",
    "PolicyCompilationError",
    "PolicyDecision",
    "PolicyIntentGraph",
    "PolicyPlan",
    "PolicyRuleEvaluation",
    "PolicyStep",
]
