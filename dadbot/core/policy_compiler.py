from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dadbot.core.semantic_primitives import evaluate_policy
from dadbot.core.semantic_primitives import hash as semantic_hash
from dadbot.core.turn_ir import PolicyInput, SemanticEvalInput, build_policy_input, hash_eval_input

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


@dataclass(frozen=True)
class PolicyEquivalenceProof:
    contract_version: str
    proof_mode: str
    equivalent: bool
    fast_output_hash: str
    full_output_hash: str
    policy_view_state_hash: str
    rule_fingerprint: str


@dataclass(frozen=True)
class SemanticDecision:
    action: str
    reason: str
    degraded: bool = False

    @staticmethod
    def allow(*, degraded: bool = False) -> SemanticDecision:
        return SemanticDecision(action="allow", reason="allow", degraded=bool(degraded))

    @staticmethod
    def deny(reason: str) -> SemanticDecision:
        return SemanticDecision(action="deny", reason=str(reason or "policy_block"), degraded=False)


class PolicyCompilationError(RuntimeError):
    """Raised when policy compilation/evaluation cannot proceed safely."""


class PolicyCompiler:
    """Compiles runtime policy handlers into an explicit, testable plan."""

    @staticmethod
    def _strict_mode() -> bool:
        raw = str(os.environ.get("DADBOT_STRICT_SAFETY_POLICY", "")).strip().lower()
        return raw in {"1", "true", "on", "yes"}

    @staticmethod
    def _semantic_tool_budget_limit() -> int:
        raw = str(os.environ.get("DADBOT_SEMANTIC_TOOL_BUDGET", "8")).strip()
        with_safety = int(raw) if raw.isdigit() else 8
        return max(1, with_safety)

    @staticmethod
    def _semantic_policy_deny_set() -> set[str]:
        raw = str(os.environ.get("DADBOT_POLICY_DENY_SET", "")).strip()
        if not raw:
            return set()
        return {item.strip().lower() for item in raw.split(",") if item.strip()}

    @staticmethod
    def evaluate_semantics(inp: SemanticEvalInput) -> SemanticDecision:
        """Thin adapter to canonical semantic policy primitive."""
        return evaluate_policy(inp)

    @staticmethod
    def full_policy_eval_debug(inp: SemanticEvalInput) -> SemanticDecision:
        """Compatibility alias for semantic evaluator; no independent execution semantics."""
        return PolicyCompiler.evaluate_semantics(inp)

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
    def _rule_fingerprint(intent_graph: PolicyIntentGraph) -> str:
        rules = [{"name": str(step.name), "kind": str(step.kind)} for step in intent_graph.rules]
        return PolicyCompiler._stable_hash(rules)

    @staticmethod
    def _build_minimal_trace(
        *,
        intent_graph: PolicyIntentGraph,
        selected_step: PolicyStep,
        selected_index: int,
        action: str,
        output_mutated: bool,
        candidate_hash: str,
        output_hash: str,
    ) -> dict[str, Any]:
        return {
            "policy": intent_graph.policy_name,
            "intent_graph_summary": PolicyCompiler._summarize_policy_input(intent_graph.policy_input),
            "selected_rule": {
                "name": selected_step.name,
                "kind": selected_step.kind,
                "index": selected_index,
            },
            "final_action": {
                "action": action,
                "step_name": selected_step.name,
                "kind": selected_step.kind,
                "output_mutated": bool(output_mutated),
                "candidate_hash": str(candidate_hash or ""),
                "output_hash": str(output_hash or ""),
            },
            "metadata": {
                "trace_schema": "policy-decision-trace-v2-minimal",
                "rule_count": len(intent_graph.rules),
                "compile_phases": ["fast_gate", "transform_output", "emit_decision"],
            },
        }

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
        return semantic_hash(payload)

    @staticmethod
    def compile(intent_graph: PolicyIntentGraph, *, trace_level: str = "full") -> PolicyDecision:
        rule_evaluations = PolicyCompiler.evaluate_rules(intent_graph)
        considered = tuple(
            intent_graph.rules[evaluation.index]
            for evaluation in rule_evaluations
            if evaluation.applicable and evaluation.index < len(intent_graph.rules)
        )
        selected_step, selected_index = PolicyCompiler.resolve_rule(intent_graph, considered)
        action, output, details, output_mutated, candidate_hash, output_hash = PolicyCompiler.transform_output(intent_graph, selected_step)
        level = str(trace_level or "full").strip().lower()
        if level == "minimal":
            trace = PolicyCompiler._build_minimal_trace(
                intent_graph=intent_graph,
                selected_step=selected_step,
                selected_index=selected_index,
                action=action,
                output_mutated=output_mutated,
                candidate_hash=candidate_hash,
                output_hash=output_hash,
            )
        else:
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
    def _audit_full_decision_consistency(
        *,
        fast_decision: PolicyDecision,
        full_decision: PolicyDecision,
        policy_input: PolicyInput,
        intent_graph: PolicyIntentGraph,
        proof_mode: str,
    ) -> PolicyEquivalenceProof:
        fast_output_hash = PolicyCompiler._stable_hash(fast_decision.output)
        full_output_hash = PolicyCompiler._stable_hash(full_decision.output)
        proof = PolicyEquivalenceProof(
            contract_version="policy-equivalence-v1",
            proof_mode=str(proof_mode or "off"),
            equivalent=(fast_output_hash == full_output_hash),
            fast_output_hash=fast_output_hash,
            full_output_hash=full_output_hash,
            policy_view_state_hash=str(
                getattr(policy_input.policy_view, "state_hash", "") or "",
            ),
            rule_fingerprint=PolicyCompiler._rule_fingerprint(intent_graph),
        )
        if not proof.equivalent:
            raise PolicyCompilationError(
                "Fast policy gate diverged from full policy evaluation output",
            )
        return proof

    @staticmethod
    def _proof_mode(audit_full: bool) -> str:
        raw = str(os.environ.get("DADBOT_POLICY_EQUIVALENCE_PROOF", "audit")).strip().lower()
        if raw in {"off", "none", "0", "false"}:
            return "off"
        if raw in {"always", "on", "1", "true"}:
            return "always"
        if raw in {"audit", "audit_only"}:
            return "audit" if bool(audit_full) else "off"
        return "audit" if bool(audit_full) else "off"

    @staticmethod
    def evaluate_safety(
        plan: PolicyPlan,
        turn_context: Any,
        candidate: Any,
        *,
        fast_gate: bool = False,
        audit_full: bool = False,
    ) -> PolicyDecision:
        # Compatibility: callers may still pass fast_gate, but policy truth is now single-path.
        _ = fast_gate
        policy_input = build_policy_input(plan.policy_name, turn_context, candidate)
        semantic_eval_input = policy_input.semantic_eval_input
        if not isinstance(semantic_eval_input, SemanticEvalInput):
            raise PolicyCompilationError("Missing semantic evaluation input projection")

        semantic_decision = PolicyCompiler.evaluate_semantics(semantic_eval_input)
        if bool(audit_full):
            semantic_debug = PolicyCompiler.full_policy_eval_debug(semantic_eval_input)
            if semantic_decision != semantic_debug:
                raise PolicyCompilationError("Semantic evaluation audit assertion failed")

        if semantic_decision.action == "deny":
            blocked_output = ("Request blocked by semantic safety policy.", False)
            return PolicyDecision(
                action="denied",
                output=blocked_output,
                step_name="semantic_eval",
                details={
                    "reason": semantic_decision.reason,
                    "degraded": bool(semantic_decision.degraded),
                },
                trace={
                    "policy": plan.policy_name,
                    "truth_source": "semantic_eval_input_v1",
                    "semantic_eval": {
                        "contract_version": "semantic-eval-v1",
                        "eval_input_hash": hash_eval_input(semantic_eval_input),
                        "decision": {
                            "action": semantic_decision.action,
                            "reason": semantic_decision.reason,
                            "degraded": bool(semantic_decision.degraded),
                        },
                    },
                },
            )

        intent_graph = PolicyCompiler.build_intent_graph(plan, policy_input)
        decision = PolicyCompiler.compile(intent_graph, trace_level="full")
        decision.trace.setdefault("truth_source", "semantic_eval_input_v1")
        decision.trace.setdefault("semantic_eval", {
            "contract_version": "semantic-eval-v1",
            "eval_input_hash": hash_eval_input(semantic_eval_input),
            "decision": {
                "action": semantic_decision.action,
                "reason": semantic_decision.reason,
                "degraded": bool(semantic_decision.degraded),
            },
            "runtime_domain_contract": {
                "policy_semantics": "pure_deterministic",
                "policy_runtime": "single_path_compile_full",
                "split_execution_disabled": True,
            },
        })
        return decision

    @staticmethod
    def evaluate_safety_input(plan: PolicyPlan, policy_input: PolicyInput) -> PolicyDecision:
        intent_graph = PolicyCompiler.build_intent_graph(plan, policy_input)
        return PolicyCompiler.compile(intent_graph, trace_level="full")


__all__ = [
    "PolicyCompilationError",
    "PolicyCompiler",
    "PolicyDecision",
    "PolicyEquivalenceProof",
    "PolicyIntentGraph",
    "PolicyPlan",
    "PolicyRuleEvaluation",
    "PolicyStep",
    "SemanticDecision",
]
