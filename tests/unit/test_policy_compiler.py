from __future__ import annotations

import asyncio
import pytest

from dadbot.core.graph import TurnContext
from dadbot.core.graph_pipeline_nodes import SafetyNode as PipelineSafetyNode
from dadbot.core.nodes import SafetyNode as RuntimeSafetyNode
from dadbot.core.policy_compiler import (
    PolicyCompilationError,
    PolicyCompiler,
    PolicyIntentGraph,
    PolicyPlan,
    PolicyStep,
)
from dadbot.core.turn_ir import build_policy_input


class _SafetyService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enforce_policies(self, turn_context: TurnContext, candidate: object) -> str:
        _ = turn_context
        self.calls.append("enforce")
        return f"safe::{candidate}"

    def validate(self, candidate: object) -> str:
        self.calls.append("validate")
        return f"validated::{candidate}"


class _ValidateOnlyService:
    def validate(self, candidate: object) -> str:
        return f"validated::{candidate}"


class _Registry:
    def __init__(self, service: object) -> None:
        self._service = service

    def get(self, _name: str) -> object:
        return self._service


def test_policy_compiler_prefers_enforce_policies() -> None:
    service = _SafetyService()
    plan = PolicyCompiler.compile_safety(service)

    decision = PolicyCompiler.evaluate_safety(
        plan,
        TurnContext(user_input="hi"),
        "candidate",
    )

    assert decision.output == "safe::candidate"
    assert decision.step_name == "enforce_policies"
    assert decision.action == "handled"
    trace = dict(decision.trace or {})
    considered = list(trace.get("considered_rules") or [])
    selected = dict(trace.get("selected_rule") or {})
    rejected = list(trace.get("rejected_alternatives") or [])
    assert considered == [
        {"name": "enforce_policies", "kind": "binary"},
        {"name": "validate", "kind": "unary"},
    ]
    assert selected.get("name") == "enforce_policies"
    assert selected.get("reason") == "first_applicable_rule_in_compiled_order"
    assert rejected == [{"name": "validate", "kind": "unary", "reason": "higher_priority_rule_selected_earlier"}]
    assert service.calls == ["enforce"]


def test_runtime_safety_node_records_policy_decision() -> None:
    service = _ValidateOnlyService()
    node = RuntimeSafetyNode(service)
    context = TurnContext(user_input="hi")
    context.state["candidate"] = "candidate"

    asyncio.run(node.run(context))

    assert context.state["safe_result"] == "validated::candidate"
    decision = dict(context.state.get("safety_policy_decision") or {})
    assert decision.get("step_name") == "validate"
    assert decision.get("action") == "handled"
    trace = dict(decision.get("trace") or {})
    assert str(trace.get("policy") or "") == "safety"
    assert str((trace.get("final_action") or {}).get("step_name") or "") == "validate"

    policy_events = list(context.state.get("policy_trace_events") or [])
    assert len(policy_events) == 1
    assert policy_events[0]["event_type"] == "policy_decision"
    assert policy_events[0]["policy"] == "safety"
    assert policy_events[0]["sequence"] == 1
    assert str((policy_events[0].get("trace") or {}).get("policy") or "") == "safety"


def test_pipeline_safety_node_passthrough_sets_flag() -> None:
    node = PipelineSafetyNode()
    context = TurnContext(user_input="hi")
    context.state["candidate"] = "candidate"

    asyncio.run(node.execute(_Registry(object()), context))

    assert context.state["safe_result"] == "candidate"
    assert context.state["safety_passthrough"]["failure_mode"] == "passthrough"
    decision = dict(context.state.get("safety_policy_decision") or {})
    trace = dict(decision.get("trace") or {})
    considered = list(trace.get("considered_rules") or [])
    assert considered == [{"name": "passthrough", "kind": "passthrough"}]
    assert str((trace.get("selected_rule") or {}).get("name") or "") == "passthrough"


def test_policy_trace_is_deterministic_for_same_inputs() -> None:
    service = _SafetyService()
    plan = PolicyCompiler.compile_safety(service)
    context_a = TurnContext(user_input="same")
    context_a.state["turn_plan"] = {"intent_type": "support", "strategy": "baseline"}
    context_b = TurnContext(user_input="same")
    context_b.state["turn_plan"] = {"intent_type": "support", "strategy": "baseline"}

    decision_a = PolicyCompiler.evaluate_safety(plan, context_a, "candidate")
    decision_b = PolicyCompiler.evaluate_safety(plan, context_b, "candidate")

    assert decision_a.trace == decision_b.trace


def test_compile_entrypoint_uses_explicit_intent_graph() -> None:
    service = _ValidateOnlyService()
    plan = PolicyCompiler.compile_safety(service)
    context = TurnContext(user_input="same")
    context.state["turn_plan"] = {"intent_type": "support", "strategy": "baseline"}

    policy_input = build_policy_input(plan.policy_name, context, "candidate")
    intent_graph = PolicyCompiler.build_intent_graph(plan, policy_input)
    decision = PolicyCompiler.compile(intent_graph)

    assert isinstance(intent_graph, PolicyIntentGraph)
    assert decision.action == "handled"
    assert decision.step_name == "validate"
    assert decision.output == "validated::candidate"
    trace = dict(decision.trace or {})
    assert str((trace.get("selected_rule") or {}).get("name") or "") == "validate"


def test_policy_compiler_accepts_explicit_policy_input_ir() -> None:
    service = _ValidateOnlyService()
    plan = PolicyCompiler.compile_safety(service)
    context = TurnContext(user_input="same")
    context.state["turn_plan"] = {"intent_type": "support", "strategy": "baseline"}
    policy_input = build_policy_input(plan.policy_name, context, "candidate")

    decision = PolicyCompiler.evaluate_safety_input(plan, policy_input)

    assert decision.action == "handled"
    assert decision.step_name == "validate"
    summary = dict((decision.trace or {}).get("intent_graph_summary") or {})
    assert summary.get("intent_type") == "support"
    assert summary.get("strategy") == "baseline"


def test_policy_compiler_trace_records_compile_phases() -> None:
    service = _ValidateOnlyService()
    plan = PolicyCompiler.compile_safety(service)
    decision = PolicyCompiler.evaluate_safety(plan, TurnContext(user_input="hi"), "candidate")

    metadata = dict((decision.trace or {}).get("metadata") or {})
    phases = list(metadata.get("compile_phases") or [])
    evaluations = list(metadata.get("rule_evaluations") or [])

    assert phases == [
        "match_rules",
        "resolve_rule",
        "transform_output",
        "emit_decision",
    ]
    assert evaluations
    assert evaluations[0]["name"] == "validate"
    assert evaluations[0]["applicable"] is True


def test_policy_compiler_unsupported_rule_kind_falls_back_to_passthrough() -> None:
    plan = PolicyPlan(
        policy_name="safety",
        steps=(
            PolicyStep(name="unsupported", kind="ternary", handler=lambda *_args: "nope"),
        ),
    )
    decision = PolicyCompiler.evaluate_safety(plan, TurnContext(user_input="hi"), "candidate")

    assert decision.action == "passthrough"
    assert decision.output == "candidate"
    assert decision.step_name == "passthrough"
    trace = dict(decision.trace or {})
    metadata = dict(trace.get("metadata") or {})
    evaluations = list(metadata.get("rule_evaluations") or [])
    assert evaluations[0]["reason"] == "unsupported_step_kind"


def test_compile_safety_strict_mode_rejects_missing_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_STRICT_SAFETY_POLICY", "1")

    with pytest.raises(PolicyCompilationError, match="missing enforce_policies/validate"):
        PolicyCompiler.compile_safety(object())


def test_compile_strict_mode_rejects_no_applicable_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DADBOT_STRICT_SAFETY_POLICY", "1")
    plan = PolicyPlan(
        policy_name="safety",
        steps=(
            PolicyStep(name="unsupported", kind="ternary", handler=lambda *_args: "nope"),
        ),
    )

    with pytest.raises(PolicyCompilationError, match="No applicable policy rules"):
        PolicyCompiler.evaluate_safety(plan, TurnContext(user_input="hi"), "candidate")
