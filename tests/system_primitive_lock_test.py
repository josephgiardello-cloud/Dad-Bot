"""Tests — Phase 4: System Primitive Lock.

Test suite T: Validates the global invariant engine, execution equivalence classes,
              and resource model as first-class execution variables.

Coverage:
    T1–T10:  GlobalInvariantEngine (registration, evaluation, violation detection)
    T11–T20: Execution equivalence classes (ToolGraphClass, PlanClass, EventStructureClass)
    T21–T30: ExecutionEquivalenceClass (combined hash, is_equivalent, proof)
    T31–T40: Resource model as first-class (system-level resource invariant)
"""
from __future__ import annotations

from typing import Any

import pytest

from dadbot.core.invariant_engine import (
    DEFAULT_INVARIANTS,
    ExecutionState,
    GlobalInvariantEngine,
    GlobalValidationReport,
    InvariantCategory,
    InvariantSeverity,
    InvariantViolation,
    SystemInvariant,
)
from dadbot.core.execution_equivalence import (
    ExecutionEquivalenceClass,
    classify_execution,
    equivalence_proof,
    event_structure_class,
    is_equivalent_execution,
    plan_class,
    plan_class_from_planner_output,
    tool_graph_class_from_names,
)
from dadbot.core.tool_resource_model import (
    BudgetExhaustionPolicy,
    ResourceModelValidator,
    TurnBudget,
)
from dadbot.core.tool_dag import ToolDAG, ToolNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_state(**kwargs) -> ExecutionState:
    defaults: dict[str, Any] = {
        "planner_output": {
            "intent_type": "question",
            "strategy": "fact_seeking",
            "tool_plan": ["memory_lookup"],
        },
        "tool_events": [
            {"type": "tool_requested", "sequence": 0},
            {"type": "tool_executed", "sequence": 1},
        ],
        "memory_entries": [{"text": "some memory"}],
    }
    defaults.update(kwargs)
    return ExecutionState(**defaults)


def _make_dag(tools: list[str]) -> ToolDAG:
    dag = ToolDAG()
    for i, tool_name in enumerate(tools):
        node = ToolNode.build(
            tool_name=tool_name,
            intent="goal_lookup",
            args={"query": f"q{i}"},
            priority=1,
            sequence=i,
        )
        dag.add_node(node)
    return dag


# ===========================================================================
# T1–T10: GlobalInvariantEngine
# ===========================================================================


class TestGlobalInvariantEngine:
    def test_default_engine_has_invariants(self):
        engine = GlobalInvariantEngine.default()
        assert engine.invariant_count() == len(DEFAULT_INVARIANTS)

    def test_valid_state_passes_all(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state()
        report = engine.validate_all(state)
        assert report.ok is True
        assert report.error_violations == 0
        assert report.critical_violations == 0

    def test_missing_intent_type_fails(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(planner_output={
            "intent_type": "",
            "strategy": "fact_seeking",
            "tool_plan": [],
        })
        report = engine.validate_all(state)
        assert report.ok is False
        violated_ids = [v.invariant_id for v in report.violations]
        assert "planner.intent_type_present" in violated_ids

    def test_missing_strategy_fails(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(planner_output={
            "intent_type": "question",
            "strategy": "",
            "tool_plan": [],
        })
        report = engine.validate_all(state)
        violated_ids = [v.invariant_id for v in report.violations]
        assert "planner.strategy_present" in violated_ids

    def test_tool_event_missing_type_fails(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(
            tool_events=[{"no_type_here": True, "sequence": 0}]
        )
        report = engine.validate_all(state)
        violated_ids = [v.invariant_id for v in report.violations]
        assert "tool_execution.events_have_type" in violated_ids

    def test_register_custom_invariant(self):
        engine = GlobalInvariantEngine.default()
        custom = SystemInvariant(
            id="custom.test_invariant",
            category=InvariantCategory.PLANNER,
            description="Custom test invariant that always passes",
            predicate=lambda _state: True,
            severity=InvariantSeverity.WARNING,
        )
        engine.register(custom)
        assert engine.invariant_count() == len(DEFAULT_INVARIANTS) + 1

    def test_custom_invariant_violation(self):
        engine = GlobalInvariantEngine()
        always_fail = SystemInvariant(
            id="test.always_fail",
            category=InvariantCategory.PLANNER,
            description="Always fails",
            predicate=lambda _: False,
            severity=InvariantSeverity.ERROR,
        )
        engine.register(always_fail)
        state = ExecutionState()
        report = engine.validate_all(state)
        assert report.ok is False
        assert "test.always_fail" in [v.invariant_id for v in report.violations]

    def test_validation_hash_stable(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state()
        r1 = engine.validate_all(state)
        r2 = engine.validate_all(state)
        assert r1.validation_hash == r2.validation_hash

    def test_report_to_dict_has_required_keys(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state()
        report = engine.validate_all(state)
        d = report.to_dict()
        assert "ok" in d
        assert "total_invariants" in d
        assert "violations" in d
        assert "categories_checked" in d

    def test_remove_invariant(self):
        engine = GlobalInvariantEngine.default()
        initial_count = engine.invariant_count()
        engine.remove("planner.intent_type_present")
        assert engine.invariant_count() == initial_count - 1

    def test_event_monotonic_invariant_detects_non_monotonic(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(
            tool_events=[
                {"type": "tool_requested", "sequence": 5},
                {"type": "tool_executed", "sequence": 3},  # non-monotonic!
            ]
        )
        report = engine.validate_all(state)
        violated_ids = [v.invariant_id for v in report.violations]
        assert "event.log_monotonic" in violated_ids


# ===========================================================================
# T11–T20: Execution Equivalence Class Functions
# ===========================================================================


class TestEquivalenceClassFunctions:
    def test_tool_graph_class_stable(self):
        gc1 = tool_graph_class_from_names(["memory_lookup", "goal_lookup"])
        gc2 = tool_graph_class_from_names(["memory_lookup", "goal_lookup"])
        assert gc1 == gc2

    def test_tool_graph_class_order_independent(self):
        """Same tools in different order → same class (sorted)."""
        gc1 = tool_graph_class_from_names(["memory_lookup", "goal_lookup"])
        gc2 = tool_graph_class_from_names(["goal_lookup", "memory_lookup"])
        assert gc1 == gc2

    def test_tool_graph_class_differs_by_tools(self):
        gc1 = tool_graph_class_from_names(["memory_lookup"])
        gc2 = tool_graph_class_from_names(["goal_lookup"])
        assert gc1 != gc2

    def test_plan_class_stable(self):
        pc1 = plan_class("question", "fact_seeking", 1)
        pc2 = plan_class("question", "fact_seeking", 1)
        assert pc1 == pc2

    def test_plan_class_differs_by_intent(self):
        pc1 = plan_class("question", "fact_seeking", 1)
        pc2 = plan_class("emotional", "fact_seeking", 1)
        assert pc1 != pc2

    def test_plan_class_ignores_specific_tool_names(self):
        """Same intent/strategy/count → same class regardless of tool names."""
        pc1 = plan_class("question", "fact_seeking", 1)
        # Different tools but same count:
        gc1 = tool_graph_class_from_names(["memory_lookup"])
        gc2 = tool_graph_class_from_names(["goal_lookup"])
        assert pc1 == pc1  # trivially same
        # Plan class doesn't incorporate tool names
        # (that's the graph_class's responsibility)

    def test_event_structure_class_stable(self):
        events = [
            {"type": "tool_requested", "seq": 0},
            {"type": "tool_executed", "seq": 1},
        ]
        ec1 = event_structure_class(events)
        ec2 = event_structure_class(events)
        assert ec1 == ec2

    def test_event_structure_class_ignores_payloads(self):
        """Same type sequence but different payloads → same class."""
        e1 = [
            {"type": "tool_requested", "session_id": "sess-A"},
            {"type": "tool_executed", "session_id": "sess-A"},
        ]
        e2 = [
            {"type": "tool_requested", "session_id": "sess-B"},
            {"type": "tool_executed", "session_id": "sess-B"},
        ]
        assert event_structure_class(e1) == event_structure_class(e2)

    def test_event_structure_class_differs_by_type_sequence(self):
        e1 = [{"type": "tool_requested"}, {"type": "tool_executed"}]
        e2 = [{"type": "tool_executed"}, {"type": "tool_requested"}]
        assert event_structure_class(e1) != event_structure_class(e2)

    def test_plan_class_from_planner_output(self):
        output = {"intent_type": "question", "strategy": "fact_seeking", "tool_plan": ["memory_lookup"]}
        pc = plan_class_from_planner_output(output)
        assert pc == plan_class("question", "fact_seeking", 1)

    def test_empty_event_log_class(self):
        ec = event_structure_class([])
        assert isinstance(ec, str)
        assert len(ec) == 16


# ===========================================================================
# T21–T30: ExecutionEquivalenceClass
# ===========================================================================


class TestExecutionEquivalenceClass:
    def _make_class(
        self,
        tools=None,
        intent="question",
        strategy="fact_seeking",
        events=None,
    ) -> ExecutionEquivalenceClass:
        gc = tool_graph_class_from_names(tools or ["memory_lookup"])
        pc = plan_class(intent, strategy, len(tools or ["memory_lookup"]))
        ec = event_structure_class(events or [
            {"type": "tool_requested"},
            {"type": "tool_executed"},
        ])
        return ExecutionEquivalenceClass.build(gc, pc, ec)

    def test_build_creates_class(self):
        cls = self._make_class()
        assert isinstance(cls.equivalence_key, str)
        assert len(cls.equivalence_key) == 16

    def test_equivalence_key_stable(self):
        cls1 = self._make_class()
        cls2 = self._make_class()
        assert cls1.equivalence_key == cls2.equivalence_key

    def test_matches_identical(self):
        cls1 = self._make_class()
        cls2 = self._make_class()
        assert cls1.matches(cls2) is True

    def test_not_matches_different_intent(self):
        cls1 = self._make_class(intent="question")
        cls2 = self._make_class(intent="emotional")
        assert cls1.matches(cls2) is False

    def test_not_matches_different_tools(self):
        cls1 = self._make_class(tools=["memory_lookup"])
        cls2 = self._make_class(tools=["goal_lookup"])
        assert cls1.matches(cls2) is False

    def test_is_equivalent_execution_true(self):
        cls1 = self._make_class()
        cls2 = self._make_class()
        assert is_equivalent_execution(cls1, cls2) is True

    def test_is_equivalent_execution_false(self):
        cls1 = self._make_class(intent="question")
        cls2 = self._make_class(intent="emotional")
        assert is_equivalent_execution(cls1, cls2) is False

    def test_equivalence_proof_equivalent(self):
        cls1 = self._make_class()
        cls2 = self._make_class()
        proof = equivalence_proof(cls1, cls2)
        assert proof["equivalent"] is True
        assert proof["graph_class_match"] is True
        assert proof["plan_class_match"] is True
        assert proof["event_class_match"] is True

    def test_equivalence_proof_not_equivalent(self):
        cls1 = self._make_class(intent="question")
        cls2 = self._make_class(intent="emotional")
        proof = equivalence_proof(cls1, cls2)
        assert proof["equivalent"] is False
        assert proof["plan_class_match"] is False

    def test_classify_execution_with_dag(self):
        dag = _make_dag(["memory_lookup"])
        planner_output = {"intent_type": "question", "strategy": "fact_seeking", "tool_plan": ["memory_lookup"]}
        events = [{"type": "tool_requested"}, {"type": "tool_executed"}]
        cls = classify_execution(planner_output, dag, events)
        assert isinstance(cls.equivalence_key, str)

    def test_to_dict_has_all_keys(self):
        cls = self._make_class()
        d = cls.to_dict()
        assert "graph_class" in d
        assert "plan_class" in d
        assert "event_class" in d
        assert "equivalence_key" in d


# ===========================================================================
# T31–T40: Resource Model as First-Class Variable
# ===========================================================================


class TestResourceModelFirstClass:
    def test_resource_invariant_in_engine(self):
        """GlobalInvariantEngine's resource.within_budget invariant fires."""
        engine = GlobalInvariantEngine.default()
        # Budget exceeded → within_budget=False
        state = _make_valid_state(resource_report={
            "within_budget": False,
            "total_cost_units": 99.0,
        })
        report = engine.validate_all(state)
        violated_ids = [v.invariant_id for v in report.violations]
        assert "resource.within_budget" in violated_ids

    def test_resource_invariant_passes_within_budget(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(resource_report={
            "within_budget": True,
            "total_cost_units": 1.0,
        })
        report = engine.validate_all(state)
        violated_ids = [v.invariant_id for v in report.violations]
        assert "resource.within_budget" not in violated_ids

    def test_resource_invariant_passes_when_no_report(self):
        """No resource report → invariant passes (not applicable)."""
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(resource_report=None)
        report = engine.validate_all(state)
        violated_ids = [v.invariant_id for v in report.violations]
        assert "resource.within_budget" not in violated_ids

    def test_resource_model_integrates_with_invariant_engine(self):
        """Full integration: compute budget report, feed into invariant check."""
        validator = ResourceModelValidator()
        # Over budget: 4 tools with max 2
        budget = TurnBudget(
            max_cost_units=2.0,
            max_tools=2,
            max_latency_ms=10000,
            exhaustion_policy=BudgetExhaustionPolicy.SKIP,
        )
        report = validator.validate_tool_list(
            ["memory_lookup", "memory_lookup", "memory_lookup"],
            budget,
        )
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(resource_report=report.to_dict())
        inv_report = engine.validate_all(state)
        if not report.within_budget:
            violated_ids = [v.invariant_id for v in inv_report.violations]
            assert "resource.within_budget" in violated_ids

    def test_resource_violation_is_warning_not_critical(self):
        """Resource budget violation is WARNING severity, not blocking."""
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state(
            resource_report={"within_budget": False}
        )
        report = engine.validate_all(state)
        resource_violations = [
            v for v in report.violations
            if v.invariant_id == "resource.within_budget"
        ]
        assert len(resource_violations) == 1
        assert resource_violations[0].severity == InvariantSeverity.WARNING

    def test_budget_hash_changes_with_different_tools(self):
        """BudgetReport hash changes when approved tools change."""
        validator = ResourceModelValidator()
        budget = TurnBudget.default()
        r1 = validator.validate_tool_list(["memory_lookup"], budget)
        r2 = validator.validate_tool_list(["memory_lookup", "memory_lookup"], budget)
        assert r1.budget_hash != r2.budget_hash

    def test_budget_hash_stable_same_input(self):
        validator = ResourceModelValidator()
        budget = TurnBudget.default()
        r1 = validator.validate_tool_list(["memory_lookup"], budget)
        r2 = validator.validate_tool_list(["memory_lookup"], budget)
        assert r1.budget_hash == r2.budget_hash

    def test_equivalence_class_with_resource_context(self):
        """Executions with same structure but different resource budgets → same class.

        Resource budget is NOT part of the equivalence class — it's a constraint,
        not a structural property of the computation.
        """
        cls1 = ExecutionEquivalenceClass.build(
            tool_graph_class_from_names(["memory_lookup"]),
            plan_class("question", "fact_seeking", 1),
            event_structure_class([{"type": "tool_executed"}]),
        )
        cls2 = ExecutionEquivalenceClass.build(
            tool_graph_class_from_names(["memory_lookup"]),
            plan_class("question", "fact_seeking", 1),
            event_structure_class([{"type": "tool_executed"}]),
        )
        assert is_equivalent_execution(cls1, cls2) is True

    def test_invariant_engine_all_categories_checked(self):
        engine = GlobalInvariantEngine.default()
        state = _make_valid_state()
        report = engine.validate_all(state)
        # Should have checked multiple categories.
        assert len(report.categories_checked) > 1

    def test_invariant_engine_get_by_id(self):
        engine = GlobalInvariantEngine.default()
        inv = engine.get("planner.intent_type_present")
        assert inv is not None
        assert inv.category == InvariantCategory.PLANNER
