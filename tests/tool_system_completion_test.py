"""Tests — Phase 2: Tool System Completion.

Test suite S: Validates tool idempotency, resource model, failure propagation,
              and conditional dependency edges.

Coverage:
    S1–S10:  Tool idempotency (registry, hash stability, violation detection)
    S11–S20: Resource model (cost catalog, budget enforcement, exhaustion policies)
    S21–S30: Failure propagation (ToolFailure types, severity, propagation policies)
    S31–S40: Conditional dependency edges (HARD/SOFT, predicates, resolution)
"""

from __future__ import annotations

import pytest

from dadbot.core.tool_algebra import (
    FailureSeverity,
    PropagationPolicy,
    ToolFailure,
    ToolFailureLog,
    ToolFailureType,
)
from dadbot.core.tool_dependency import (
    PREDICATE_ALWAYS_TRUE,
    PREDICATE_HAS_GOALS,
    PREDICATE_RESULT_NONEMPTY,
    PREDICATE_STATUS_OK,
    ConditionalEdge,
    DependencyResolver,
    DependencyType,
    NodeStatus,
)
from dadbot.core.tool_idempotency import (
    IdempotencyViolationError,
    ToolIdempotencyRegistry,
    canonical_request_hash,
    canonical_result_class,
    infer_output_type,
)
from dadbot.core.tool_resource_model import (
    BudgetExhaustionPolicy,
    ResourceModelValidator,
    ToolCostCatalog,
    ToolCostEntry,
    TurnBudget,
)

# ===========================================================================
# S1–S10: Tool Idempotency
# ===========================================================================


class TestToolIdempotency:
    def test_canonical_request_hash_stable(self):
        h1 = canonical_request_hash("memory_lookup", {"query": "goals"}, "goal_lookup")
        h2 = canonical_request_hash("memory_lookup", {"query": "goals"}, "goal_lookup")
        assert h1 == h2

    def test_canonical_request_hash_differs_by_tool_name(self):
        h1 = canonical_request_hash("memory_lookup", {"q": "x"})
        h2 = canonical_request_hash("goal_lookup", {"q": "x"})
        assert h1 != h2

    def test_canonical_request_hash_differs_by_args(self):
        h1 = canonical_request_hash("memory_lookup", {"query": "a"})
        h2 = canonical_request_hash("memory_lookup", {"query": "b"})
        assert h1 != h2

    def test_canonical_result_class_stable(self):
        c1 = canonical_result_class("memory_lookup", "ok", "list")
        c2 = canonical_result_class("memory_lookup", "ok", "list")
        assert c1 == c2

    def test_infer_output_type_coverage(self):
        assert infer_output_type(None) == "null"
        assert infer_output_type({}) == "dict"
        assert infer_output_type([]) == "list"
        assert infer_output_type("text") == "str"
        assert infer_output_type(42) == "number"
        assert infer_output_type(True) == "bool"

    def test_registry_registers_first_call(self):
        reg = ToolIdempotencyRegistry()
        rec = reg.register("memory_lookup", {"query": "goals"}, "ok", ["goal 1"])
        assert rec.hit_count == 1
        assert rec.result_class is not None

    def test_registry_increments_hit_count(self):
        reg = ToolIdempotencyRegistry()
        reg.register("memory_lookup", {"query": "goals"}, "ok", ["goal 1"])
        rec2 = reg.register("memory_lookup", {"query": "goals"}, "ok", ["goal 2"])
        assert rec2.hit_count == 2

    def test_registry_same_class_no_violation(self):
        reg = ToolIdempotencyRegistry()
        reg.register("memory_lookup", {"query": "goals"}, "ok", ["result 1"])
        reg.register("memory_lookup", {"query": "goals"}, "ok", ["result 2"])
        assert reg.registry_size() == 1

    def test_registry_raises_on_class_change(self):
        """Same request but different result type → idempotency violation."""
        reg = ToolIdempotencyRegistry()
        reg.register("memory_lookup", {"query": "goals"}, "ok", [])  # list
        with pytest.raises(IdempotencyViolationError) as exc_info:
            reg.register("memory_lookup", {"query": "goals"}, "ok", {})  # dict
        assert exc_info.value.stored_class != exc_info.value.new_class

    def test_is_cache_hit_true_after_register(self):
        reg = ToolIdempotencyRegistry()
        reg.register("memory_lookup", {"query": "x"}, "ok", [])
        assert reg.is_cache_hit("memory_lookup", {"query": "x"}) is True

    def test_is_cache_hit_false_before_register(self):
        reg = ToolIdempotencyRegistry()
        assert reg.is_cache_hit("memory_lookup", {"query": "new"}) is False

    def test_idempotency_proof_registered(self):
        reg = ToolIdempotencyRegistry()
        reg.register("memory_lookup", {"query": "g"}, "ok", [])
        proof = reg.idempotency_proof("memory_lookup", {"query": "g"})
        assert proof["ok"] is True
        assert proof["registered"] is True
        assert proof["hit_count"] >= 1

    def test_idempotency_proof_unregistered(self):
        reg = ToolIdempotencyRegistry()
        proof = reg.idempotency_proof("memory_lookup", {"query": "never"})
        assert proof["ok"] is False
        assert proof["registered"] is False

    def test_assert_idempotent_no_raise_on_match(self):
        reg = ToolIdempotencyRegistry()
        reg.register("memory_lookup", {"q": "x"}, "ok", [])
        reg.assert_idempotent("memory_lookup", {"q": "x"}, "ok", [])

    def test_assert_idempotent_raise_on_mismatch(self):
        reg = ToolIdempotencyRegistry()
        reg.register("memory_lookup", {"q": "x"}, "ok", [])
        with pytest.raises(IdempotencyViolationError):
            reg.assert_idempotent("memory_lookup", {"q": "x"}, "ok", {})


# ===========================================================================
# S11–S20: Resource Model
# ===========================================================================


class TestResourceModel:
    def test_default_catalog_has_memory_lookup(self):
        catalog = ToolCostCatalog.default()
        entry = catalog.get("memory_lookup")
        assert entry.tool_name == "memory_lookup"
        assert entry.cost_units > 0

    def test_default_budget_created(self):
        budget = TurnBudget.default()
        assert budget.max_tools > 0
        assert budget.max_cost_units > 0
        assert budget.exhaustion_policy == BudgetExhaustionPolicy.DEGRADE

    def test_strict_budget_created(self):
        budget = TurnBudget.strict()
        assert budget.max_tools == 1
        assert budget.exhaustion_policy == BudgetExhaustionPolicy.SKIP

    def test_within_budget_single_tool(self):
        validator = ResourceModelValidator()
        report = validator.validate_tool_list(["memory_lookup"], TurnBudget.default())
        assert report.within_budget is True
        assert len(report.approved_tools) == 1

    def test_empty_plan_within_budget(self):
        validator = ResourceModelValidator()
        report = validator.validate_tool_list([], TurnBudget.default())
        assert report.within_budget is True
        assert report.tool_count == 0

    def test_degrade_policy_drops_excess_tools(self):
        """DEGRADE: only tools within budget are approved; extras go to exhausted."""
        catalog = ToolCostCatalog.default()
        # memory_lookup costs 1.0, so max 2 with budget of 2.0
        budget = TurnBudget(
            max_cost_units=2.0,
            max_tools=2,
            max_latency_ms=10000,
            exhaustion_policy=BudgetExhaustionPolicy.DEGRADE,
        )
        validator = ResourceModelValidator(catalog)
        report = validator.validate_tool_list(["memory_lookup", "memory_lookup", "memory_lookup"], budget)
        assert len(report.approved_tools) == 2
        assert len(report.exhausted_tools) == 1

    def test_skip_policy_rejects_all_if_over_budget(self):
        """SKIP: if plan exceeds budget, reject all tools."""
        budget = TurnBudget(
            max_cost_units=0.5,
            max_tools=1,
            max_latency_ms=10000,
            exhaustion_policy=BudgetExhaustionPolicy.SKIP,
        )
        validator = ResourceModelValidator()
        report = validator.validate_tool_list(["memory_lookup"], budget)
        assert len(report.approved_tools) == 0
        assert len(report.exhausted_tools) == 1

    def test_compress_policy_approves_all_but_marks_exhausted(self):
        """COMPRESS: all tools are approved but exhausted list populated if over budget."""
        budget = TurnBudget(
            max_cost_units=0.1,
            max_tools=10,
            max_latency_ms=10000,
            exhaustion_policy=BudgetExhaustionPolicy.COMPRESS,
        )
        validator = ResourceModelValidator()
        report = validator.validate_tool_list(["memory_lookup", "memory_lookup"], budget)
        assert len(report.approved_tools) == 2  # COMPRESS allows all
        assert len(report.exhausted_tools) == 2  # But marks as exhausted

    def test_budget_report_has_hash(self):
        validator = ResourceModelValidator()
        report = validator.validate_tool_list(["memory_lookup"], TurnBudget.default())
        assert isinstance(report.budget_hash, str)
        assert len(report.budget_hash) == 64

    def test_budget_report_to_dict(self):
        validator = ResourceModelValidator()
        report = validator.validate_tool_list(["memory_lookup"], TurnBudget.default())
        d = report.to_dict()
        assert "within_budget" in d
        assert "total_cost_units" in d
        assert "suggested_action" in d

    def test_custom_catalog_register(self):
        catalog = ToolCostCatalog.default()
        custom = ToolCostEntry(tool_name="my_custom_tool", cost_units=5.0, latency_ms_estimate=250)
        catalog.register(custom)
        assert catalog.get("my_custom_tool").cost_units == 5.0


# ===========================================================================
# S21–S30: Failure Propagation
# ===========================================================================


class TestFailurePropagation:
    def test_validation_error_factory(self):
        f = ToolFailure.validation_error("tool-001", "invalid args")
        assert f.failure_type == ToolFailureType.VALIDATION
        assert f.severity == FailureSeverity.MEDIUM
        assert f.propagation_policy == PropagationPolicy.SKIP
        assert not f.recoverable

    def test_execution_error_factory_recoverable(self):
        f = ToolFailure.execution_error("tool-002", "timeout", recoverable=True)
        assert f.propagation_policy == PropagationPolicy.RETRY
        assert f.recoverable is True

    def test_execution_error_factory_unrecoverable(self):
        f = ToolFailure.execution_error("tool-003", "crash", recoverable=False)
        assert f.propagation_policy == PropagationPolicy.FALLBACK
        assert f.recoverable is False

    def test_unsupported_tool_factory(self):
        f = ToolFailure.unsupported_tool("tool-004", "os.system")
        assert f.failure_type == ToolFailureType.UNSUPPORTED
        assert f.severity == FailureSeverity.HIGH
        assert f.propagation_policy == PropagationPolicy.HALT
        assert f.should_halt() is True

    def test_should_skip(self):
        f = ToolFailure.validation_error("t", "bad")
        assert f.should_skip() is True
        assert f.should_halt() is False

    def test_should_retry(self):
        f = ToolFailure.execution_error("t", "err", recoverable=True)
        assert f.should_retry() is True

    def test_failure_log_has_halt(self):
        log = ToolFailureLog()
        log.append(ToolFailure.unsupported_tool("t", "bad"))
        assert log.has_halt() is True

    def test_failure_log_recoverable_list(self):
        log = ToolFailureLog()
        log.append(ToolFailure.validation_error("t1", "bad"))
        log.append(ToolFailure.execution_error("t2", "err", recoverable=True))
        recoverable = log.recoverable_failures()
        assert len(recoverable) == 1
        assert recoverable[0].tool_id == "t2"

    def test_failure_log_to_list(self):
        log = ToolFailureLog()
        log.append(ToolFailure.validation_error("t1", "err"))
        items = log.to_list()
        assert len(items) == 1
        assert items[0]["failure_type"] == "validation"

    def test_failure_to_dict(self):
        f = ToolFailure.validation_error("t", "bad args")
        d = f.to_dict()
        assert d["failure_type"] == "validation"
        assert d["propagation_policy"] == "skip"


# ===========================================================================
# S31–S40: Conditional Dependency Edges
# ===========================================================================


class TestToolDependency:
    def _make_resolver(self) -> DependencyResolver:
        resolver = DependencyResolver()
        resolver.add_edge(
            ConditionalEdge(
                source_node_id="node-A",
                target_node_id="node-B",
                predicate_key=PREDICATE_STATUS_OK,
                dependency_type=DependencyType.HARD,
            )
        )
        return resolver

    def test_hard_dep_blocked_when_source_missing(self):
        resolver = self._make_resolver()
        result = resolver.resolve_node("node-B", outputs={})
        assert result.status == NodeStatus.BLOCKED

    def test_hard_dep_runnable_when_source_ok(self):
        resolver = self._make_resolver()
        result = resolver.resolve_node(
            "node-B",
            outputs={"node-A": {"status": "ok", "data": []}},
        )
        assert result.status == NodeStatus.RUNNABLE

    def test_hard_dep_blocked_when_predicate_fails(self):
        resolver = self._make_resolver()
        result = resolver.resolve_node(
            "node-B",
            outputs={"node-A": {"status": "error", "data": []}},
        )
        assert result.status == NodeStatus.BLOCKED

    def test_soft_dep_degraded_when_source_missing(self):
        resolver = DependencyResolver()
        resolver.add_edge(
            ConditionalEdge(
                source_node_id="node-A",
                target_node_id="node-B",
                predicate_key=PREDICATE_STATUS_OK,
                dependency_type=DependencyType.SOFT,
                fallback_input={"fallback": True},
            )
        )
        result = resolver.resolve_node("node-B", outputs={})
        assert result.status == NodeStatus.SOFT_DEGRADED

    def test_soft_dep_uses_fallback_input(self):
        resolver = DependencyResolver()
        resolver.add_edge(
            ConditionalEdge(
                source_node_id="node-A",
                target_node_id="node-B",
                predicate_key=PREDICATE_STATUS_OK,
                dependency_type=DependencyType.SOFT,
                fallback_input={"fallback": True},
            )
        )
        result = resolver.resolve_node("node-B", outputs={})
        assert result.effective_input == {"fallback": True}

    def test_no_edges_always_runnable(self):
        resolver = DependencyResolver()
        result = resolver.resolve_node("node-X", outputs={})
        assert result.status == NodeStatus.RUNNABLE

    def test_predicate_always_true(self):
        resolver = DependencyResolver()
        resolver.add_edge(
            ConditionalEdge(
                source_node_id="node-A",
                target_node_id="node-B",
                predicate_key=PREDICATE_ALWAYS_TRUE,
                dependency_type=DependencyType.HARD,
            )
        )
        result = resolver.resolve_node("node-B", outputs={"node-A": None})
        assert result.status == NodeStatus.RUNNABLE

    def test_predicate_result_nonempty(self):
        resolver = DependencyResolver()
        resolver.add_edge(
            ConditionalEdge(
                source_node_id="src",
                target_node_id="tgt",
                predicate_key=PREDICATE_RESULT_NONEMPTY,
                dependency_type=DependencyType.HARD,
            )
        )
        # Non-empty output → RUNNABLE
        result = resolver.resolve_node("tgt", outputs={"src": ["some", "data"]})
        assert result.status == NodeStatus.RUNNABLE
        # Empty output → BLOCKED
        result2 = resolver.resolve_node("tgt", outputs={"src": []})
        assert result2.status == NodeStatus.BLOCKED

    def test_runnable_node_ids_filters_blocked(self):
        resolver = self._make_resolver()
        runnable = resolver.runnable_node_ids(
            outputs={"node-A": {"status": "ok"}},
            node_ids=["node-B"],
        )
        assert "node-B" in runnable

    def test_dependency_graph_hash_stable(self):
        resolver1 = self._make_resolver()
        resolver2 = self._make_resolver()
        assert resolver1.dependency_graph_hash() == resolver2.dependency_graph_hash()

    def test_has_goals_predicate(self):
        edge = ConditionalEdge(
            source_node_id="s",
            target_node_id="t",
            predicate_key=PREDICATE_HAS_GOALS,
            dependency_type=DependencyType.HARD,
        )
        assert edge.evaluate({"goals": ["goal1"]}) is True
        assert edge.evaluate({"goals": []}) is False
        assert edge.evaluate({}) is False
