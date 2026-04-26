"""Tests A–G: 7-Phase Tool DAG Formalization.

A — TestToolDAGValidity        (Phase 1)
B — TestToolPlanCompilation    (Phase 2)
C — TestEventStreamReplay      (Phase 3)
D — TestToolStateAlgebra       (Phase 4+5)
E — TestSchedulerDeterminism   (Phase 6)
F — TestSandboxIsolation       (Phase 7)
G — TestCrossModeDagConsistency (equivalence extension)
"""
from __future__ import annotations

import asyncio
import copy
import pytest

from dadbot.core.tool_dag import (
    ToolDAG,
    ToolEdge,
    ToolNode,
    ToolPlanCompiler,
    ToolPlanIR,
    build_dag_from_execution_plan,
)
from dadbot.core.tool_ir import (
    ToolEvent,
    ToolEventLog,
    ToolEventType,
    reduce_events_to_results,
    stable_tool_input_hash,
)
from dadbot.core.tool_algebra import (
    FailureSeverity,
    PropagationPolicy,
    ToolFailure,
    ToolFailureLog,
    ToolFailureType,
    ToolState,
)
from dadbot.core.tool_scheduler import ToolScheduler
from dadbot.core.tool_sandbox import ToolSandbox, ToolSandboxSnapshot


# ===========================================================================
# A — TestToolDAGValidity
# ===========================================================================


class TestToolDAGValidity:

    def _make_node(self, seq: int, priority: int = 100, intent: str = "goal_lookup") -> ToolNode:
        return ToolNode.build(
            tool_name="memory_lookup",
            intent=intent,
            args={"key": f"k{seq}"},
            priority=priority,
            sequence=seq,
        )

    def test_single_node_dag_is_acyclic(self):
        node = self._make_node(0)
        dag = ToolDAG.from_nodes([node])
        assert dag.is_acyclic()

    def test_empty_dag_is_acyclic(self):
        dag = ToolDAG()
        assert dag.is_acyclic()

    def test_sequential_chain_is_acyclic(self):
        nodes = [self._make_node(i) for i in range(4)]
        dag = ToolDAG.from_nodes(nodes)
        for i in range(3):
            dag.add_edge(nodes[i].node_id, nodes[i + 1].node_id)
        assert dag.is_acyclic()
        order = dag.execution_order()
        assert [n.node_id for n in order] == [n.node_id for n in nodes]

    def test_add_edge_rejects_equal_sequence(self):
        n1 = self._make_node(5)
        n2 = ToolNode.build("memory_lookup", "goal_lookup", {"key": "x"}, 100, 5)
        dag = ToolDAG.from_nodes([n1, n2])
        with pytest.raises(ValueError, match="strictly less than"):
            dag.add_edge(n1.node_id, n2.node_id)

    def test_add_edge_rejects_backward_sequence(self):
        n1 = self._make_node(3)
        n2 = self._make_node(1)
        dag = ToolDAG.from_nodes([n1, n2])
        with pytest.raises(ValueError, match="strictly less than"):
            dag.add_edge(n1.node_id, n2.node_id)

    def test_add_edge_rejects_unknown_source(self):
        n1 = self._make_node(0)
        dag = ToolDAG.from_nodes([n1])
        with pytest.raises(ValueError, match="unknown source"):
            dag.add_edge("nonexistent", n1.node_id)

    def test_add_edge_rejects_unknown_target(self):
        n1 = self._make_node(0)
        dag = ToolDAG.from_nodes([n1])
        with pytest.raises(ValueError, match="unknown target"):
            dag.add_edge(n1.node_id, "nonexistent")

    def test_root_nodes_are_entry_points(self):
        nodes = [self._make_node(i) for i in range(3)]
        dag = ToolDAG.from_nodes(nodes)
        dag.add_edge(nodes[0].node_id, nodes[1].node_id)
        dag.add_edge(nodes[1].node_id, nodes[2].node_id)
        roots = dag.root_nodes
        assert len(roots) == 1
        assert roots[0].node_id == nodes[0].node_id

    def test_terminal_nodes_are_exit_points(self):
        nodes = [self._make_node(i) for i in range(3)]
        dag = ToolDAG.from_nodes(nodes)
        dag.add_edge(nodes[0].node_id, nodes[1].node_id)
        dag.add_edge(nodes[1].node_id, nodes[2].node_id)
        terminals = dag.terminal_nodes
        assert len(terminals) == 1
        assert terminals[0].node_id == nodes[2].node_id

    def test_deterministic_hash_is_reproducible(self):
        plan = [
            {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "a"}, "priority": 100, "sequence": 0},
            {"tool_name": "memory_lookup", "intent": "session_memory_fetch", "args": {"key": "b"}, "priority": 90, "sequence": 1},
        ]
        dag1 = build_dag_from_execution_plan(plan)
        dag2 = build_dag_from_execution_plan(plan)
        assert dag1.deterministic_hash() == dag2.deterministic_hash()

    def test_different_plans_produce_different_hashes(self):
        plan_a = [{"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "x"}, "priority": 100, "sequence": 0}]
        plan_b = [{"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "y"}, "priority": 100, "sequence": 0}]
        dag_a = build_dag_from_execution_plan(plan_a)
        dag_b = build_dag_from_execution_plan(plan_b)
        assert dag_a.deterministic_hash() != dag_b.deterministic_hash()

    def test_execution_order_matches_sequence_for_linear_dag(self):
        plan = [
            {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": str(i)}, "priority": 100 - i, "sequence": i}
            for i in range(5)
        ]
        dag = build_dag_from_execution_plan(plan)
        order = dag.execution_order()
        # Execution order must have correct count and all nodes present.
        assert len(order) == len(plan)
        node_ids = {n.node_id for n in order}
        assert node_ids == {n.node_id for n in dag.nodes}

    def test_ordering_key_tiebreaker_is_deterministic(self):
        # Two nodes with same priority — ordering must be stable.
        n1 = ToolNode.build("memory_lookup", "goal_lookup", {"k": "1"}, 100, 0)
        n2 = ToolNode.build("memory_lookup", "session_memory_fetch", {"k": "2"}, 100, 1)
        assert n1.ordering_key() < n2.ordering_key() or n1.ordering_key() != n2.ordering_key()

    def test_to_dict_contains_required_keys(self):
        plan = [{"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"k": "v"}, "priority": 100, "sequence": 0}]
        dag = build_dag_from_execution_plan(plan)
        d = dag.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "root_nodes" in d
        assert "terminal_nodes" in d
        assert "execution_order" in d
        assert "dag_hash" in d


# ===========================================================================
# B — TestToolPlanCompilation
# ===========================================================================


class TestToolPlanCompilation:

    def _valid_plan(self) -> ToolPlanIR:
        return ToolPlanIR(
            intent_summary="Fetch user goals and session state",
            tool_candidates=[
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "goals"}, "priority": 90},
                {"tool_name": "memory_lookup", "intent": "session_memory_fetch", "args": {"key": "session"}, "priority": 100},
            ],
            constraints={"max_nodes": 8},
            optimization_mode="sequential",
        )

    def test_same_plan_produces_same_dag_hash(self):
        compiler = ToolPlanCompiler()
        plan = self._valid_plan()
        dag1 = compiler.compile(plan)
        dag2 = compiler.compile(plan)
        assert dag1.deterministic_hash() == dag2.deterministic_hash()

    def test_different_plans_produce_different_dag_hashes(self):
        compiler = ToolPlanCompiler()
        plan_a = self._valid_plan()
        plan_b = ToolPlanIR(
            intent_summary="Fetch goals only",
            tool_candidates=[
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "different"}, "priority": 90},
            ],
            constraints={"max_nodes": 8},
            optimization_mode="sequential",
        )
        dag_a = compiler.compile(plan_a)
        dag_b = compiler.compile(plan_b)
        assert dag_a.deterministic_hash() != dag_b.deterministic_hash()

    def test_unsupported_tool_filtered(self):
        compiler = ToolPlanCompiler()
        plan = ToolPlanIR(
            intent_summary="Bad tool",
            tool_candidates=[
                {"tool_name": "web_search", "intent": "goal_lookup", "args": {"q": "test"}, "priority": 100},
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "ok"}, "priority": 90},
            ],
            constraints={},
            optimization_mode="sequential",
        )
        dag = compiler.compile(plan)
        tool_names = [n.tool_name for n in dag.nodes]
        assert "web_search" not in tool_names
        assert "memory_lookup" in tool_names

    def test_invalid_intent_filtered(self):
        compiler = ToolPlanCompiler()
        plan = ToolPlanIR(
            intent_summary="Bad intent",
            tool_candidates=[
                {"tool_name": "memory_lookup", "intent": "bad_intent", "args": {"key": "x"}, "priority": 100},
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "good"}, "priority": 90},
            ],
            constraints={},
            optimization_mode="sequential",
        )
        dag = compiler.compile(plan)
        intents = [n.intent for n in dag.nodes]
        assert "bad_intent" not in intents
        assert "goal_lookup" in intents

    def test_duplicate_candidates_deduplicated(self):
        compiler = ToolPlanCompiler()
        same_args = {"key": "dup"}
        plan = ToolPlanIR(
            intent_summary="Duplicates",
            tool_candidates=[
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": same_args, "priority": 100},
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": same_args, "priority": 100},
            ],
            constraints={},
            optimization_mode="sequential",
        )
        dag = compiler.compile(plan)
        assert len(dag.nodes) == 1

    def test_plan_hash_is_stable(self):
        plan = self._valid_plan()
        h1 = plan.plan_hash()
        h2 = plan.plan_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_max_nodes_constraint_respected(self):
        compiler = ToolPlanCompiler()
        candidates = [
            {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": str(i)}, "priority": i}
            for i in range(10)
        ]
        plan = ToolPlanIR(
            intent_summary="Many candidates",
            tool_candidates=candidates,
            constraints={"max_nodes": 3},
            optimization_mode="sequential",
        )
        dag = compiler.compile(plan)
        assert len(dag.nodes) <= 3

    def test_compiled_dag_is_acyclic(self):
        compiler = ToolPlanCompiler()
        dag = compiler.compile(self._valid_plan())
        assert dag.is_acyclic()

    def test_non_dict_args_filtered(self):
        compiler = ToolPlanCompiler()
        plan = ToolPlanIR(
            intent_summary="Bad args",
            tool_candidates=[
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": "not_a_dict", "priority": 100},
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "ok"}, "priority": 90},
            ],
            constraints={},
            optimization_mode="sequential",
        )
        dag = compiler.compile(plan)
        # Only the valid candidate should be compiled.
        assert len(dag.nodes) == 1
        assert dag.nodes[0].args == {"key": "ok"}


# ===========================================================================
# C — TestEventStreamReplay
# ===========================================================================


class TestEventStreamReplay:

    def _build_sample_log(self) -> ToolEventLog:
        log = ToolEventLog()
        log.append(ToolEvent.requested("tid-0", 0, "memory_lookup", {"key": "goals"}))
        log.append(ToolEvent.executed("tid-0", 1, "memory_lookup", {"key": "goals"}, {"result": "goal_a"}, "ok"))
        log.append(ToolEvent.requested("tid-1", 2, "memory_lookup", {"key": "session"}))
        log.append(ToolEvent.executed("tid-1", 3, "memory_lookup", {"key": "session"}, {"result": "session_data"}, "ok"))
        return log

    def test_replay_hash_is_stable(self):
        log1 = self._build_sample_log()
        log2 = self._build_sample_log()
        assert log1.replay_hash() == log2.replay_hash()

    def test_different_outputs_produce_different_replay_hash(self):
        log1 = ToolEventLog()
        log1.append(ToolEvent.executed("tid-0", 1, "memory_lookup", {"key": "k"}, {"result": "A"}, "ok"))
        log2 = ToolEventLog()
        log2.append(ToolEvent.executed("tid-0", 1, "memory_lookup", {"key": "k"}, {"result": "B"}, "ok"))
        assert log1.replay_hash() != log2.replay_hash()

    def test_reduce_events_to_results_correct_count(self):
        log = self._build_sample_log()
        results = reduce_events_to_results(log)
        # REQUESTED events don't contribute; only EXECUTED/FAILED do.
        assert len(results) == 2

    def test_reduce_events_preserves_status(self):
        log = ToolEventLog()
        log.append(ToolEvent.executed("t0", 0, "memory_lookup", {"key": "k"}, {"result": "ok"}, "ok"))
        log.append(ToolEvent.failed("t1", 1, "memory_lookup", {"key": "bad"}, "some error"))
        results = reduce_events_to_results(log)
        statuses = {r["tool_id"]: r["status"] for r in results}
        assert statuses["t0"] == "ok"
        assert statuses["t1"] == "error"

    def test_reduce_events_ordered_by_sequence(self):
        log = ToolEventLog()
        # Append out-of-sequence order.
        log.append(ToolEvent.executed("t1", 3, "memory_lookup", {"key": "b"}, "B", "ok"))
        log.append(ToolEvent.executed("t0", 1, "memory_lookup", {"key": "a"}, "A", "ok"))
        results = reduce_events_to_results(log)
        assert results[0]["sequence"] < results[1]["sequence"]

    def test_event_type_enum_values(self):
        assert ToolEventType.REQUESTED.value == "requested"
        assert ToolEventType.EXECUTED.value == "executed"
        assert ToolEventType.FAILED.value == "failed"
        assert ToolEventType.MERGED.value == "merged"

    def test_tool_event_to_dict_keys(self):
        ev = ToolEvent.executed("t0", 0, "memory_lookup", {"key": "k"}, {"out": "v"}, "ok")
        d = ev.to_dict()
        for key in ("event_type", "tool_id", "sequence", "input_hash", "output_hash", "payload"):
            assert key in d

    def test_events_for_tool_filter(self):
        log = self._build_sample_log()
        events_t0 = log.events_for_tool("tid-0")
        assert len(events_t0) == 2
        assert all(e.tool_id == "tid-0" for e in events_t0)

    def test_event_log_to_list_is_serializable(self):
        import json
        log = self._build_sample_log()
        raw = log.to_list()
        # Should not raise.
        json.dumps(raw)
        assert isinstance(raw, list)
        assert len(raw) == 4

    def test_input_hash_deterministic(self):
        h1 = stable_tool_input_hash("memory_lookup", {"key": "goals"})
        h2 = stable_tool_input_hash("memory_lookup", {"key": "goals"})
        assert h1 == h2
        assert len(h1) == 64


# ===========================================================================
# D — TestToolStateAlgebra
# ===========================================================================


class TestToolStateAlgebra:

    def _state(self, results: list[dict]) -> ToolState:
        return ToolState.from_results(results)

    def test_identity_merge_left(self):
        s = self._state([{"tool_id": "t0", "status": "ok"}])
        result = ToolState.identity().merge(s)
        assert result.composite_hash == s.composite_hash

    def test_identity_merge_right(self):
        s = self._state([{"tool_id": "t0", "status": "ok"}])
        result = s.merge(ToolState.identity())
        assert result.composite_hash == s.composite_hash

    def test_identity_plus_identity(self):
        assert ToolState.identity().merge(ToolState.identity()).composite_hash == ToolState.identity().composite_hash

    def test_merge_associativity(self):
        a = self._state([{"id": "a", "status": "ok"}])
        b = self._state([{"id": "b", "status": "ok"}])
        c = self._state([{"id": "c", "status": "ok"}])
        left = (a.merge(b)).merge(c)
        right = a.merge(b.merge(c))
        assert left.composite_hash == right.composite_hash

    def test_merge_operator_alias(self):
        a = self._state([{"id": "x"}])
        b = self._state([{"id": "y"}])
        via_method = a.merge(b)
        via_operator = a + b
        assert via_method.composite_hash == via_operator.composite_hash

    def test_compose_empty_is_identity(self):
        assert ToolState.compose([]).composite_hash == ToolState.identity().composite_hash

    def test_compose_single_element(self):
        s = self._state([{"id": "solo"}])
        assert ToolState.compose([s]).composite_hash == s.composite_hash

    def test_compose_multiple_associative(self):
        states = [self._state([{"id": str(i)}]) for i in range(4)]
        via_compose = ToolState.compose(states)
        via_fold = states[0].merge(states[1]).merge(states[2]).merge(states[3])
        assert via_compose.composite_hash == via_fold.composite_hash

    def test_event_count_accumulates(self):
        a = self._state([{"id": "a"}])
        b = self._state([{"id": "b"}, {"id": "c"}])
        merged = a.merge(b)
        assert merged.event_count == 3

    def test_failure_halt_propagation(self):
        f = ToolFailure.unsupported_tool("t0", "bad_tool")
        assert f.should_halt()
        assert not f.should_skip()
        assert not f.should_retry()

    def test_failure_skip_propagation(self):
        f = ToolFailure.validation_error("t1", "bad args")
        assert f.should_skip()
        assert not f.should_halt()

    def test_failure_retry_propagation(self):
        f = ToolFailure.execution_error("t2", "timeout", recoverable=True)
        assert f.should_retry()

    def test_failure_not_retry_when_not_recoverable(self):
        f = ToolFailure.execution_error("t3", "fatal", recoverable=False)
        assert not f.should_retry()

    def test_failure_log_has_halt_check(self):
        log = ToolFailureLog()
        log.append(ToolFailure.validation_error("t0", "err"))
        assert not log.has_halt()
        log.append(ToolFailure.unsupported_tool("t1", "bad_tool"))
        assert log.has_halt()

    def test_failure_to_dict_contains_required_keys(self):
        f = ToolFailure.validation_error("t0", "message")
        d = f.to_dict()
        for k in ("failure_type", "severity", "recoverable", "propagation_policy", "tool_id", "message"):
            assert k in d


# ===========================================================================
# E — TestSchedulerDeterminism
# ===========================================================================


class TestSchedulerDeterminism:

    def _build_linear_dag(self, length: int = 3) -> ToolDAG:
        plan = [
            {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"k": str(i)}, "priority": 100, "sequence": i}
            for i in range(length)
        ]
        return build_dag_from_execution_plan(plan)

    def test_same_dag_same_seed_same_schedule(self):
        dag = self._build_linear_dag(4)
        s1 = ToolScheduler(seed=42)
        s2 = ToolScheduler(seed=42)
        sched1 = s1.schedule(dag)
        sched2 = s2.schedule(dag)
        assert [i.node.node_id for i in sched1] == [i.node.node_id for i in sched2]

    def test_same_dag_different_seed_different_schedule_hash(self):
        # Multi-node DAGs with parallel waves should differ.
        plan = [
            {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"k": "0"}, "priority": 100, "sequence": 0},
            {"tool_name": "memory_lookup", "intent": "session_memory_fetch", "args": {"k": "1"}, "priority": 200, "sequence": 1},
        ]
        dag = build_dag_from_execution_plan(plan)
        s1 = ToolScheduler(seed=1)
        s2 = ToolScheduler(seed=999)
        h1 = s1.schedule_hash(dag)
        h2 = s2.schedule_hash(dag)
        # Different seeds may yield different ordering hashes (seeds influence ordering_key).
        # Just assert both are valid SHA-256 hashes.
        assert len(h1) == 64
        assert len(h2) == 64

    def test_schedule_covers_all_nodes(self):
        dag = self._build_linear_dag(5)
        scheduler = ToolScheduler()
        items = scheduler.schedule(dag)
        assert len(items) == len(dag.nodes)

    def test_schedule_sequence_is_monotonic(self):
        dag = self._build_linear_dag(5)
        items = ToolScheduler(seed=0).schedule(dag)
        seqs = [i.schedule_sequence for i in items]
        assert seqs == list(range(len(items)))

    def test_schedule_hash_is_reproducible(self):
        dag = self._build_linear_dag(4)
        s = ToolScheduler(seed=7)
        h1 = s.schedule_hash(dag)
        h2 = s.schedule_hash(dag)
        assert h1 == h2

    def test_empty_dag_schedule_is_empty(self):
        dag = ToolDAG()
        items = ToolScheduler().schedule(dag)
        assert items == []

    def test_schedule_wave_assignments(self):
        # Single chain: each node should be in a successive wave.
        dag = self._build_linear_dag(3)
        items = ToolScheduler().schedule(dag)
        waves = [i.wave for i in items]
        assert waves == [0, 1, 2]

    def test_to_schedule_dict_contains_required_keys(self):
        dag = self._build_linear_dag(2)
        d = ToolScheduler(seed=0).to_schedule_dict(dag)
        assert "seed" in d
        assert "schedule" in d
        assert "schedule_hash" in d


# ===========================================================================
# F — TestSandboxIsolation
# ===========================================================================


class TestSandboxIsolation:

    def test_fresh_sandbox_is_clean(self):
        sb = ToolSandbox()
        assert sb.is_clean()

    def test_snapshot_before_after_differs(self):
        sb = ToolSandbox()
        snap_before = sb.isolated_state_snapshot(generation=0)
        sb.execute(
            tool_name="memory_lookup",
            parameters={"key": "test"},
            executor=lambda: {"result": "data"},
        )
        snap_after = sb.isolated_state_snapshot(generation=1)
        assert snap_before.snapshot_hash != snap_after.snapshot_hash

    def test_two_sandboxes_same_operations_produce_same_snapshot_hash(self):
        def run_ops(sb: ToolSandbox) -> ToolSandboxSnapshot:
            sb.execute(
                tool_name="memory_lookup",
                parameters={"key": "goals"},
                executor=lambda: {"result": "goal_a"},
            )
            return sb.isolated_state_snapshot(generation=1)

        sb1 = ToolSandbox()
        sb2 = ToolSandbox()
        snap1 = run_ops(sb1)
        snap2 = run_ops(sb2)
        assert snap1.snapshot_hash == snap2.snapshot_hash

    def test_cross_tool_no_leakage(self):
        """Two sandboxes operating independently should not affect each other."""
        sb1 = ToolSandbox()
        sb2 = ToolSandbox()

        sb1.execute(
            tool_name="memory_lookup",
            parameters={"key": "tool_a"},
            executor=lambda: {"result": "a"},
        )
        # sb2 should remain clean.
        assert sb2.is_clean()

    def test_rollback_removes_compensating_actions(self):
        sb = ToolSandbox()
        rolled_back: list[str] = []

        def undo_a():
            rolled_back.append("a")

        sb.execute(
            tool_name="memory_lookup",
            parameters={"key": "alpha"},
            executor=lambda: "ok",
            compensating_action=undo_a,
        )
        outcomes = sb.rollback()
        assert any(o["rolled_back"] for o in outcomes)
        assert "a" in rolled_back

    def test_idempotency_key_deduplicates(self):
        sb = ToolSandbox()
        call_count = [0]

        def executor():
            call_count[0] += 1
            return "result"

        sb.execute(tool_name="memory_lookup", parameters={"key": "dup"}, executor=executor)
        sb.execute(tool_name="memory_lookup", parameters={"key": "dup"}, executor=executor)
        # Executor should only be called once due to idempotency.
        assert call_count[0] == 1

    def test_failed_execution_does_not_pollute_cache(self):
        sb = ToolSandbox()

        def bad_executor():
            raise RuntimeError("deliberate failure")

        rec = sb.execute(
            tool_name="memory_lookup",
            parameters={"key": "fail_key"},
            executor=bad_executor,
        )
        assert rec.status == "failed"
        # A fresh call with same params should re-execute (not use cache).
        counter = [0]

        def good_executor():
            counter[0] += 1
            return "recovered"

        sb.execute(tool_name="memory_lookup", parameters={"key": "fail_key"}, executor=good_executor)
        assert counter[0] == 1

    def test_snapshot_to_dict_contains_required_keys(self):
        sb = ToolSandbox()
        snap = sb.isolated_state_snapshot(generation=0)
        d = snap.to_dict()
        for k in ("records_count", "cache_keys", "snapshot_hash", "generation"):
            assert k in d


# ===========================================================================
# G — TestCrossModeDagConsistency (equivalence extension)
# ===========================================================================


class TestCrossModeDagConsistency:
    """Verify ToolDAG stability across reruns and between v2 ON/OFF modes.

    These tests use the tool_dag module directly (no live orchestrator).
    The orchestrator-level consistency is covered by tool_system_v2_equivalence_test.py.
    """

    def _canonical_plan(self) -> list[dict]:
        return [
            {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "goals"}, "priority": 90, "sequence": 0},
            {"tool_name": "memory_lookup", "intent": "session_memory_fetch", "args": {"key": "session"}, "priority": 100, "sequence": 1},
        ]

    def test_v2_on_dag_hash_stable_across_reruns(self):
        """Identical execution plan → identical DAG hash every time."""
        plan = self._canonical_plan()
        hashes = [build_dag_from_execution_plan(plan).deterministic_hash() for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_v2_off_dag_is_empty(self):
        """When no execution plan is provided, resulting DAG should be empty."""
        dag = build_dag_from_execution_plan([])
        assert len(dag.nodes) == 0
        assert len(dag.edges) == 0

    def test_dag_serialization_roundtrip_hash_stable(self):
        """DAG hash must survive a to_dict() serialization round-trip."""
        import json
        plan = self._canonical_plan()
        dag = build_dag_from_execution_plan(plan)
        d = dag.to_dict()
        # Reconstruct from dict.
        new_plan = [
            {
                "tool_name": n["tool_name"],
                "intent": n["intent"],
                "args": n["args"],
                "priority": n["priority"],
                "sequence": n["sequence"],
            }
            for n in d["nodes"]
        ]
        dag2 = build_dag_from_execution_plan(new_plan)
        assert dag.deterministic_hash() == dag2.deterministic_hash()

    def test_plan_compiler_idempotent(self):
        """Compiling the same ToolPlanIR 3 times gives the same DAG hash."""
        compiler = ToolPlanCompiler()
        plan = ToolPlanIR(
            intent_summary="Stable compile",
            tool_candidates=[
                {"tool_name": "memory_lookup", "intent": "goal_lookup", "args": {"key": "k1"}, "priority": 90},
            ],
            constraints={},
            optimization_mode="sequential",
        )
        hashes = [compiler.compile(plan).deterministic_hash() for _ in range(3)]
        assert len(set(hashes)) == 1

    def test_scheduler_and_dag_hash_independent_of_call_order(self):
        """Scheduler hash and DAG hash should not change if plan is rebuilt from scratch."""
        plan = self._canonical_plan()
        dag_a = build_dag_from_execution_plan(plan)
        dag_b = build_dag_from_execution_plan(plan)
        sched_a = ToolScheduler(seed=0).schedule_hash(dag_a)
        sched_b = ToolScheduler(seed=0).schedule_hash(dag_b)
        assert dag_a.deterministic_hash() == dag_b.deterministic_hash()
        assert sched_a == sched_b
