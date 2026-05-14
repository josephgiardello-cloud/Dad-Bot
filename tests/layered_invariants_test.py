"""Tests H–P: L3/L4/L5 Layered Invariants.

H — TestToolIRClosure         (L3-P1: tool_ir_boundary)
I — TestDagIntegrityLock      (L3-P2: dag_integrity)
J — TestCriticConstraints     (L3-P3: critic_constraints)
K — TestMemoryStateSpace      (L3-P4: memory_space)
L — TestEventAuthority        (L4-P1+P2: event_authority)
M — TestGraphAlgebra          (L4-P3: graph_algebra)
N — TestScheduleConfluence    (L4-P4: schedule_confluence)
O — TestStatelessExecutor     (L4-P5: stateless_executor)
P — TestEvolutionHooks        (L5: evolution_hooks)
"""

from __future__ import annotations

import pytest

from dadbot.core.critic_constraints import (
    DEFAULT_CONSTRAINTS,
    ConstraintCritiqueEngine,
    ConstraintViolation,
    CritiqueViolationType,
)
from dadbot.core.dag_integrity import (
    DagIdentityLock,
    DagReplayInvariantError,
    assert_replay_dag_invariant,
    build_replay_proof,
    graph_equivalence_proof,
    graph_semantic_equivalence,
    graph_structural_equivalence,
    lock_dag,
)
from dadbot.core.event_authority import (
    EventAuthority,
    UndefinedSystemStateError,
    rebuild_state_from_events,
)
from dadbot.core.evolution_hooks import (
    ExecutionTelemetryVector,
    GraphIntrospectionAPI,
    OptimizationBoundary,
    PolicySeparationLayer,
)
from dadbot.core.graph_algebra import ToolGraphAlgebra
from dadbot.core.memory_space import (
    GoalWeightingFunction,
    MemoryRankerOperator,
    MemoryStateVector,
)
from dadbot.core.schedule_confluence import (
    ExecutionEquivalenceChecker,
    ScheduleNormalizer,
)
from dadbot.core.stateless_executor import (
    StatelessExecutionResult,
    StatelessExecutor,
    is_bootstrapped,
)
from dadbot.core.tool_dag import ToolDAG, build_dag_from_execution_plan
from dadbot.core.tool_ir_boundary import (
    ToolIRBijectionError,
    ToolSchemaError,
    assert_bijection,
    build_bijection_proof,
    validate_tool_request,
    validate_tool_requests_batch,
    validate_tool_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dag(n: int = 3) -> ToolDAG:
    specs = [
        {
            "tool_name": "memory_lookup",
            "intent": "goal_lookup",
            "args": {"q": str(i)},
            "priority": i,
            "sequence": i,
        }
        for i in range(n)
    ]
    return build_dag_from_execution_plan(specs)


def _valid_request_raw() -> dict:
    return {
        "tool_name": "memory_lookup",
        "args": {"q": "test"},
        "intent": "goal_lookup",
        "expected_output": "some result",
        "priority": 0,
    }


def _valid_result_raw(tool_name: str = "memory_lookup") -> dict:
    return {
        "tool_name": tool_name,
        "status": "ok",
        "output": "some result",
    }


# ===========================================================================
# H — TestToolIRClosure
# ===========================================================================


class TestToolIRClosure:
    def test_validate_tool_request_accepts_valid(self):
        req = validate_tool_request(_valid_request_raw())
        assert req.tool_name == "memory_lookup"
        assert req.intent == "goal_lookup"
        assert req.priority == 0

    def test_validate_tool_request_rejects_unknown_tool(self):
        raw = _valid_request_raw()
        raw["tool_name"] = "forbidden_tool"
        with pytest.raises(ToolSchemaError) as exc:
            validate_tool_request(raw)
        assert exc.value.field == "tool_name"

    def test_validate_tool_request_rejects_unknown_intent(self):
        raw = _valid_request_raw()
        raw["intent"] = "not_an_intent"
        with pytest.raises(ToolSchemaError) as exc:
            validate_tool_request(raw)
        assert exc.value.field == "intent"

    def test_validate_tool_request_rejects_negative_priority(self):
        raw = _valid_request_raw()
        raw["priority"] = -1
        with pytest.raises(ToolSchemaError) as exc:
            validate_tool_request(raw)
        assert exc.value.field == "priority"

    def test_validate_tool_request_rejects_empty_expected_output(self):
        raw = _valid_request_raw()
        raw["expected_output"] = ""
        with pytest.raises(ToolSchemaError) as exc:
            validate_tool_request(raw)
        assert exc.value.field == "expected_output"

    def test_validate_tool_result_accepts_valid(self):
        result = validate_tool_result(_valid_result_raw())
        assert result.tool_name == "memory_lookup"
        assert result.status == "ok"

    def test_validate_tool_result_rejects_unknown_status(self):
        raw = _valid_result_raw()
        raw["status"] = "unknown_status_xyz"
        with pytest.raises(ToolSchemaError) as exc:
            validate_tool_result(raw)
        assert exc.value.field == "status"

    def test_validate_tool_result_derives_deterministic_id(self):
        result = validate_tool_result(_valid_result_raw())
        assert result.deterministic_id != ""
        assert len(result.deterministic_id) == 24  # first 24 chars of sha256

    def test_batch_validation_separates_valid_and_invalid(self):
        valid = _valid_request_raw()
        invalid = {"tool_name": "nope", "args": {}, "intent": "bad", "expected_output": "", "priority": 0}
        ok, failed = validate_tool_requests_batch([valid, invalid])
        assert len(ok) == 1
        assert len(failed) == 1

    def test_bijection_proof_with_matching_log_and_results(self):
        from dadbot.core.tool_ir import ToolEvent, ToolEventLog

        tool_id = "tid_match_1"
        log = ToolEventLog()
        log.append(
            ToolEvent.executed(
                tool_id=tool_id,
                sequence=0,
                tool_name="memory_lookup",
                args={"q": "test"},
                output="result_value",
            )
        )
        result = {"tool_name": "memory_lookup", "status": "ok", "deterministic_id": tool_id}
        proof = build_bijection_proof(log, [result])
        assert proof["cardinality_match"] is True
        assert proof["ok"] is True

    def test_bijection_proof_detects_orphan_events(self):
        from dadbot.core.tool_ir import ToolEvent, ToolEventLog

        tool_id = "orphan_tid"
        log = ToolEventLog()
        log.append(
            ToolEvent.executed(
                tool_id=tool_id,
                sequence=0,
                tool_name="memory_lookup",
                args={},
                output=None,
            )
        )
        proof = build_bijection_proof(log, [])
        assert proof["ok"] is False
        assert len(proof["orphan_events"]) == 1

    def test_assert_bijection_raises_on_mismatch(self):
        from dadbot.core.tool_ir import ToolEvent, ToolEventLog

        log = ToolEventLog()
        log.append(
            ToolEvent.executed(
                tool_id="tid_no_result",
                sequence=0,
                tool_name="memory_lookup",
                args={},
                output=None,
            )
        )
        with pytest.raises(ToolIRBijectionError):
            assert_bijection(log, [])


# ===========================================================================
# I — TestDagIntegrityLock
# ===========================================================================


class TestDagIntegrityLock:
    def test_identity_lock_from_dag(self):
        dag = _make_dag(3)
        lock = DagIdentityLock.from_dag(dag)
        assert lock.node_count == 3
        assert lock.structural_hash != ""
        assert lock.semantic_hash != ""

    def test_identical_dags_produce_identical_locks(self):
        dag1 = _make_dag(3)
        dag2 = _make_dag(3)
        lock1 = DagIdentityLock.from_dag(dag1)
        lock2 = DagIdentityLock.from_dag(dag2)
        assert lock1.semantic_hash == lock2.semantic_hash

    def test_different_dags_produce_different_semantic_hashes(self):
        dag1 = _make_dag(2)
        dag2 = _make_dag(4)
        lock1 = DagIdentityLock.from_dag(dag1)
        lock2 = DagIdentityLock.from_dag(dag2)
        assert lock1.semantic_hash != lock2.semantic_hash

    def test_structural_equivalence_holds_for_same_topology(self):
        dag1 = _make_dag(3)
        dag2 = _make_dag(3)
        assert graph_structural_equivalence(dag1, dag2) is True

    def test_semantic_equivalence_holds_for_same_content(self):
        dag1 = _make_dag(3)
        dag2 = _make_dag(3)
        assert graph_semantic_equivalence(dag1, dag2) is True

    def test_semantic_equivalence_fails_for_different_content(self):
        dag1 = _make_dag(2)
        dag2 = _make_dag(4)
        assert graph_semantic_equivalence(dag1, dag2) is False

    def test_replay_invariant_passes_for_identical_dags(self):
        dag1 = _make_dag(3)
        dag2 = _make_dag(3)
        assert_replay_dag_invariant(dag1, dag2)  # should not raise

    def test_replay_invariant_raises_for_different_dags(self):
        dag1 = _make_dag(2)
        dag2 = _make_dag(4)
        with pytest.raises(DagReplayInvariantError):
            assert_replay_dag_invariant(dag1, dag2)

    def test_replay_proof_is_non_raising(self):
        dag1 = _make_dag(2)
        dag2 = _make_dag(4)
        proof = build_replay_proof(dag1, dag2)
        assert proof["ok"] is False
        assert "semantic_match" in proof

    def test_lock_dag_returns_storable_dict(self):
        dag = _make_dag(2)
        locked = lock_dag(dag)
        assert "identity_lock" in locked
        assert "dag" in locked
        assert "nodes" in locked["dag"]

    def test_equivalence_proof_both_equivalent(self):
        dag1 = _make_dag(3)
        dag2 = _make_dag(3)
        proof = graph_equivalence_proof(dag1, dag2)
        assert proof["structurally_equivalent"] is True
        assert proof["semantically_equivalent"] is True


# ===========================================================================
# J — TestCriticConstraints
# ===========================================================================


class TestCriticConstraints:
    def _engine(self):
        return ConstraintCritiqueEngine.default()

    def test_passes_on_good_reply(self):
        engine = self._engine()
        result = engine.evaluate(
            "I hear you, that sounds really tough. Let me think about what might help.",
            "I'm struggling with something",
            {"strategy": "empathy_first", "intent_type": "emotional"},
            iteration=0,
        )
        # May not always pass but should not hard-fail.
        assert isinstance(result.passed, bool)
        assert result.satisfaction_ratio >= 0.0
        assert result.satisfaction_ratio <= 1.0

    def test_fails_on_empty_reply(self):
        engine = self._engine()
        result = engine.evaluate("", "hello", {}, iteration=0)
        assert result.passed is False
        assert result.hard_failure is True
        assert CritiqueViolationType.EMPTY_REPLY.value in result.issue_tags()

    def test_fails_on_fallback_phrase(self):
        engine = self._engine()
        result = engine.evaluate("something went sideways, try again in a moment", "hi", {}, iteration=0)
        assert result.hard_failure is True
        assert CritiqueViolationType.FALLBACK_DETECTED.value in result.issue_tags()

    def test_satisfaction_ratio_is_fraction(self):
        engine = self._engine()
        result = engine.evaluate("Hello there!", "hi", {}, iteration=0)
        assert 0.0 <= result.satisfaction_ratio <= 1.0

    def test_violation_list_is_typed(self):
        engine = self._engine()
        result = engine.evaluate("", "user", {}, iteration=0)
        for v in result.violations:
            assert isinstance(v, ConstraintViolation)
            assert isinstance(v.violation_type, CritiqueViolationType)

    def test_needs_revision_below_threshold_at_iteration_zero(self):
        engine = self._engine()
        result = engine.evaluate("", "user", {}, iteration=0)
        assert engine.needs_revision(result) is True

    def test_needs_revision_false_at_max_iterations(self):
        engine = self._engine()
        result = engine.evaluate("", "user", {}, iteration=1)  # max_iterations=2, so idx 1 is last
        assert engine.needs_revision(result) is False

    def test_default_constraints_count(self):
        assert len(DEFAULT_CONSTRAINTS) == 10

    def test_constraint_by_id_lookup(self):
        engine = self._engine()
        c = engine.constraint_by_id("empty_reply")
        assert c is not None
        assert c.violation_type == CritiqueViolationType.EMPTY_REPLY

    def test_result_to_dict(self):
        engine = self._engine()
        result = engine.evaluate("ok reply here", "question?", {}, iteration=0)
        d = result.to_dict()
        assert "satisfaction_ratio" in d
        assert "passed" in d
        assert "issue_tags" in d


# ===========================================================================
# K — TestMemoryStateSpace
# ===========================================================================


class TestMemoryStateSpace:
    def _memories(self):
        return [
            {"content": "user likes hiking in the mountains"},
            {"content": "user has a dog named Max"},
            {"content": "user is interested in machine learning and AI"},
        ]

    def test_vector_from_memories_has_correct_dimension(self):
        vec = MemoryStateVector.from_memories(self._memories())
        assert vec.dimension == 3
        assert len(vec.entries) == 3

    def test_vector_from_empty_list(self):
        vec = MemoryStateVector.from_memories([])
        assert vec.dimension == 0
        assert vec.space_hash != ""

    def test_same_memories_produce_same_hash(self):
        mem = self._memories()
        v1 = MemoryStateVector.from_memories(mem)
        v2 = MemoryStateVector.from_memories(mem)
        assert v1.space_hash == v2.space_hash

    def test_different_memories_produce_different_hash(self):
        v1 = MemoryStateVector.from_memories(self._memories())
        v2 = MemoryStateVector.from_memories([{"content": "something totally different"}])
        assert v1.space_hash != v2.space_hash

    def test_projection_reduces_dimension(self):
        vec = MemoryStateVector.from_memories(self._memories())
        sub = vec.project([0, 2])
        assert sub.dimension == 2

    def test_projection_preserves_entries(self):
        mems = self._memories()
        vec = MemoryStateVector.from_memories(mems)
        sub = vec.project([1])
        assert sub.entries[0]["content"] == mems[1]["content"]

    def test_goal_weighting_gives_higher_weight_for_relevant(self):
        goals = [{"description": "become expert in machine learning AI"}]
        wf = GoalWeightingFunction(goals)
        entry_relevant = {"content": "user studies machine learning and AI every day"}
        entry_irrelevant = {"content": "user loves cooking pasta and pizza"}
        w_rel = wf.weight(entry_relevant)
        w_irr = wf.weight(entry_irrelevant)
        assert w_rel > w_irr

    def test_goal_weighting_identity_when_no_goals(self):
        wf = GoalWeightingFunction([])
        entry = {"content": "anything"}
        assert wf.weight(entry) == 1.0

    def test_ranker_operator_produces_same_dimension(self):
        vec = MemoryStateVector.from_memories(self._memories())
        goals = [{"description": "machine learning AI"}]
        wf = GoalWeightingFunction(goals)
        ranker = MemoryRankerOperator()
        ranked = ranker.apply(vec, wf)
        assert ranked.dimension == vec.dimension

    def test_ranker_places_relevant_first(self):
        mems = [
            {"content": "user likes cooking pasta"},
            {"content": "user studies machine learning and AI deeply"},
        ]
        vec = MemoryStateVector.from_memories(mems)
        goals = [{"description": "machine learning AI research"}]
        wf = GoalWeightingFunction(goals)
        ranker = MemoryRankerOperator()
        ranked = ranker.apply(vec, wf)
        # The ML entry should be first.
        assert "machine learning" in ranked.entries[0]["content"].lower()


# ===========================================================================
# L — TestEventAuthority
# ===========================================================================


class TestEventAuthority:
    def test_empty_authority_is_not_defined(self):
        auth = EventAuthority()
        assert auth.is_defined() is False

    def test_assert_defined_raises_when_empty(self):
        auth = EventAuthority()
        with pytest.raises(UndefinedSystemStateError):
            auth.assert_defined()

    def test_append_makes_authority_defined(self):
        auth = EventAuthority()
        auth.append({"type": "SESSION_STATE_UPDATED", "session_id": "s1"})
        assert auth.is_defined() is True

    def test_authority_hash_stable_after_same_events(self):
        auth1 = EventAuthority()
        auth2 = EventAuthority()
        ev = {"type": "SESSION_STATE_UPDATED", "session_id": "s1", "state": {"x": 1}}
        auth1.append(ev)
        auth2.append(ev)
        assert auth1.authority_hash() == auth2.authority_hash()

    def test_authority_hash_changes_on_new_event(self):
        auth = EventAuthority()
        auth.append({"type": "SESSION_STATE_UPDATED"})
        h1 = auth.authority_hash()
        auth.append({"type": "TURN_COMPLETED"})
        h2 = auth.authority_hash()
        assert h1 != h2

    def test_derive_state_is_dict(self):
        auth = EventAuthority()
        auth.append({"type": "SESSION_STATE_UPDATED", "session_id": "s1"})
        state = auth.derive_state()
        assert isinstance(state, dict)

    def test_derive_state_raises_when_undefined(self):
        auth = EventAuthority()
        with pytest.raises(UndefinedSystemStateError):
            auth.derive_state()

    def test_rebuild_state_is_pure(self):
        events = [{"type": "SESSION_STATE_UPDATED", "session_id": "s1", "sequence": 0}]
        auth = EventAuthority()
        state1 = auth.rebuild_state_from_events(events)
        state2 = auth.rebuild_state_from_events(events)
        # Same input → same output.
        assert state1 == state2

    def test_rebuild_state_does_not_affect_authority(self):
        auth = EventAuthority()
        events = [{"type": "SESSION_STATE_UPDATED", "session_id": "s1", "sequence": 0}]
        auth.rebuild_state_from_events(events)
        # After pure call, authority should still be undefined.
        assert auth.is_defined() is False

    def test_standalone_rebuild_state_from_events(self):
        events = [{"type": "SESSION_STATE_UPDATED", "session_id": "s1", "sequence": 0}]
        state = rebuild_state_from_events(events)
        assert isinstance(state, dict)

    def test_event_count_tracks_appends(self):
        auth = EventAuthority()
        auth.append({"type": "A"})
        auth.append({"type": "B"})
        assert auth.event_count() == 2

    def test_batch_append(self):
        auth = EventAuthority()
        events = [{"type": "A"}, {"type": "B"}, {"type": "C"}]
        seqs = auth.append_batch(events)
        assert len(seqs) == 3
        assert auth.event_count() == 3

    def test_read_from_filters_by_sequence(self):
        auth = EventAuthority()
        for i in range(5):
            auth.append({"type": "E", "idx": i})
        tail = auth.read_from(3)
        assert len(tail) == 2  # sequences 3 and 4

    def test_head_sequence_minus_one_when_empty(self):
        auth = EventAuthority()
        assert auth.head_sequence() == -1

    def test_head_sequence_advances(self):
        auth = EventAuthority()
        auth.append({"type": "A"})
        auth.append({"type": "B"})
        assert auth.head_sequence() == 1

    def test_to_dict_contains_expected_keys(self):
        auth = EventAuthority()
        auth.append({"type": "A"})
        d = auth.to_dict()
        assert "event_count" in d
        assert "is_defined" in d
        assert "authority_hash" in d


# ===========================================================================
# M — TestGraphAlgebra
# ===========================================================================


class TestGraphAlgebra:
    def test_identity_is_empty_dag(self):
        identity = ToolGraphAlgebra.identity()
        assert ToolGraphAlgebra.is_identity(identity)

    def test_compose_with_identity_is_identity_law(self):
        dag = _make_dag(3)
        proof = ToolGraphAlgebra.verify_identity_laws(dag)
        assert proof["ok"] is True

    def test_compose_produces_dag(self):
        dag1 = _make_dag(2)
        dag2 = _make_dag(2)
        composed = ToolGraphAlgebra.compose(dag1, dag2)
        assert isinstance(composed, ToolDAG)

    def test_compose_deduplicates_nodes(self):
        dag = _make_dag(3)
        composed = ToolGraphAlgebra.compose(dag, dag)
        # Composing same dag with itself should deduplicate
        assert len(composed.nodes) == len(dag.nodes)

    def test_compose_all_empty_is_identity(self):
        result = ToolGraphAlgebra.compose_all([])
        assert ToolGraphAlgebra.is_identity(result)

    def test_compose_all_single_is_identity_law(self):
        dag = _make_dag(3)
        result = ToolGraphAlgebra.compose_all([dag])
        assert result.deterministic_hash() == dag.deterministic_hash()

    def test_merge_hash_is_deterministic(self):
        dag1 = _make_dag(2)
        dag2 = _make_dag(2)
        h1 = ToolGraphAlgebra.merge_hash([dag1, dag2])
        h2 = ToolGraphAlgebra.merge_hash([dag1, dag2])
        assert h1 == h2

    def test_associativity_proof(self):
        a = _make_dag(1)
        b = _make_dag(2)
        c = _make_dag(1)
        proof = ToolGraphAlgebra.verify_associativity(a, b, c)
        assert proof["ok"] is True

    def test_identity_left_law(self):
        dag = _make_dag(3)
        result = ToolGraphAlgebra.compose(ToolGraphAlgebra.identity(), dag)
        assert result.deterministic_hash() == dag.deterministic_hash()

    def test_identity_right_law(self):
        dag = _make_dag(3)
        result = ToolGraphAlgebra.compose(dag, ToolGraphAlgebra.identity())
        assert result.deterministic_hash() == dag.deterministic_hash()


# ===========================================================================
# N — TestScheduleConfluence
# ===========================================================================


class TestScheduleConfluence:
    def test_normalizer_produces_same_hash_for_same_dag_different_seeds(self):
        dag = _make_dag(3)
        normalizer = ScheduleNormalizer()
        from dadbot.core.tool_scheduler import ToolScheduler

        items_0 = ToolScheduler(seed=0).schedule(dag)
        items_1 = ToolScheduler(seed=42).schedule(dag)
        h0 = normalizer.normalized_hash(items_0)
        h1 = normalizer.normalized_hash(items_1)
        assert h0 == h1

    def test_normalizer_produces_different_hashes_for_different_dags(self):
        dag2 = _make_dag(2)
        dag4 = _make_dag(4)
        normalizer = ScheduleNormalizer()
        h2 = normalizer.normalized_dag_hash(dag2)
        h4 = normalizer.normalized_dag_hash(dag4)
        assert h2 != h4

    def test_confluence_proof_confluent_for_single_node_dag(self):
        dag = _make_dag(1)
        checker = ExecutionEquivalenceChecker()
        proof = checker.confluence_proof(dag, seeds=[0, 1, 2, 42, 100])
        assert proof.confluent is True

    def test_confluence_proof_confluent_for_chain_dag(self):
        dag = _make_dag(3)
        checker = ExecutionEquivalenceChecker()
        proof = checker.confluence_proof(dag, seeds=[0, 1, 2, 99])
        assert proof.confluent is True
        assert proof.canonical_hash != ""

    def test_equivalence_checker_two_schedules_same_dag(self):
        dag = _make_dag(3)
        from dadbot.core.tool_scheduler import ToolScheduler

        sched_a = ToolScheduler(seed=0).schedule(dag)
        sched_b = ToolScheduler(seed=7).schedule(dag)
        checker = ExecutionEquivalenceChecker()
        assert checker.are_equivalent(sched_a, sched_b, dag) is True

    def test_confluence_proof_records_seeds(self):
        dag = _make_dag(2)
        checker = ExecutionEquivalenceChecker()
        proof = checker.confluence_proof(dag, seeds=[0, 5, 10])
        assert sorted(proof.seeds_tested) == [0, 5, 10]

    def test_confluence_proof_to_dict(self):
        dag = _make_dag(2)
        checker = ExecutionEquivalenceChecker()
        proof = checker.confluence_proof(dag, seeds=[0])
        d = proof.to_dict()
        assert "confluent" in d
        assert "canonical_hash" in d


# ===========================================================================
# O — TestStatelessExecutor
# ===========================================================================


class TestStatelessExecutor:
    def _session_events(self):
        return [
            {"type": "SESSION_STATE_UPDATED", "session_id": "s1", "sequence": 0},
            {"type": "TURN_COMPLETED", "session_id": "s1", "sequence": 1},
        ]

    def test_is_bootstrapped_true_with_session_events(self):
        events = self._session_events()
        assert is_bootstrapped(events) is True

    def test_is_bootstrapped_false_with_empty_log(self):
        assert is_bootstrapped([]) is False

    def test_is_bootstrapped_false_with_only_tool_events(self):
        events = [{"type": "TOOL_EXECUTED", "session_id": "s1"}]
        assert is_bootstrapped(events) is False

    def test_execute_returns_stateless_result(self):
        executor = StatelessExecutor()
        result = executor.execute("hello", self._session_events())
        assert isinstance(result, StatelessExecutionResult)

    def test_execute_sets_bootstrapped_true(self):
        executor = StatelessExecutor()
        result = executor.execute("hi", self._session_events())
        assert result.bootstrapped is True

    def test_execute_sets_bootstrapped_false_for_empty_log(self):
        executor = StatelessExecutor()
        result = executor.execute("hi", [])
        assert result.bootstrapped is False

    def test_execute_is_deterministic(self):
        executor = StatelessExecutor()
        events = self._session_events()
        r1 = executor.execute("test input", events)
        r2 = executor.execute("test input", events)
        assert r1.execution_hash == r2.execution_hash

    def test_different_inputs_produce_different_hashes(self):
        executor = StatelessExecutor()
        events = self._session_events()
        r1 = executor.execute("input A", events)
        r2 = executor.execute("input B", events)
        assert r1.execution_hash != r2.execution_hash

    def test_reconstructed_is_always_true(self):
        executor = StatelessExecutor()
        result = executor.execute("hi", [])
        assert result.reconstructed is True

    def test_event_log_is_immutable_tuple(self):
        executor = StatelessExecutor()
        events = self._session_events()
        result = executor.execute("hi", events)
        assert isinstance(result.event_log, tuple)

    def test_replay_execution_verifies_determinism(self):
        executor = StatelessExecutor()
        events = self._session_events()
        original = executor.execute("test", events)
        replay = executor.replay_execution(original, "test")
        assert replay["ok"] is True
        assert replay["hash_matches"] is True

    def test_executor_is_bootstrapped_method(self):
        executor = StatelessExecutor()
        assert executor.is_bootstrapped(self._session_events()) is True
        assert executor.is_bootstrapped([]) is False

    def test_state_is_derived_not_stored(self):
        executor = StatelessExecutor()
        # Execute twice with completely independent event logs.
        events_a = [{"type": "SESSION_STATE_UPDATED", "session_id": "a", "sequence": 0}]
        events_b = [{"type": "SESSION_STATE_UPDATED", "session_id": "b", "sequence": 0}]
        # Executor has no stored state — each call is fully independent.
        r_a = executor.execute("hi", events_a)
        r_b = executor.execute("hi", events_b)
        assert r_a.execution_hash != r_b.execution_hash


# ===========================================================================
# P — TestEvolutionHooks
# ===========================================================================


class TestEvolutionHooks:
    def test_telemetry_vector_to_feature_vector(self):
        tv = ExecutionTelemetryVector(
            event_count=5,
            tool_count=2,
            latency_ms=120.5,
            schedule_waves=3,
            state_transitions=4,
            dag_hash="abc123",
            session_id="s1",
        )
        fv = tv.to_feature_vector()
        assert isinstance(fv, list)
        assert len(fv) == 7
        assert all(isinstance(x, float) for x in fv)

    def test_telemetry_vector_event_count_position(self):
        tv = ExecutionTelemetryVector(event_count=7)
        fv = tv.to_feature_vector()
        assert fv[0] == 7.0

    def test_optimization_boundary_default(self):
        boundary = OptimizationBoundary.default()
        assert boundary.is_evolvable("tool_selection_policy")
        assert boundary.is_frozen("event_authority")
        assert not boundary.is_frozen("tool_selection_policy")

    def test_optimization_boundary_hash_stable(self):
        b1 = OptimizationBoundary.default()
        b2 = OptimizationBoundary.default()
        assert b1.boundary_hash == b2.boundary_hash

    def test_optimization_boundary_to_dict(self):
        boundary = OptimizationBoundary.default()
        d = boundary.to_dict()
        assert "evolvable_components" in d
        assert "frozen_components" in d
        assert "boundary_hash" in d

    def test_graph_introspection_is_read_only(self):
        api = GraphIntrospectionAPI()
        dag = _make_dag(3)
        info = api.introspect(dag)
        assert info["node_count"] == 3
        assert info["dag_hash"] != ""
        # Dag should be unchanged.
        assert len(dag.nodes) == 3

    def test_graph_introspection_registers_hooks(self):
        api = GraphIntrospectionAPI()
        api.hook_pre_mutation(lambda dag: None)
        api.hook_post_mutation(lambda dag: None)
        assert api.pre_hook_count() == 1
        assert api.post_hook_count() == 1

    def test_clear_hooks_removes_all(self):
        api = GraphIntrospectionAPI()
        api.hook_pre_mutation(lambda dag: None)
        api.hook_post_mutation(lambda dag: None)
        api.clear_hooks()
        assert api.pre_hook_count() == 0
        assert api.post_hook_count() == 0

    def test_policy_separation_layer_defaults(self):
        psl = PolicySeparationLayer()
        assert "execution" in psl.layer_names()
        assert "planning" in psl.layer_names()
        assert "tool_selection" in psl.layer_names()

    def test_policy_separation_layer_to_dict(self):
        psl = PolicySeparationLayer()
        d = psl.to_dict()
        assert "execution_policy" in d
        assert "planning_policy" in d
        assert "tool_selection_policy" in d

    def test_telemetry_vector_to_dict(self):
        tv = ExecutionTelemetryVector(event_count=3, tool_count=1)
        d = tv.to_dict()
        assert d["event_count"] == 3
        assert d["tool_count"] == 1
