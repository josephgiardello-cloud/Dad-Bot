from __future__ import annotations

import pytest

from dadbot.core.adversarial_closure import explore_closure
from dadbot.core.adversarial_closure import classify_closure_report, ClosureUnsafeClass
from dadbot.core.distributed_correctness import DistributedCorrectnessModel, NodeRole
from dadbot.core.global_transition_invariants import (
    GlobalTransitionInvariantEnforcer,
    TransitionBoundaryView,
)
from dadbot.core.runtime_errors import AuthorityViolation, InvariantViolation


pytestmark = pytest.mark.unit


class TestGlobalTransitionInvariantEnforcer:
    def test_accepts_valid_running_to_completed_transition(self):
        enforcer = GlobalTransitionInvariantEnforcer()
        enforcer.enforce(
            TransitionBoundaryView(
                session_id="s1",
                trace_id="t1",
                before_state="running",
                after_state="completed",
                before_causal_step_count=3,
                after_causal_step_count=4,
                turn_truth_ok=True,
            ),
        )

    def test_rejects_non_monotonic_causal_step_count(self):
        enforcer = GlobalTransitionInvariantEnforcer()
        with pytest.raises(InvariantViolation):
            enforcer.enforce(
                TransitionBoundaryView(
                    session_id="s1",
                    trace_id="t1",
                    before_state="running",
                    after_state="running",
                    before_causal_step_count=5,
                    after_causal_step_count=4,
                ),
            )

    def test_rejects_unknown_after_state(self):
        enforcer = GlobalTransitionInvariantEnforcer()
        with pytest.raises(InvariantViolation):
            enforcer.enforce(
                TransitionBoundaryView(
                    session_id="s1",
                    trace_id="t1",
                    before_state="running",
                    after_state="teleporting",
                    before_causal_step_count=1,
                    after_causal_step_count=2,
                ),
            )

    def test_rejects_completed_without_turn_truth(self):
        enforcer = GlobalTransitionInvariantEnforcer()
        with pytest.raises(InvariantViolation):
            enforcer.enforce(
                TransitionBoundaryView(
                    session_id="s1",
                    trace_id="t1",
                    before_state="running",
                    after_state="completed",
                    before_causal_step_count=1,
                    after_causal_step_count=2,
                    turn_truth_ok=None,
                ),
            )


class TestDistributedCorrectnessModel:
    def test_current_authority_uses_epoch_and_tiebreak(self):
        model = DistributedCorrectnessModel()
        model.register_node(
            node_id="node-b",
            epoch=1,
            lease_until_ms=2000,
            role=NodeRole.LEADER,
            state_hash="h1",
        )
        model.register_node(
            node_id="node-a",
            epoch=2,
            lease_until_ms=2000,
            role=NodeRole.LEADER,
            state_hash="h2",
        )

        authority = model.current_authority(now_ms=1000)
        assert authority is not None
        assert authority.node_id == "node-a"

    def test_detects_split_brain_same_epoch_active_leaders(self):
        model = DistributedCorrectnessModel()
        model.register_node(
            node_id="n1",
            epoch=7,
            lease_until_ms=3000,
            role=NodeRole.LEADER,
            state_hash="h",
        )
        model.register_node(
            node_id="n2",
            epoch=7,
            lease_until_ms=3000,
            role=NodeRole.LEADER,
            state_hash="h",
        )

        assert model.detect_split_brain(now_ms=2000) == ["n1", "n2"]
        with pytest.raises(AuthorityViolation):
            model.enforce_no_split_brain(now_ms=2000)

    def test_reconcile_reports_divergent_nodes(self):
        model = DistributedCorrectnessModel()
        model.register_node(
            node_id="leader",
            epoch=3,
            lease_until_ms=5000,
            role=NodeRole.LEADER,
            state_hash="H1",
        )
        model.register_node(
            node_id="follower-ok",
            epoch=3,
            lease_until_ms=5000,
            role=NodeRole.FOLLOWER,
            state_hash="H1",
        )
        model.register_node(
            node_id="follower-bad",
            epoch=3,
            lease_until_ms=5000,
            role=NodeRole.FOLLOWER,
            state_hash="H2",
        )

        plan = model.reconcile(now_ms=1000)
        assert plan.authoritative_node == "leader"
        assert plan.divergent_nodes == ["follower-bad"]
        assert plan.converged is False


class _ToyConvergingSystem:
    def initial_state(self):
        return {"x": 0, "y": 0}

    def enabled_actions(self, state):
        if self.is_terminal(state):
            return []
        return ["inc-x", "inc-y"]

    def step(self, state, action):
        nxt = dict(state)
        if action == "inc-x":
            nxt["x"] = min(1, int(nxt["x"]) + 1)
        if action == "inc-y":
            nxt["y"] = min(1, int(nxt["y"]) + 1)
        return nxt

    def is_terminal(self, state):
        return int(state.get("x", 0)) == 1 and int(state.get("y", 0)) == 1


class _ToyDivergingSystem:
    def initial_state(self):
        return {"s": 0}

    def enabled_actions(self, state):
        if self.is_terminal(state):
            return []
        return ["left", "right"]

    def step(self, state, action):
        if action == "left":
            return {"s": 1}
        return {"s": 2}

    def is_terminal(self, state):
        return int(state.get("s", 0)) in {1, 2}


class TestAdversarialClosure:
    def test_closure_converged_for_commutative_end_state(self):
        report = explore_closure(_ToyConvergingSystem(), max_depth=4)
        assert report.converged is True
        assert len(report.terminal_hashes) == 1

    def test_closure_finds_counterexample_for_divergent_terminal_states(self):
        report = explore_closure(_ToyDivergingSystem(), max_depth=2)
        assert report.converged is False
        assert len(report.terminal_hashes) == 2
        assert report.counterexample is not None

    def test_classification_marks_divergent_terminal_set_as_unsafe_non_crashing(self):
        report = explore_closure(_ToyDivergingSystem(), max_depth=2)
        classification = classify_closure_report(report)
        assert classification.classification == ClosureUnsafeClass.UNSAFE_NON_CRASHING_DIVERGENCE
        assert classification.unsafe is True

    def test_classification_marks_converged_terminal_set_as_safe(self):
        report = explore_closure(_ToyConvergingSystem(), max_depth=4)
        classification = classify_closure_report(report)
        assert classification.classification == ClosureUnsafeClass.SAFE
        assert classification.unsafe is False
