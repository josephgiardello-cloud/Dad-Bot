"""Tests for dadbot/core/memory_set_invariants.py.

Covers:
  Immediate  shrink and salience invariants (fire / pass)
  Gap C      causal ordering enforcement
  Gap D      lifecycle state machine transitions
"""

from __future__ import annotations

import pytest

from dadbot.core.memory_set_invariants import (
    CAUSAL_STEP_ORDER,
    MemoryLifecycleState,
    MemorySetInvariantViolation,
    assert_causal_order,
    assert_causal_transition,
    assert_memory_set_invariants,
    assert_memory_set_salience_invariant,
    assert_memory_set_shrink_invariant,
    record_causal_step,
    record_causal_step_locked,
    validate_lifecycle_transition,
)

pytestmark = pytest.mark.unit

# ── helpers ──────────────────────────────────────────────────────────────────

def _entry(memory_id: str, summary: str = "", importance_score: float = 0.5, **extra) -> dict:
    e: dict = {"memory_id": memory_id, "summary": summary or memory_id, "importance_score": importance_score}
    e.update(extra)
    return e


# ── Shrink invariant ──────────────────────────────────────────────────────────

class TestShrinkInvariant:
    def test_no_entries_before_is_a_noop(self):
        # Empty before-set must never raise.
        assert_memory_set_shrink_invariant([], [_entry("a")])

    def test_stable_set_passes(self):
        before = [_entry("a"), _entry("b")]
        after = [_entry("a"), _entry("b")]
        assert_memory_set_shrink_invariant(before, after)  # no raise

    def test_addition_only_passes(self):
        before = [_entry("a")]
        after = [_entry("a"), _entry("b")]
        assert_memory_set_shrink_invariant(before, after)  # no raise

    def test_silent_removal_raises(self):
        before = [_entry("a"), _entry("b", summary="important thing")]
        after = [_entry("a")]
        with pytest.raises(MemorySetInvariantViolation, match="shrink"):
            assert_memory_set_shrink_invariant(before, after)

    def test_removal_with_decay_entries_passes(self):
        b_entry = _entry("b", summary="forgettable")
        before = [_entry("a"), b_entry]
        after = [_entry("a")]
        # Explicitly signal b is decayed.
        assert_memory_set_shrink_invariant(before, after, decay_entries=[b_entry])

    def test_removal_with_inline_marker_passes(self):
        b_entry = _entry("b", _memory_decay_marker=True)
        before = [_entry("a"), b_entry]
        after = [_entry("a")]
        assert_memory_set_shrink_invariant(before, after)  # inline marker on before entry

    def test_removal_with_decay_reason_string_passes(self):
        b_entry = _entry("b", decay_reason="user-requested forgetting")
        before = [_entry("a"), b_entry]
        after = [_entry("a")]
        assert_memory_set_shrink_invariant(before, after)

    def test_context_tag_appears_in_message(self):
        before = [_entry("a")]
        after = []
        with pytest.raises(MemorySetInvariantViolation, match="my_context"):
            assert_memory_set_shrink_invariant(before, after, context="my_context")


# ── Salience invariant ────────────────────────────────────────────────────────

class TestSalienceInvariant:
    def test_stable_salience_passes(self):
        before = [_entry("a", importance_score=0.8)]
        after = [_entry("a", importance_score=0.8)]
        assert_memory_set_salience_invariant(before, after)

    def test_salience_increase_passes(self):
        before = [_entry("a", importance_score=0.5)]
        after = [_entry("a", importance_score=0.9)]
        assert_memory_set_salience_invariant(before, after)

    def test_small_drop_below_threshold_passes(self):
        before = [_entry("a", importance_score=0.8)]
        after = [_entry("a", importance_score=0.7)]  # 0.1 drop < 0.15
        assert_memory_set_salience_invariant(before, after)

    def test_large_drop_without_signal_raises(self):
        before = [_entry("a", importance_score=0.9)]
        after = [_entry("a", importance_score=0.5)]  # 0.4 drop
        with pytest.raises(MemorySetInvariantViolation, match="salience"):
            assert_memory_set_salience_invariant(before, after)

    def test_large_drop_with_decay_entry_passes(self):
        a_before = _entry("a", importance_score=0.9)
        a_after = _entry("a", importance_score=0.5)
        assert_memory_set_salience_invariant([a_before], [a_after], decay_entries=[a_before])

    def test_large_drop_with_inline_marker_passes(self):
        a_after = _entry("a", importance_score=0.5, _memory_decay_marker=True)
        before = [_entry("a", importance_score=0.9)]
        assert_memory_set_salience_invariant(before, [a_after])

    def test_missing_entry_not_checked_here(self):
        # Removed entries are the shrink invariant's domain; salience check skips them.
        before = [_entry("a", importance_score=0.9), _entry("b", importance_score=0.7)]
        after = [_entry("a", importance_score=0.9)]  # b gone → shrink invariant fires
        assert_memory_set_salience_invariant(before, after)  # no raise here

    def test_custom_threshold(self):
        before = [_entry("a", importance_score=0.8)]
        after = [_entry("a", importance_score=0.75)]  # 0.05 drop > strict threshold 0.01
        with pytest.raises(MemorySetInvariantViolation):
            assert_memory_set_salience_invariant(before, after, salience_drop_threshold=0.01)

    def test_weight_field_used_as_fallback_salience(self):
        before = [{"memory_id": "x", "weight": 0.9}]
        after = [{"memory_id": "x", "weight": 0.4}]  # 0.5 drop
        with pytest.raises(MemorySetInvariantViolation, match="salience"):
            assert_memory_set_salience_invariant(before, after)


# ── Combined assert_memory_set_invariants ────────────────────────────────────

class TestCombinedInvariants:
    def test_shrink_caught_first(self):
        before = [_entry("a"), _entry("b")]
        after = [_entry("a")]  # silent removal
        with pytest.raises(MemorySetInvariantViolation, match="shrink"):
            assert_memory_set_invariants(before, after)

    def test_salience_caught_when_no_shrink(self):
        before = [_entry("a", importance_score=0.9)]
        after = [_entry("a", importance_score=0.3)]
        with pytest.raises(MemorySetInvariantViolation, match="salience"):
            assert_memory_set_invariants(before, after)

    def test_clean_state_passes(self):
        before = [_entry("a", importance_score=0.8), _entry("b", importance_score=0.5)]
        after = [_entry("a", importance_score=0.8), _entry("b", importance_score=0.5)]
        assert_memory_set_invariants(before, after)  # no raise


# ── Gap C: Causal ordering ────────────────────────────────────────────────────

class TestCausalOrdering:
    def test_valid_full_sequence_passes(self):
        assert_causal_order(CAUSAL_STEP_ORDER)

    def test_subset_in_order_passes(self):
        assert_causal_order(["retrieval", "tool_execution", "persistence"])

    def test_out_of_order_raises(self):
        with pytest.raises(MemorySetInvariantViolation, match="Causal order violation"):
            assert_causal_order(["tool_execution", "retrieval"])

    def test_unknown_steps_are_ignored(self):
        # 'logging' is not a known causal step; must not cause a violation.
        assert_causal_order(["retrieval", "logging", "planning"])

    def test_empty_steps_passes(self):
        assert_causal_order([])

    def test_record_causal_step_appends(self):
        state: dict = {}
        record_causal_step(state, "retrieval")
        record_causal_step(state, "planning")
        assert state["_causal_step_log"] == ["retrieval", "planning"]

    def test_record_then_assert_valid(self):
        state: dict = {}
        for step in ["retrieval", "planning", "tool_execution"]:
            record_causal_step(state, step)
        assert_causal_order(state["_causal_step_log"])  # no raise

    def test_record_then_assert_invalid(self):
        state: dict = {}
        record_causal_step(state, "tool_execution")
        record_causal_step(state, "retrieval")
        with pytest.raises(MemorySetInvariantViolation):
            assert_causal_order(state["_causal_step_log"])


class TestCausalTransitionLock:
    def test_transition_allows_ordered_step(self):
        assert_causal_transition(["retrieval"], "planning")

    def test_transition_rejects_out_of_order_step(self):
        with pytest.raises(MemorySetInvariantViolation, match="Causal transition violation"):
            assert_causal_transition(["retrieval"], "persistence")

    def test_locked_record_rejects_missing_retrieval(self):
        state: dict = {}
        with pytest.raises(MemorySetInvariantViolation, match="Causal transition violation"):
            record_causal_step_locked(state, "planning")

    def test_locked_record_accepts_valid_sequence(self):
        state: dict = {}
        record_causal_step_locked(state, "retrieval")
        record_causal_step_locked(state, "planning")
        record_causal_step_locked(state, "tool_execution")
        record_causal_step_locked(state, "post_tool_refresh")
        record_causal_step_locked(state, "persistence")
        assert state["_causal_step_log"] == [
            "retrieval",
            "planning",
            "tool_execution",
            "post_tool_refresh",
            "persistence",
        ]


# ── Gap D: Lifecycle state machine ────────────────────────────────────────────

class TestLifecycleStateMachine:
    @pytest.mark.parametrize("from_s,to_s", [
        ("active", "reinforced"),
        ("active", "decaying"),
        ("reinforced", "active"),
        ("reinforced", "decaying"),
        ("decaying", "archived"),
        ("decaying", "active"),    # recovery
        ("archived", "expired"),
        ("archived", "active"),   # resurrection
    ])
    def test_allowed_transitions(self, from_s: str, to_s: str):
        validate_lifecycle_transition(from_s, to_s)  # no raise

    @pytest.mark.parametrize("from_s,to_s", [
        ("active", "expired"),     # cannot skip to expired
        ("expired", "active"),     # expired is terminal
        ("expired", "decaying"),
        ("reinforced", "archived"),# must pass through decaying
        ("archived", "reinforced"),
    ])
    def test_prohibited_transitions_raise(self, from_s: str, to_s: str):
        with pytest.raises(MemorySetInvariantViolation, match="Prohibited lifecycle transition"):
            validate_lifecycle_transition(from_s, to_s)

    def test_invalid_state_name_raises_value_error(self):
        with pytest.raises(ValueError):
            validate_lifecycle_transition("active", "zombie")

    def test_enum_values_accepted(self):
        validate_lifecycle_transition(
            MemoryLifecycleState.ACTIVE,
            MemoryLifecycleState.DECAYING,
        )  # no raise

    def test_context_tag_in_message(self):
        with pytest.raises(MemorySetInvariantViolation, match="my_ctx"):
            validate_lifecycle_transition("expired", "active", context="my_ctx")
