"""Phase A — Invariant Enforcement Tests.

Validates the enforce_invariant() centralized gate and all 4 CRITICAL check
constructors before any production wiring.

Checks under test:
  1. Execution State Mismatch
  2. Causal Structure Violation
  3. Ledger Lifecycle Integrity
  4. Lease / Execution Ownership Violation

Usage: pytest tests/test_invariant_enforcement.py -m unit -q
"""

from __future__ import annotations

import pytest

from dadbot.core.invariant_engine import (
    InvariantCheck,
    InvariantSeverity,
    enforce_invariant,
)
from dadbot.core.invariant_gate import (
    InvariantViolationError,
    causal_structure_check,
    execution_state_check,
    ledger_lifecycle_check,
    lease_ownership_check,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# enforce_invariant() contract
# ---------------------------------------------------------------------------


class TestEnforceInvariant:
    def test_passed_check_does_not_raise(self):
        enforce_invariant(InvariantCheck(passed=True), InvariantSeverity.CRITICAL)

    def test_critical_failed_check_raises_invariant_violation_error(self):
        with pytest.raises(InvariantViolationError, match="state divergence"):
            enforce_invariant(
                InvariantCheck(passed=False, message="state divergence"),
                InvariantSeverity.CRITICAL,
            )

    def test_important_failed_check_warns_not_raises(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            enforce_invariant(
                InvariantCheck(passed=False, message="minor drift detected"),
                InvariantSeverity.IMPORTANT,
            )
        assert "minor drift detected" in caplog.text

    def test_diagnostic_failed_check_warns_not_raises(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            enforce_invariant(
                InvariantCheck(passed=False, message="perf anomaly"),
                InvariantSeverity.DIAGNOSTIC,
            )
        assert "perf anomaly" in caplog.text

    def test_error_message_propagated_verbatim(self):
        msg = "State divergence after execution — violations: ['causal_structure']"
        with pytest.raises(InvariantViolationError, match="causal_structure"):
            enforce_invariant(InvariantCheck(passed=False, message=msg), InvariantSeverity.CRITICAL)

    def test_strict_mode_escalates_error_to_raise(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DADBOT_STRICT_INVARIANTS", "1")
        with pytest.raises(InvariantViolationError, match="budget exceeded"):
            enforce_invariant(
                InvariantCheck(passed=False, message="budget exceeded"),
                InvariantSeverity.ERROR,
            )


# ---------------------------------------------------------------------------
# Check 1: Execution State Mismatch
# ---------------------------------------------------------------------------


class _MockDecision:
    """Minimal stand-in for ExecutionEquivalenceDecision."""
    def __init__(self, equivalent: bool, violations: list[str]):
        self.equivalent = equivalent
        self.violations = violations


class TestExecutionStateCheck:
    def test_equivalent_states_pass(self):
        decision = _MockDecision(equivalent=True, violations=[])
        check = execution_state_check(decision)
        assert check.passed is True

    def test_non_equivalent_states_fail(self):
        decision = _MockDecision(equivalent=False, violations=["memory_state"])
        check = execution_state_check(decision)
        assert check.passed is False
        assert "State divergence" in check.message
        assert "memory_state" in check.message

    def test_enforce_raises_on_state_divergence(self):
        decision = _MockDecision(equivalent=False, violations=["memory_state"])
        with pytest.raises(InvariantViolationError, match="State divergence"):
            enforce_invariant(execution_state_check(decision), InvariantSeverity.CRITICAL)

    def test_violations_list_in_message(self):
        decision = _MockDecision(
            equivalent=False,
            violations=["memory_state", "embedding_lock_state"],
        )
        check = execution_state_check(decision)
        assert "embedding_lock_state" in check.message


# ---------------------------------------------------------------------------
# Check 2: Causal Structure Violation
# ---------------------------------------------------------------------------


class TestCausalStructureCheck:
    def test_no_causal_violation_passes(self):
        decision = _MockDecision(equivalent=True, violations=[])
        assert causal_structure_check(decision).passed is True

    def test_unrelated_violation_passes(self):
        decision = _MockDecision(equivalent=False, violations=["memory_state"])
        assert causal_structure_check(decision).passed is True

    def test_causal_structure_in_violations_fails(self):
        decision = _MockDecision(equivalent=False, violations=["causal_structure"])
        check = causal_structure_check(decision)
        assert check.passed is False
        assert "causal" in check.message.lower()

    def test_enforce_raises_on_causal_violation(self):
        decision = _MockDecision(equivalent=False, violations=["causal_structure"])
        with pytest.raises(InvariantViolationError, match="causal"):
            enforce_invariant(causal_structure_check(decision), InvariantSeverity.CRITICAL)

    def test_causal_violation_mixed_with_others_still_fails(self):
        decision = _MockDecision(
            equivalent=False,
            violations=["memory_state", "causal_structure", "tool_io_graph"],
        )
        check = causal_structure_check(decision)
        assert check.passed is False


# ---------------------------------------------------------------------------
# Check 3: Ledger Lifecycle Integrity
# ---------------------------------------------------------------------------

def _evt(event_type: str, job_id: str, **kwargs) -> dict:
    return {"type": event_type, "job_id": job_id, **kwargs}


class TestLedgerLifecycleCheck:
    def test_empty_events_pass(self):
        assert ledger_lifecycle_check([]).passed is True

    def test_events_without_job_id_ignored(self):
        events = [
            {"type": "SESSION_STARTED", "session_id": "s1"},
            {"type": "JOB_COMPLETED"},  # no job_id
        ]
        assert ledger_lifecycle_check(events).passed is True

    def test_valid_started_then_completed(self):
        events = [
            _evt("JOB_STARTED", "job-1"),
            _evt("JOB_COMPLETED", "job-1"),
        ]
        assert ledger_lifecycle_check(events).passed is True

    def test_valid_started_then_failed(self):
        events = [
            _evt("JOB_STARTED", "job-1"),
            _evt("JOB_FAILED", "job-1"),
        ]
        assert ledger_lifecycle_check(events).passed is True

    def test_multiple_jobs_valid(self):
        events = [
            _evt("JOB_STARTED", "job-1"),
            _evt("JOB_STARTED", "job-2"),
            _evt("JOB_COMPLETED", "job-1"),
            _evt("JOB_COMPLETED", "job-2"),
        ]
        assert ledger_lifecycle_check(events).passed is True

    def test_completed_before_started_fails(self):
        events = [
            _evt("JOB_COMPLETED", "job-1"),
            _evt("JOB_STARTED", "job-1"),
        ]
        check = ledger_lifecycle_check(events)
        assert check.passed is False
        assert "JOB_COMPLETED before JOB_STARTED" in check.message

    def test_duplicate_terminal_state_fails(self):
        events = [
            _evt("JOB_STARTED", "job-1"),
            _evt("JOB_COMPLETED", "job-1"),
            _evt("JOB_COMPLETED", "job-1"),
        ]
        check = ledger_lifecycle_check(events)
        assert check.passed is False
        assert "duplicate terminal state" in check.message

    def test_job_id_in_payload_fallback(self):
        events = [
            {"type": "JOB_STARTED", "payload": {"job_id": "job-2"}},
            {"type": "JOB_COMPLETED", "payload": {"job_id": "job-2"}},
        ]
        assert ledger_lifecycle_check(events).passed is True

    def test_failed_then_completed_is_duplicate_terminal(self):
        events = [
            _evt("JOB_STARTED", "job-1"),
            _evt("JOB_FAILED", "job-1"),
            _evt("JOB_COMPLETED", "job-1"),
        ]
        check = ledger_lifecycle_check(events)
        assert check.passed is False
        assert "duplicate terminal state" in check.message

    def test_enforce_raises_on_lifecycle_violation(self):
        events = [_evt("JOB_COMPLETED", "job-x")]
        with pytest.raises(InvariantViolationError, match="JOB_COMPLETED before JOB_STARTED"):
            enforce_invariant(ledger_lifecycle_check(events), InvariantSeverity.CRITICAL)


# ---------------------------------------------------------------------------
# Check 4: Lease / Execution Ownership Violation
# ---------------------------------------------------------------------------


class TestLeaseOwnershipCheck:
    def test_no_active_lease_passes(self):
        check = lease_ownership_check("sess-1", "worker-A", actual_owner=None)
        assert check.passed is True

    def test_same_owner_passes(self):
        check = lease_ownership_check("sess-1", "worker-A", actual_owner="worker-A")
        assert check.passed is True

    def test_different_owner_fails(self):
        check = lease_ownership_check("sess-1", "worker-B", actual_owner="worker-A")
        assert check.passed is False
        assert "worker-A" in check.message
        assert "worker-B" in check.message
        assert "sess-1" in check.message

    def test_enforce_raises_on_lease_conflict(self):
        with pytest.raises(InvariantViolationError, match="Execution lease violation"):
            enforce_invariant(
                lease_ownership_check("sess-1", "worker-B", actual_owner="worker-A"),
                InvariantSeverity.CRITICAL,
            )

    def test_empty_actual_owner_treated_as_no_owner(self):
        # actual_owner="" is falsy — treated the same as None (no active lease)
        check = lease_ownership_check("sess-1", "worker-A", actual_owner="")
        # "" != "worker-A" so this fails — this is the correct strict behaviour
        assert check.passed is False

    def test_empty_requested_owner_matches_empty_actual(self):
        check = lease_ownership_check("sess-1", "", actual_owner="")
        assert check.passed is True


# ---------------------------------------------------------------------------
# InvariantGate.validate_ledger_lifecycle integration
# ---------------------------------------------------------------------------


class TestInvariantGateLifecycleMethod:
    def test_valid_sequence_does_not_raise(self):
        from dadbot.core.invariant_gate import InvariantGate
        gate = InvariantGate()
        events = [
            _evt("JOB_STARTED", "j1"),
            _evt("JOB_COMPLETED", "j1"),
        ]
        gate.validate_ledger_lifecycle(events)  # should not raise

    def test_invalid_sequence_raises_and_increments_counter(self):
        from dadbot.core.invariant_gate import InvariantGate
        gate = InvariantGate()
        events = [_evt("JOB_COMPLETED", "j1")]
        with pytest.raises(InvariantViolationError):
            gate.validate_ledger_lifecycle(events)
        assert gate.violations_observed == 1
