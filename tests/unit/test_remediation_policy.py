from __future__ import annotations

import pytest

from dadbot.core.invariant_gate import InvariantGate, RemediationAction


pytestmark = pytest.mark.unit


def test_validation_contract_violation_replans_before_downgrade():
    first = InvariantGate.decide_remediation(
        "validation_contract_violation",
        attempt=0,
        max_attempts=1,
    )
    exhausted = InvariantGate.decide_remediation(
        "validation_contract_violation",
        attempt=1,
        max_attempts=1,
    )

    assert first.action is RemediationAction.REPLAN
    assert exhausted.action is RemediationAction.DOWNGRADE


def test_retryable_tool_failure_retries_before_hard_fail():
    first = InvariantGate.decide_remediation(
        "retryable_tool_failure",
        attempt=0,
        max_attempts=1,
    )
    exhausted = InvariantGate.decide_remediation(
        "retryable_tool_failure",
        attempt=1,
        max_attempts=1,
    )

    assert first.action is RemediationAction.RETRY
    assert exhausted.action is RemediationAction.HARD_FAIL


def test_invariant_violation_always_hard_fails():
    decision = InvariantGate.decide_remediation("invariant_violation")

    assert decision.action is RemediationAction.HARD_FAIL