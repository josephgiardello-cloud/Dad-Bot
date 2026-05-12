from __future__ import annotations

import pytest

from dadbot.core.contract_evaluator import validate_sovereign_ledger_transition
from dadbot.core.contracts_adapter import ContractViolationError
from dadbot.core.runtime_types import (
    ExecutionIdentity,
    ToolDeterminismClass,
    ToolExecutionStatus,
    ToolInvocation,
    ToolSideEffectClass,
    ToolSpec,
)
from dadbot.core.tool_registry import ToolExecutionContext, ToolRegistry

pytestmark = pytest.mark.unit


def test_impossible_state_tripwire_hard_halts_invalid_mutation_transition_and_permission_spoof() -> None:
    before = {
        "session_id": "s-tripwire",
        "trace_id": "t-tripwire",
        "execution_mode": "live",
        "execution_state": "running",
        "execution_status": "running",
        "causal_step_count": 3,
        "metadata": {},
    }

    after_invalid_mutation = {
        "session_id": "s-tripwire",
        "trace_id": "t-tripwire",
        "execution_mode": "live",
        "execution_state": "completed",
        "execution_status": "completed",
        "turn_truth_ok": True,
        "invariance_hash": "h1",
        "causal_step_count": 4,
        "metadata": {
            "ledger_mutations": [
                {
                    "op": "append_history",
                    "payload": "invalid-payload-type",
                    "source": "test",
                }
            ]
        },
    }

    with pytest.raises(ContractViolationError, match="ledger_mutation"):
        validate_sovereign_ledger_transition(before, after_invalid_mutation)

    after_invalid_transition = {
        "session_id": "s-tripwire",
        "trace_id": "t-tripwire",
        "execution_mode": "live",
        "execution_state": "submitted",
        "execution_status": "running",
        "causal_step_count": 2,
        "metadata": {},
    }

    with pytest.raises(ContractViolationError, match="invalid execution_state transition|causal_step_count"):
        validate_sovereign_ledger_transition(before, after_invalid_transition)

    registry = ToolRegistry()
    admin_tool = ToolSpec(
        name="admin_op",
        version="1.0.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.LOGGED,
        required_permissions=frozenset({"tool.admin"}),
    )
    registry.register(
        admin_tool,
        lambda invocation: (_ for _ in ()).throw(RuntimeError(f"must not execute: {invocation.invocation_id}")),
    )

    context = ToolExecutionContext(registry)
    invocation = ToolInvocation(
        invocation_id="inv-tripwire-spoof",
        tool_spec=admin_tool,
        arguments={
            "caller_identity": {
                "id": "spoofed-user",
                "permissions": ["tool.admin"],
            }
        },
        caller=ExecutionIdentity(
            caller_trace_id="trace-tripwire",
            caller_role="agent",
            caller_context="{}",
        ),
    )

    result = context.execute(invocation)
    assert result.status == ToolExecutionStatus.DENIED
    assert "permission denied" in str(result.error).lower()
