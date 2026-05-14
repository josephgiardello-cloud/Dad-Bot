"""Tests for canonical runtime types.

Validates that the new type model:
- Removes Any from critical paths
- Enables deterministic hashing
- Supports serialization/audit trails
- Carries invariants correctly
"""

from __future__ import annotations

import pytest

from dadbot.core.runtime_types import (
    CanonicalPayload,
    ExecutionIdentity,
    PolicyDecision,
    PolicyEffect,
    PolicyEffectType,
    RecoveryAction,
    RecoveryStrategy,
    ToolDeterminismClass,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
)

pytestmark = pytest.mark.unit


def test_tool_spec_defines_determinism_and_side_effects():
    spec = ToolSpec(
        name="weather_lookup",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
        capabilities=frozenset({"query_weather", "location_resolution"}),
    )

    assert spec.is_idempotent() is True
    assert spec.has_side_effects() is False
    assert "query_weather" in spec.capabilities


def test_tool_spec_with_side_effects():
    spec = ToolSpec(
        name="send_notification",
        version="1.0.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.LOGGED,
        required_permissions=frozenset({"send_messages"}),
    )

    assert spec.is_idempotent() is True
    assert spec.has_side_effects() is True
    assert "send_messages" in spec.required_permissions


def test_execution_identity_tracks_caller():
    identity = ExecutionIdentity(
        caller_trace_id="trace-123",
        caller_role="agent",
        caller_context="turn_execution",
    )

    assert identity.trace_id == "trace-123"
    assert identity.role == "agent"
    identity_dict = identity.to_dict()
    assert identity_dict["trace_id"] == "trace-123"

    restored = ExecutionIdentity.from_dict(identity_dict)
    assert restored.trace_id == "trace-123"
    assert restored.role == "agent"


def test_canonical_payload_provides_deterministic_hashing():
    payload_a = CanonicalPayload({"name": "Alice", "age": 30}, payload_type="person")
    payload_b = CanonicalPayload({"age": 30, "name": "Alice"}, payload_type="person")

    # Same content, different order → same hash (deterministic)
    assert payload_a.content_hash == payload_b.content_hash


def test_canonical_payload_hash_changes_on_content_mutation():
    payload_before = CanonicalPayload({"count": 1}, payload_type="counter")
    payload_after = CanonicalPayload({"count": 2}, payload_type="counter")

    assert payload_before.content_hash != payload_after.content_hash


def test_tool_invocation_carries_spec_and_caller():
    spec = ToolSpec(
        name="lookup",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    caller = ExecutionIdentity("trace-xyz", "agent")
    invocation = ToolInvocation(
        invocation_id="inv-001",
        tool_spec=spec,
        arguments={"query": "weather"},
        caller=caller,
    )

    assert invocation.tool_spec.name == "lookup"
    assert invocation.caller.trace_id == "trace-xyz"
    invocation_dict = invocation.to_dict()
    assert invocation_dict["tool_spec"]["name"] == "lookup"


def test_tool_invocation_argument_hash_is_deterministic():
    spec = ToolSpec(
        name="test",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    inv_a = ToolInvocation(
        invocation_id="inv-1",
        tool_spec=spec,
        arguments={"x": 1, "y": 2},
    )
    inv_b = ToolInvocation(
        invocation_id="inv-1",
        tool_spec=spec,
        arguments={"y": 2, "x": 1},
    )

    assert inv_a.argument_hash() == inv_b.argument_hash()


def test_tool_result_marks_replayability():
    payload = CanonicalPayload({"data": "result"}, payload_type="output")
    result = ToolResult(
        tool_name="lookup",
        invocation_id="inv-001",
        status=ToolExecutionStatus.OK,
        payload=payload,
        latency_ms=50.0,
        replay_safe=True,
    )

    assert result.succeeded() is True
    assert result.is_replayable() is True


def test_tool_result_failed_is_not_replayable():
    result = ToolResult(
        tool_name="lookup",
        invocation_id="inv-001",
        status=ToolExecutionStatus.ERROR,
        error="network timeout",
        replay_safe=False,
    )

    assert result.failed() is True
    assert result.is_replayable() is False


def test_policy_effect_tracks_mutations():
    effect = PolicyEffect(
        effect_type=PolicyEffectType.STRIP_FACET,
        source_rule="enforce_tone",
        before_hash="abc123",
        after_hash="def456",
        reason="sarcasm_removed",
    )

    assert effect.is_mutation() is True
    effect_dict = effect.to_dict()
    assert effect_dict["effect_type"] == "strip_facet"


def test_policy_effect_no_mutation_when_hashes_match():
    effect = PolicyEffect(
        effect_type=PolicyEffectType.DENY_TOOL,
        source_rule="permission_check",
        before_hash="abc123",
        after_hash="abc123",
        reason="no_change",
    )

    assert effect.is_mutation() is False


def test_policy_decision_collects_effects_and_output():
    effect1 = PolicyEffect(
        effect_type=PolicyEffectType.STRIP_FACET,
        source_rule="rule_1",
        before_hash="h1",
        after_hash="h2",
    )
    effect2 = PolicyEffect(
        effect_type=PolicyEffectType.REWRITE_OUTPUT,
        source_rule="rule_2",
        before_hash="h2",
        after_hash="h3",
    )
    output = CanonicalPayload({"text": "safe response"}, payload_type="text")
    decision = PolicyDecision(
        matched_rules=["rule_1", "rule_2"],
        emitted_effects=[effect1, effect2],
        final_output=output,
        output_was_modified=True,
    )

    assert len(decision.matched_rules) == 2
    assert len(decision.emitted_effects) == 2
    assert decision.output_was_modified is True
    decision_dict = decision.to_dict()
    assert len(decision_dict["emitted_effects"]) == 2


def test_recovery_action_is_bounded():
    action = RecoveryAction(
        strategy=RecoveryStrategy.RETRY_SAME,
        fault_trace_id="trace-fail",
        affected_tool_name="lookup",
        bounded_attempts=3,
    )

    assert action.is_bounded() is True
    assert action.bounded_attempts == 3
    action_dict = action.to_dict()
    assert action_dict["strategy"] == "retry_same"


def test_recovery_action_with_checkpoint():
    action = RecoveryAction(
        strategy=RecoveryStrategy.REPLAY_CHECKPOINT,
        fault_trace_id="trace-fail",
        affected_tool_name="lookup",
        checkpoint_id="ckpt-xyz",
        replay_from_invocation_id="inv-001",
    )

    assert action.checkpoint_id == "ckpt-xyz"
    assert action.replay_from_invocation_id == "inv-001"


def test_tool_spec_serialization_roundtrip():
    spec = ToolSpec(
        name="complex_tool",
        version="2.1.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.STATEFUL,
        capabilities=frozenset({"a", "b", "c"}),
        required_permissions=frozenset({"admin", "write"}),
        max_retries=5,
        timeout_seconds=30.0,
        description="A complex tool",
    )

    spec_dict = spec.to_dict()
    assert spec_dict["name"] == "complex_tool"
    assert spec_dict["version"] == "2.1.0"
    assert sorted(spec_dict["capabilities"]) == ["a", "b", "c"]
    assert sorted(spec_dict["required_permissions"]) == ["admin", "write"]


def test_canonical_payload_handles_nested_structures():
    nested = {
        "user": {"name": "Alice", "age": 30},
        "items": [1, 2, 3],
        "metadata": {"created": "2026-05-01"},
    }
    payload = CanonicalPayload(nested, payload_type="complex_record")

    hash1 = payload.content_hash
    payload2 = CanonicalPayload(nested, payload_type="complex_record")
    hash2 = payload2.content_hash

    assert hash1 == hash2


def test_execution_identity_missing_fields_use_defaults():
    identity = ExecutionIdentity(caller_trace_id="trace-000")

    assert identity.trace_id == "trace-000"
    assert identity.role == "agent"
    assert identity.context == ""
