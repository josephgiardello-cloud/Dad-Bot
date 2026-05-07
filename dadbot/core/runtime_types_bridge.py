"""Integration bridge: replacing Any/dict paths with canonical runtime types.

This module documents and demonstrates how the new runtime types eliminate
implicit contracts throughout the existing codebase.

Key integration points:
1. ExternalToolRuntime → uses ToolSpec, ToolInvocation, ToolResult
2. PolicyCompiler → emits PolicyEffect, PolicyDecision
3. PersistenceService → records ToolInvocation, ToolResult, PolicyEffect
4. RecoveryStrategies → use RecoveryAction with bounded retries
5. Audit/Replay → use typed effects to reconstruct system state
"""

from __future__ import annotations

from typing import Any

from dadbot.core.runtime_types import (
    CanonicalPayload,
    ExecutionIdentity,
    PolicyDecision,
    PolicyEffect,
    PolicyEffectType,
    RecoveryAction,
    RecoveryStrategy,
    ToolDeterminismClass,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
    ToolStatus,
)


# ============================================================================
# BEFORE: Implicit dict-based contracts (current state)
# ============================================================================


def _current_tool_execution_model() -> None:
    """Current: tools return dicts with implicit schema."""
    # What we have now:
    result_dict: dict[str, Any] = {
        "tool_name": "lookup",
        "status": "ok",
        "output": {"data": "..."},
        "latency_ms": 50.0,
        "metadata": {"http_status": 200},
        # But where is the contract? What fields are required?
        # What does "replay_safe" mean? Is it in metadata or separate?
    }
    # Problem: Any path, implicit schema, unclear invariants


def _current_policy_model() -> None:
    """Current: policy effects buried in trace dicts."""
    # What we have now:
    trace: dict[str, Any] = {
        "policy": "safety",
        "selected_rule": {"name": "enforce_policies", "kind": "binary"},
        "final_action": {
            "action": "handled",
            "output_mutated": True,  # Is this always present?
            "candidate_hash": "abc123",  # What if it's missing?
            "output_hash": "def456",
        },
        # Problem: Effects are buried in nested dicts
        # No way to iterate effects, compose them, or audit them
    }


def _current_recovery_model() -> None:
    """Current: recovery is binary (halt or retry)."""
    # What we have now:
    if_divergence_detected: bool = True
    if if_divergence_detected:
        # Only option: halt and block commit
        # No intermediate strategies: retry, degrade, escalate, repair
        # No bounded attempts or explicit recovery plan
        raise RuntimeError("Memory authority divergence detected; commit blocked")


# ============================================================================
# AFTER: Typed, canonical runtime model (new state)
# ============================================================================


def example_tool_spec_definition() -> ToolSpec:
    """Define a tool with explicit semantics."""
    return ToolSpec(
        name="schedule_meeting",
        version="1.0.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.LOGGED,  # Mutates calendar
        capabilities=frozenset({"time_slots", "calendar_check"}),
        required_permissions=frozenset({"calendar_write"}),
        timeout_seconds=15.0,
        description="Schedule a meeting and return confirmation",
    )


def example_tool_invocation_with_identity() -> ToolInvocation:
    """Invoke a tool with full execution context."""
    spec = example_tool_spec_definition()
    caller = ExecutionIdentity(
        caller_trace_id="turn-2026-05-07-xyz",
        caller_role="agent",
        caller_context="reply_planning",
    )
    return ToolInvocation(
        invocation_id="inv-001-schedule-meeting",
        tool_spec=spec,
        arguments={
            "attendee": "alice@example.com",
            "time_slot": "2026-05-10T14:00:00Z",
            "duration_minutes": 30,
        },
        caller=caller,
        timeout_override_seconds=20.0,  # Override tool default
    )


def example_tool_result_with_replayability() -> ToolResult:
    """Tool result carries replay semantics."""
    invocation = example_tool_invocation_with_identity()
    payload = CanonicalPayload(
        content={
            "confirmation_id": "meet-001",
            "attendee": "alice@example.com",
            "scheduled_time": "2026-05-10T14:00:00Z",
        },
        payload_type="meeting_confirmation",
    )
    return ToolResult(
        tool_name="schedule_meeting",
        invocation_id=invocation.invocation_id,
        status=ToolStatus.OK,
        payload=payload,
        latency_ms=120.0,
        attempts=1,
        replay_safe=True,  # This is deterministic, can be cached
        effects=[],  # Side effects logged separately
        metadata={"http_status": 200, "service": "calendar-api"},
    )


def example_policy_decision_with_effects() -> PolicyDecision:
    """Policy decision emits typed effects."""
    # Original candidate output
    original_candidate = CanonicalPayload(
        content="I'll definitely schedule that—worst case it overwrites your other meetings 😂",
        payload_type="reply_candidate",
    )
    original_hash = original_candidate.content_hash

    # Policy evaluation emits effects
    effect_strip_facet = PolicyEffect(
        effect_type=PolicyEffectType.STRIP_FACET,
        source_rule="enforce_professional_tone",
        before_hash=original_hash,
        after_hash="new_hash_1",
        reason="Remove sarcasm and humor",
    )
    effect_rewrite = PolicyEffect(
        effect_type=PolicyEffectType.REWRITE_OUTPUT,
        source_rule="enforce_accuracy",
        before_hash="new_hash_1",
        after_hash="new_hash_2",
        reason="Add explicit confirmation and no-overwrite guarantee",
    )

    final_output = CanonicalPayload(
        content="I'll schedule that meeting for you at 2026-05-10T14:00:00Z. This will not overwrite existing meetings.",
        payload_type="reply_candidate",
    )

    return PolicyDecision(
        matched_rules=["enforce_professional_tone", "enforce_accuracy"],
        emitted_effects=[effect_strip_facet, effect_rewrite],
        final_output=final_output,
        output_was_modified=True,
        trace={
            "policy": "safety",
            "evaluated_at": "2026-05-07T10:00:00Z",
        },
    )


def example_recovery_action_with_strategy() -> RecoveryAction:
    """Recovery is explicit, bounded, and stratified."""
    # When tool execution fails:
    return RecoveryAction(
        strategy=RecoveryStrategy.RETRY_SAME,
        fault_trace_id="turn-2026-05-07-xyz",
        affected_tool_name="schedule_meeting",
        bounded_attempts=3,  # Bounded, not infinite
        metadata={
            "failure_reason": "timeout_on_calendar_api",
            "backoff_ms": 500,
        },
    )


def example_recovery_with_fallback() -> RecoveryAction:
    """Recovery can escalate to fallback strategy."""
    return RecoveryAction(
        strategy=RecoveryStrategy.DEGRADE_GRACEFULLY,
        fault_trace_id="turn-2026-05-07-xyz",
        affected_tool_name="schedule_meeting",
        bounded_attempts=1,
        metadata={
            "fallback_tool": "send_text_reminder",
            "reason": "calendar_api unavailable; use manual scheduling reminder",
        },
    )


def example_recovery_with_checkpoint_replay() -> RecoveryAction:
    """Recovery can replay from a known-good checkpoint."""
    return RecoveryAction(
        strategy=RecoveryStrategy.REPLAY_CHECKPOINT,
        fault_trace_id="turn-2026-05-07-xyz",
        affected_tool_name="schedule_meeting",
        checkpoint_id="ckpt-2026-05-07-09:45:00Z",
        replay_from_invocation_id="inv-001-schedule-meeting",
        bounded_attempts=1,
        metadata={
            "reason": "divergence detected; replaying from known-good state",
            "checkpoint_age_seconds": 300,
        },
    )


# ============================================================================
# KEY ADVANTAGES OF TYPED MODEL
# ============================================================================


def advantages_summary() -> None:
    """Why the canonical types matter."""
    print("BEFORE (dict/Any):")
    print("  - Tool results are untyped dicts")
    print("  - Policy effects are buried in nested traces")
    print("  - Recovery is binary (halt or retry)")
    print("  - No audit trail of what changed")
    print("  - Replay semantics are implicit")
    print("  - Can't compose or analyze effects")
    print()
    print("AFTER (canonical types):")
    print("  - ToolSpec defines contracts explicitly")
    print("  - ToolInvocation carries execution identity")
    print("  - ToolResult marks replayability")
    print("  - PolicyEffect is first-class, auditable")
    print("  - RecoveryAction is stratified with bounds")
    print("  - Effects form a causality chain (before/after hash)")
    print("  - Audit log is queryable, composable")
    print("  - Determinism and side-effects are explicit")
    print()
    print("STRATEGIC GAIN:")
    print("  - Tool registration becomes extensible (Phase B)")
    print("  - Policy becomes declarative (Phase C)")
    print("  - Recovery becomes systematic (Phase D)")
    print("  - Agent planning becomes grounded (Phase E)")
