"""Canonical runtime type model for DadBot.

This module defines the semantic primitives that all subsystems (tools, policy,
recovery, replay, audit) use to communicate. No implicit contracts, no dicts.

The goal is to make the entire runtime auditable, composable, and replayable
through typed entities that carry their own invariants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ToolDeterminismClass(str, Enum):
    """Execution determinism classification for tools."""

    READ_ONLY = "read_only"  # Same input → same output, no mutations
    DETERMINISTIC = "deterministic"  # Deterministic but may have side effects
    NONDETERMINISTIC = "nondeterministic"  # Output cannot be replayed


class ToolSideEffectClass(str, Enum):
    """Side-effect classification for tools."""

    PURE = "pure"  # No side effects, no external mutations
    LOGGED = "logged"  # Side effects are recorded, can be audited
    STATEFUL = "stateful"  # Mutates system state, requires special replay handling


class ToolExecutionStatus(str, Enum):
    """Execution status for tool invocations (canonical Phase A/B semantics).
    
    Distinct from tool_ir.ToolStatus (SUCCESS/RETRY/CONTRACT_VIOLATION/FATAL).
    This enum represents the runtime status of a ToolInvocation.
    """

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED = "denied"
    DEGRADED = "degraded"
    SKIPPED = "skipped"


class PolicyEffectType(str, Enum):
    """Types of effects that policy rules can emit."""

    DENY_TOOL = "deny_tool"  # Block tool invocation
    REWRITE_OUTPUT = "rewrite_output"  # Transform tool output
    REQUIRE_APPROVAL = "require_approval"  # Escalate to human
    STRIP_FACET = "strip_facet"  # Remove personality/tone
    FORCE_DEGRADATION = "force_degradation"  # Downgrade to fallback


class RecoveryStrategy(str, Enum):
    """Strategies for recovering from faults."""

    RETRY_SAME = "retry_same"  # Retry with same input
    REPLAY_CHECKPOINT = "replay_checkpoint"  # Replay from saved state
    DEGRADE_GRACEFULLY = "degrade_gracefully"  # Use fallback capability
    ESCALATE_PERMISSION = "escalate_permission"  # Request higher privilege
    HALT_SAFE = "halt_safe"  # Safe termination


class ExecutionIdentity:
    """Opaque identity for tracking who invoked a tool/policy/recovery action."""

    def __init__(self, caller_trace_id: str, caller_role: str = "agent", caller_context: str = "") -> None:
        self.trace_id = str(caller_trace_id).strip()
        self.role = str(caller_role).strip()
        self.context = str(caller_context).strip()

    def to_dict(self) -> dict[str, str]:
        return {
            "trace_id": self.trace_id,
            "role": self.role,
            "context": self.context,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExecutionIdentity:
        return ExecutionIdentity(
            caller_trace_id=str(data.get("trace_id") or ""),
            caller_role=str(data.get("role") or "agent"),
            caller_context=str(data.get("context") or ""),
        )


class CanonicalPayload:
    """Typed, hashable payload for tool results.
    
    Replaces dict[str, Any] to:
    - Enable deterministic hashing for idempotency/replay
    - Carry shape invariants
    - Support safe serialization
    """

    def __init__(self, content: Any, payload_type: str = "unknown") -> None:
        self._content = content
        self._type = str(payload_type).strip()
        self._hash = self._compute_hash()

    def _compute_hash(self) -> str:
        try:
            serialized = json.dumps(
                self._content,
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            )
        except (TypeError, ValueError):
            serialized = str(self._content)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def content(self) -> Any:
        return self._content

    @property
    def payload_type(self) -> str:
        return self._type

    @property
    def content_hash(self) -> str:
        return self._hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self._content,
            "type": self._type,
            "hash": self._hash,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CanonicalPayload:
        return CanonicalPayload(
            content=data.get("content"),
            payload_type=str(data.get("type") or "unknown"),
        )


@dataclass(frozen=True)
class ToolSpec:
    """Specification of a tool's capabilities, behavior, and contracts.
    
    Replaces ad-hoc metadata dicts with a canonical tool definition.
    """

    name: str
    version: str
    determinism: ToolDeterminismClass
    side_effect_class: ToolSideEffectClass
    capabilities: frozenset[str] = field(default_factory=frozenset)
    required_permissions: frozenset[str] = field(default_factory=frozenset)
    max_retries: int = 3
    timeout_seconds: float = 10.0
    description: str = ""

    def is_idempotent(self) -> bool:
        """Tool is safe to replay with same input."""
        return self.determinism in {
            ToolDeterminismClass.READ_ONLY,
            ToolDeterminismClass.DETERMINISTIC,
        }

    def has_side_effects(self) -> bool:
        """Tool may mutate system state."""
        return self.side_effect_class != ToolSideEffectClass.PURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "determinism": self.determinism.value,
            "side_effect_class": self.side_effect_class.value,
            "capabilities": sorted(self.capabilities),
            "required_permissions": sorted(self.required_permissions),
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "description": self.description,
        }


@dataclass
class ToolInvocation:
    """Canonical representation of a tool invocation.
    
    Replaces implicit tool-call dicts with explicit, traceable invocations.
    """

    invocation_id: str
    tool_spec: ToolSpec
    arguments: dict[str, Any] = field(default_factory=dict)
    caller: ExecutionIdentity | None = None
    requested_at_ms: float = 0.0
    timeout_override_seconds: float | None = None

    def effective_timeout_seconds(self) -> float:
        """Timeout to apply: override if set, else tool spec default."""
        if self.timeout_override_seconds is not None:
            return float(self.timeout_override_seconds)
        return float(self.tool_spec.timeout_seconds)

    def argument_hash(self) -> str:
        """Deterministic hash of invocation arguments for idempotency."""
        serialized = json.dumps(
            self.arguments,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "tool_spec": self.tool_spec.to_dict(),
            "arguments": dict(self.arguments),
            "caller": self.caller.to_dict() if self.caller else None,
            "requested_at_ms": self.requested_at_ms,
            "timeout_override_seconds": self.timeout_override_seconds,
        }


@dataclass
class ToolResult:
    """Canonical representation of a tool execution result.
    
    Typed, auditable, replayable. Replaces ToolExecutionResult dicts.
    """

    tool_name: str
    invocation_id: str
    status: ToolExecutionStatus
    payload: CanonicalPayload | None = None
    error: str = ""
    latency_ms: float = 0.0
    attempts: int = 1
    replay_safe: bool = False
    effects: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def succeeded(self) -> bool:
        return self.status == ToolExecutionStatus.OK

    def failed(self) -> bool:
        return self.status in (ToolExecutionStatus.ERROR, ToolExecutionStatus.TIMEOUT, ToolExecutionStatus.DENIED)

    def is_replayable(self) -> bool:
        """Result can be safely replayed for idempotency."""
        return self.replay_safe and self.succeeded()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "invocation_id": self.invocation_id,
            "status": self.status.value,
            "payload": self.payload.to_dict() if self.payload else None,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "replay_safe": self.replay_safe,
            "effects": list(self.effects),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PolicyEffect:
    """Effect emitted by a policy rule evaluation.
    
    Replaces implicit policy mutations with explicit, auditable effects.
    """

    effect_type: PolicyEffectType
    source_rule: str
    before_hash: str
    after_hash: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_mutation(self) -> bool:
        """Effect changes output or behavior."""
        return self.before_hash != self.after_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_type": self.effect_type.value,
            "source_rule": self.source_rule,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class PolicyDecision:
    """Result of policy evaluation with explicit effects.
    
    Replaces procedural policy traces with declarative decisions.
    """

    matched_rules: list[str] = field(default_factory=list)
    emitted_effects: list[PolicyEffect] = field(default_factory=list)
    final_output: CanonicalPayload | None = None
    output_was_modified: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_rules": list(self.matched_rules),
            "emitted_effects": [e.to_dict() for e in self.emitted_effects],
            "final_output": self.final_output.to_dict() if self.final_output else None,
            "output_was_modified": self.output_was_modified,
            "trace": dict(self.trace),
        }


@dataclass
class RecoveryAction:
    """Action to take when a fault is detected.
    
    Replaces implicit halt/retry logic with explicit recovery strategies.
    """

    strategy: RecoveryStrategy
    fault_trace_id: str
    affected_tool_name: str
    checkpoint_id: str | None = None
    replay_from_invocation_id: str | None = None
    escalation_level: int = 0
    bounded_attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_bounded(self) -> bool:
        """Recovery has explicit attempt limits."""
        return self.bounded_attempts > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "fault_trace_id": self.fault_trace_id,
            "affected_tool_name": self.affected_tool_name,
            "checkpoint_id": self.checkpoint_id,
            "replay_from_invocation_id": self.replay_from_invocation_id,
            "escalation_level": self.escalation_level,
            "bounded_attempts": self.bounded_attempts,
            "metadata": dict(self.metadata),
        }


class Effect(Protocol):
    """Protocol for all runtime effects (tools, policy, recovery)."""

    def to_dict(self) -> dict[str, Any]:
        ...
