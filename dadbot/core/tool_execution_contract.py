"""Formal tool execution contract: strict input/output schema, failure taxonomy, replay semantics.

This module defines the canonical interface for tool execution with determinism guarantees,
unified failure classification, and idempotency semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    """Unified failure taxonomy with recovery implications."""

    # ✓ Retryable: transient network/rate conditions
    NETWORK_TIMEOUT = "network_timeout"  # Can retry, may succeed on repeat
    NETWORK_RATE_LIMIT = "network_rate_limit"  # Throttled; retry with backoff
    NETWORK_TEMPORARY = "network_temporary"  # DNS, connection drop, temporary server error
    NETWORK_UNKNOWN = "network_unknown"  # Unclassified network failure

    # ✗ Non-retryable: permanent/policy failures
    CLIENT_ERROR = "client_error"  # 4xx; input/auth problem; retry won't help
    PERMISSION_DENIED = "permission_denied"  # Not authorized; policy blocked
    RESOURCE_UNAVAILABLE = "resource_unavailable"  # Tool not registered, version mismatch
    ISOLATION_VIOLATED = "isolation_violated"  # Resource limits exceeded
    TOOL_INTERNAL_ERROR = "tool_internal_error"  # Handler raised exception
    SCHEMA_VALIDATION = "schema_validation"  # Input did not match contract
    TIMEOUT_EXCEEDED = "timeout_exceeded"  # Hard timeout (non-retryable)

    # ◐ Partial/degraded: operation succeeded with caveats
    PARTIAL_OUTPUT = "partial_output"  # Tool returned 206 Partial Content
    DEGRADED_CONFIDENCE = "degraded_confidence"  # Operation succeeded but quality reduced (e.g., latency spike)

    # ◆ Unknown/other
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    """Severity classification for failure decision-making."""

    TRANSIENT = "transient"  # Retry likely to succeed
    PERMANENT = "permanent"  # Retry will fail; need fallback/escalation
    PARTIAL = "partial"  # Output usable but degraded
    UNKNOWN = "unknown"  # Unable to classify


@dataclass(frozen=True)
class FailureTaxonomyEntry:
    """Unified failure classification with recovery metadata."""

    failure_class: FailureClass
    severity: FailureSeverity
    description: str
    retryable: bool
    escalation_key: str  # Category for policy decision engine (e.g., "rate_limit_exceeded", "auth_required")


# Canonical failure taxonomy bindings
_FAILURE_TAXONOMY: dict[FailureClass, FailureTaxonomyEntry] = {
    FailureClass.NETWORK_TIMEOUT: FailureTaxonomyEntry(
        failure_class=FailureClass.NETWORK_TIMEOUT,
        severity=FailureSeverity.TRANSIENT,
        description="Network I/O timeout; may succeed on retry with backoff",
        retryable=True,
        escalation_key="network_timeout_retryable",
    ),
    FailureClass.NETWORK_RATE_LIMIT: FailureTaxonomyEntry(
        failure_class=FailureClass.NETWORK_RATE_LIMIT,
        severity=FailureSeverity.TRANSIENT,
        description="Rate limited (HTTP 429); retry with exponential backoff",
        retryable=True,
        escalation_key="rate_limit_backoff",
    ),
    FailureClass.NETWORK_TEMPORARY: FailureTaxonomyEntry(
        failure_class=FailureClass.NETWORK_TEMPORARY,
        severity=FailureSeverity.TRANSIENT,
        description="Transient network failure (DNS, connection drop, 5xx); retry likely to succeed",
        retryable=True,
        escalation_key="network_transient_retry",
    ),
    FailureClass.NETWORK_UNKNOWN: FailureTaxonomyEntry(
        failure_class=FailureClass.NETWORK_UNKNOWN,
        severity=FailureSeverity.TRANSIENT,
        description="Unclassified network failure; retry conservatively",
        retryable=True,
        escalation_key="network_unknown_retry",
    ),
    FailureClass.CLIENT_ERROR: FailureTaxonomyEntry(
        failure_class=FailureClass.CLIENT_ERROR,
        severity=FailureSeverity.PERMANENT,
        description="Client error (HTTP 4xx); input validation or auth issue; do not retry same input",
        retryable=False,
        escalation_key="client_error_noretry",
    ),
    FailureClass.PERMISSION_DENIED: FailureTaxonomyEntry(
        failure_class=FailureClass.PERMISSION_DENIED,
        severity=FailureSeverity.PERMANENT,
        description="Permission denied by policy; escalate to approval gate or skip",
        retryable=False,
        escalation_key="permission_denied_escalate",
    ),
    FailureClass.RESOURCE_UNAVAILABLE: FailureTaxonomyEntry(
        failure_class=FailureClass.RESOURCE_UNAVAILABLE,
        severity=FailureSeverity.PERMANENT,
        description="Tool not registered or version incompatible; try fallback tool",
        retryable=False,
        escalation_key="tool_unavailable_fallback",
    ),
    FailureClass.ISOLATION_VIOLATED: FailureTaxonomyEntry(
        failure_class=FailureClass.ISOLATION_VIOLATED,
        severity=FailureSeverity.PERMANENT,
        description="Resource limits exceeded (CPU, memory, I/O); do not retry with same estimate",
        retryable=False,
        escalation_key="isolation_violated_abort",
    ),
    FailureClass.TOOL_INTERNAL_ERROR: FailureTaxonomyEntry(
        failure_class=FailureClass.TOOL_INTERNAL_ERROR,
        severity=FailureSeverity.PERMANENT,
        description="Tool handler raised exception; indicates handler bug or input edge case",
        retryable=False,
        escalation_key="tool_error_debug",
    ),
    FailureClass.SCHEMA_VALIDATION: FailureTaxonomyEntry(
        failure_class=FailureClass.SCHEMA_VALIDATION,
        severity=FailureSeverity.PERMANENT,
        description="Input did not satisfy contract schema; caller error",
        retryable=False,
        escalation_key="schema_validation_fail",
    ),
    FailureClass.TIMEOUT_EXCEEDED: FailureTaxonomyEntry(
        failure_class=FailureClass.TIMEOUT_EXCEEDED,
        severity=FailureSeverity.PERMANENT,
        description="Hard timeout exceeded; operation aborted; do not retry with same timeout",
        retryable=False,
        escalation_key="hard_timeout_abort",
    ),
    FailureClass.PARTIAL_OUTPUT: FailureTaxonomyEntry(
        failure_class=FailureClass.PARTIAL_OUTPUT,
        severity=FailureSeverity.PARTIAL,
        description="Tool returned partial content (HTTP 206/207); output usable but incomplete",
        retryable=False,
        escalation_key="partial_output_accepted",
    ),
    FailureClass.DEGRADED_CONFIDENCE: FailureTaxonomyEntry(
        failure_class=FailureClass.DEGRADED_CONFIDENCE,
        severity=FailureSeverity.PARTIAL,
        description="Operation succeeded with degraded confidence (latency spike, quality reduction)",
        retryable=False,
        escalation_key="degraded_accepted",
    ),
    FailureClass.UNKNOWN: FailureTaxonomyEntry(
        failure_class=FailureClass.UNKNOWN,
        severity=FailureSeverity.UNKNOWN,
        description="Failure reason unknown; treat conservatively as transient with limit",
        retryable=True,
        escalation_key="unknown_limited_retry",
    ),
}


def taxonomy_entry_for(failure_class: FailureClass) -> FailureTaxonomyEntry:
    """Retrieve taxonomy entry; validates classification exists."""
    entry = _FAILURE_TAXONOMY.get(failure_class)
    if entry is None:
        raise ValueError(f"Unknown failure class: {failure_class}")
    return entry


@dataclass(frozen=True)
class ToolInputSchema:
    """Strict contract for required input fields and their types/constraints."""

    required_fields: frozenset[str]  # e.g., {"action", "target"}
    optional_fields: frozenset[str]  # e.g., {"timeout_seconds", "metadata"}
    field_types: dict[str, type | tuple[type, ...]]  # Validation: {field_name: (str, int, ...)}
    max_payload_bytes: int = 1_000_000  # Default 1MB

    def validate(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """Validate input against schema; return (valid, error_message)."""
        if not isinstance(payload, dict):
            return False, "payload must be dict"
        if len(str(payload).encode()) > self.max_payload_bytes:
            return False, f"payload exceeds {self.max_payload_bytes} bytes"

        for field in self.required_fields:
            if field not in payload:
                return False, f"missing required field: {field}"

        for field, value in payload.items():
            if field not in self.required_fields and field not in self.optional_fields:
                return False, f"unexpected field: {field}"

            expected_types = self.field_types.get(field)
            if expected_types is not None:
                if not isinstance(value, expected_types):
                    type_names = expected_types if isinstance(expected_types, tuple) else (expected_types,)
                    return False, f"field {field} has wrong type; expected {type_names}, got {type(value).__name__}"

        return True, "ok"


@dataclass(frozen=True)
class ToolOutputSchema:
    """Contract for successful output shape and constraints."""

    required_fields: frozenset[str]  # e.g., {"result", "metadata"}
    optional_fields: frozenset[str]
    field_types: dict[str, type | tuple[type, ...]]
    max_output_bytes: int = 10_000_000  # Default 10MB

    def validate(self, output: Any) -> tuple[bool, str]:
        """Validate output against schema."""
        if not isinstance(output, dict):
            return False, "output must be dict"
        if len(str(output).encode()) > self.max_output_bytes:
            return False, f"output exceeds {self.max_output_bytes} bytes"

        for field in self.required_fields:
            if field not in output:
                return False, f"missing required field in output: {field}"

        for field, value in output.items():
            if field not in self.required_fields and field not in self.optional_fields:
                return False, f"unexpected field in output: {field}"

            expected_types = self.field_types.get(field)
            if expected_types is not None:
                if not isinstance(value, expected_types):
                    type_names = expected_types if isinstance(expected_types, tuple) else (expected_types,)
                    return False, f"output field {field} has wrong type; expected {type_names}, got {type(value).__name__}"

        return True, "ok"


@dataclass(frozen=True)
class ReplaySemantics:
    """Formal semantics for idempotency, determinism, and cache validity."""

    # Determinism boundaries: conditions that guarantee same result on replay
    idempotency_key_factors: frozenset[str]
    """Fields that contribute to idempotency key (e.g., tool_name, action, target).
    Replay with same key should return cached result even if time has passed."""

    policy_context_factors: frozenset[str]
    """Policy fields (permissions, approval state) that are part of idempotency boundary.
    Different policy context = different idempotency key even if input is identical."""

    cache_ttl_seconds: float
    """How long is a cached result valid? Null = indefinite (until explicit invalidation)."""

    determinism_guarantee: str
    """Semantic: "strict" = exactly same output, "soft" = logically equivalent output, "none" = non-deterministic."""

    def is_replay_valid(self, *, cached_result_age_seconds: float) -> bool:
        """Check if cached result is still valid based on TTL."""
        if self.cache_ttl_seconds < 0:
            return False  # Cache disabled
        if self.cache_ttl_seconds == 0:
            return True  # Indefinite cache
        return cached_result_age_seconds < self.cache_ttl_seconds


@dataclass(frozen=True)
class ToolExecutionContract:
    """Canonical execution contract for a tool: schema + failure taxonomy + replay semantics."""

    tool_name: str
    version: str
    """Tool version (semver). Part of contract identity; version mismatch = different contract."""

    input_schema: ToolInputSchema
    output_schema: ToolOutputSchema
    replay_semantics: ReplaySemantics

    description: str = ""
    """Human-readable contract description."""

    supported_failure_classes: frozenset[FailureClass] = field(
        default_factory=lambda: frozenset(FailureClass),
    )
    """Which failures is this tool expected to produce? Enables strict failure validation."""

    max_total_attempts: int = 3
    """Max retry attempts; failure policy engine uses this to decide escalation point."""

    def validate_input(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """Validate input before execution."""
        return self.input_schema.validate(payload)

    def validate_output(self, output: Any) -> tuple[bool, str]:
        """Validate output after successful execution."""
        return self.output_schema.validate(output)

    def classify_failure(self, failure_class: FailureClass) -> FailureTaxonomyEntry:
        """Get taxonomy entry for failure; validates against supported_failure_classes."""
        if failure_class not in self.supported_failure_classes and failure_class != FailureClass.UNKNOWN:
            raise ValueError(
                f"Tool {self.tool_name}@{self.version} does not document support for {failure_class}; "
                f"supported: {self.supported_failure_classes}",
            )
        return taxonomy_entry_for(failure_class)


# Example contract (can be used as a template)
DEFAULT_GENERIC_TOOL_CONTRACT = ToolExecutionContract(
    tool_name="generic",
    version="1.0.0",
    description="Generic tool with minimal constraints; use as fallback or test fixture",
    input_schema=ToolInputSchema(
        required_fields=frozenset(),
        optional_fields=frozenset({"_idempotency_key", "_timeout_seconds", "metadata"}),
        field_types={},
    ),
    output_schema=ToolOutputSchema(
        required_fields=frozenset(),
        optional_fields=frozenset({"result", "metadata", "status"}),
        field_types={},
    ),
    replay_semantics=ReplaySemantics(
        idempotency_key_factors=frozenset({"tool_name", "_idempotency_key"}),
        policy_context_factors=frozenset({"approval_granted", "session_permissions"}),
        cache_ttl_seconds=0,  # Indefinite cache
        determinism_guarantee="soft",
    ),
    supported_failure_classes=frozenset(FailureClass),
)


__all__ = [
    "DEFAULT_GENERIC_TOOL_CONTRACT",
    "FailureClass",
    "FailureSeverity",
    "FailureTaxonomyEntry",
    "ReplaySemantics",
    "ToolExecutionContract",
    "ToolInputSchema",
    "ToolOutputSchema",
    "taxonomy_entry_for",
]
