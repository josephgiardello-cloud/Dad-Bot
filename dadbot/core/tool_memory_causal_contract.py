"""Tool→Memory Causal Contract.

Defines the explicit rules governing how tool execution results are written
into long-term memory.  Every execution that passes through
ContractAwareToolRuntime produces a CausalMemoryEntry that records:

  - which tool ran (name + contract version)
  - how many attempts were made
  - what the outcome was (status, failure_class, severity)
  - what the failure policy engine decided (policy_action)
  - a stable causal key for deduplication and tracing

Callers set a CausalWritePolicy to control which outcomes are persisted.
The default policy (ALWAYS) writes both successes and failures so that the
memory layer captures the full causal history of tool executions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Write policy
# ---------------------------------------------------------------------------


class CausalWritePolicy(Enum):
    """Controls which tool execution outcomes are written to memory."""

    ALWAYS = "always"
    """Persist every outcome — success, partial, degraded, error, timeout."""

    SUCCESS_ONLY = "success_only"
    """Persist only fully-successful (OK) executions."""

    FAILURE_ONLY = "failure_only"
    """Persist only non-OK outcomes."""

    NEVER = "never"
    """Do not write to memory at all (useful for read-only/diagnostic tools)."""


# ---------------------------------------------------------------------------
# Causal memory entry
# ---------------------------------------------------------------------------


@dataclass
class CausalMemoryEntry:
    """Structured record of a single tool execution event for memory ingestion.

    This is the canonical causal unit: every field is traceable back to a
    specific execution, contract, and (optionally) policy decision.
    """

    tool_name: str
    contract_version: str
    attempt: int
    status: str                         # ToolExecutionStatus.value
    causal_key: str                     # Stable unique key: tool:version:timestamp_ms
    timestamp_ms: int
    latency_ms: float

    # Populated for non-OK outcomes
    failure_class: str | None = None    # FailureClass.value
    failure_severity: str | None = None  # FailureSeverity.value
    failure_retryable: bool | None = None

    # Populated when FailurePolicyEngine was consulted
    policy_action: str | None = None    # PolicyAction.value

    # Output / error surface
    output_preview: str = ""
    error: str = ""

    # Forward any extra metadata from the enriched result
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_sink_event(self) -> dict[str, Any]:
        """Convert to the dict format expected by ToolResultMemorySink.__call__."""
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "attempts": self.attempt,
            "latency_ms": self.latency_ms,
            "degraded_reason": "",
            "error": self.error,
            "output_preview": self.output_preview,
            "captured_at_ms": self.timestamp_ms,
            "metadata": {
                "contract_version": self.contract_version,
                "failure_class": self.failure_class,
                "failure_severity": self.failure_severity,
                "failure_retryable": self.failure_retryable,
                "policy_action": self.policy_action,
                "causal_key": self.causal_key,
                **self.extra_metadata,
            },
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _safe_preview(value: Any, *, max_chars: int = 500) -> str:
    text = str(value)
    return text[:max_chars] + "..." if len(text) > max_chars else text


def build_causal_entry(
    result: Any,                          # ToolExecutionResult
    contract: Any,                        # ToolExecutionContract
    attempt: int,
    *,
    policy_action: str | None = None,
    clock_fn: Any = None,                 # injectable for tests
) -> CausalMemoryEntry:
    """Build a CausalMemoryEntry from an execution result and its contract.

    Args:
        result: ToolExecutionResult (possibly enriched with failure metadata).
        contract: ToolExecutionContract used to validate the execution.
        attempt: 1-based attempt number for this execution.
        policy_action: PolicyAction.value from FailurePolicyEngine.decide(), if consulted.
        clock_fn: Optional callable returning current epoch-ms (for test determinism).

    Returns:
        CausalMemoryEntry ready for emission.
    """
    ts_ms = int((clock_fn() if clock_fn is not None else time.time()) * 1000)
    causal_key = f"{contract.tool_name}:{contract.version}:{ts_ms}"

    # Pull taxonomy metadata written by normalize_result_failure_class()
    metadata = dict(result.metadata or {})
    failure_class = metadata.get("failure_class")
    failure_severity = metadata.get("failure_severity")
    failure_retryable = metadata.get("failure_retryable")

    # Everything else goes to extra_metadata (but skip the already-extracted keys)
    _extracted = {"failure_class", "failure_severity", "failure_retryable", "escalation_key", "failure_description"}
    extra = {k: v for k, v in metadata.items() if k not in _extracted}

    output_preview = _safe_preview(result.output) if result.output is not None else ""

    return CausalMemoryEntry(
        tool_name=contract.tool_name,
        contract_version=contract.version,
        attempt=attempt,
        status=result.status.value,
        causal_key=causal_key,
        timestamp_ms=ts_ms,
        latency_ms=float(result.latency_ms or 0.0),
        failure_class=failure_class,
        failure_severity=failure_severity,
        failure_retryable=failure_retryable,
        policy_action=policy_action,
        output_preview=output_preview,
        error=result.error or "",
        extra_metadata=extra,
    )


# ---------------------------------------------------------------------------
# Write gating
# ---------------------------------------------------------------------------


def should_write_entry(entry: CausalMemoryEntry, policy: CausalWritePolicy) -> bool:
    """Return True if the entry should be written under the given policy."""
    if policy == CausalWritePolicy.NEVER:
        return False
    if policy == CausalWritePolicy.ALWAYS:
        return True
    success_status = "ok"  # ToolExecutionStatus.OK.value
    if policy == CausalWritePolicy.SUCCESS_ONLY:
        return entry.status == success_status
    if policy == CausalWritePolicy.FAILURE_ONLY:
        return entry.status != success_status
    return True


def emit_causal_entry(
    sink: Any,                           # ToolResultMemorySink or any callable(dict)
    entry: CausalMemoryEntry,
    policy: CausalWritePolicy = CausalWritePolicy.ALWAYS,
) -> bool:
    """Emit a CausalMemoryEntry to the memory sink if policy allows.

    Args:
        sink: Any callable that accepts a dict event (e.g., ToolResultMemorySink).
        entry: The causal entry to emit.
        policy: Controls whether the entry is written.

    Returns:
        True if the entry was written, False if skipped by policy.
    """
    if not should_write_entry(entry, policy):
        return False
    event = entry.to_sink_event()
    sink(event)
    return True


__all__ = [
    "CausalMemoryEntry",
    "CausalWritePolicy",
    "build_causal_entry",
    "emit_causal_entry",
    "should_write_entry",
]
