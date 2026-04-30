"""Tool State Algebra (Phase 4) and Tool Failure Algebra (Phase 5).

Phase 4 — ToolState as a Monoid:
  - Identity element: ToolState.identity()
  - Associative merge: state_a.merge(state_b) == state_b.merge(state_a) for
    commutative cases; general merge is left-biased for sequence ordering
  - Deterministic composition: ToolState.compose([s1, s2, ...])

Phase 5 — ToolFailure Algebra:
  - Replaces boolean success/failure with structured ToolFailure
  - Typed failure categories, severity levels, recoverability, propagation policy
  - Enables partial execution graphs and resilience modeling
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


# ---------------------------------------------------------------------------
# Phase 4: ToolState — a monoid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolState:
    """Immutable tool execution state as a monoid.

    Monoid laws:
      1. Identity:     ToolState.identity().merge(s) == s.merge(ToolState.identity()) == s
      2. Associativity: (a.merge(b)).merge(c) == a.merge(b.merge(c))

    The ``composite_hash`` is derived from the ordered results list, so
    equal states produce equal hashes.  Merge concatenates results in
    left-then-right order and recomputes the composite hash.
    """

    results: tuple[dict[str, Any], ...]
    event_count: int
    composite_hash: str

    @staticmethod
    def identity() -> ToolState:
        """The monoid identity element (empty state)."""
        return ToolState(
            results=(),
            event_count=0,
            composite_hash=_stable_hash({"results": [], "event_count": 0}),
        )

    @classmethod
    def from_results(cls, results: list[dict[str, Any]]) -> ToolState:
        """Construct a ToolState from a list of tool result dicts."""
        frozen = tuple(dict(r) for r in (results or []))
        return cls(
            results=frozen,
            event_count=len(frozen),
            composite_hash=_stable_hash(
                {"results": [dict(r) for r in frozen], "event_count": len(frozen)},
            ),
        )

    def merge(self, other: ToolState) -> ToolState:
        """Associative merge: combine two ToolStates into one.

        Results are concatenated left-then-right and re-sequenced.
        The composite hash is recomputed from the merged result set.
        """
        merged_results = self.results + other.results
        merged_event_count = self.event_count + other.event_count
        return ToolState(
            results=merged_results,
            event_count=merged_event_count,
            composite_hash=_stable_hash(
                {
                    "results": [dict(r) for r in merged_results],
                    "event_count": merged_event_count,
                },
            ),
        )

    def __add__(self, other: ToolState) -> ToolState:
        """Operator alias for merge: state_a + state_b."""
        return self.merge(other)

    @classmethod
    def compose(cls, states: list[ToolState]) -> ToolState:
        """Left-fold over a list of ToolStates using merge.

        compose([]) == identity()
        compose([s]) == s
        compose([s1, s2, s3]) == s1.merge(s2).merge(s3)
        """
        result = cls.identity()
        for s in states or []:
            result = result.merge(s)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [dict(r) for r in self.results],
            "event_count": self.event_count,
            "composite_hash": self.composite_hash,
        }


# ---------------------------------------------------------------------------
# Phase 5: ToolFailure Algebra
# ---------------------------------------------------------------------------


class ToolFailureType(enum.Enum):
    """Categorical classification of tool failure causes."""

    VALIDATION = "validation"  # Invalid args / unsupported tool
    EXECUTION = "execution"  # Runtime error during dispatch
    TIMEOUT = "timeout"  # Execution exceeded time budget
    UNSUPPORTED = "unsupported"  # Tool not in allowed set
    STATE_CORRUPTION = "state_corruption"  # Output violated state integrity


class FailureSeverity(enum.Enum):
    """Impact level of a tool failure."""

    LOW = "low"  # Non-blocking; result omitted gracefully
    MEDIUM = "medium"  # Partial degradation; inference may proceed
    HIGH = "high"  # Significant capability loss; should warn
    CRITICAL = "critical"  # Execution cannot continue safely


class PropagationPolicy(enum.Enum):
    """How a failure propagates through the execution graph."""

    HALT = "halt"  # Stop entire execution graph
    SKIP = "skip"  # Skip this node; continue graph
    RETRY = "retry"  # Retry this node (if retries remain)
    FALLBACK = "fallback"  # Use fallback output and continue


@dataclass(frozen=True)
class ToolFailure:
    """Structured tool failure replacing boolean success/failure.

    Enables:
    - Partial execution graphs (SKIP policy)
    - Resilience modeling (recoverable + RETRY policy)
    - Non-binary inference flows (severity gradient)
    - Structured error propagation (propagation_policy)
    """

    failure_type: ToolFailureType
    severity: FailureSeverity
    recoverable: bool
    propagation_policy: PropagationPolicy
    tool_id: str
    message: str

    @classmethod
    def validation_error(cls, tool_id: str, message: str) -> ToolFailure:
        return cls(
            failure_type=ToolFailureType.VALIDATION,
            severity=FailureSeverity.MEDIUM,
            recoverable=False,
            propagation_policy=PropagationPolicy.SKIP,
            tool_id=tool_id,
            message=message,
        )

    @classmethod
    def execution_error(
        cls,
        tool_id: str,
        message: str,
        *,
        recoverable: bool = True,
    ) -> ToolFailure:
        return cls(
            failure_type=ToolFailureType.EXECUTION,
            severity=FailureSeverity.HIGH,
            recoverable=recoverable,
            propagation_policy=PropagationPolicy.RETRY if recoverable else PropagationPolicy.FALLBACK,
            tool_id=tool_id,
            message=message,
        )

    @classmethod
    def unsupported_tool(cls, tool_id: str, tool_name: str) -> ToolFailure:
        return cls(
            failure_type=ToolFailureType.UNSUPPORTED,
            severity=FailureSeverity.HIGH,
            recoverable=False,
            propagation_policy=PropagationPolicy.HALT,
            tool_id=tool_id,
            message=f"Tool {tool_name!r} is not in the allowed set",
        )

    def should_halt(self) -> bool:
        return self.propagation_policy == PropagationPolicy.HALT

    def should_skip(self) -> bool:
        return self.propagation_policy == PropagationPolicy.SKIP

    def should_retry(self) -> bool:
        return self.propagation_policy == PropagationPolicy.RETRY and self.recoverable

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "propagation_policy": self.propagation_policy.value,
            "tool_id": self.tool_id,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# ToolFailureLog: ordered collection for partial-execution tracking
# ---------------------------------------------------------------------------


@dataclass
class ToolFailureLog:
    """Ordered log of ToolFailures encountered during execution."""

    failures: list[ToolFailure] = field(default_factory=list)

    def append(self, failure: ToolFailure) -> None:
        self.failures.append(failure)

    def has_halt(self) -> bool:
        return any(f.should_halt() for f in self.failures)

    def critical_failures(self) -> list[ToolFailure]:
        return [f for f in self.failures if f.severity == FailureSeverity.CRITICAL]

    def recoverable_failures(self) -> list[ToolFailure]:
        return [f for f in self.failures if f.recoverable]

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.failures]


__all__ = [
    "FailureSeverity",
    "PropagationPolicy",
    "ToolFailure",
    "ToolFailureLog",
    "ToolFailureType",
    "ToolState",
]
