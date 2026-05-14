from __future__ import annotations

from typing import Any


class RuntimeErrorBase(RuntimeError):
    """Base class for runtime-layer failures that should be machine-reasonable."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})


class RuntimeExecutionError(RuntimeErrorBase):
    """Raised when turn execution fails outside a narrower domain subtype."""


class DadbotRuntimeError(RuntimeErrorBase):
    """Top-level runtime contract error for Dadbot execution boundaries."""


class AuthorityError(DadbotRuntimeError):
    """Raised when an authority boundary is violated."""


class InvariantError(DadbotRuntimeError):
    """Raised when an invariant boundary is violated."""


class BackpressureError(DadbotRuntimeError):
    """Raised when runtime backpressure is applied."""


class ReplayError(DadbotRuntimeError):
    """Raised when replay or idempotency validation fails."""


class AuthorityViolation(AuthorityError):
    """Raised when a runtime authority contract is violated."""


class InvariantViolation(InvariantError):
    """Raised when a runtime invariant is violated."""


class SystemInvariantViolation(InvariantViolation):
    """Raised when a system-level invariant boundary is violated."""


class CanonicalInvariantViolation(SystemInvariantViolation):
    """Raised when canonical single-path execution invariants are violated."""


class ConfigurationError(RuntimeErrorBase):
    """Raised when required runtime configuration is missing or invalid."""


class TransientExecutionError(RuntimeExecutionError):
    """Raised for retryable runtime execution failures (timeouts/backpressure/transient faults)."""


class ReplayMismatch(ReplayError):
    """Raised when replay output diverges from recorded execution shape."""


class ReplayInvariantViolation(ReplayError):
    """Raised when replay mode cannot satisfy an invariant without live execution."""


class ProjectionMismatch(RuntimeErrorBase):
    """Raised when projected trace/state diverges from authoritative execution state."""


class PersistenceFailure(RuntimeErrorBase):
    """Raised when persistence boundaries fail in strict/authoritative paths."""


class ExecutionStageError(RuntimeErrorBase):
    """Raised for failures scoped to a specific execution stage or pipeline step."""


class PartialCommitError(RuntimeErrorBase):
    """Raised when effects were partially committed and reconciliation is required."""


class PoisonExecutionError(RuntimeErrorBase):
    """Raised when a poisoned payload/tool result is detected and must be quarantined."""


NON_FATAL_RUNTIME_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    LookupError,
    OSError,
)


__all__ = [
    "AuthorityError",
    "AuthorityViolation",
    "BackpressureError",
    "CanonicalInvariantViolation",
    "ConfigurationError",
    "DadbotRuntimeError",
    "NON_FATAL_RUNTIME_EXCEPTIONS",
    "ExecutionStageError",
    "InvariantViolation",
    "InvariantError",
    "PartialCommitError",
    "PersistenceFailure",
    "PoisonExecutionError",
    "ProjectionMismatch",
    "SystemInvariantViolation",
    "ReplayError",
    "ReplayInvariantViolation",
    "ReplayMismatch",
    "RuntimeErrorBase",
    "RuntimeExecutionError",
    "TransientExecutionError",
]
