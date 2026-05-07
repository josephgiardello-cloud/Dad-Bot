from __future__ import annotations

from typing import Any


class RuntimeErrorBase(RuntimeError):
    """Base class for runtime-layer failures that should be machine-reasonable."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})


class RuntimeExecutionError(RuntimeErrorBase):
    """Raised when turn execution fails outside a narrower domain subtype."""


class InvariantViolation(RuntimeErrorBase):
    """Raised when a runtime invariant is violated."""


class ProjectionMismatch(RuntimeErrorBase):
    """Raised when projected trace/state diverges from authoritative execution state."""


class PersistenceFailure(RuntimeErrorBase):
    """Raised when persistence boundaries fail in strict/authoritative paths."""


class ExecutionStageError(RuntimeErrorBase):
    """Raised for failures scoped to a specific execution stage or pipeline step."""


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
    "ExecutionStageError",
    "InvariantViolation",
    "NON_FATAL_RUNTIME_EXCEPTIONS",
    "PersistenceFailure",
    "ProjectionMismatch",
    "RuntimeErrorBase",
    "RuntimeExecutionError",
]
