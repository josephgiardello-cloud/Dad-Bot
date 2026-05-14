"""Sealed type-safe schemas for execution results (Phase 1: Hardening)."""

from typing import Any, NotRequired, TypedDict


class DegradationInfo(TypedDict):
    """Sealed degradation tracking schema"""
    count: int
    items: list[dict[str, Any]]


class TimeoutInfo(TypedDict):
    """Sealed timeout state schema"""
    seconds: float
    timed_out: bool


class OutputsInfo(TypedDict):
    """Sealed execution outputs schema"""
    response: str
    should_end: bool
    semantic_eval_input_hash: str


FailureInfo = TypedDict(
    "FailureInfo",
    {
        "class": str,
        "type": str,
        "message": str,
        "source": str,
        "retryable": bool,
    },
)


class ExecutionResultEnvelope(TypedDict):
    """
    Sealed execution result envelope.
    
    This is the canonical shape for all execution results flowing through the system.
    All four invariants are enforced at construction time.
    """
    status: str
    degradation: DegradationInfo
    failure: FailureInfo
    timeout: TimeoutInfo
    outputs: OutputsInfo


class ExecutionResultEnvelopeCompat(ExecutionResultEnvelope, total=False):
    """Compatibility shape that preserves legacy passthrough fields."""

    initial_result: NotRequired[Any]
    result: NotRequired[Any]
    candidates: NotRequired[Any]
