from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True)
class MutationEvent:
    event_type: str
    payload: dict[str, Any]
    source: str


def emit_event(event_type: str, payload: dict[str, Any] | None = None, *, source: str = "") -> MutationEvent:
    """Create an immutable mutation event envelope."""
    return MutationEvent(
        event_type=str(event_type or "MUTATION_EVENT"),
        payload=deepcopy(dict(payload or {})),
        source=str(source or "kernel"),
    )


def apply_event(event: MutationEvent, state: _T, applier: Callable[[_T, MutationEvent], _T]) -> _T:
    """Apply a mutation only through an event envelope.

    This enforces the "event first, then mutate" contract and uses copy-on-write
    semantics for mutable state payloads.
    """
    if not isinstance(event, MutationEvent):
        raise RuntimeError("kernel_mutation_gate.apply_event requires MutationEvent")
    if not callable(applier):
        raise RuntimeError("kernel_mutation_gate.apply_event requires callable applier")
    return applier(deepcopy(state), event)
