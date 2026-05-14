from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class TransitionSystem(Protocol):
    def initial_state(self) -> Any: ...
    def enabled_actions(self, state: Any) -> list[Any]: ...
    def step(self, state: Any, action: Any) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...


@dataclass(frozen=True)
class ClosureCounterexample:
    path: list[str]
    final_hash: str


@dataclass(frozen=True)
class ClosureReport:
    explored_states: int
    explored_edges: int
    converged: bool
    terminal_hashes: list[str]
    counterexample: ClosureCounterexample | None = None


class ClosureUnsafeClass(str, Enum):
    SAFE = "safe"
    NO_TERMINAL_REACHED = "no_terminal_reached"
    UNSAFE_NON_CRASHING_DIVERGENCE = "unsafe_non_crashing_divergence"


@dataclass(frozen=True)
class ClosureClassification:
    classification: ClosureUnsafeClass
    reason: str
    unsafe: bool


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def explore_closure(
    system: TransitionSystem,
    *,
    max_depth: int = 6,
    max_states: int = 50_000,
) -> ClosureReport:
    """Exhaustively explore interleavings up to bounded depth/state count.

    Convergence means all terminal states reached in the explored region reduce to
    a single canonical hash.
    """
    initial = system.initial_state()
    initial_hash = _stable_hash(initial)

    queue: deque[tuple[Any, int, list[str]]] = deque([(initial, 0, [])])
    seen: set[str] = {initial_hash}

    explored_edges = 0
    terminal_hashes: dict[str, list[str]] = {}

    while queue and len(seen) < max_states:
        state, depth, path = queue.popleft()
        if system.is_terminal(state):
            h = _stable_hash(state)
            terminal_hashes.setdefault(h, path)
            continue

        if depth >= max_depth:
            continue

        for action in system.enabled_actions(state):
            next_state = system.step(state, action)
            next_hash = _stable_hash(next_state)
            explored_edges += 1
            if next_hash in seen:
                continue
            seen.add(next_hash)
            queue.append((next_state, depth + 1, [*path, str(action)]))

    hashes = sorted(terminal_hashes.keys())
    if len(hashes) <= 1:
        return ClosureReport(
            explored_states=len(seen),
            explored_edges=explored_edges,
            converged=True,
            terminal_hashes=hashes,
            counterexample=None,
        )

    worst = hashes[-1]
    return ClosureReport(
        explored_states=len(seen),
        explored_edges=explored_edges,
        converged=False,
        terminal_hashes=hashes,
        counterexample=ClosureCounterexample(path=terminal_hashes[worst], final_hash=worst),
    )


def classify_closure_report(report: ClosureReport) -> ClosureClassification:
    """Classify closure exploration into deterministic safety classes."""
    terminal_hashes = sorted(str(item) for item in list(report.terminal_hashes or []))
    if not terminal_hashes:
        return ClosureClassification(
            classification=ClosureUnsafeClass.NO_TERMINAL_REACHED,
            reason="bounded exploration did not reach any terminal state",
            unsafe=True,
        )
    if bool(report.converged):
        return ClosureClassification(
            classification=ClosureUnsafeClass.SAFE,
            reason="all explored terminal states converged to one canonical hash",
            unsafe=False,
        )
    return ClosureClassification(
        classification=ClosureUnsafeClass.UNSAFE_NON_CRASHING_DIVERGENCE,
        reason="multiple terminal hashes observed under bounded adversarial interleavings",
        unsafe=True,
    )


__all__ = [
    "TransitionSystem",
    "ClosureCounterexample",
    "ClosureReport",
    "ClosureUnsafeClass",
    "ClosureClassification",
    "explore_closure",
    "classify_closure_report",
]
