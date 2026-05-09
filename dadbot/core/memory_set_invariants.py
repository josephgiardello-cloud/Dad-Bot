"""memory_set_invariants.py — Strict invariant enforcement for memory-set transitions.

Addresses four previously-open architectural gaps:

  Immediate  Shrink + salience invariants: prohibit entry loss or salience drop
             unless an explicit decay/downgrade signal is present.

  Gap A      Global lifecycle constraint: invariants run at every merge boundary,
             not just tool-gate time.

  Gap C      Causal ordering: record and assert the retrieval→plan→execute→
             post-tool→persist step sequence as a DAG invariant.

  Gap D      Memory decay state machine: formal lifecycle enum with only the
             allowed transitions permitted; all others raise.

  (Gap B — single truth resolution — lives in turn_service.resolve_turn_truth.)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from dadbot.core.execution_memory_view import _memory_entry_signature


# ── Exception ────────────────────────────────────────────────────────────────

class MemorySetInvariantViolation(ValueError):
    """Raised when a prohibited memory-set transition is detected."""


# ── Salience helpers ─────────────────────────────────────────────────────────

_SALIENCE_FIELDS = ("importance_score", "impact_score", "weight")

# A decay/downgrade signal is present when ANY of these boolean keys is truthy
# OR any of the reason keys holds a non-empty string.
_DECAY_MARKER_KEYS = ("_memory_decay_marker", "_downgrade", "decay_marker", "decayed")
_DECAY_REASON_KEYS = ("decay_reason", "downgrade_reason", "_decay_reason")


def _entry_salience(entry: dict[str, Any]) -> float:
    """Return the first numeric salience field found on an entry, else 0.0."""
    for field in _SALIENCE_FIELDS:
        val = entry.get(field)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def _entry_has_decay_signal(entry: dict[str, Any]) -> bool:
    """Return True if the entry carries an explicit decay/downgrade signal."""
    for key in _DECAY_MARKER_KEYS:
        if entry.get(key):
            return True
    for key in _DECAY_REASON_KEYS:
        if str(entry.get(key) or "").strip():
            return True
    return False


# ── Shrink invariant ─────────────────────────────────────────────────────────

def assert_memory_set_shrink_invariant(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    decay_entries: list[dict[str, Any]] | None = None,
    context: str = "",
) -> None:
    """Assert no entries vanish from the retrieval set without a decay/downgrade signal.

    Permitted removals
    ------------------
    * Entries explicitly listed in ``decay_entries``.
    * Entries that carry an inline decay/downgrade marker (``_memory_decay_marker``,
      ``decay_reason``, etc.).

    Raises
    ------
    MemorySetInvariantViolation
        If any entry was silently dropped.
    """
    if not before:
        return

    tag = f" [{context}]" if context else ""

    # Build a set of signatures that are permitted to disappear.
    decay_sigs: set[str] = set()
    for entry in list(decay_entries or []):
        if isinstance(entry, dict):
            decay_sigs.add(_memory_entry_signature(entry))
    for entry in list(before or []):
        if isinstance(entry, dict) and _entry_has_decay_signal(entry):
            decay_sigs.add(_memory_entry_signature(entry))

    before_by_sig: dict[str, dict[str, Any]] = {
        _memory_entry_signature(e): e for e in before if isinstance(e, dict)
    }
    after_sigs: set[str] = {
        _memory_entry_signature(e) for e in after if isinstance(e, dict)
    }

    violations: list[str] = []
    for sig, entry in before_by_sig.items():
        if sig not in after_sigs and sig not in decay_sigs:
            label = str(
                entry.get("summary")
                or entry.get("content")
                or entry.get("memory_id")
                or sig[:12]
            )
            violations.append(label)

    if violations:
        raise MemorySetInvariantViolation(
            f"Prohibited memory-set shrink{tag}: {len(violations)} entries removed without "
            f"a decay/downgrade signal: {violations[:5]}"
        )


# ── Salience invariant ────────────────────────────────────────────────────────

def assert_memory_set_salience_invariant(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    decay_entries: list[dict[str, Any]] | None = None,
    salience_drop_threshold: float = 0.15,
    context: str = "",
) -> None:
    """Assert no entry loses significant salience without a decay/downgrade signal.

    A salience drop ≥ ``salience_drop_threshold`` on an entry without a decay
    marker is a prohibited transition.
    """
    if not before:
        return

    tag = f" [{context}]" if context else ""

    decay_sigs: set[str] = set()
    for entry in list(decay_entries or []):
        if isinstance(entry, dict):
            decay_sigs.add(_memory_entry_signature(entry))

    before_by_sig: dict[str, dict[str, Any]] = {
        _memory_entry_signature(e): e for e in before if isinstance(e, dict)
    }
    after_by_sig: dict[str, dict[str, Any]] = {
        _memory_entry_signature(e): e for e in after if isinstance(e, dict)
    }

    violations: list[str] = []
    for sig, before_entry in before_by_sig.items():
        after_entry = after_by_sig.get(sig)
        if after_entry is None:
            continue  # Handled by shrink invariant.
        if sig in decay_sigs or _entry_has_decay_signal(after_entry):
            continue
        before_sal = _entry_salience(before_entry)
        after_sal = _entry_salience(after_entry)
        drop = before_sal - after_sal
        if drop >= salience_drop_threshold:
            label = str(
                before_entry.get("summary")
                or before_entry.get("content")
                or sig[:12]
            )
            violations.append(f"{label!r}: {before_sal:.3f} → {after_sal:.3f}")

    if violations:
        raise MemorySetInvariantViolation(
            f"Prohibited salience loss{tag}: {len(violations)} entries lost "
            f"≥{salience_drop_threshold} salience without decay signal: {violations[:5]}"
        )


# ── Combined check (Gap A — call at every merge boundary) ────────────────────

def assert_memory_set_invariants(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    decay_entries: list[dict[str, Any]] | None = None,
    salience_drop_threshold: float = 0.15,
    context: str = "",
) -> None:
    """Run both shrink and salience invariants.

    This is the canonical entry point for Gap A enforcement.  Call it at every
    reconciliation/merge boundary — not only at tool-gate time.

    Raises MemorySetInvariantViolation on the first violation found.
    """
    assert_memory_set_shrink_invariant(
        before,
        after,
        decay_entries=decay_entries,
        context=context,
    )
    assert_memory_set_salience_invariant(
        before,
        after,
        decay_entries=decay_entries,
        salience_drop_threshold=salience_drop_threshold,
        context=context,
    )


# ── Gap C — Causal ordering ───────────────────────────────────────────────────

# Canonical causal step order for a turn execution.
CAUSAL_STEP_ORDER: list[str] = [
    "retrieval",
    "planning",
    "tool_execution",
    "post_tool_refresh",
    "persistence",
]

_CAUSAL_STEP_POSITIONS: dict[str, int] = {
    step: idx for idx, step in enumerate(CAUSAL_STEP_ORDER)
}

_CAUSAL_ALLOWED_NEXT: dict[str | None, set[str]] = {
    None: {"retrieval"},
    "retrieval": {"retrieval", "planning"},
    "planning": {"planning", "tool_execution", "persistence"},
    "tool_execution": {"tool_execution", "post_tool_refresh", "persistence"},
    "post_tool_refresh": {"post_tool_refresh", "persistence"},
    "persistence": {"persistence"},
}


def record_causal_step(state: dict[str, Any], step_name: str) -> None:
    """Append a causal step to the turn state log.

    Call this at each lifecycle boundary so ``assert_causal_order`` can
    validate the sequence later.
    """
    steps: list[str] = list(state.get("_causal_step_log") or [])
    steps.append(str(step_name))
    state["_causal_step_log"] = steps


def assert_causal_transition(
    steps: list[str],
    next_step: str,
    *,
    context: str = "",
) -> None:
    """Assert that appending ``next_step`` is valid for the current lifecycle state."""
    tag = f" [{context}]" if context else ""
    normalized_next = str(next_step or "").strip()
    if normalized_next not in _CAUSAL_STEP_POSITIONS:
        raise MemorySetInvariantViolation(
            f"Causal transition violation{tag}: unknown step {normalized_next!r}",
        )

    last_known: str | None = None
    for step in reversed(list(steps or [])):
        if step in _CAUSAL_STEP_POSITIONS:
            last_known = step
            break

    allowed = set(_CAUSAL_ALLOWED_NEXT.get(last_known, set()))
    if normalized_next not in allowed:
        raise MemorySetInvariantViolation(
            f"Causal transition violation{tag}: cannot transition from "
            f"{last_known!r} to {normalized_next!r}; allowed={sorted(allowed)}"
        )


def record_causal_step_locked(
    state: dict[str, Any],
    step_name: str,
    *,
    context: str = "",
) -> None:
    """Record a step only if it respects the causal transition graph."""
    steps: list[str] = list(state.get("_causal_step_log") or [])
    assert_causal_transition(steps, step_name, context=context)
    steps.append(str(step_name))
    state["_causal_step_log"] = steps


def assert_causal_order(
    steps: list[str],
    *,
    context: str = "",
) -> None:
    """Assert that the recorded causal steps appear in valid DAG order.

    Unknown step names are silently ignored so this is forward-compatible.
    Only steps whose names appear in ``CAUSAL_STEP_ORDER`` are validated.

    Raises
    ------
    MemorySetInvariantViolation
        If any known step occurs *before* a known step that has already been
        recorded with a higher canonical position.
    """
    tag = f" [{context}]" if context else ""
    last_position = -1
    for step in steps:
        pos = _CAUSAL_STEP_POSITIONS.get(step, -1)
        if pos < 0:
            continue
        if pos < last_position:
            raise MemorySetInvariantViolation(
                f"Causal order violation{tag}: step '{step}' (position {pos}) "
                f"appeared after a step at position {last_position}; "
                f"expected order: {CAUSAL_STEP_ORDER}"
            )
        last_position = pos


# ── Gap D — Memory lifecycle state machine ────────────────────────────────────

class MemoryLifecycleState(str, Enum):
    """Formal states in the memory entry lifecycle."""

    ACTIVE = "active"
    REINFORCED = "reinforced"
    DECAYING = "decaying"
    ARCHIVED = "archived"
    EXPIRED = "expired"


# Only these (from, to) pairs are permitted.  Any other transition is a contract
# violation.  Decay transitions require an explicit signal (see Gap A).
_ALLOWED_LIFECYCLE_TRANSITIONS: frozenset[tuple[MemoryLifecycleState, MemoryLifecycleState]] = frozenset(
    {
        (MemoryLifecycleState.ACTIVE, MemoryLifecycleState.REINFORCED),
        (MemoryLifecycleState.ACTIVE, MemoryLifecycleState.DECAYING),
        (MemoryLifecycleState.REINFORCED, MemoryLifecycleState.ACTIVE),
        (MemoryLifecycleState.REINFORCED, MemoryLifecycleState.DECAYING),
        (MemoryLifecycleState.DECAYING, MemoryLifecycleState.ARCHIVED),
        (MemoryLifecycleState.DECAYING, MemoryLifecycleState.ACTIVE),   # recovery
        (MemoryLifecycleState.ARCHIVED, MemoryLifecycleState.EXPIRED),
        (MemoryLifecycleState.ARCHIVED, MemoryLifecycleState.ACTIVE),   # resurrection
    }
)


def validate_lifecycle_transition(
    from_state: str | MemoryLifecycleState,
    to_state: str | MemoryLifecycleState,
    *,
    context: str = "",
) -> None:
    """Assert that the lifecycle transition (from_state → to_state) is permitted.

    Raises
    ------
    MemorySetInvariantViolation
        If the transition is not in the allowed set.
    ValueError
        If either state string is not a valid ``MemoryLifecycleState``.
    """
    tag = f" [{context}]" if context else ""
    try:
        from_s = from_state if isinstance(from_state, MemoryLifecycleState) else MemoryLifecycleState(str(from_state).lower())
        to_s = to_state if isinstance(to_state, MemoryLifecycleState) else MemoryLifecycleState(str(to_state).lower())
    except ValueError as exc:
        raise ValueError(
            f"Invalid lifecycle state{tag}: {exc}"
        ) from exc

    if (from_s, to_s) not in _ALLOWED_LIFECYCLE_TRANSITIONS:
        raise MemorySetInvariantViolation(
            f"Prohibited lifecycle transition{tag}: {from_s.value!r} → {to_s.value!r}. "
            f"Allowed transitions from '{from_s.value}': "
            f"{[to.value for (f, to) in _ALLOWED_LIFECYCLE_TRANSITIONS if f == from_s]}"
        )
