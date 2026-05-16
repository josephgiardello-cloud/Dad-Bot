from __future__ import annotations

import time
from typing import Any

from dadbot.core.control_plane_feedback import _coerce_float
from dadbot.core.runtime_errors import InvariantViolation


SUBMIT_TURN_PHASE_ORDER: tuple[str, ...] = (
    "preflight",
    "register",
    "execution",
    "drain",
    "finalize",
)


def submit_turn_phase_order() -> tuple[str, ...]:
    return SUBMIT_TURN_PHASE_ORDER


def _phase_name(entry: Any) -> str:
    if isinstance(entry, tuple) and len(entry) >= 1:
        return str(entry[0] or "").strip()
    return str(entry or "").strip()


def _phase_ts(entry: Any) -> float | None:
    if isinstance(entry, tuple) and len(entry) >= 2:
        return _coerce_float(entry[1])
    return None


def _validate_submit_turn_phase_progress(phase_trace: list[Any]) -> None:
    expected = submit_turn_phase_order()
    observed = tuple(_phase_name(item) for item in list(phase_trace))
    if len(observed) > len(expected):
        raise InvariantViolation(
            "submit_turn phase ordering violated",
            context={
                "expected_order": list(expected),
                "observed": list(observed),
                "reason": "observed phases exceed expected order length",
            },
        )
    prefix = expected[: len(observed)]
    if observed != prefix:
        raise InvariantViolation(
            "submit_turn phase ordering violated",
            context={
                "expected_prefix": list(prefix),
                "observed": list(observed),
            },
        )

    observed_ts = [_phase_ts(item) for item in list(phase_trace)]
    for index in range(1, len(observed_ts)):
        prev_ts = observed_ts[index - 1]
        curr_ts = observed_ts[index]
        if prev_ts is None or curr_ts is None:
            continue
        if curr_ts < prev_ts:
            raise InvariantViolation(
                "submit_turn phase ordering violated",
                context={
                    "expected_prefix": list(prefix),
                    "observed": list(observed),
                    "reason": "phase timestamps are non-monotonic",
                    "phase_index": index,
                },
            )


def _append_submit_turn_phase(phase_trace: list[tuple[str, float]], phase: str) -> None:
    next_index = len(phase_trace)
    expected = submit_turn_phase_order()
    if next_index >= len(expected):
        raise InvariantViolation(
            "submit_turn phase ordering violated",
            context={
                "expected_order": list(expected),
                "observed": [name for name, _ in list(phase_trace)],
                "attempted_phase": str(phase or "").strip(),
                "reason": "attempted to append phase past terminal finalize phase",
            },
        )

    expected_phase = expected[next_index]
    normalized_phase = str(phase or "").strip()
    if normalized_phase != expected_phase:
        raise InvariantViolation(
            "submit_turn phase ordering violated",
            context={
                "expected_phase": expected_phase,
                "observed_phase": normalized_phase,
                "phase_index": next_index,
            },
        )

    phase_trace.append((normalized_phase, float(time.time())))
    _validate_submit_turn_phase_progress(phase_trace)


def _assert_submit_turn_phase_trace_complete(phase_trace: list[tuple[str, float]]) -> None:
    observed = tuple(name for name, _ts in list(phase_trace))
    expected = submit_turn_phase_order()
    if observed != expected:
        raise InvariantViolation(
            "submit_turn phase ordering violated",
            context={
                "expected_order": list(expected),
                "observed": list(observed),
                "phase_trace": [
                    {"phase": str(name), "timestamp": float(ts)}
                    for name, ts in list(phase_trace)
                ],
            },
        )


def _assert_submit_turn_phase_boundary(
    *,
    phase_trace: list[tuple[str, float]],
    expected_phase: str,
    operation: str,
) -> None:
    current = str(phase_trace[-1][0] if phase_trace else "")
    if current != str(expected_phase or "").strip():
        raise InvariantViolation(
            "submit_turn phase boundary violation",
            context={
                "operation": str(operation or ""),
                "expected_phase": str(expected_phase or ""),
                "current_phase": current,
                "phase_trace": [
                    {"phase": str(name), "timestamp": float(ts)}
                    for name, ts in list(phase_trace)
                ],
            },
        )


__all__ = [
    "SUBMIT_TURN_PHASE_ORDER",
    "submit_turn_phase_order",
    "_append_submit_turn_phase",
    "_assert_submit_turn_phase_boundary",
    "_assert_submit_turn_phase_trace_complete",
]