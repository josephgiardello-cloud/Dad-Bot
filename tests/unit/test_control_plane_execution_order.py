from __future__ import annotations

import pytest
import time

from dadbot.core.control_plane import submit_turn_phase_order
from dadbot.core.control_plane import _validate_submit_turn_phase_progress
from dadbot.core.control_plane import _assert_submit_turn_phase_trace_complete
from dadbot.core.runtime_errors import InvariantViolation


def test_submit_turn_phase_order_is_single_authority():
    assert submit_turn_phase_order() == (
        "preflight",
        "register",
        "execution",
        "drain",
        "finalize",
    )


def test_submit_turn_phase_progress_accepts_valid_prefix() -> None:
    _validate_submit_turn_phase_progress(
        [
            ("preflight", time.time()),
            ("register", time.time() + 0.001),
        ],
    )


def test_submit_turn_phase_progress_rejects_out_of_order() -> None:
    with pytest.raises(InvariantViolation):
        _validate_submit_turn_phase_progress(
            [
                "preflight",
                "execution",
            ],
        )


def test_submit_turn_phase_trace_requires_strict_equivalence() -> None:
    with pytest.raises(InvariantViolation):
        _assert_submit_turn_phase_trace_complete(
            [
                ("preflight", time.time()),
                ("register", time.time() + 0.001),
                ("execution", time.time() + 0.002),
            ],
        )
