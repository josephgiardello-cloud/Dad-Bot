"""Injectable event clock for deterministic replay.

All domain logic that needs a timestamp must call ``now()`` rather than
``time.time()`` directly.  In production the two are equivalent.  In tests
the clock can be frozen to a fixed value so that two replay runs produce
bit-identical write logs.

Usage
-----
Production (default — no change required)::

    from dadbot.core.event_clock import now
    ts = now()  # identical to time.time()

Tests::

    from dadbot.core.event_clock import set_event_clock, reset_event_clock
    set_event_clock(lambda: 1_700_000_000.0)
    # ... run turn ...
    reset_event_clock()
"""
from __future__ import annotations

import time
from collections.abc import Callable

_event_clock: Callable[[], float] = time.time


def now() -> float:
    """Return the current event time from the injectable clock."""
    return _event_clock()


def set_event_clock(clock_fn: Callable[[], float]) -> None:
    """Override the event clock.  Call ``reset_event_clock`` when done."""
    global _event_clock
    _event_clock = clock_fn


def reset_event_clock() -> None:
    """Restore the default wall-clock source."""
    global _event_clock
    _event_clock = time.time
