"""Fault injection, error classification, and deterministic retry policy.

ErrorClassification:
  Labels every exception as RETRYABLE, TERMINAL, or COMPENSATING so
  the caller can route to the appropriate recovery path.

FaultInjector:
  Registers named failure points with deterministic arm-count or
  probabilistic triggers.  Used in tests to simulate crash scenarios
  without monkey-patching or mocking.

RetryPolicy:
  Configurable max-attempts + exponential backoff with jitter.

FaultBoundary:
  Context manager that classifies exceptions and invokes per-class handlers.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from enum import Enum
from threading import RLock
from typing import Any, Callable, Type


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorClassification(Enum):
    RETRYABLE    = "retryable"     # transient; safe to retry with backoff
    TERMINAL     = "terminal"      # permanent; escalate / dead-letter
    COMPENSATING = "compensating"  # partial success; undo required before retry


class RetryableError(RuntimeError):
    """Transient error — safe to retry."""
    classification = ErrorClassification.RETRYABLE


class TerminalError(RuntimeError):
    """Permanent error — do not retry."""
    classification = ErrorClassification.TERMINAL


class CompensatingActionRequired(RuntimeError):
    """Partial success — compensating action required before any retry."""
    classification = ErrorClassification.COMPENSATING


# ---------------------------------------------------------------------------
# Error classifier registry
# ---------------------------------------------------------------------------

class ErrorClassifier:
    """Maps exception types to ErrorClassification values via isinstance checks."""

    _DEFAULT_MAP: dict[Type[Exception], ErrorClassification] = {
        RetryableError:              ErrorClassification.RETRYABLE,
        TerminalError:               ErrorClassification.TERMINAL,
        CompensatingActionRequired:  ErrorClassification.COMPENSATING,
        ConnectionError:             ErrorClassification.RETRYABLE,
        TimeoutError:                ErrorClassification.RETRYABLE,
        OSError:                     ErrorClassification.RETRYABLE,
        PermissionError:             ErrorClassification.TERMINAL,
        ValueError:                  ErrorClassification.TERMINAL,
        TypeError:                   ErrorClassification.TERMINAL,
        RuntimeError:                ErrorClassification.TERMINAL,
        KeyError:                    ErrorClassification.TERMINAL,
    }

    def __init__(self) -> None:
        self._map: dict[Type[Exception], ErrorClassification] = dict(self._DEFAULT_MAP)

    def register(self, exc_type: Type[Exception], classification: ErrorClassification) -> None:
        self._map[exc_type] = classification

    def classify(self, exc: BaseException) -> ErrorClassification:
        for t, c in self._map.items():
            if isinstance(exc, t):
                return c
        return ErrorClassification.TERMINAL


_CLASSIFIER = ErrorClassifier()


def classify_error(exc: BaseException) -> ErrorClassification:
    return _CLASSIFIER.classify(exc)


def register_error_class(exc_type: Type[Exception], classification: ErrorClassification) -> None:
    _CLASSIFIER.register(exc_type, classification)


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class RetryPolicy:
    """Deterministic retry with exponential backoff and error classification."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay_seconds: float = 0.1,
        max_delay_seconds: float = 10.0,
        backoff_factor: float = 2.0,
        jitter: bool = False,
    ) -> None:
        self._max_attempts    = max(1, int(max_attempts))
        self._base_delay      = max(0.0, float(base_delay_seconds))
        self._max_delay       = max(0.0, float(max_delay_seconds))
        self._backoff_factor  = max(1.0, float(backoff_factor))
        self._jitter          = bool(jitter)

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        """Return True if the error should be retried (attempt is 1-based)."""
        if attempt >= self._max_attempts:
            return False
        return classify_error(exc) == ErrorClassification.RETRYABLE

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait before the given retry attempt (1-based)."""
        delay = self._base_delay * (self._backoff_factor ** (attempt - 1))
        delay = min(delay, self._max_delay)
        if self._jitter:
            delay *= (0.5 + random.random() * 0.5)
        return delay

    def execute(
        self,
        fn: Callable,
        *args,
        on_retry: Callable[[int, BaseException], None] | None = None,
        **kwargs,
    ) -> Any:
        """Execute fn(*args, **kwargs) with retry semantics."""
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not self.should_retry(exc, attempt):
                    raise
                if on_retry:
                    on_retry(attempt, exc)
                wait = self.delay_for(attempt)
                if wait > 0:
                    time.sleep(wait)
        assert last_exc is not None
        raise last_exc


# ---------------------------------------------------------------------------
# FaultInjector
# ---------------------------------------------------------------------------

class FaultInjector:
    """Named failure-point registry for chaos-engineering and tests.

    Usage (tests — deterministic)::

        injector = FaultInjector()
        injector.arm("ledger.append", count=1)
        with pytest.raises(RetryableError):
            injector.check("ledger.append")

    Usage (production — probabilistic)::

        injector = FaultInjector()
        injector.register("ledger.append", probability=0.01)
        injector.check("ledger.append")  # 1% chance of RetryableError
    """

    def __init__(self) -> None:
        self._lock           = RLock()
        self._armed:       dict[str, int]              = defaultdict(int)
        self._probability: dict[str, float]            = {}
        self._exc_types:   dict[str, Type[Exception]]  = {}
        self._triggered:   dict[str, int]              = defaultdict(int)

    def arm(
        self,
        name: str,
        *,
        count: int = 1,
        exc_type: Type[Exception] = RetryableError,
    ) -> None:
        with self._lock:
            self._armed[name] += max(1, int(count))
            self._exc_types[name] = exc_type

    def register(
        self,
        name: str,
        *,
        probability: float,
        exc_type: Type[Exception] = RetryableError,
    ) -> None:
        with self._lock:
            self._probability[name] = max(0.0, min(1.0, float(probability)))
            self._exc_types[name] = exc_type

    def disarm(self, name: str) -> None:
        with self._lock:
            self._armed.pop(name, None)
            self._probability.pop(name, None)

    def check(self, name: str) -> None:
        """Raise the configured exception if fault triggers.  No-op otherwise."""
        with self._lock:
            exc_type = self._exc_types.get(name, RetryableError)
            if self._armed.get(name, 0) > 0:
                self._armed[name] -= 1
                self._triggered[name] += 1
                raise exc_type(f"Fault injected at [{name}]")
            prob = self._probability.get(name, 0.0)
            if prob > 0.0 and random.random() < prob:
                self._triggered[name] += 1
                raise exc_type(f"Probabilistic fault injected at [{name}]")

    def triggered_count(self, name: str) -> int:
        with self._lock:
            return self._triggered.get(name, 0)

    def reset(self) -> None:
        with self._lock:
            self._armed.clear()
            self._probability.clear()
            self._triggered.clear()


_DEFAULT_INJECTOR = FaultInjector()


def get_fault_injector() -> FaultInjector:
    return _DEFAULT_INJECTOR


# ---------------------------------------------------------------------------
# FaultBoundary context manager
# ---------------------------------------------------------------------------

class FaultBoundary:
    """Classify exceptions and route to appropriate handlers.

    Usage::

        with FaultBoundary(
            "ledger.write",
            on_retryable=lambda e: metrics.increment("ledger.write.retried"),
            on_terminal=lambda e: logger.error("terminal", exc=e),
        ):
            ledger.write(event)
    """

    def __init__(
        self,
        name: str = "",
        *,
        on_retryable:    Callable[[Exception], None] | None = None,
        on_terminal:     Callable[[Exception], None] | None = None,
        on_compensating: Callable[[Exception], None] | None = None,
        injector: FaultInjector | None = None,
    ) -> None:
        self._name           = name
        self._on_retryable   = on_retryable
        self._on_terminal    = on_terminal
        self._on_compensating = on_compensating
        self._injector       = injector or _DEFAULT_INJECTOR
        self.classification: ErrorClassification | None = None

    def __enter__(self) -> "FaultBoundary":
        if self._name:
            self._injector.check(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is None:
            return False
        c = classify_error(exc_val)
        self.classification = c
        handler = {
            ErrorClassification.RETRYABLE:    self._on_retryable,
            ErrorClassification.TERMINAL:     self._on_terminal,
            ErrorClassification.COMPENSATING: self._on_compensating,
        }.get(c)
        if handler is not None:
            handler(exc_val)
            return True  # suppress
        return False  # re-raise
