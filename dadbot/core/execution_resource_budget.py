"""Execution resource budget and concurrency boundary model.

Design contract
---------------
Every active ``TurnGraph.execute()`` call must acquire a concurrency slot
before the execution pipeline begins.  When all slots are exhausted the
``BackpressurePolicy`` decides whether to reject the turn immediately or raise
a ``ConcurrencyBudgetExceeded`` error.

Architectural role
------------------
::

    Kernel (execution runtime)
        └── ExecutionResourceBudget       ← this module
               ├── ConcurrencyBudget      (max concurrent turns, per-stage parallelism)
               ├── ResourceAccounter      (tracks in-flight executions)
               ├── BackpressurePolicy     (reject / raise on budget overflow)
               └── ExecutionResourceGuard (context manager for slot acquisition)

Integration
-----------
Wire the budget into the runtime adapter or app layer:

::

    from dadbot.core.execution_resource_budget import ExecutionResourceBudget

    budget = ExecutionResourceBudget(
        max_concurrent_turns=10,
        max_stage_parallelism=4,
    )

    async with budget.acquire(trace_id=turn_context.trace_id):
        result = await graph.execute(turn_context)

The graph itself is unaware of the budget — resource accounting is a
cross-cutting concern managed at the **kernel/adapter** boundary.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConcurrencyBudgetExceeded(RuntimeError):
    """Raised when a new execution cannot be admitted due to concurrency limits."""

    def __init__(self, *, max_concurrent: int, current_inflight: int, trace_id: str = "") -> None:
        self.max_concurrent = int(max_concurrent)
        self.current_inflight = int(current_inflight)
        self.trace_id = str(trace_id or "")
        super().__init__(
            f"Concurrency budget exceeded: max_concurrent={max_concurrent}, "
            f"current_inflight={current_inflight}, trace_id={trace_id!r}"
        )


class BackpressureSignal(RuntimeError):
    """Raised when the backpressure policy rejects a turn due to resource pressure.

    Callers can catch this to return a graceful degraded response instead of
    propagating the error.
    """

    def __init__(self, *, reason: str, retry_after_ms: float = 0.0, trace_id: str = "") -> None:
        self.reason = str(reason or "")
        self.retry_after_ms = float(retry_after_ms or 0.0)
        self.trace_id = str(trace_id or "")
        super().__init__(
            f"Backpressure: {self.reason} (retry_after_ms={self.retry_after_ms:.0f}, "
            f"trace_id={self.trace_id!r})"
        )


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConcurrencyBudget:
    """Immutable concurrency limits for turn execution.

    Attributes
    ----------
    max_concurrent_turns:
        Maximum number of turns that may execute concurrently across all
        sessions.  Additional turns are rejected via the backpressure policy.
    max_stage_parallelism:
        Maximum number of pipeline stages that may execute in parallel within
        a single turn (used by future parallel-node scheduling).
    soft_limit_turns:
        When inflight count reaches this value the backpressure policy emits a
        warning but does not reject.  Must be ≤ ``max_concurrent_turns``.
    """

    max_concurrent_turns: int = 10
    max_stage_parallelism: int = 4
    soft_limit_turns: int = 8

    def __post_init__(self) -> None:
        if self.max_concurrent_turns < 1:
            raise ValueError("max_concurrent_turns must be >= 1")
        if self.max_stage_parallelism < 1:
            raise ValueError("max_stage_parallelism must be >= 1")
        if self.soft_limit_turns > self.max_concurrent_turns:
            raise ValueError("soft_limit_turns must be <= max_concurrent_turns")


@dataclass(frozen=True)
class BackpressurePolicy:
    """Policy controlling behaviour when the concurrency budget is exceeded.

    Attributes
    ----------
    action:
        ``"reject"`` — raise ``BackpressureSignal`` immediately.
        ``"raise"`` — raise ``ConcurrencyBudgetExceeded`` (harder error).
        ``"wait"`` — await a free slot up to ``wait_timeout_ms``.
    wait_timeout_ms:
        Only used when ``action="wait"``.  If a slot does not free up within
        this duration, falls back to ``"reject"``.
    retry_after_hint_ms:
        Hint included in ``BackpressureSignal.retry_after_ms`` to tell callers
        how long to back off before retrying.
    """

    action: str = "reject"
    wait_timeout_ms: float = 5000.0
    retry_after_hint_ms: float = 1000.0

    def __post_init__(self) -> None:
        if self.action not in ("reject", "raise", "wait"):
            raise ValueError(f"BackpressurePolicy.action must be one of 'reject', 'raise', 'wait'; got {self.action!r}")


# ---------------------------------------------------------------------------
# Resource accounting ledger
# ---------------------------------------------------------------------------

@dataclass
class _InflightRecord:
    trace_id: str
    session_id: str
    started_at: float = field(default_factory=time.monotonic)


class ResourceAccounter:
    """Tracks in-flight turn executions and enforces the concurrency budget.

    Thread-safe via ``asyncio.Semaphore`` (single event loop assumed for async
    execution).  Also exposes synchronous ``try_acquire`` / ``release`` for
    use in sync runtimes.
    """

    def __init__(self, budget: ConcurrencyBudget) -> None:
        self._budget = budget
        self._semaphore = asyncio.Semaphore(budget.max_concurrent_turns)
        self._inflight: dict[str, _InflightRecord] = {}
        self._lock = asyncio.Lock()
        self._total_admitted: int = 0
        self._total_rejected: int = 0

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def try_acquire_async(self, *, trace_id: str, session_id: str = "") -> bool:
        """Attempt to acquire a concurrency slot (non-blocking).

        Returns ``True`` if a slot was acquired, ``False`` if the budget is at
        capacity.
        """
        acquired = self._semaphore.locked() is False and await asyncio.wait_for(
            asyncio.shield(self._semaphore.acquire()), timeout=0
        ) if False else self._semaphore._value > 0  # type: ignore[attr-defined]
        if not acquired:
            self._total_rejected += 1
            return False
        await self._semaphore.acquire()
        await self._record_inflight(trace_id=trace_id, session_id=session_id)
        return True

    async def acquire_async(
        self,
        *,
        trace_id: str,
        session_id: str = "",
        timeout_ms: float = 0.0,
    ) -> None:
        """Acquire a concurrency slot, blocking up to *timeout_ms* ms.

        Raises ``ConcurrencyBudgetExceeded`` if *timeout_ms* == 0 and the
        budget is full.
        """
        if timeout_ms > 0:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError as exc:
                self._total_rejected += 1
                raise ConcurrencyBudgetExceeded(
                    max_concurrent=self._budget.max_concurrent_turns,
                    current_inflight=self.inflight_count,
                    trace_id=trace_id,
                ) from exc
        else:
            if self._semaphore._value <= 0:  # type: ignore[attr-defined]
                self._total_rejected += 1
                raise ConcurrencyBudgetExceeded(
                    max_concurrent=self._budget.max_concurrent_turns,
                    current_inflight=self.inflight_count,
                    trace_id=trace_id,
                )
            await self._semaphore.acquire()
        await self._record_inflight(trace_id=trace_id, session_id=session_id)

    async def release_async(self, trace_id: str) -> None:
        """Release a concurrency slot for *trace_id*."""
        async with self._lock:
            self._inflight.pop(trace_id, None)
        self._semaphore.release()

    async def _record_inflight(self, *, trace_id: str, session_id: str) -> None:
        async with self._lock:
            self._inflight[trace_id] = _InflightRecord(
                trace_id=str(trace_id),
                session_id=str(session_id or ""),
            )
            self._total_admitted += 1
        if self.inflight_count >= self._budget.soft_limit_turns:
            logger.warning(
                "ExecutionResourceBudget: soft limit reached — inflight=%d, soft_limit=%d",
                self.inflight_count,
                self._budget.soft_limit_turns,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    @property
    def available_slots(self) -> int:
        return self._budget.max_concurrent_turns - self.inflight_count

    @property
    def total_admitted(self) -> int:
        return self._total_admitted

    @property
    def total_rejected(self) -> int:
        return self._total_rejected

    def snapshot(self) -> dict[str, Any]:
        return {
            "inflight_count": self.inflight_count,
            "available_slots": self.available_slots,
            "max_concurrent_turns": self._budget.max_concurrent_turns,
            "soft_limit_turns": self._budget.soft_limit_turns,
            "total_admitted": self._total_admitted,
            "total_rejected": self._total_rejected,
            "inflight_trace_ids": list(self._inflight.keys()),
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class ExecutionResourceBudget:
    """Unified resource accounting and concurrency boundary enforcement.

    Usage (async context manager — preferred)::

        budget = ExecutionResourceBudget(max_concurrent_turns=10)
        async with budget.acquire(trace_id=turn_context.trace_id):
            result = await graph.execute(turn_context)

    The graph is unaware of this layer.  Wire it in the runtime adapter or
    app layer that dispatches turns.
    """

    def __init__(
        self,
        *,
        max_concurrent_turns: int = 10,
        max_stage_parallelism: int = 4,
        soft_limit_turns: int | None = None,
        backpressure_action: str = "reject",
        backpressure_wait_timeout_ms: float = 5000.0,
        backpressure_retry_after_ms: float = 1000.0,
    ) -> None:
        _soft = soft_limit_turns if soft_limit_turns is not None else max(1, max_concurrent_turns - 2)
        self._budget = ConcurrencyBudget(
            max_concurrent_turns=max_concurrent_turns,
            max_stage_parallelism=max_stage_parallelism,
            soft_limit_turns=min(_soft, max_concurrent_turns),
        )
        self._policy = BackpressurePolicy(
            action=backpressure_action,
            wait_timeout_ms=backpressure_wait_timeout_ms,
            retry_after_hint_ms=backpressure_retry_after_ms,
        )
        self._accounter = ResourceAccounter(self._budget)

    @contextlib.asynccontextmanager
    async def acquire(
        self,
        *,
        trace_id: str,
        session_id: str = "",
    ) -> AsyncIterator[None]:
        """Async context manager that acquires and releases a concurrency slot.

        Raises ``BackpressureSignal`` or ``ConcurrencyBudgetExceeded`` based on
        the configured ``BackpressurePolicy`` if the budget is exhausted.

        Usage::

            async with budget.acquire(trace_id=turn_context.trace_id):
                result = await graph.execute(turn_context)
        """
        try:
            if self._policy.action == "wait":
                await self._accounter.acquire_async(
                    trace_id=trace_id,
                    session_id=session_id,
                    timeout_ms=self._policy.wait_timeout_ms,
                )
            elif self._policy.action == "raise":
                await self._accounter.acquire_async(
                    trace_id=trace_id,
                    session_id=session_id,
                    timeout_ms=0,
                )
            else:  # "reject"
                try:
                    await self._accounter.acquire_async(
                        trace_id=trace_id,
                        session_id=session_id,
                        timeout_ms=0,
                    )
                except ConcurrencyBudgetExceeded as exc:
                    raise BackpressureSignal(
                        reason=str(exc),
                        retry_after_ms=self._policy.retry_after_hint_ms,
                        trace_id=trace_id,
                    ) from exc
        except (BackpressureSignal, ConcurrencyBudgetExceeded):
            raise

        try:
            yield
        finally:
            await self._accounter.release_async(trace_id)

    @property
    def budget(self) -> ConcurrencyBudget:
        return self._budget

    @property
    def policy(self) -> BackpressurePolicy:
        return self._policy

    def snapshot(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of current resource state."""
        return {
            "budget": {
                "max_concurrent_turns": self._budget.max_concurrent_turns,
                "max_stage_parallelism": self._budget.max_stage_parallelism,
                "soft_limit_turns": self._budget.soft_limit_turns,
            },
            "policy": {
                "action": self._policy.action,
                "wait_timeout_ms": self._policy.wait_timeout_ms,
                "retry_after_hint_ms": self._policy.retry_after_hint_ms,
            },
            "accounter": self._accounter.snapshot(),
        }
