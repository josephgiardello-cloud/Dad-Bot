"""Tool execution sandbox for the DadBot agentic pipeline.

Provides three guarantees that were absent from the previous dispatch map:

1. **Idempotency** â€” each tool invocation is keyed by a stable content-hash of
   (tool_name, parameters).  If the same key is presented again within the same
   sandbox instance the cached result is returned without re-executing the tool.
   This prevents duplicate reminders / double web-lookups caused by retries.

2. **Failure isolation** â€” every executor call is wrapped in a structured try/except.
   Tool failures never propagate as uncaught exceptions into the turn pipeline;
   they are recorded as ``ToolExecutionRecord`` entries with ``status="failed"``
   and the pipeline falls through gracefully.

3. **Rollback semantics** â€” each successful execution registers a compensating
   action via ``register_compensating_action``.  Calling ``rollback()`` runs every
   registered compensating action in LIFO order.  Compensating actions are
   best-effort: individual failures are logged but do not abort the rollback.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


def _idempotency_key(tool_name: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tool": str(tool_name or ""), "parameters": dict(parameters or {})},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


@dataclass
class ToolExecutionRecord:
    tool_name: str
    idempotency_key: str
    status: str          # "succeeded" | "failed" | "cached"
    result: Any          # return value of the executor, or None on failure
    error: str           # exception message, or "" on success
    compensating_action: Callable[[], None] | None = field(default=None, repr=False)


class ToolSandbox:
    """Single-turn execution sandbox for all agentic tool calls.

    Usage
    -----
    sandbox = ToolSandbox()

    result, observation = sandbox.execute(
        tool_name="set_reminder",
        parameters={"title": "Call dentist", "due_text": "tomorrow"},
        executor=lambda: bot.add_reminder("Call dentist", "tomorrow"),
        compensating_action=lambda: bot.delete_reminder(reminder_id),
    )

    # On any downstream failure:
    sandbox.rollback()
    """

    def __init__(self) -> None:
        self._cache: dict[str, ToolExecutionRecord] = {}
        self._records: list[ToolExecutionRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        executor: Callable[[], Any],
        compensating_action: Callable[[], None] | None = None,
    ) -> ToolExecutionRecord:
        """Execute the tool with idempotency, isolation, and rollback registration.

        Returns a ``ToolExecutionRecord``.  The caller is responsible for
        interpreting ``status`` and ``result``; this method never raises.
        """
        params = dict(parameters or {})
        key = _idempotency_key(str(tool_name or ""), params)

        # Idempotency: return cached result for duplicate requests.
        if key in self._cache:
            cached = self._cache[key]
            record = ToolExecutionRecord(
                tool_name=str(tool_name or ""),
                idempotency_key=key,
                status="cached",
                result=cached.result,
                error="",
                compensating_action=None,
            )
            self._records.append(record)
            logger.debug("ToolSandbox: idempotency HIT tool=%r key=%s", tool_name, key)
            return record

        # Failure isolation: execute inside a full try/except boundary.
        try:
            result = executor()
            status = "succeeded"
            error = ""
        except Exception as exc:
            result = None
            status = "failed"
            error = str(exc) or type(exc).__name__
            logger.warning("ToolSandbox: tool %r raised: %s", tool_name, exc)

        record = ToolExecutionRecord(
            tool_name=str(tool_name or ""),
            idempotency_key=key,
            status=status,
            result=result,
            error=error,
            compensating_action=compensating_action if status == "succeeded" else None,
        )

        if status == "succeeded":
            self._cache[key] = record

        self._records.append(record)
        logger.debug("ToolSandbox: tool=%r key=%s status=%s", tool_name, key, status)
        return record

    def rollback(self) -> list[dict[str, Any]]:
        """Run all compensating actions in LIFO order.

        Returns a list of rollback outcome records.  Never raises.
        """
        outcomes: list[dict[str, Any]] = []
        for record in reversed(self._records):
            if record.compensating_action is None:
                continue
            outcome: dict[str, Any] = {
                "tool": record.tool_name,
                "key": record.idempotency_key,
                "rolled_back": False,
                "error": "",
            }
            try:
                record.compensating_action()
                outcome["rolled_back"] = True
                logger.info("ToolSandbox: rolled back tool=%r", record.tool_name)
            except Exception as exc:
                outcome["error"] = str(exc) or type(exc).__name__
                logger.warning("ToolSandbox: rollback failed for tool=%r: %s", record.tool_name, exc)
            outcomes.append(outcome)
        return outcomes

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable summary of the sandbox state."""
        return {
            "executed_count": len(self._records),
            "cached_count": sum(1 for r in self._records if r.status == "cached"),
            "failed_count": sum(1 for r in self._records if r.status == "failed"),
            "succeeded_count": sum(1 for r in self._records if r.status == "succeeded"),
            "records": [
                {
                    "tool": r.tool_name,
                    "key": r.idempotency_key,
                    "status": r.status,
                    "error": r.error,
                    "has_compensating_action": r.compensating_action is not None,
                }
                for r in self._records
            ],
        }

    def isolated_state_snapshot(self, generation: int = 0) -> "ToolSandboxSnapshot":
        """Capture an immutable ToolSandboxSnapshot for isolation testing.

        Two sandboxes executing the same tool sequence must produce equal
        ``snapshot_hash`` values, proving no cross-tool state leakage.
        """
        return ToolSandboxSnapshot.capture(self, generation)

    def is_clean(self) -> bool:
        """True iff no tools have been executed yet (fresh sandbox)."""
        return len(self._records) == 0

    @contextmanager
    def transaction(
        self,
        *,
        tool_name: str,
        parameters: "dict[str, Any] | None" = None,
    ) -> "Iterator[ToolTransaction]":
        """Context manager: explicit transaction semantics for a tool call.

        On clean exit the transaction is committed.  On any exception the
        transaction auto-rolls-back compensating actions registered during it.
        """
        txn = ToolTransaction(
            sandbox=self,
            tool_name=str(tool_name or ""),
            parameters=dict(parameters or {}),
        )
        try:
            yield txn
            txn._mark_committed()
        except Exception:
            txn._auto_rollback()
            raise


class ToolTransaction:
    """Explicit transaction wrapper for a single ToolSandbox tool call.

    Obtained via ToolSandbox.transaction().  Provides begin/commit/rollback
    semantics and auto-rollback on exception.
    """

    def __init__(
        self,
        *,
        sandbox: "ToolSandbox",
        tool_name: str,
        parameters: dict,
    ) -> None:
        self._sandbox = sandbox
        self._tool_name = tool_name
        self._parameters = parameters
        self._record: "ToolExecutionRecord | None" = None
        self._committed: bool = False
        self._rolled_back: bool = False
        # Snapshot record count at begin so rollback only affects THIS transaction.
        self._records_at_begin: int = len(sandbox._records)

    def execute(
        self,
        executor,
        *,
        compensating_action=None,
    ) -> "ToolExecutionRecord":
        """Execute the tool call through the sandbox."""
        self._record = self._sandbox.execute(
            tool_name=self._tool_name,
            parameters=self._parameters,
            executor=executor,
            compensating_action=compensating_action,
        )
        return self._record

    @property
    def result(self):
        return self._record.result if self._record is not None else None

    @property
    def status(self) -> str:
        return self._record.status if self._record is not None else "not_started"

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def rolled_back(self) -> bool:
        return self._rolled_back

    def _mark_committed(self) -> None:
        self._committed = True

    def _auto_rollback(self) -> list:
        """Roll back compensating actions added during this transaction."""
        if self._rolled_back:
            return []
        self._rolled_back = True
        txn_records = list(reversed(self._sandbox._records[self._records_at_begin:]))
        outcomes = []
        for record in txn_records:
            if record.compensating_action is None:
                continue
            outcome = {
                "tool": record.tool_name,
                "key": record.idempotency_key,
                "rolled_back": False,
                "error": "",
            }
            try:
                record.compensating_action()
                outcome["rolled_back"] = True
            except Exception as exc:
                outcome["error"] = str(exc) or type(exc).__name__
            outcomes.append(outcome)
        return outcomes


# ---------------------------------------------------------------------------
# Phase 7: ToolSandboxSnapshot — isolated state snapshot for cross-tool isolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSandboxSnapshot:
    """Immutable, content-addressed snapshot of a ToolSandbox's execution records.

    Enables:
    - Cross-tool isolation checks: snapshots taken before/after each tool should
      only differ by that tool's records.
    - Rollback comparison: verify that rollback restored the exact prior state.
    - Determinism validation: same sequence of operations → same snapshot_hash.
    """

    records_count: int
    cache_keys: tuple[str, ...]
    snapshot_hash: str
    generation: int

    @classmethod
    def capture(cls, sandbox: "ToolSandbox", generation: int) -> "ToolSandboxSnapshot":
        cache_keys = tuple(sorted(sandbox._cache.keys()))
        payload = json.dumps(
            {
                "records_count": len(sandbox._records),
                "cache_keys": list(cache_keys),
                "generation": generation,
            },
            sort_keys=True,
        )
        snap_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return cls(
            records_count=len(sandbox._records),
            cache_keys=cache_keys,
            snapshot_hash=snap_hash,
            generation=generation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_count": self.records_count,
            "cache_keys": list(self.cache_keys),
            "snapshot_hash": self.snapshot_hash,
            "generation": self.generation,
        }
