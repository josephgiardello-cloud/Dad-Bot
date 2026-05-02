"""Tool execution sandbox internals for the DadBot agentic pipeline.

This module is core-private. External callers must use the runtime execution
spine exposed by ``dadbot.core.tool_executor`` or the runtime contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from dadbot.core.kernel_locks import KernelToolIdempotencyRegistry

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
    status: str
    result: Any
    error: str
    compensating_action: Callable[[], None] | None = field(default=None, repr=False)


_SANDBOX_INSTANTIATION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "dadbot.core.tool_executor",
        "dadbot.core.testing.tool_runtime_test_adapter",
    }
)


class _ToolSandbox:
    """Single-turn execution sandbox for all agentic tool calls.

    Instantiation is restricted to the allowlist at runtime.  This converts
    the CI RULE16_TOOL_SANDBOX_ISOLATION convention into a hard runtime guard
    that fires *before* any forbidden use can proceed.  Attempting to
    instantiate ``_ToolSandbox`` outside the spine raises ``RuntimeError``
    immediately.
    """

    def __init__(self) -> None:
        import sys as _sys

        frame = _sys._getframe(1)
        caller_module = frame.f_globals.get("__name__", "")
        if caller_module not in _SANDBOX_INSTANTIATION_ALLOWLIST:
            raise RuntimeError(
                f"_ToolSandbox may only be instantiated from the kernel execution spine "
                f"(dadbot.core.tool_executor). "
                f"Use dadbot.core.tool_executor.execute_tool() instead. "
                f"Blocked caller: {caller_module!r}",
            )
        self._cache: dict[str, ToolExecutionRecord] = {}
        self._records: list[ToolExecutionRecord] = []

    def execute(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        executor: Callable[[], Any],
        compensating_action: Callable[[], None] | None = None,
    ) -> ToolExecutionRecord:
        """Execute the tool with idempotency, isolation, and rollback registration."""
        params = dict(parameters or {})
        key = str(params.get("_idempotency_key") or "").strip()
        if not key:
            key = _idempotency_key(str(tool_name or ""), params)

        shared_cached = KernelToolIdempotencyRegistry.get(key)
        if isinstance(shared_cached, ToolExecutionRecord):
            record = ToolExecutionRecord(
                tool_name=str(tool_name or ""),
                idempotency_key=key,
                status="cached",
                result=shared_cached.result,
                error="",
                compensating_action=None,
            )
            self._records.append(record)
            logger.debug("ToolSandbox: global idempotency HIT tool=%r key=%s", tool_name, key)
            return record

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

        try:
            result = executor()
            status = "succeeded"
            error = ""
        except Exception as exc:  # noqa: BLE001
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
            KernelToolIdempotencyRegistry.put(key, record)

        self._records.append(record)
        logger.debug("ToolSandbox: tool=%r key=%s status=%s", tool_name, key, status)
        return record

    def rollback(self) -> list[dict[str, Any]]:
        """Run all compensating actions in LIFO order."""
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
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = str(exc) or type(exc).__name__
                logger.warning(
                    "ToolSandbox: rollback failed for tool=%r: %s",
                    record.tool_name,
                    exc,
                )
            outcomes.append(outcome)
        return outcomes

    def snapshot(self) -> dict[str, Any]:
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

    def isolated_state_snapshot(self, generation: int = 0) -> ToolSandboxSnapshot:
        return ToolSandboxSnapshot.capture(self, generation)

    def is_clean(self) -> bool:
        return len(self._records) == 0

    @contextmanager
    def transaction(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> Iterator[_ToolTransaction]:
        txn = _ToolTransaction(
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


class _ToolTransaction:
    """Explicit transaction wrapper for a single private tool sandbox call."""

    def __init__(
        self,
        *,
        sandbox: _ToolSandbox,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> None:
        self._sandbox = sandbox
        self._tool_name = tool_name
        self._parameters = parameters
        self._record: ToolExecutionRecord | None = None
        self._committed = False
        self._rolled_back = False
        self._records_at_begin = len(sandbox._records)

    def execute(
        self,
        executor: Callable[[], Any],
        *,
        compensating_action: Callable[[], None] | None = None,
    ) -> ToolExecutionRecord:
        self._record = self._sandbox.execute(
            tool_name=self._tool_name,
            parameters=self._parameters,
            executor=executor,
            compensating_action=compensating_action,
        )
        return self._record

    @property
    def result(self) -> Any:
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

    def _auto_rollback(self) -> list[dict[str, Any]]:
        if self._rolled_back:
            return []
        self._rolled_back = True
        txn_records = list(reversed(self._sandbox._records[self._records_at_begin :]))
        outcomes: list[dict[str, Any]] = []
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
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = str(exc) or type(exc).__name__
            outcomes.append(outcome)
        return outcomes


@dataclass(frozen=True)
class ToolSandboxSnapshot:
    records_count: int
    cache_keys: tuple[str, ...]
    snapshot_hash: str
    generation: int

    @classmethod
    def capture(cls, sandbox: _ToolSandbox, generation: int) -> ToolSandboxSnapshot:
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