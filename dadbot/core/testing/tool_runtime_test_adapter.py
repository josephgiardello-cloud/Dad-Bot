from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from dadbot.core._tool_sandbox import _ToolSandbox, _ToolTransaction, ToolSandboxSnapshot
from dadbot.core.action_mixin import DadBotActionMixin
from dadbot.core.kernel_locks import KernelToolIdempotencyRegistry


class _ToolRuntimeStub(DadBotActionMixin):
    pass


@dataclass
class ToolRuntimeTransactionAdapter:
    _txn: _ToolTransaction

    def execute(
        self,
        executor: Callable[[], Any],
        *,
        compensating_action: Callable[[], None] | None = None,
    ):
        KernelToolIdempotencyRegistry.clear()
        return self._txn.execute(executor, compensating_action=compensating_action)

    @property
    def result(self) -> Any:
        return self._txn.result

    @property
    def status(self) -> str:
        return self._txn.status

    @property
    def committed(self) -> bool:
        return self._txn.committed

    @property
    def rolled_back(self) -> bool:
        return self._txn.rolled_back


class ToolRuntimeTestAdapter:
    """Test-only adapter over the runtime execution spine and private sandbox."""

    def __init__(self, runtime: Any | None = None) -> None:
        self.runtime = runtime or _ToolRuntimeStub()
        self._sandbox = _ToolSandbox()
        KernelToolIdempotencyRegistry.clear()

    def execute_tool(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        executor: Callable[[], Any],
        compensating_action: Callable[[], None] | None = None,
    ):
        KernelToolIdempotencyRegistry.clear()
        return self.runtime.execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            executor=executor,
            compensating_action=compensating_action,
        )

    def execute(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        executor: Callable[[], Any],
        compensating_action: Callable[[], None] | None = None,
    ):
        KernelToolIdempotencyRegistry.clear()
        return self._sandbox.execute(
            tool_name=tool_name,
            parameters=parameters,
            executor=executor,
            compensating_action=compensating_action,
        )

    def rollback(self) -> list[dict[str, Any]]:
        return self._sandbox.rollback()

    def snapshot(self) -> dict[str, Any]:
        return self._sandbox.snapshot()

    def isolated_state_snapshot(self, generation: int = 0) -> ToolSandboxSnapshot:
        return self._sandbox.isolated_state_snapshot(generation=generation)

    def is_clean(self) -> bool:
        return self._sandbox.is_clean()

    @contextmanager
    def transaction(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> Iterator[ToolRuntimeTransactionAdapter]:
        with self._sandbox.transaction(tool_name=tool_name, parameters=parameters) as txn:
            yield ToolRuntimeTransactionAdapter(txn)

    def make_transaction(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> ToolRuntimeTransactionAdapter:
        return ToolRuntimeTransactionAdapter(
            _ToolTransaction(
                sandbox=self._sandbox,
                tool_name=str(tool_name or ""),
                parameters=dict(parameters or {}),
            )
        )