from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExecutionKernelSpec(Protocol):
    strict: bool

    def validate(
        self,
        stage: str,
        operation: str,
        context: Any,
        *,
        mutation_outside_save_node: bool = False,
    ) -> Any: ...

    async def run(
        self,
        turn_context: Any,
        pipeline: Iterable[tuple[str, Any]],
        execute_stage: Any,
    ) -> Any: ...


class ExecutionKernelContractViolation(RuntimeError):
    """Raised when a kernel object does not satisfy ExecutionKernelSpec."""


def validate_execution_kernel_spec(
    kernel: Any,
    *,
    raise_on_failure: bool = False,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(kernel, ExecutionKernelSpec):
        issues.append("kernel does not satisfy ExecutionKernelSpec structure")

    validate_fn = getattr(kernel, "validate", None)
    if not callable(validate_fn):
        issues.append("kernel.validate missing or not callable")

    run_fn = getattr(kernel, "run", None)
    if run_fn is None or not inspect.iscoroutinefunction(run_fn):
        issues.append("kernel.run must be an async callable")

    if not hasattr(kernel, "strict"):
        issues.append("kernel.strict missing")

    if raise_on_failure and issues:
        raise ExecutionKernelContractViolation("; ".join(issues))

    return issues


__all__ = [
    "ExecutionKernelContractViolation",
    "ExecutionKernelSpec",
    "validate_execution_kernel_spec",
]
