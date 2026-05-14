from __future__ import annotations

from contextlib import contextmanager, AbstractContextManager
from contextvars import ContextVar

_KERNEL_SCOPE: ContextVar[bool] = ContextVar("dadbot_kernel_scope", default=False)


class KernelBoundaryViolation(RuntimeError):
    """Raised when runtime execution escapes the enforced kernel scope."""


class KernelBoundary:
    """Shared kernel-boundary scope primitive across layers.

    Placed outside dadbot.core so higher-level layers (e.g. registry) can
    enforce kernel scope without violating import direction constraints.
    """

    @staticmethod
    def in_scope() -> bool:
        return bool(_KERNEL_SCOPE.get())

    @staticmethod
    def assert_scope(operation: str) -> None:
        if not KernelBoundary.in_scope():
            raise KernelBoundaryViolation(
                f"Kernel boundary violation: {operation} is illegal outside KernelGateway scope",
            )

    @staticmethod
    @contextmanager
    def open_scope() -> AbstractContextManager[None]:
        token = _KERNEL_SCOPE.set(True)
        try:
            yield
        finally:
            _KERNEL_SCOPE.reset(token)


__all__ = ["KernelBoundary", "KernelBoundaryViolation"]
