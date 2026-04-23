from __future__ import annotations

from contextvars import ContextVar


_current_execution_token: ContextVar[str] = ContextVar("_current_execution_token", default="")


class ControlPlaneExecutionBoundary:
    """Context boundary proving graph execution came from the control plane."""

    @staticmethod
    def current() -> str:
        return _current_execution_token.get() or ""

    @staticmethod
    def bind(execution_token: str) -> "_ExecutionBoundaryScope":
        token = str(execution_token or "").strip()
        return _ExecutionBoundaryScope(token)


class _ExecutionBoundaryScope:
    def __init__(self, token: str) -> None:
        self._token = token
        self._ctx_token = None

    def __enter__(self) -> str:
        self._ctx_token = _current_execution_token.set(self._token)
        return self._token

    def __exit__(self, *_) -> None:
        if self._ctx_token is not None:
            _current_execution_token.reset(self._ctx_token)
