from __future__ import annotations

import os
from contextvars import ContextVar

_current_execution_token: ContextVar[str] = ContextVar(
    "_current_execution_token",
    default="",
)
_current_model_gateway: ContextVar[str] = ContextVar(
    "_current_model_gateway",
    default="",
)
_current_memory_write_owner: ContextVar[str] = ContextVar(
    "_current_memory_write_owner",
    default="",
)


CANONICAL_EXECUTION_KERNEL = "dadbot.core.orchestrator.DadBotOrchestrator.handle_turn"


def canonical_execution_kernel() -> str:
    return CANONICAL_EXECUTION_KERNEL


class RuntimeExecutionViolation(RuntimeError):
    """Raised when a non-production execution surface is invoked."""


class ModelGatewayViolation(RuntimeError):
    """Raised when model runtime is called outside ModelPort."""


class MemoryWriteSurfaceViolation(RuntimeError):
    """Raised when memory writes bypass MemoryManager.mutate_memory_store()."""


def experimental_runtime_enabled() -> bool:
    return str(
        os.environ.get("DADBOT_ENABLE_EXPERIMENTAL_RUNTIME", "0"),
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enforce_execution_role(*, module: str, role: str) -> None:
    normalized = str(role or "").strip().lower()
    if normalized == "production_kernel":
        return
    if normalized == "experimental" and experimental_runtime_enabled():
        return
    raise RuntimeExecutionViolation(
        f"Execution denied for module={module!r} role={role!r}; production execution path is orchestrator-only",
    )


def enforce_model_gateway(*, caller: str) -> None:
    normalized = str(caller or "").strip()
    bound_gateway = str(_current_model_gateway.get() or "").strip()
    if normalized == "ModelPort" and bound_gateway == "ModelPort":
        return
    raise ModelGatewayViolation(
        f"Model gateway violation: caller={normalized or '<unknown>'!r}; "
        f"bound_gateway={bound_gateway or '<none>'!r}; "
        "only ModelPort may access runtime_client",
    )


def enforce_memory_write_owner(*, owner: str) -> None:
    normalized = str(owner or "").strip()
    bound_owner = str(_current_memory_write_owner.get() or "").strip()
    if normalized == "MemoryManager" and bound_owner == "MemoryManager":
        return
    raise MemoryWriteSurfaceViolation(
        f"Memory write denied for owner={normalized or '<unknown>'!r}; "
        f"bound_owner={bound_owner or '<none>'!r}; "
        "only MemoryManager.mutate_memory_store() may mutate memory",
    )


class ModelGatewayScope:
    """Context boundary proving runtime_client is being called through ModelPort."""

    @staticmethod
    def current() -> str:
        return _current_model_gateway.get() or ""

    @staticmethod
    def bind(caller: str) -> _ModelGatewayScope:
        token = str(caller or "").strip()
        return _ModelGatewayScope(token)


class _ModelGatewayScope:
    def __init__(self, caller: str) -> None:
        self._caller = caller
        self._ctx_token = None

    def __enter__(self) -> str:
        self._ctx_token = _current_model_gateway.set(self._caller)
        return self._caller

    def __exit__(self, *_) -> None:
        if self._ctx_token is not None:
            _current_model_gateway.reset(self._ctx_token)


class MemoryWriteOwnerScope:
    """Context boundary proving memory writes are routed through MemoryManager."""

    @staticmethod
    def current() -> str:
        return _current_memory_write_owner.get() or ""

    @staticmethod
    def bind(owner: str) -> _MemoryWriteOwnerScope:
        token = str(owner or "").strip()
        return _MemoryWriteOwnerScope(token)


class _MemoryWriteOwnerScope:
    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._ctx_token = None

    def __enter__(self) -> str:
        self._ctx_token = _current_memory_write_owner.set(self._owner)
        return self._owner

    def __exit__(self, *_) -> None:
        if self._ctx_token is not None:
            _current_memory_write_owner.reset(self._ctx_token)


class ControlPlaneExecutionBoundary:
    """Context boundary proving graph execution came from the control plane."""

    @staticmethod
    def current() -> str:
        return _current_execution_token.get() or ""

    @staticmethod
    def bind(execution_token: str) -> _ExecutionBoundaryScope:
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
