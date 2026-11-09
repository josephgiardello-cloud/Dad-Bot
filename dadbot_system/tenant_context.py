"""Unwired request-scoped tenant context contract for Phase 2."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from .tenant_identity import is_versioned_tenant_id


class TenantContextMissingError(RuntimeError):
    """Raised when tenant-scoped code runs without an established context."""


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Immutable holder for an opaque, issued tenant ID."""

    tenant_id: str

    def __post_init__(self) -> None:
        if not is_versioned_tenant_id(self.tenant_id):
            raise ValueError("TenantContext requires an issued canonical tenant ID")


_tenant_context: ContextVar[TenantContext | None] = ContextVar("dadbot_tenant_context", default=None)


def require_tenant_context() -> TenantContext:
    context = _tenant_context.get()
    if context is None:
        raise TenantContextMissingError("Tenant context is required")
    return context


@contextmanager
def tenant_context_scope(context: TenantContext) -> Iterator[None]:
    token: Token[TenantContext | None] = _tenant_context.set(context)
    try:
        yield
    finally:
        _tenant_context.reset(token)
