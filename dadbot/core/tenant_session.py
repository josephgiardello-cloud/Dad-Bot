from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TenantSessionIdentity:
    tenant_id: str
    session_id: str


_ACTIVE_TENANT_SESSION: contextvars.ContextVar[TenantSessionIdentity | None] = contextvars.ContextVar(
    "dadbot_active_tenant_session",
    default=None,
)


def current_tenant_session() -> TenantSessionIdentity | None:
    return _ACTIVE_TENANT_SESSION.get()


def current_tenant_id() -> str | None:
    identity = current_tenant_session()
    return identity.tenant_id if identity is not None else None


def resolve_tenant_id(
    tenant_id: str | None = None,
    *,
    default_tenant_id: str | None = None,
) -> str:
    explicit = str(tenant_id or "").strip()
    active = current_tenant_id()
    if explicit and active and explicit != active:
        raise ValueError("tenant identity does not match the active session scope")
    resolved = explicit or active or str(default_tenant_id or "").strip()
    if not resolved:
        raise ValueError("tenant_id is required for session access")
    return resolved


@contextmanager
def tenant_session_scope(
    tenant_id: str,
    session_id: str,
    *,
    allow_tenant_switch: bool = False,
) -> Iterator[TenantSessionIdentity]:
    tenant = str(tenant_id or "").strip()
    session = str(session_id or "").strip()
    if not tenant:
        raise ValueError("tenant_id is required for session scope")
    if not session:
        raise ValueError("session_id is required for session scope")
    active = current_tenant_id()
    if active and active != tenant and not allow_tenant_switch:
        raise ValueError("tenant identity does not match the active session scope")
    identity = TenantSessionIdentity(tenant_id=tenant, session_id=session)
    token = _ACTIVE_TENANT_SESSION.set(identity)
    try:
        yield identity
    finally:
        _ACTIVE_TENANT_SESSION.reset(token)
