"""Canonical tenant identifiers and trusted provisioning interfaces."""

from __future__ import annotations

import base64
import logging
import re
import secrets
import time
from collections.abc import Sequence
from typing import NewType, Protocol

TenantId = NewType("TenantId", str)

_VERSIONED_TENANT_ID = re.compile(r"t1_[a-z2-7]{26}\Z", re.ASCII)
_RETAINED_LEGACY_TENANT_ID = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)+\Z", re.ASCII)
_RETAINED_DEFAULT_TENANT_ID = "default"
_MAX_TENANT_ID_LENGTH = 64
_logger = logging.getLogger(__name__)


class TenantIdentityResolver(Protocol):
    def resolve_tenant_alias(self, *, issuer: str, alias: str) -> str | None: ...


class TenantIdentityProvisioner(Protocol):
    def provision_tenant_identity(
        self,
        *,
        issuer: str,
        subject: str,
        tenant_id: str,
        scopes: set[str],
        aliases: Sequence[str],
    ) -> None: ...


def is_versioned_tenant_id(value: object) -> bool:
    return isinstance(value, str) and _VERSIONED_TENANT_ID.fullmatch(value) is not None


def log_legacy_tenant_id_acceptance(
    value: object,
    *,
    issuer: str = "unbound",
    subject: str = "unbound",
    acceptance_path: str = "tenant_identity.parse_tenant_id",
) -> None:
    """Record accepted legacy identifiers without asserting ownership."""
    if is_versioned_tenant_id(value):
        return
    _logger.warning(
        "Legacy tenant identity accepted during compatibility window",
        extra={
            "tenant_identity_class": "default" if value == _RETAINED_DEFAULT_TENANT_ID else "legacy",
            "tenant_id": str(value),
            "tenant_issuer": str(issuer or "unbound"),
            "tenant_subject": str(subject or "unbound"),
            "tenant_accepted_at_unix": time.time(),
            "tenant_acceptance_path": acceptance_path,
        },
    )


def parse_tenant_id(
    value: object,
    *,
    issuer: str = "unbound",
    subject: str = "unbound",
    acceptance_path: str = "tenant_identity.parse_tenant_id",
) -> TenantId:
    """Validate an exact v1 opaque ID or explicitly retained legacy ID."""
    if not isinstance(value, str) or not value.isascii() or len(value) > _MAX_TENANT_ID_LENGTH:
        raise ValueError("tenant identity must be canonical ASCII with no rewriting")
    if _VERSIONED_TENANT_ID.fullmatch(value):
        return TenantId(value)
    if value.startswith("t1_"):
        raise ValueError("tenant identity must be canonical; aliases require explicit resolution")
    if value == _RETAINED_DEFAULT_TENANT_ID or _RETAINED_LEGACY_TENANT_ID.fullmatch(value):
        log_legacy_tenant_id_acceptance(
            value,
            issuer=issuer,
            subject=subject,
            acceptance_path=acceptance_path,
        )
        return TenantId(value)
    raise ValueError("tenant identity must be canonical; aliases require explicit resolution")


def require_tenant_id(
    value: object,
    *,
    issuer: str = "unbound",
    subject: str = "unbound",
    acceptance_path: str = "tenant_identity.require_tenant_id",
) -> str:
    return str(
        parse_tenant_id(
            value,
            issuer=issuer,
            subject=subject,
            acceptance_path=acceptance_path,
        )
    )


def mint_tenant_id(*, random_bytes=secrets.token_bytes) -> TenantId:
    encoded = base64.b32encode(random_bytes(16)).decode("ascii").rstrip("=").lower()
    return parse_tenant_id(f"t1_{encoded}")


def resolve_tenant_reference(
    value: object,
    *,
    issuer: str,
    resolver: TenantIdentityResolver,
) -> str:
    """Resolve a canonical ID or exact registered legacy alias without rewriting."""
    if isinstance(value, str):
        try:
            return require_tenant_id(
                value,
                issuer=issuer,
                subject="alias-resolution",
                acceptance_path="tenant_identity.resolve_tenant_reference",
            )
        except ValueError:
            pass
        if not value or value != value.strip():
            raise ValueError("tenant reference is missing or not an exact alias")
        resolved = resolver.resolve_tenant_alias(issuer=issuer, alias=value)
        if resolved is not None:
            return require_tenant_id(
                resolved,
                issuer=issuer,
                subject="alias-resolution",
                acceptance_path="tenant_identity.resolve_tenant_reference",
            )
    raise ValueError("tenant reference is not canonical or explicitly registered")


def issue_tenant_identity(
    provisioner: TenantIdentityProvisioner,
    *,
    issuer: str,
    subject: str,
    scopes: set[str],
    aliases: Sequence[str] = (),
) -> str:
    """Mint an opaque ID and atomically register its identity, membership and aliases."""
    tenant_id = str(mint_tenant_id())
    exact_aliases = tuple(dict.fromkeys(aliases))
    provisioner.provision_tenant_identity(
        issuer=issuer,
        subject=subject,
        tenant_id=tenant_id,
        scopes=set(scopes),
        aliases=exact_aliases,
    )
    return tenant_id
