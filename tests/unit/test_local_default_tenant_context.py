import asyncio

import pytest

from dadbot_system.execution_identity import require_scoped_tenant_id
from dadbot_system.tenant_context import (
    LocalDefaultPrincipalContext,
    TenantContextMissingError,
    local_default_context_scope,
    require_request_context,
    require_tenant_context,
)


def test_local_default_context_is_separate_from_canonical_context() -> None:
    with local_default_context_scope(enabled=True):
        context = require_request_context()
        assert isinstance(context, LocalDefaultPrincipalContext)
        assert context.tenant_id == "default"
        with pytest.raises(TenantContextMissingError):
            require_tenant_context()

    with pytest.raises(TenantContextMissingError):
        require_request_context()


def test_local_default_context_requires_explicit_gate() -> None:
    with pytest.raises(ValueError, match="compatibility is disabled"):
        with local_default_context_scope(enabled=False):
            pass


def test_scoped_default_tenant_is_gated_to_nonproduction(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "test")
    monkeypatch.delenv("DADBOT_ALLOW_LEGACY_DEFAULT_TENANT", raising=False)
    with pytest.raises(ValueError, match="gated non-production"):
        require_scoped_tenant_id("default")

    monkeypatch.setenv("DADBOT_ALLOW_LEGACY_DEFAULT_TENANT", "true")
    assert require_scoped_tenant_id("default") == "default"

    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")
    with pytest.raises(ValueError, match="gated non-production"):
        require_scoped_tenant_id("default")


def test_local_default_context_resets_after_exception() -> None:
    with pytest.raises(RuntimeError, match="request failed"):
        with local_default_context_scope(enabled=True):
            raise RuntimeError("request failed")

    with pytest.raises(TenantContextMissingError):
        require_request_context()


@pytest.mark.asyncio
async def test_concurrent_local_default_contexts_are_isolated() -> None:
    async def resolve() -> LocalDefaultPrincipalContext:
        with local_default_context_scope(enabled=True):
            context = require_request_context()
            assert isinstance(context, LocalDefaultPrincipalContext)
            await asyncio.sleep(0)
            assert require_request_context() is context
            return context

    first, second = await asyncio.gather(resolve(), resolve())

    assert first is not second
    with pytest.raises(TenantContextMissingError):
        require_request_context()
