import asyncio

import pytest

from dadbot_system.tenant_context import (
    TenantContext,
    TenantContextMissingError,
    require_tenant_context,
    tenant_context_scope,
)
from dadbot_system.tenant_identity import mint_tenant_id


def test_tenant_context_is_immutable_and_requires_issued_id():
    context = TenantContext(str(mint_tenant_id()))

    assert context.tenant_id.startswith("t1_")
    with pytest.raises((AttributeError, TypeError)):
        context.tenant_id = "t1_invalid"
    with pytest.raises(ValueError, match="issued canonical"):
        TenantContext("tenant-a")


def test_tenant_context_resolution_fails_closed_without_scope():
    with pytest.raises(TenantContextMissingError, match="required"):
        require_tenant_context()


def test_tenant_context_scope_resets_after_use():
    context = TenantContext(str(mint_tenant_id()))

    with tenant_context_scope(context):
        assert require_tenant_context() is context

    with pytest.raises(TenantContextMissingError):
        require_tenant_context()


@pytest.mark.asyncio
async def test_tenant_context_isolated_between_async_tasks():
    contexts = [TenantContext(str(mint_tenant_id())) for _ in range(2)]

    async def resolve(context):
        with tenant_context_scope(context):
            await asyncio.sleep(0)
            return require_tenant_context()

    resolved = await asyncio.gather(*(resolve(context) for context in contexts))

    assert resolved == contexts
