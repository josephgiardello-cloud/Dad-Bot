import asyncio

import pytest

from dadbot.core.execution_resource_budget import (
    ConcurrencyBudget,
    ConcurrencyBudgetExceeded,
    ResourceAccounter,
)

pytestmark = pytest.mark.unit


def test_resource_accounter_try_acquire_async_is_non_blocking_and_rejects_when_full():
    async def _exercise() -> None:
        accounter = ResourceAccounter(
            ConcurrencyBudget(max_concurrent_turns=1, max_stage_parallelism=1, soft_limit_turns=1)
        )

        first = await accounter.try_acquire_async(trace_id="t-1", session_id="s-1")
        second = await accounter.try_acquire_async(trace_id="t-2", session_id="s-1")

        assert first is True
        assert second is False
        assert accounter.total_rejected == 1

        await accounter.release_async("t-1")

        third = await accounter.try_acquire_async(trace_id="t-3", session_id="s-1")
        assert third is True
        await accounter.release_async("t-3")

    asyncio.run(_exercise())


def test_resource_accounter_acquire_async_zero_timeout_raises_when_full():
    async def _exercise() -> None:
        accounter = ResourceAccounter(
            ConcurrencyBudget(max_concurrent_turns=1, max_stage_parallelism=1, soft_limit_turns=1)
        )

        await accounter.acquire_async(trace_id="t-1", session_id="s-1", timeout_ms=0)

        with pytest.raises(ConcurrencyBudgetExceeded):
            await accounter.acquire_async(trace_id="t-2", session_id="s-1", timeout_ms=0)

        await accounter.release_async("t-1")

    asyncio.run(_exercise())
