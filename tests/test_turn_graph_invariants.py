from __future__ import annotations

import asyncio

import pytest

from dadbot.core.execution_firewall import ExecutionFirewall
from dadbot.core.execution_kernel import ExecutionKernel
from dadbot.core.graph import FatalTurnError, MutationIntent, MutationQueue, TurnContext, TurnGraph
from dadbot.core.invariant_registry import InvariantRegistry
from dadbot.core.nodes import TemporalNode

pytestmark = pytest.mark.integration


def test_mutation_queue_retains_pending_after_failed_drain() -> None:
    queue = MutationQueue()
    queue.bind_owner("trace-a")
    temporal = {"wall_time": "2026-01-01T00:00:00", "wall_date": "2026-01-01"}
    queue.queue(
        MutationIntent(
            type="memory",
            payload={"op": "save_mood_state", "mood": "neutral", "temporal": temporal},
        )
    )
    queue.queue(
        MutationIntent(
            type="relationship",
            payload={"op": "update", "user_input": "x", "mood": "neutral", "temporal": temporal},
        )
    )

    def _fail_first(_intent: MutationIntent) -> None:
        raise RuntimeError("boom")

    with pytest.raises(FatalTurnError):
        queue.drain(_fail_first, hard_fail_on_error=True)

    # Failed drain must keep intents pending for explicit visibility.
    assert queue.is_empty() is False
    assert queue.size() == 2


def test_mutation_queue_rejects_cross_turn_owner_rebind() -> None:
    queue = MutationQueue()
    queue.bind_owner("trace-a")
    with pytest.raises(RuntimeError, match="cross-turn reuse"):
        queue.bind_owner("trace-b")


def test_turn_graph_raises_when_save_stage_missing() -> None:
    graph = TurnGraph(registry={})
    graph.add_node("temporal", TemporalNode())

    context = TurnContext(user_input="hello")

    with pytest.raises(RuntimeError, match="Kernel validation failed"):
        asyncio.run(graph.execute(context))


def test_kernel_validation_blocks_turns_without_registry() -> None:
    graph = TurnGraph(registry=None)

    with pytest.raises(RuntimeError, match="Kernel validation failed"):
        graph._enforce_kernel_invariants(
            turn_context=TurnContext(user_input="hello"),
            stage="test",
            operation="test",
            mutation_outside_save_node=True,
        )


def test_non_strict_injected_kernel_cannot_bypass_validation_gate() -> None:
    graph = TurnGraph(registry=None)
    graph.set_execution_kernel(
        ExecutionKernel(
            firewall=ExecutionFirewall(),
            invariant_registry=InvariantRegistry(),
            quarantine=None,
            strict=False,
        ),
    )

    with pytest.raises(RuntimeError, match="Kernel validation failed"):
        graph._enforce_kernel_invariants(
            turn_context=TurnContext(user_input="hello"),
            stage="test",
            operation="test",
            mutation_outside_save_node=True,
        )


def test_kernel_validation_shadow_override_is_explicit_and_nonproduction(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_ALLOW_KERNEL_VALIDATION_SHADOW", "1")
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "development")
    graph = TurnGraph(registry=None)

    graph._enforce_kernel_invariants(
        turn_context=TurnContext(user_input="hello"),
        stage="test",
        operation="test",
        mutation_outside_save_node=True,
    )


def test_kernel_validation_shadow_override_is_rejected_in_production(monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_ALLOW_KERNEL_VALIDATION_SHADOW", "1")
    monkeypatch.setenv("DADBOT_ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="cannot be enabled in production"):
        TurnGraph(registry=None)
