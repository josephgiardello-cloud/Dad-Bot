"""Execution semantics contract tests for TurnGraph.

These tests enforce explicit runtime behavior for:
- kernel rejection semantics (default skip vs configured abort)
- failure taxonomy emission
- persistence contract validation modes
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
from harness.kernel_mock import MockRegistry

from dadbot.core.graph import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    KernelRejectionSemantics,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    TurnContext,
    TurnGraph,
)


class _RejectInferenceKernel:
    async def execute_step(self, turn_context, step_name: str, step_fn):
        if step_name == "inference":
            policy = SimpleNamespace(reason="policy denied inference")
            return SimpleNamespace(status="rejected", error="", policy=policy)
        await step_fn()
        return SimpleNamespace(status="ok", error="", policy=None)


def _build_canonical(registry) -> TurnGraph:
    graph = TurnGraph(registry=registry)
    stages = [
        ("temporal", TemporalNode()),
        ("health", HealthNode()),
        ("context_builder", ContextBuilderNode()),
        ("inference", InferenceNode()),
        ("safety", SafetyNode()),
        ("reflection", ReflectionNode()),
        ("save", SaveNode()),
    ]
    previous = None
    for name, node in stages:
        graph.add_node(name, node)
        if previous:
            graph.set_edge(previous, name)
        previous = name
    return graph


def test_kernel_rejection_default_semantics_skip_and_continue():
    registry = MockRegistry()
    graph = _build_canonical(registry)
    graph.set_kernel(_RejectInferenceKernel())

    ctx = TurnContext(user_input="hello")
    result = asyncio.run(graph.execute(ctx))

    assert isinstance(result, tuple)
    assert len(result) >= 2
    contracts = list(ctx.state.get("kernel_rejection_contract") or [])
    assert contracts, "Expected explicit kernel rejection contract record"
    first = contracts[0]
    assert first["stage"] == "inference"
    assert first["action"] == "skip_stage"
    assert first["invalidate_downstream"] is False


def test_kernel_rejection_abort_semantics_fail_fast():
    registry = MockRegistry()
    graph = _build_canonical(registry)
    graph.set_kernel(_RejectInferenceKernel())
    graph.set_kernel_rejection_semantics(
        "inference",
        KernelRejectionSemantics(
            retryable=False,
            state_mutation_allowed=False,
            invalidate_downstream=True,
            persistence_behavior="persist_rejection_event",
            action="abort_turn",
        ),
    )

    ctx = TurnContext(user_input="hello")
    with pytest.raises(RuntimeError, match="abort semantics"):
        asyncio.run(graph.execute(ctx))

    taxonomy = dict(ctx.state.get("failure_taxonomy") or {})
    assert taxonomy.get("severity"), "Expected standardized failure taxonomy payload"


def test_persistence_contract_strict_mode_enforced():
    class _MinimalPersistence:
        # Deliberately incomplete: only save_turn exists.
        def save_turn(self, _ctx, _result):
            return None

    registry = MockRegistry()
    registry.persistence = _MinimalPersistence()
    graph = _build_canonical(registry)

    # Non-strict: contract issues are captured, execution may continue.
    ctx = TurnContext(user_input="hello")
    ctx.metadata["persistence_contract_strict"] = False
    asyncio.run(graph.execute(ctx))
    payload = dict(ctx.state.get("persistence_contract") or {})
    assert payload.get("ok") is False
    assert "save_graph_checkpoint" in list(payload.get("missing") or [])
    assert "save_turn_event" in list(payload.get("missing") or [])

    # Strict: contract violation aborts execution.
    strict_ctx = TurnContext(user_input="hello")
    strict_ctx.metadata["persistence_contract_strict"] = True
    with pytest.raises(RuntimeError, match="Persistence service contract violation"):
        asyncio.run(graph.execute(strict_ctx))


def test_execution_trace_contract_emitted_on_success():
    registry = MockRegistry()
    graph = _build_canonical(registry)

    ctx = TurnContext(user_input="hello")
    asyncio.run(graph.execute(ctx))

    contract = dict(ctx.state.get("execution_trace_contract") or {})
    assert contract.get("version") == "1.0"
    assert int(contract.get("event_count", 0)) > 0
    assert str(contract.get("trace_hash") or "")
    events = list(ctx.state.get("execution_trace") or [])
    assert events
    assert events[0].get("event_type") == "turn_start"


def test_execution_trace_expected_hash_mismatch_raises():
    registry = MockRegistry()
    graph = _build_canonical(registry)

    ctx = TurnContext(user_input="hello")
    ctx.metadata["expected_execution_trace_hash"] = "not-the-real-hash"
    with pytest.raises(RuntimeError, match="Execution trace determinism mismatch"):
        asyncio.run(graph.execute(ctx))
