"""Phase 4 Full-Path Execution Trace Validator.

Verifies end-to-end execution trace completeness and ordering using a
self-contained graph with mock services — no real DadBot instance required.

Checks:
  - All canonical stages appear in ctx.stage_traces
  - Stage ordering matches canonical pipeline contract
  - mutation_queue is empty after SaveNode drains it
  - execution_trace_contract is present and non-empty in ctx.state
  - execution_identity is sealed into ctx.state and ctx.metadata
  - trace_hash is a non-empty hex string
  - Phase transitions follow PLAN → ACT → OBSERVE → RESPOND ordering
  - Short-circuit path still seals execution_identity
  - Failure path still seals execution_identity (identity emitted before re-raise)
"""

from __future__ import annotations

from typing import Any

import pytest

from dadbot.core.graph import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    TurnContext,
    TurnGraph,
    TurnPhase,
)

try:
    from tests.harness.kernel_mock import MockRegistry
except ImportError:
    from harness.kernel_mock import MockRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_graph(registry: Any) -> TurnGraph:
    """Construct a canonical 7-stage graph wired through explicit edges."""
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


def _stage_names(ctx: TurnContext) -> list[str]:
    return [str(t.stage or "") for t in (ctx.stage_traces or [])]


# ---------------------------------------------------------------------------
# Core full-pipeline trace test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_execution_trace():
    """All canonical stages must appear in stage_traces in the correct order."""
    registry = MockRegistry()
    graph = _build_graph(registry)
    ctx = TurnContext(user_input="hello phase 4")

    await graph.execute(ctx)

    stages = _stage_names(ctx)

    # Must include every canonical stage
    for required in ("temporal", "health", "context_builder", "inference", "safety", "reflection", "save"):
        assert required in stages, f"Stage '{required}' missing from execution trace; got {stages!r}"

    # Strict ordering: temporal first, then mutation-capable stages
    assert stages.index("temporal") < stages.index("inference")
    assert stages.index("inference") < stages.index("safety")
    assert stages.index("safety") < stages.index("reflection")
    assert stages.index("reflection") < stages.index("save")

    # Mutation queue drained by SaveNode
    assert ctx.mutation_queue.is_empty(), (
        f"mutation_queue must be empty after SaveNode; snapshot={ctx.mutation_queue.snapshot()}"
    )


# ---------------------------------------------------------------------------
# Execution trace contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_trace_contract_present_and_complete():
    """execution_trace_contract must be sealed into state and metadata after execute()."""
    registry = MockRegistry()
    graph = _build_graph(registry)
    ctx = TurnContext(user_input="trace contract check")

    await graph.execute(ctx)

    contract = ctx.state.get("execution_trace_contract")
    assert isinstance(contract, dict), "execution_trace_contract must be a dict in ctx.state"
    assert contract.get("event_count", 0) > 0, "event_count must be > 0"
    trace_hash = str(contract.get("trace_hash") or "")
    assert len(trace_hash) == 64, f"trace_hash must be a 64-char hex string, got {trace_hash!r}"
    assert contract["version"] == "1.0"

    meta_contract = ctx.metadata.get("execution_trace_contract")
    assert isinstance(meta_contract, dict), "execution_trace_contract must be in ctx.metadata too"
    assert meta_contract["trace_hash"] == trace_hash, "state and metadata trace_hash must be identical"


# ---------------------------------------------------------------------------
# ExecutionIdentity sealing (GAP-1 hard runtime contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_identity_sealed_after_execute():
    """execution_identity must be sealed in ctx.state and ctx.metadata after execute()."""
    registry = MockRegistry()
    graph = _build_graph(registry)
    ctx = TurnContext(user_input="identity seal check")

    await graph.execute(ctx)

    identity = ctx.state.get("execution_identity")
    assert isinstance(identity, dict), "execution_identity must be a dict in ctx.state"
    assert identity.get("fingerprint"), "execution_identity must include 'fingerprint'"
    assert identity.get("trace_hash"), "execution_identity must include 'trace_hash'"
    assert identity.get("trace_id") == ctx.trace_id

    meta_identity = ctx.metadata.get("execution_identity")
    assert isinstance(meta_identity, dict), "execution_identity must be in ctx.metadata"
    assert meta_identity["fingerprint"] == identity["fingerprint"]


@pytest.mark.asyncio
async def test_execution_identity_emitted_as_persistence_event():
    """execution_identity event must be emitted to the persistence service."""
    registry = MockRegistry()
    service = registry.get("persistence_service")
    graph = _build_graph(registry)
    ctx = TurnContext(user_input="identity event check")

    await graph.execute(ctx)

    identity_events = [e for e in service.events if e.get("event_type") == "execution_identity"]
    assert len(identity_events) >= 1, (
        "At least one 'execution_identity' event must be emitted to the persistence service"
    )
    identity_payload = identity_events[-1].get("identity", {})
    assert identity_payload.get("fingerprint"), "execution_identity event must carry the fingerprint"


@pytest.mark.asyncio
async def test_expected_execution_fingerprint_mismatch_raises():
    """When metadata['expected_execution_fingerprint'] is set, execute() must raise on mismatch."""
    from dadbot.core.execution_identity import ExecutionIdentityViolation

    registry = MockRegistry()
    graph = _build_graph(registry)
    ctx = TurnContext(
        user_input="fingerprint mismatch test",
        metadata={"expected_execution_fingerprint": "deliberately_wrong_fingerprint"},
    )

    with pytest.raises(ExecutionIdentityViolation):
        await graph.execute(ctx)


# ---------------------------------------------------------------------------
# Short-circuit path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_identity_sealed_on_short_circuit():
    """execution_identity must be sealed even when the pipeline short-circuits.

    Uses a canonical temporal node + a save node that short-circuits, so all
    fidelity invariants are satisfied before the early exit fires.
    """

    class _ShortCircuitSaveNode:
        """Marks fidelity.save, then short-circuits the pipeline."""

        name = "save"

        async def execute(self, _registry: Any, turn_context: Any) -> None:
            turn_context.fidelity.save = True
            turn_context.short_circuit = True
            turn_context.short_circuit_result = ("short", False)

    # Build: temporal → inference (stub) → short-circuit save
    class _StubInferenceNode:
        name = "inference"

        async def execute(self, _registry: Any, turn_context: Any) -> None:
            turn_context.state["candidate"] = ("stub", False)
            turn_context.fidelity.inference = True

    class _StubSafetyNode:
        name = "safety"

        async def execute(self, _registry: Any, turn_context: Any) -> None:
            turn_context.state["safe_result"] = turn_context.state.get("candidate", ("stub", False))

    # Use non-canonical names for stub middle stages so the ordering gate is
    # bypassed — only "temporal" (first) and "save" (last) need to be canonical
    # so the fidelity projector can set fidelity.temporal / fidelity.save.
    graph = TurnGraph(
        registry=None,
        nodes=[
            TemporalNode(),
            _StubInferenceNode(),
            _StubSafetyNode(),
            _ShortCircuitSaveNode(),
        ],
    )
    # Patch stub names to non-canonical after instantiation so ordering gate skips them.
    graph.nodes[1].name = "sc_inference_stub"
    graph.nodes[2].name = "sc_safety_stub"
    ctx = TurnContext(user_input="short circuit trace")

    result = await graph.execute(ctx)
    assert result == ("short", False)

    identity = ctx.state.get("execution_identity")
    assert isinstance(identity, dict) and identity.get("fingerprint"), (
        "execution_identity must be sealed even on short-circuit path"
    )


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_identity_sealed_on_failure_path():
    """execution_identity must be sealed before the exception is re-raised."""

    class _BoomNode:
        name = "temporal"

        async def execute(self, _registry: Any, turn_context: Any) -> None:
            turn_context.state.setdefault("temporal", turn_context.temporal.to_dict())
            turn_context.metadata.setdefault("temporal", turn_context.temporal.to_dict())
            raise RuntimeError("intentional failure for identity seal test")

    graph = TurnGraph(registry=None, nodes=[_BoomNode()])
    ctx = TurnContext(user_input="failure path trace")

    with pytest.raises(RuntimeError, match="intentional failure"):
        await graph.execute(ctx)

    identity = ctx.state.get("execution_identity")
    assert isinstance(identity, dict) and identity.get("fingerprint"), (
        "execution_identity must be sealed even on the failure exit path"
    )


# ---------------------------------------------------------------------------
# Phase ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_transitions_follow_canonical_order():
    """Phase history must advance monotonically PLAN → ACT → OBSERVE → RESPOND."""
    _PHASE_ORDER = [p.value for p in (TurnPhase.PLAN, TurnPhase.ACT, TurnPhase.OBSERVE, TurnPhase.RESPOND)]

    registry = MockRegistry()
    graph = _build_graph(registry)
    ctx = TurnContext(user_input="phase order check")

    await graph.execute(ctx)

    phase_vals = [entry["to"] for entry in ctx.phase_history]
    seen_idx = -1
    for phase in phase_vals:
        if phase not in _PHASE_ORDER:
            continue
        idx = _PHASE_ORDER.index(phase)
        assert idx >= seen_idx, (
            f"Phase regression in history: {phase!r} after index {seen_idx}; full history: {ctx.phase_history}"
        )
        seen_idx = idx


# ---------------------------------------------------------------------------
# No duplicate stage execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_stage_executed_twice_in_full_pipeline():
    """Every stage name must appear at most once in stage_traces."""
    registry = MockRegistry()
    graph = _build_graph(registry)
    ctx = TurnContext(user_input="duplicate stage check")

    await graph.execute(ctx)

    names = _stage_names(ctx)
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    assert not duplicates, f"Stages executed more than once: {duplicates}"
