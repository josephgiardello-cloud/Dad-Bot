"""GraphRunner — execution wrapper with timing and artifact capture.

Wraps TurnGraph.execute() and returns a RunResult that bundles the
result, any exception, elapsed time, fidelity state, and all
persistence artifacts recorded during the run.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from dadbot.core.graph import TurnContext, TurnFidelity, TurnGraph


def confluence_key_for_turn(session_id: str, user_input: str, *, prefix: str = "test") -> str:
    """Generate a deterministic confluence key for test turns.

    Mirrors the pattern used in production stress tests.  The prefix identifies
    the test context; session_id + user_input provides semantic uniqueness so
    every distinct turn carries an explicit, policy-compliant key.
    """
    digest = hashlib.sha256(f"{session_id}:{user_input}".encode()).hexdigest()[:24]
    return f"{prefix}:{digest}"


@dataclass
class RunResult:
    """Collected output from a single graph run."""

    result: Any  # FinalizedTurnResult or None on error
    error: Exception | None  # Exception if graph raised, else None
    elapsed_ms: float
    trace_id: str
    fidelity: TurnFidelity
    mutation_snapshot: dict[str, Any]
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    composition_contract: dict[str, Any] = field(default_factory=dict)
    confluence_report: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def phase_transitions(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == "phase_transition"]

    @property
    def checkpoint_hashes(self) -> list[str]:
        return [c.get("checkpoint_hash", "") for c in self.checkpoints]

    def assert_succeeded(self) -> None:
        if self.error is not None:
            raise AssertionError(f"GraphRunner: expected success but got {type(self.error).__name__}: {self.error}")

    def assert_phase_sequence(self, *expected: str) -> None:
        actual = [t.get("to") for t in self.phase_transitions]
        assert actual == list(expected), f"Phase sequence mismatch: expected {list(expected)}, got {actual}"


class GraphRunner:
    """Test wrapper that runs a TurnGraph and collects all observable artifacts.

    Usage::

        registry = MockRegistry()
        graph = build_canonical_graph(registry)
        ctx = TurnFactory().build_turn(seed=42)
        runner = GraphRunner()
        run = runner.run(graph, ctx, registry)
        run.assert_succeeded()
    """

    def run(
        self,
        graph: TurnGraph,
        ctx: TurnContext,
        registry: Any = None,
    ) -> RunResult:
        """Synchronous entry point — wraps the async execute() call."""
        return asyncio.run(self._run_async(graph, ctx, registry))

    async def run_async(
        self,
        graph: TurnGraph,
        ctx: TurnContext,
        registry: Any = None,
    ) -> RunResult:
        return await self._run_async(graph, ctx, registry)

    async def _run_async(
        self,
        graph: TurnGraph,
        ctx: TurnContext,
        registry: Any = None,
    ) -> RunResult:
        started = time.perf_counter()
        error: Exception | None = None
        result = None
        try:
            result = await graph.execute(ctx)
        except Exception as exc:
            error = exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

        # Pull persistence artifacts from registry if available
        checkpoints: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        composition_contract: dict[str, Any] = {}
        confluence_report: dict[str, Any] = {}
        if registry is not None and hasattr(registry, "persistence"):
            checkpoints = list(registry.persistence.checkpoints)
            events = list(registry.persistence.events)
        state = getattr(ctx, "state", None)
        if isinstance(state, dict):
            composition_contract = dict(state.get("last_execution_composition_contract") or {})
            confluence_report = dict(state.get("last_execution_confluence_report") or {})

        return RunResult(
            result=result,
            error=error,
            elapsed_ms=elapsed_ms,
            trace_id=ctx.trace_id,
            fidelity=ctx.fidelity,
            mutation_snapshot=ctx.mutation_queue.snapshot(),
            checkpoints=checkpoints,
            events=events,
            composition_contract=composition_contract,
            confluence_report=confluence_report,
        )
