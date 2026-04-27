"""MockRegistry and MockPersistenceService — full-fidelity test doubles.

MockRegistry supplies all services required by canonical graph nodes
(TemporalNode, HealthNode, ContextBuilderNode, InferenceNode, SafetyNode,
ReflectionNode, SaveNode) without any external dependencies.

MockPersistenceService is stateful across calls — the same instance is
returned every time registry.get("persistence_service") is called,
so checkpoint lists and event logs accumulate correctly across a full turn.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class MockPersistenceService:
    """Records checkpoints, events, and drained mutations across a full turn."""

    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.drained: list[Any] = []
        self.finalize_calls: int = 0
        self.save_turn_calls: int = 0

    # Called by SaveNode when finalize_turn is available
    def finalize_turn(self, ctx: Any, result: Any) -> Any:
        self.finalize_calls += 1
        # Drain the mutation queue — this is the canonical SaveNode commit path.
        def _drain_executor(intent: Any) -> None:
            self.drained.append(intent)
        ctx.mutation_queue.drain(_drain_executor, hard_fail_on_error=False)
        self.events.append({"event_type": "finalize_turn", "trace_id": ctx.trace_id})
        return result if result is not None else ("finalized", False)

    # Fallback when finalize_turn raises
    def save_turn(self, ctx: Any, result: Any) -> None:
        self.save_turn_calls += 1
        self.events.append({"event_type": "save_turn", "trace_id": ctx.trace_id})

    # Called by TurnGraph._emit_checkpoint
    def save_graph_checkpoint(self, payload: dict[str, Any], **_kw: Any) -> None:
        self.checkpoints.append(payload)

    # Called by TurnGraph._emit_checkpoint (event leg)
    def save_turn_event(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)

    def phase_transition_events(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == "phase_transition"]

    def checkpoint_sequence(self) -> list[str]:
        """Return (stage, status) pairs in order, for assertion."""
        return [(c.get("stage", "?"), c.get("status", "?")) for c in self.checkpoints]


class MockRegistry:
    """Full-fidelity mock registry supplying all graph-node services.

    Args:
        response:      The string the mock inference returns.
        fail_services: Set of service keys that should raise on ``get()``.
                       Use this to simulate dependency failures.
    """

    def __init__(
        self,
        *,
        response: str = "Mock Dad response",
        fail_services: set[str] | None = None,
    ) -> None:
        self._response = response
        self._fail: set[str] = fail_services or set()
        self.persistence = MockPersistenceService()

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN001
        if key in self._fail:
            raise RuntimeError(f"[MockRegistry] injected failure: service={key!r}")

        if key == "maintenance_service":
            return SimpleNamespace(tick=lambda ctx: {"status": "ok", "ticks": 1})

        if key == "context_service":
            return SimpleNamespace(
                build_context=lambda ctx: {
                    "memory": [],
                    "profile": {"name": "Test User"},
                    "source": "mock",
                }
            )

        if key == "agent_service":
            resp = self._response

            class _Agent:
                async def run_agent(self, ctx: Any, rich_context: Any) -> tuple[str, bool]:
                    return (resp, False)

            return _Agent()

        if key == "safety_service":
            return SimpleNamespace(
                enforce_policies=lambda ctx, candidate: (
                    candidate if candidate is not None else ("safe_fallback", False)
                )
            )

        if key == "reflection":
            return SimpleNamespace(
                reflect_after_turn=lambda *args, **kw: {"status": "ok", "reflected": True}
            )

        if key == "persistence_service":
            return self.persistence

        if key == "telemetry":
            return None

        return default
