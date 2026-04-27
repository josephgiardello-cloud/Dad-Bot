"""Graph side-effects orchestrator — single boundary for all persistence and UX
side effects emitted during graph execution.

Design contract
---------------
``TurnGraph`` contains NO direct calls to persistence or UX layers.  Every
side effect (checkpoint emission, phase transition, kernel rejection events,
identity sealing, UX health projection) is routed through
``GraphSideEffectsOrchestrator``.

Architectural role
------------------
::

    Graph (topology only)
        └── GraphSideEffectsOrchestrator   ← this module
               ├── GraphPersistenceEventAdapter  (persistence boundary)
               ├── TurnUxProjector               (UX projection boundary)
               └── ExecutionPolicyEngine          (policy queries)

The graph calls the orchestrator.  The orchestrator decides which adapter or
projector handles the side effect.  No execution decision logic lives here —
only routing and delegation.
"""
from __future__ import annotations

from typing import Any, Callable

from dadbot.core.execution_identity import ExecutionIdentity
from dadbot.core.execution_policy import (
    ExecutionPolicyEngine,
    KernelRejectionSemantics,
    PersistenceServiceContract,
)
from dadbot.core.persistence_event_adapter import GraphPersistenceEventAdapter
from dadbot.core.ux_projection import TurnUxProjector


class GraphSideEffectsOrchestrator:
    """Routes all graph execution side effects to the appropriate adapter.

    This is the *only* object that ``TurnGraph`` uses to interact with
    persistence or UX layers.  The graph itself imports nothing from those
    layers directly.

    Parameters
    ----------
    persistence_event_adapter:
        Adapter responsible for shaping and emitting all persistence events.
    ux_projector:
        Projector that assembles user-facing health/status payloads from raw
        execution timing and state.
    policy_engine:
        Policy engine used for rejection-semantics queries and persistence
        contract validation.
    json_safe:
        The same ``_json_safe`` helper that the graph uses for payload
        serialisation — injected to avoid coupling this module to the graph.
    """

    def __init__(
        self,
        *,
        persistence_event_adapter: GraphPersistenceEventAdapter,
        ux_projector: TurnUxProjector,
        policy_engine: ExecutionPolicyEngine,
        json_safe: Callable[[Any], Any],
    ) -> None:
        self._persistence = persistence_event_adapter
        self._ux = ux_projector
        self._policy = policy_engine
        self._json_safe = json_safe

    # ------------------------------------------------------------------
    # Policy queries (delegation — no interpretation in the graph)
    # ------------------------------------------------------------------

    def rejection_semantics_for_stage(self, stage: str) -> KernelRejectionSemantics:
        """Return the rejection semantics for *stage* from the policy engine."""
        return self._policy.rejection_semantics_for_stage(stage)

    def set_kernel_rejection_semantics(self, stage: str, semantics: KernelRejectionSemantics) -> None:
        """Override rejection semantics for *stage*."""
        self._policy.set_kernel_rejection_semantics(stage, semantics)

    def validate_persistence_service_contract(
        self, turn_context: Any, service: Any
    ) -> None:
        """Validate the persistence service and stamp the result into context."""
        payload = self._policy.validate_persistence_service_contract(
            service,
            strict_mode=bool(
                (getattr(turn_context, "metadata", None) or {}).get(
                    "persistence_contract_strict", False
                )
            ),
        )
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        if isinstance(state, dict):
            state["persistence_contract"] = payload
        if isinstance(metadata, dict):
            metadata["persistence_contract"] = dict(payload)

    # ------------------------------------------------------------------
    # Persistence side effects
    # ------------------------------------------------------------------

    def emit_checkpoint(
        self,
        *,
        registry: Any,
        turn_context: Any,
        stage: str,
        status: str,
        error: str | None = None,
        active_stage: str = "",
        checkpoint_snapshot_fn: Callable[..., dict[str, Any]],
    ) -> None:
        """Emit a graph checkpoint event to the persistence service."""
        if registry is None:
            return
        service = registry.get("persistence_service")
        self.validate_persistence_service_contract(turn_context, service)
        if service is None:
            return
        determinism_lock = dict(
            (getattr(turn_context, "metadata", None) or {}).get("determinism") or {}
        )
        checkpoint = checkpoint_snapshot_fn(
            stage=stage,
            status=status,
            error=error,
            advance_chain=bool(
                (getattr(turn_context, "metadata", None) or {}).get(
                    "checkpoint_every_node", False
                )
            ),
        )
        self._persistence.emit_graph_checkpoint(
            service=service,
            turn_context=turn_context,
            stage=stage,
            status=status,
            error=str(error or "").strip(),
            active_stage=str(active_stage or ""),
            determinism_lock=determinism_lock,
            checkpoint=checkpoint,
        )

    def emit_phase_transition(
        self,
        *,
        registry: Any,
        turn_context: Any,
        stage: str,
        transition: dict[str, Any],
    ) -> None:
        """Emit a phase-transition event to the persistence service."""
        if registry is None:
            return
        service = registry.get("persistence_service")
        if service is None:
            return
        determinism_lock = dict(
            (getattr(turn_context, "metadata", None) or {}).get("determinism") or {}
        )
        self._persistence.emit_phase_transition(
            service=service,
            turn_context=turn_context,
            stage=stage,
            transition=transition,
            determinism_lock=determinism_lock,
        )

    def emit_kernel_rejection(
        self,
        *,
        registry: Any,
        turn_context: Any,
        stage: str,
        reason: str,
        semantics: KernelRejectionSemantics,
    ) -> None:
        """Emit a kernel-rejection event to the persistence service."""
        if registry is None:
            return
        service = registry.get("persistence_service")
        if service is None:
            return
        self._persistence.emit_kernel_rejection(
            service=service,
            turn_context=turn_context,
            stage=stage,
            reason=reason,
            semantics=semantics,
        )

    def emit_execution_identity(
        self,
        *,
        registry: Any,
        turn_context: Any,
        identity: ExecutionIdentity,
    ) -> None:
        """Emit the execution identity as a durable persistence event."""
        if registry is None:
            return
        service = registry.get("persistence_service")
        if service is None:
            return
        self._persistence.emit_execution_identity(
            service=service,
            turn_context=turn_context,
            identity=identity,
        )

    # ------------------------------------------------------------------
    # UX / observability side effects
    # ------------------------------------------------------------------

    def project_turn_health(
        self,
        turn_context: Any,
        *,
        total_latency_ms: float,
        failed: bool,
        stage_duration_lookup: Callable[[Any, str], float],
    ) -> None:
        """Assemble and stamp UX health state into turn context.

        NOTE: Although this is called from within ``TurnGraph.execute()`` for
        backward compatibility, the preferred pattern is to invoke this through
        ``TurnUxProjectionGateway`` after execution completes.  The gateway
        provides a fully decoupled post-execution projection path.
        """
        self._ux.project(
            turn_context,
            total_latency_ms=float(total_latency_ms or 0.0),
            failed=bool(failed),
            stage_duration_lookup=stage_duration_lookup,
        )

    def mark_structural_degradation(self, turn_context: Any, reason: str) -> None:
        """Mark structural degradation in turn context via the UX projector."""
        self._ux.mark_structural_degradation(turn_context, reason)
