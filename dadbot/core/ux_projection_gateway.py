"""UX Projection Gateway — post-execution UX assembly, decoupled from execution.

Design contract
---------------
The UX layer is a **pure projection system**.  It reads from finalised turn
state / metadata *after* execution completes and assembles user-facing health
and feedback payloads.  It does NOT participate in execution semantics.

Architectural role
------------------
::

    TurnGraph.execute()  →  returns FinalizedTurnResult
                           (UX fields are NOT assembled inside execute())

    caller / runtime adapter:
        gateway = TurnUxProjectionGateway()
        gateway.project(turn_context, total_latency_ms=..., failed=False)
        health = turn_context.state["turn_health_state"]   # ← stamped here

The gateway reads the stage-timing traces from ``turn_context.stage_traces``
and delegates to ``TurnUxProjector.project()`` — the same projector that
``GraphSideEffectsOrchestrator`` uses when ``inline_ux_projection`` is enabled
for backward compatibility.

Separation boundary
-------------------
- ``TurnUxProjectionGateway`` may only import from UX and data modules.
- It must NOT import from ``dadbot.core.graph``.
- The graph must NOT import from this module.
- The runtime adapter / app layer wires them together.

Usage
-----
::

    from dadbot.core.ux_projection_gateway import TurnUxProjectionGateway

    gateway = TurnUxProjectionGateway()
    gateway.project(turn_context, total_latency_ms=elapsed_ms, failed=False)
    health_dict = turn_context.state.get("turn_health_state", {})
    ux_feedback  = turn_context.state.get("ux_feedback", {})
"""

from __future__ import annotations

from typing import Any

from dadbot.core.ux_projection import (  # noqa: F401 (re-export)
    TurnHealthState,
    TurnUxProjector,
)


def _stage_duration_ms_from_context(turn_context: Any, stage_name: str) -> float:
    """Extract total elapsed ms for *stage_name* from ``turn_context.stage_traces``."""
    total = 0.0
    for trace in list(getattr(turn_context, "stage_traces", None) or []):
        if str(getattr(trace, "stage", "") or "").strip().lower() == str(stage_name or "").strip().lower():
            total += float(getattr(trace, "duration_ms", 0.0) or 0.0)
    return round(total, 3)


class TurnUxProjectionGateway:
    """Post-execution UX assembly gateway.

    Assembles all user-facing health and feedback payloads from a finalised
    ``TurnContext`` without touching execution semantics.

    This gateway is the **canonical entry point** for UX assembly.  The
    ``GraphSideEffectsOrchestrator`` provides an inline equivalent for
    backward compatibility, but new code should use this gateway exclusively.

    Parameters
    ----------
    degraded_latency_threshold_ms:
        Total turn latency above which the turn is flagged as
        ``DEGRADED_PERFORMANCE``.
    degraded_inference_threshold_ms:
        Inference stage latency threshold.
    degraded_memory_threshold_ms:
        Memory operations latency threshold.
    degraded_graph_sync_threshold_ms:
        Graph sync latency threshold.

    """

    def __init__(
        self,
        *,
        degraded_latency_threshold_ms: float = 2500.0,
        degraded_inference_threshold_ms: float = 2200.0,
        degraded_memory_threshold_ms: float = 1200.0,
        degraded_graph_sync_threshold_ms: float = 1200.0,
    ) -> None:
        self._projector = TurnUxProjector(
            degraded_latency_threshold_ms=degraded_latency_threshold_ms,
            degraded_inference_threshold_ms=degraded_inference_threshold_ms,
            degraded_memory_threshold_ms=degraded_memory_threshold_ms,
            degraded_graph_sync_threshold_ms=degraded_graph_sync_threshold_ms,
        )

    def project(
        self,
        turn_context: Any,
        *,
        total_latency_ms: float,
        failed: bool,
    ) -> None:
        """Assemble and stamp UX payloads onto *turn_context*.

        After this call:
        - ``turn_context.state["turn_health_state"]`` contains the health dict.
        - ``turn_context.state["ux_feedback"]`` contains the feedback dict.
        - ``turn_context.metadata["turn_health_state"]`` is populated.

        Parameters
        ----------
        turn_context:
            The finalised ``TurnContext`` produced by ``TurnGraph.execute()``.
            Must have ``stage_traces``, ``state``, ``metadata``, ``fidelity``.
        total_latency_ms:
            Wall-clock duration of the full turn execution in milliseconds.
        failed:
            True when the turn ended in an unhandled exception.

        """
        self._projector.project(
            turn_context,
            total_latency_ms=float(total_latency_ms or 0.0),
            failed=bool(failed),
            stage_duration_lookup=_stage_duration_ms_from_context,
        )

    def project_safe(
        self,
        turn_context: Any,
        *,
        total_latency_ms: float,
        failed: bool,
    ) -> dict[str, Any]:
        """Project UX state and return the health payload dict.

        Like :meth:`project` but additionally returns the assembled
        ``turn_health_state`` dict — useful for testing and for callers that
        do not want to reach into ``turn_context.state`` directly.

        Returns
        -------
        dict
            The ``turn_health_state`` payload stamped into the context.
            Returns an empty dict if projection fails silently.

        """
        try:
            self.project(turn_context, total_latency_ms=total_latency_ms, failed=failed)
        except Exception:  # noqa: BLE001
            return {}
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            return dict(state.get("turn_health_state") or {})
        return {}

    @staticmethod
    def mark_structural_degradation(turn_context: Any, reason: str) -> None:
        """Mark structural degradation in turn context (delegated to projector)."""
        TurnUxProjector.mark_structural_degradation(turn_context, reason)
