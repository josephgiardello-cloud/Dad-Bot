from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_firewall import ExecutionFirewall, FirewallContext
from dadbot.core.invariant_registry import InvariantRegistry


@dataclass
class KernelValidationResult:
    ok: bool
    stage: str
    operation: str
    reason: str = ""


class ExecutionKernel:
    """Central authority layer for runtime execution validation.

    In shadow mode (strict=False), callers should wrap validate() and log
    violations without propagating them.
    """

    def __init__(
        self,
        firewall: ExecutionFirewall,
        invariant_registry: InvariantRegistry,
        quarantine: Any,
        *,
        strict: bool = False,
    ):
        self.firewall = firewall
        self.invariants = invariant_registry
        self.quarantine = quarantine
        self.strict = bool(strict)

    @staticmethod
    def _has_temporal(context: Any) -> bool:
        temporal = getattr(context, "temporal", None)
        if temporal is None:
            return False
        wall_time = str(getattr(temporal, "wall_time", "") or "").strip()
        wall_date = str(getattr(temporal, "wall_date", "") or "").strip()
        return bool(wall_time and wall_date)

    def validate(
        self,
        stage: str,
        operation: str,
        context: Any,
        *,
        mutation_outside_save_node: bool = False,
    ) -> KernelValidationResult:
        try:
            # Invariant checks are centralized here.
            self.invariants.check_invariants(
                context,
                stage=stage,
                operation=operation,
                mutation_outside_save_node=mutation_outside_save_node,
            )

            # Firewall checks are centralized here.
            firewall_context = FirewallContext(
                trace_id=str(getattr(context, "trace_id", "") or ""),
                stage=str(stage or ""),
                mutation_outside_save_node=bool(mutation_outside_save_node),
                temporal_missing=not self._has_temporal(context),
                metadata=dict(getattr(context, "metadata", {}) or {}),
            )
            self.firewall.enforce_execution_firewall(operation, firewall_context)
            return KernelValidationResult(
                ok=True,
                stage=str(stage or ""),
                operation=str(operation or ""),
            )
        except Exception as exc:
            if self.strict:
                raise
            return KernelValidationResult(
                ok=False,
                stage=str(stage or ""),
                operation=str(operation or ""),
                reason=str(exc),
            )

    async def run(
        self,
        turn_context: Any,
        pipeline: Iterable[tuple[str, Any]],
        execute_stage: Callable[[str, Any, Any], Awaitable[Any]],
    ) -> Any:
        """Experimental-only kernel loop.

        Production graph execution must not use this path. It remains only to
        satisfy legacy contracts that require an async ``run`` surface.
        """
        if str(os.getenv("DADBOT_ENABLE_EXPERIMENTAL_KERNEL_RUN", "")).strip().lower() not in {"1", "true", "yes"}:
            raise RuntimeError(
                "ExecutionKernel.run is experimental and disabled. "
                "Use TurnGraph canonical execution path."
            )
        for stage_name, stage_obj in pipeline:
            turn_context = await execute_stage(stage_name, stage_obj, turn_context)
            if bool(getattr(turn_context, "short_circuit", False)):
                return turn_context
        return turn_context


__all__ = ["ExecutionKernel", "KernelValidationResult"]
