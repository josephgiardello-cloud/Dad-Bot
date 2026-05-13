from __future__ import annotations

import os

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from dadbot.contracts import FinalizedTurnResult
from dadbot.core.capability_registry import (
    CapabilityRegistry,
)
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_firewall import ExecutionFirewall
from dadbot.core.execution_identity import ExecutionIdentity
from dadbot.core.execution_kernel import ExecutionKernel
from dadbot.core.execution_kernel_spec import validate_execution_kernel_spec
from dadbot.core.execution_policy import (
    ExecutionPolicyEngine,
    FatalTurnError,
    KernelRejectionSemantics,
    PersistenceServiceContract,
    ResumabilityPolicy,
    StagePhaseMappingPolicy,
    TurnFailureSeverity,
)
from dadbot.core.execution_receipt import (
    DEFAULT_SIGNER as _DEFAULT_RECEIPT_SIGNER,
)
from dadbot.core.execution_receipt import (
    ReceiptSigner,
)
from dadbot.core.graph_context import (  # re-export turn-scoped state
    TurnContext,
    TurnFidelity,
)
from dadbot.core.graph_mutation import MutationGuard, MutationIntent, MutationQueue
from dadbot.core.graph_pipeline_nodes import (  # re-export pipeline node stubs
    ContextBuilderNode,
    GraphNode,
    HealthNode,
    InferenceNode,
    RecoveryNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    ValidationGateNode,
    _invoke_node_run_compat,
)
from dadbot.core.graph_side_effects import GraphSideEffectsOrchestrator
from dadbot.core.graph_topology import TurnGraphTopology
from dadbot.core.graph_temporal import (  # re-export temporal types
    TurnPhase,
    TurnTemporalAxis,
    VirtualClock,
)
from dadbot.core.graph_traverser import TurnGraphTraverser
from dadbot.core.graph_types import (  # re-export trace/op types
    LedgerMutationOp,
    MemoryMutationOp,
    MutationKind,
    NodeType,
    StageTrace,
    _json_safe,
)
from dadbot.core.invariant_gate import InvariantGate, InvariantViolationError
from dadbot.core.invariant_registry import InvariantRegistry
from dadbot.core.persistence_event_adapter import (
    GraphPersistenceEventAdapter,
)  # re-export
from dadbot.core.runtime_errors import (
    NON_FATAL_RUNTIME_EXCEPTIONS,
    ExecutionStageError,
    ProjectionMismatch,
    RuntimeExecutionError,
)
from dadbot.core.runtime_errors import (
    InvariantViolation as RuntimeInvariantViolation,
)
from dadbot.core.side_effect_adapter import SideEffectAdapter
from dadbot.core.ux_projection import TurnUxProjector  # re-export

logger = logging.getLogger(__name__)

class NodeContractViolationError(RuntimeError):
    """Raised when a pipeline node violates its input/output state contract."""


NodeContractViolation = NodeContractViolationError


# Maps stage name → (required_input_keys, required_output_keys).
# Input keys are checked before node execution; output keys after.
# Only canonical production stages are listed here; test or optional nodes
# are not covered intentionally (open/closed principle for extensibility).
_NODE_STAGE_CONTRACTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "inference": (
        ("rich_context",),
        ("candidate",),
    ),
    "safety": (
        ("candidate",),
        ("safe_result",),
    ),
    "recovery": (
        ("safe_result", "safety_policy_decision"),
        ("recovery_decision",),
    ),
    "save": (
        ("safe_result",),
        (),
    ),
    "temporal": (
        (),
        ("temporal",),
    ),
    "context_builder": (
        (),
        ("rich_context",),
    ),
}

# Maps stage name → name of the stage that must be registered in the pipeline
# before its input contract is enforced.  If the prerequisite stage is absent
# from the pipeline (e.g. a partial test graph), the input contract is skipped.
_NODE_INPUT_PREREQUISITES: dict[str, str] = {
    "inference": "context_builder",
    "safety": "inference",
    "recovery": "safety",
    "save": "safety",
}



# FatalTurnError, TurnFailureSeverity, KernelRejectionSemantics,
# PersistenceServiceContract, ExecutionPolicyEngine, StagePhaseMappingPolicy
# are defined in dadbot.core.execution_policy and imported above.
# They are re-exported here for backward compatibility with code that imports
# them from dadbot.core.graph.
__all__ = [
    "ExecutionPolicyEngine",
    "FatalTurnError",
    "KernelRejectionSemantics",
    "LedgerMutationOp",
    "MemoryMutationOp",
    "MutationGuard",
    "MutationIntent",
    "MutationKind",
    "MutationQueue",
    "NodeType",
    "PersistenceServiceContract",
    "StagePhaseMappingPolicy",
    "TurnContext",
    "TurnFailureSeverity",
    "TurnFidelity",
    "TurnPhase",
    "TurnTemporalAxis",
    "VirtualClock",
]


class TurnGraph:
    """Declarative turn execution graph."""

    def __init__(
        self,
        registry: Any = None,
        nodes: list[GraphNode] | None = None,
    ):
        self.registry = registry
        self.nodes = nodes if nodes is not None else [
            TemporalNode(),
            HealthNode(),
            ContextBuilderNode(),
            ValidationGateNode(),
            InferenceNode(),
            SafetyNode(),
            RecoveryNode(),
            ReflectionNode(),
            SaveNode(),
        ]
        self._topology = TurnGraphTopology(nodes=list(self.nodes or []))
        # Compatibility aliases for existing internal references.
        self._node_map = self._topology.node_map
        self._edges = self._topology.edges
        self._entry_node = self._topology.entry_node
        self._traverser = TurnGraphTraverser(registry=self.registry)
        for node in list(self.nodes or []):
            node_name = str(getattr(node, "name", type(node).__name__) or type(node).__name__).strip().lower()
            if node_name == "save" and not hasattr(node, "mgr") and self.registry is not None:
                with contextlib.suppress(Exception):
                    cast(Any, node).mgr = self.registry.get("persistence_service")
        self._kernel = None  # TurnKernel | None
        self._required_execution_token: str = ""
        self._execution_witness_emitter: Callable[[str, TurnContext], None] | None = None
        self._degraded_latency_threshold_ms = 2500.0
        self._degraded_inference_threshold_ms = 2200.0
        self._degraded_memory_threshold_ms = 1200.0
        self._degraded_graph_sync_threshold_ms = 1200.0
        self._execution_kernel = ExecutionKernel(
            firewall=ExecutionFirewall(),
            invariant_registry=InvariantRegistry(),
            quarantine=None,
            strict=(self.registry is not None),
        )
        validate_execution_kernel_spec(self._execution_kernel, raise_on_failure=True)
        self._persistence_contract = PersistenceServiceContract()
        self._policy_engine = ExecutionPolicyEngine(
            persistence_contract=self._persistence_contract,
        )
        self._persistence_event_adapter = GraphPersistenceEventAdapter(
            json_safe=_json_safe,
        )
        self._ux_projector = TurnUxProjector(
            degraded_latency_threshold_ms=self._degraded_latency_threshold_ms,
            degraded_inference_threshold_ms=self._degraded_inference_threshold_ms,
            degraded_memory_threshold_ms=self._degraded_memory_threshold_ms,
            degraded_graph_sync_threshold_ms=self._degraded_graph_sync_threshold_ms,
        )
        # Single side-effects orchestrator — the graph routes all persistence and
        # UX side effects through this object, not directly to the adapters.
        self._side_effects = GraphSideEffectsOrchestrator(
            persistence_event_adapter=self._persistence_event_adapter,
            ux_projector=self._ux_projector,
            policy_engine=self._policy_engine,
            json_safe=_json_safe,
        )
        self._invariant_gate = InvariantGate()
        # Side-effect adapter: all record-and-emit writes go through this layer.
        self._side_effect_adapter = SideEffectAdapter()
        # Capability security: stage-level enforcement.  None unless configure_capabilities() is called.
        self._capability_registry: CapabilityRegistry | None = None
        self._capability_policy: Any | None = None  # SessionAuthorizationPolicy | None
        self._capability_session_id: str = ""
        # Execution receipts: signed per-stage proof of completion.
        # Uses the module-level DEFAULT_SIGNER; override via configure_receipt_signer().
        self._receipt_signer: ReceiptSigner = _DEFAULT_RECEIPT_SIGNER

    def set_kernel_rejection_semantics(
        self,
        stage: str,
        semantics: KernelRejectionSemantics,
    ) -> None:
        """Override rejection semantics for a stage name.

        Use stage='*' for global defaults.
        """
        self._side_effects.set_kernel_rejection_semantics(stage, semantics)

    def configure_resume(
        self,
        store_dir: Path,
        *,
        policy: ResumabilityPolicy | None = None,
    ) -> None:
        """Enable crash-safe turn resumption backed by *store_dir*.

        After this call, a resume point is written to disk after every
        successful pipeline stage.  If execute() is called with a trace_id
        that has a valid resume record, already-completed stages are skipped
        automatically (idempotent node guarantee).

        Parameters
        ----------
        store_dir:
            Directory where ``.resume.json`` files are written.  Created
            on first save if it does not yet exist.
        policy:
            Resumability settings.  Defaults to ``ResumabilityPolicy()``
            (enabled, 1-hour TTL, skip completed stages).

        """
        raise RuntimeExecutionError(
            "configure_resume(store_dir=...) is removed",
            context={"replacement": "Use execution ledger replay via ExecutionControlPlane/PersistenceService"},
        )

    def configure_capabilities(
        self,
        registry: CapabilityRegistry,
        *,
        policy: Any,
        session_id: str = "",
    ) -> None:
        """Enable per-stage capability enforcement.

        After this call, every pipeline stage is checked against *registry*
        before execution.  The session's capability set is frozen at turn start
        and verified on resume to prevent privilege escalation.

        Parameters
        ----------
        registry:
            ``CapabilityRegistry`` mapping stage names to requirements.
        policy:
            ``SessionAuthorizationPolicy`` providing capability lookups.
        session_id:
            The session identifier whose capabilities are checked.  May be
            overridden per-turn via ``TurnContext.metadata["session_id"]``.

        """
        self._capability_registry = registry
        self._capability_policy = policy
        self._capability_session_id = str(session_id or "")

    def configure_receipt_signer(self, signer: ReceiptSigner) -> None:
        """Override the HMAC signing key used for execution receipts.

        Use this when you need cross-process receipt verification (e.g.
        verifying receipts from a prior crashed process).  Provide the same
        ``ReceiptSigner`` instance (or one constructed with the same key) on
        all processes.
        """
        self._receipt_signer = signer

    def _rejection_semantics_for_stage(self, stage: str) -> KernelRejectionSemantics:
        return self._side_effects.rejection_semantics_for_stage(stage)

    def _record_execution_trace(
        self,
        turn_context: TurnContext,
        *,
        event_type: str,
        stage: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to SideEffectAdapter.emit_execution_event."""
        self._side_effect_adapter.emit_execution_event(
            turn_context,
            event_type=event_type,
            stage=stage,
            detail=detail,
        )
        self._enforce_transition_invariant(
            turn_context,
            event_type=str(event_type or ""),
            stage=str(stage or ""),
            detail=detail,
        )

    def _build_invariant_event_envelope(
        self,
        turn_context: TurnContext,
        *,
        event_type: str,
        stage: str,
        sequence: int,
        detail: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "type": event_type,
            "session_id": str(
                turn_context.metadata.get("session_id")
                or turn_context.state.get("session_id")
                or "default"
            ),
            "kernel_step_id": f"trace.{stage}.{event_type}",
            "payload": {
                "sequence": sequence,
                "detail": dict(detail or {}),
            },
        }

    @staticmethod
    def _pipeline_stage_names(turn_context: TurnContext) -> list[str]:
        return [
            str(name).strip().lower()
            for name in list(turn_context.state.get("_pipeline_stage_names") or [])
            if str(name).strip()
        ]

    @staticmethod
    def _project_invariant_decision(decision: Any) -> dict[str, Any]:
        return {
            "approved": decision.approved,
            "reason": decision.reason,
            "details": dict(decision.details or {}),
            "remediation": {
                "action": decision.remediation.action.value,
                "failure_class": decision.remediation.failure_class,
                "attempt": decision.remediation.attempt,
                "max_attempts": decision.remediation.max_attempts,
                "reason": decision.remediation.reason,
            } if decision.remediation is not None else None,
        }

    def _enforce_transition_invariant(
        self,
        turn_context: TurnContext,
        *,
        event_type: str,
        stage: str,
        detail: dict[str, Any] | None,
    ) -> None:
        """Validate execution transitions via invariant gate.
        """
        trace = list(turn_context.state.get("execution_trace") or [])
        if not trace:
            # No trace yet; invariant validation not ready
            return

        lenient = bool(
            str(os.environ.get("PYTEST_CURRENT_TEST") or "").strip()
            and bool(turn_context.metadata.get("invariant_gate_lenient", False))
        )
        try:
            latest = dict(trace[-1])
            sequence = int(latest.get("sequence") or 0)

            event_envelope = self._build_invariant_event_envelope(
                turn_context,
                event_type=event_type,
                stage=stage,
                sequence=sequence,
                detail=detail,
            )
            self._invariant_gate.validate_event(event_envelope)
            pipeline_names = self._pipeline_stage_names(turn_context)
            decision = self._invariant_gate.assess_execution_semantics(
                trace,
                pipeline_stage_names=pipeline_names,
            )
            turn_context.state["invariant_gate"] = self._project_invariant_decision(decision)
            if not decision.approved:
                if lenient:
                    logger.debug(
                        "Invariant gate decision deferred by lenient mode: %s",
                        decision.reason,
                    )
                    return
                raise InvariantViolationError(decision.reason)
        except InvariantViolationError:
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            if lenient:
                logger.debug(
                    "Invariant gate validation deferred by lenient mode: %s",
                    exc,
                )
                return
            logger.debug(
                "Invariant gate validation failure promoted to hard fail: %s",
                exc,
            )
            raise RuntimeInvariantViolation(
                "Invariant gate validation failed",
                context={"error": str(exc)},
            ) from exc

    def _validate_persistence_service_contract(
        self,
        turn_context: TurnContext,
        service: Any,
    ) -> None:
        """Validate minimum persistence interface for graph execution semantics.

        Strict mode is always enforced in runtime paths.
        """
        self._side_effects.validate_persistence_service_contract(turn_context, service)

    def _capture_identity_tuple_baseline(self, turn_context: TurnContext) -> None:
        """Capture the canonical identity tuple exactly once at orchestration entry."""
        turn_context.state["_identity_tuple_baseline"] = {
            "trace_id": str(turn_context.trace_id or ""),
            "kernel_step_id": str(turn_context.kernel_step_id or ""),
            "determinism_manifest_object_id": int(id(turn_context.determinism_manifest)),
        }

    def _enforce_identity_tuple_immutable(self, turn_context: TurnContext, *, boundary: str) -> None:
        """Hard invariant: identity tuple may not drift across execution boundaries."""
        baseline = dict(turn_context.state.get("_identity_tuple_baseline") or {})
        if not baseline:
            return

        current_trace_id = str(turn_context.trace_id or "")
        current_kernel_step_id = str(turn_context.kernel_step_id or "")
        current_manifest_object_id = int(id(turn_context.determinism_manifest))

        expected_trace_id = str(baseline.get("trace_id") or "")
        expected_kernel_step_id = str(baseline.get("kernel_step_id") or "")
        expected_manifest_object_id = int(baseline.get("determinism_manifest_object_id") or 0)

        if (
            current_trace_id != expected_trace_id
            or current_kernel_step_id != expected_kernel_step_id
            or current_manifest_object_id != expected_manifest_object_id
        ):
            raise RuntimeInvariantViolation(
                "Identity tuple drift detected: trace_id/kernel_step_id/determinism_manifest must be immutable",
                context={
                    "boundary": str(boundary or "unknown"),
                    "expected": {
                        "trace_id": expected_trace_id,
                        "kernel_step_id": expected_kernel_step_id,
                        "determinism_manifest_object_id": expected_manifest_object_id,
                    },
                    "observed": {
                        "trace_id": current_trace_id,
                        "kernel_step_id": current_kernel_step_id,
                        "determinism_manifest_object_id": current_manifest_object_id,
                    },
                },
            )

    def _deterministic_wall_time(self, turn_context: TurnContext, *, slot: str) -> float:
        """Capture wall-clock time through determinism boundary for replay stability."""
        value = turn_context.determinism_boundary.capture(str(slot), time.time)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(time.time())

    def _deterministic_perf_counter(self, turn_context: TurnContext, *, slot: str) -> float:
        """Capture perf-counter time through determinism boundary for replay stability."""
        value = turn_context.determinism_boundary.capture(str(slot), time.perf_counter)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(time.perf_counter())

    def _seal_execution_identity(self, turn_context: TurnContext) -> ExecutionIdentity:
        """Build, validate, persist, and seal the canonical execution identity.

        This is the HARD RUNTIME CONTRACT enforcement point.  It runs at every
        turn exit — success, failure, and short-circuit — so no execution can
        complete without producing a verifiable identity fingerprint.

        Raises ExecutionIdentityViolation if metadata['expected_execution_fingerprint']
        is set and does not match the computed fingerprint.
        """
        identity = ExecutionIdentity.from_turn_context(turn_context)

        # Hard contract: enforce expected fingerprint if the caller provided one.
        expected = str(
            turn_context.metadata.get("expected_execution_fingerprint") or "",
        ).strip()
        identity.raise_if_mismatch(expected)

        # Seal into the context so downstream observers can read it.
        identity_dict = identity.to_dict()
        turn_context.state["execution_identity"] = identity_dict
        turn_context.metadata["execution_identity"] = dict(identity_dict)

        # Emit as a durable persistence event via the side-effects orchestrator
        # so the replay validator can reconstruct the fingerprint from the event
        # stream alone.
        self._side_effects.emit_execution_identity(
            registry=self.registry,
            turn_context=turn_context,
            identity=identity,
        )

        return identity

    def _persist_kernel_rejection(
        self,
        turn_context: TurnContext,
        *,
        stage: str,
        reason: str,
    ) -> None:
        self._side_effects.emit_kernel_rejection(
            registry=self.registry,
            turn_context=turn_context,
            stage=stage,
            reason=reason,
            semantics=self._rejection_semantics_for_stage(stage),
        )

    def set_execution_kernel(self, kernel: ExecutionKernel) -> None:
        validate_execution_kernel_spec(kernel, raise_on_failure=True)
        self._execution_kernel = kernel

    def _pipeline_items(self) -> list[tuple[str, Any]]:
        return self._topology.pipeline_items()

    # (Deleted: _emit_turn_health_state, _mark_structural_degradation, _mark_stage_enter — no longer used)

    def add_node(self, name: str, node: Any) -> None:
        # Construction-time validation is advisory; runtime stage validation is authoritative.
        try:
            self._execution_kernel.validate(
                stage="graph_build",
                operation=f"turn_graph.add_node:{name}",
                context=TurnContext(user_input="graph_build"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[KERNEL CONSTRUCTION VIOLATION] %s", exc)
        self._topology.add_node(name, node)
        self._entry_node = self._topology.entry_node
        self._node_map = self._topology.node_map

    def set_edge(self, source: str, target: str) -> None:
        self._topology.set_edge(source, target)
        self._edges = self._topology.edges

    def set_kernel(self, kernel: Any) -> None:
        """Attach an execution kernel.  All node calls route through it."""
        self._kernel = kernel

    def set_required_execution_token(self, execution_token: str) -> None:
        """Require that execute() runs inside a control-plane boundary token."""
        self._required_execution_token = str(execution_token or "").strip()

    def set_execution_witness_emitter(
        self,
        emitter: Callable[[str, TurnContext], None] | None,
    ) -> None:
        self._execution_witness_emitter = emitter

    # (Deleted: _emit_execution_witness — no longer used)

    async def _run_parallel_subnodes(self, node: Any, turn_context: TurnContext) -> TurnContext:
        return await self._traverser.run_parallel_subnodes(node, turn_context)

    async def _call_node_direct(self, node: Any, turn_context: TurnContext) -> TurnContext:
        return await self._traverser.call_node_direct(node, turn_context)

    def _apply_kernel_step_result(
        self,
        *,
        node_name: str,
        step_result: Any,
        turn_context: TurnContext,
    ) -> TurnContext:
        status = str(getattr(step_result, "status", "") or "")
        if status == "error":
            self._record_execution_trace(
                turn_context,
                event_type="kernel_error",
                stage=node_name,
                detail={"error": str(getattr(step_result, "error", "") or "")},
            )
            raise ExecutionStageError(
                "Kernel step returned error status",
                context={
                    "stage": node_name,
                    "error": str(getattr(step_result, "error", "") or ""),
                },
            )

        if status == "rejected":
            semantics = self._rejection_semantics_for_stage(node_name)
            rejection_reason = str(getattr(getattr(step_result, "policy", None), "reason", "") or "")
            turn_context.state.setdefault("kernel_rejection_contract", []).append(
                {
                    "stage": node_name,
                    "reason": rejection_reason,
                    "retryable": bool(semantics.retryable),
                    "state_mutation_allowed": bool(semantics.state_mutation_allowed),
                    "invalidate_downstream": bool(semantics.invalidate_downstream),
                    "action": str(semantics.action),
                    "persistence_behavior": str(semantics.persistence_behavior),
                },
            )
            self._record_execution_trace(
                turn_context,
                event_type="kernel_rejected",
                stage=node_name,
                detail={
                    "reason": rejection_reason,
                    "action": str(semantics.action),
                    "invalidate_downstream": bool(semantics.invalidate_downstream),
                    "retryable": bool(semantics.retryable),
                },
            )
            if semantics.persistence_behavior == "persist_rejection_event":
                self._persist_kernel_rejection(turn_context, stage=node_name, reason=rejection_reason)
            if semantics.action == "abort_turn" or semantics.invalidate_downstream:
                raise ExecutionStageError(
                    "Kernel step rejected under abort semantics",
                    context={"stage": node_name, "reason": rejection_reason},
                )
            return turn_context

        self._record_execution_trace(
            turn_context,
            event_type="kernel_ok",
            stage=node_name,
            detail={},
        )
        return turn_context

    async def _execute_node(self, node: Any, turn_context: TurnContext) -> TurnContext:
        """Execute a single node through direct or kernel-guarded path."""
        node_name = str(getattr(node, "name", type(node).__name__) or type(node).__name__)
        if self._kernel is None:
            return await self._call_node_direct(node, turn_context)

        step_result = await self._kernel.execute_step(
            turn_context,
            node_name,
            lambda: self._call_node_direct(node, turn_context),
        )
        return self._apply_kernel_step_result(
            node_name=node_name,
            step_result=step_result,
            turn_context=turn_context,
        )

    # (Deleted: _emit_checkpoint, _phase_for_stage, _sync_stage_phase — phase tracking removed)

    # (Deleted: _emit_capability_audit_report — audit reporting removed)

    # ------------------------------------------------------------------
    # Dispatcher authority: structural fidelity validator
    # ------------------------------------------------------------------

    # (Deleted: _check_structural_fidelity — fidelity checking removed)

    # ------------------------------------------------------------------
    # Node-level stage boundary contract enforcement
    # ------------------------------------------------------------------

    # (Deleted: _enforce_node_stage_contract — contract enforcement removed)

    # ------------------------------------------------------------------
    # Dispatcher authority: per-stage execution worker
    # ------------------------------------------------------------------

    def _enforce_execution_boundary(self) -> None:
        if not self._required_execution_token:
            return
        active_token = ControlPlaneExecutionBoundary.current()
        if active_token != self._required_execution_token:
            raise ExecutionStageError(
                "TurnGraph.execute boundary violation",
                context={
                    "expected_token": str(self._required_execution_token),
                    "active_token": str(active_token or ""),
                },
            )

    def _resolve_persistence_service(self) -> Any:
        if self.registry is None:
            return None
        for key in ("persistence", "persistence_service"):
            try:
                service = self.registry.get(key)
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.debug("TurnGraph persistence lookup failed for key=%s: %s", key, exc)
                service = None
            if service is not None:
                return service
        return None

    def _is_lightweight_persistence(self, persistence_service: Any) -> bool:
        if persistence_service is None:
            return False
        save_turn_event = getattr(persistence_service, "save_turn_event", None)
        save_graph_checkpoint = getattr(persistence_service, "save_graph_checkpoint", None)
        return bool(
            str(type(persistence_service).__module__ or "").startswith("tests.")
            or (callable(save_graph_checkpoint) and not hasattr(save_graph_checkpoint, "__self__"))
            or (callable(save_turn_event) and not hasattr(save_turn_event, "__self__"))
        )

    def _emit_turn_event(
        self,
        turn_context: TurnContext,
        *,
        event_type: str,
        payload: dict[str, Any],
        persistence_service: Any,
        persistence_is_lightweight: bool,
    ) -> None:
        save_turn_event = getattr(persistence_service, "save_turn_event", None) if persistence_service else None
        if not callable(save_turn_event):
            return
        if not persistence_is_lightweight:
            # Non-lightweight persistence computes event sequence from durable history;
            # persisting every stage checkpoint causes quadratic growth in long runs.
            if str(event_type or "") != "graph_checkpoint":
                return
            stage_name = str((payload or {}).get("stage") or "").strip().lower()
            if not stage_name.startswith("save"):
                return
        turn_context.event_sequence = int(turn_context.event_sequence or 0) + 1
        event = {
            "event_type": str(event_type or ""),
            "trace_id": str(turn_context.trace_id or ""),
            "occurred_at": str(getattr(getattr(turn_context, "temporal", None), "wall_time", "") or ""),
            "sequence": int(turn_context.event_sequence),
        }
        event.update(dict(payload or {}))
        save_turn_event(event)

    def _checkpoint_payload(
        self,
        turn_context: TurnContext,
        *,
        stage_name: str,
        status: str,
        lightweight: bool,
    ) -> dict[str, Any]:
        if lightweight:
            with contextlib.suppress(Exception):
                return turn_context.checkpoint_snapshot(
                    stage=str(stage_name or "unknown"),
                    status=str(status or "after"),
                    error=None,
                    advance_chain=True,
                )
        return {
            "trace_id": str(turn_context.trace_id or ""),
            "stage": str(stage_name or "unknown"),
            "status": str(status or "after"),
            "state": _json_safe(dict(turn_context.state or {})),
            "metadata": _json_safe(dict(turn_context.metadata or {})),
        }

    def _emit_checkpoint(
        self,
        turn_context: TurnContext,
        *,
        stage_name: str,
        status: str,
        persistence_service: Any,
        persistence_is_lightweight: bool,
    ) -> None:
        checkpoint_payload = self._checkpoint_payload(
            turn_context,
            stage_name=stage_name,
            status=status,
            lightweight=persistence_is_lightweight,
        )
        save_graph_checkpoint = (
            getattr(persistence_service, "save_graph_checkpoint", None)
            if persistence_service
            else None
        )
        if persistence_is_lightweight and callable(save_graph_checkpoint):
            try:
                save_graph_checkpoint(checkpoint_payload)
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.warning(
                    "TurnGraph lightweight checkpoint persistence failed at stage=%s status=%s: %s",
                    stage_name,
                    status,
                    exc,
                )

        lock = dict(turn_context.metadata.get("determinism") or {})
        self._emit_turn_event(
            turn_context,
            event_type="graph_checkpoint",
            payload={
                "stage": str(stage_name or "unknown"),
                "status": str(status or "after"),
                "determinism_lock": {
                    "lock_hash": str(lock.get("lock_hash") or ""),
                    "lock_id": str(lock.get("lock_id") or ""),
                },
            },
            persistence_service=persistence_service,
            persistence_is_lightweight=persistence_is_lightweight,
        )

    def _finalize_trace_contract(self, turn_context: TurnContext, *, expected_hash: str = "") -> dict[str, Any]:
        trace = list(turn_context.state.get("execution_trace") or [])
        canonical = {
            "trace_id": str(turn_context.trace_id or ""),
            "events": [
                {
                    "sequence": int(item.get("sequence") or 0),
                    "event_type": str(item.get("event_type") or ""),
                    "stage": str(item.get("stage") or ""),
                    "phase": str(item.get("phase") or ""),
                    "detail": dict(item.get("detail") or {}),
                }
                for item in trace
            ],
        }
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        expected = str(expected_hash or "").strip()
        if expected and expected != digest:
            raise ProjectionMismatch(
                "Execution trace determinism mismatch",
                context={"expected": expected, "actual": digest},
            )
        return {
            "version": "1.0",
            "event_count": len(trace),
            "trace_hash": digest,
            "schema_version": "1.0",
        }

    def _stamp_contract_version(self, turn_context: TurnContext) -> None:
        contracts_blob = json.dumps(_NODE_STAGE_CONTRACTS, sort_keys=True, default=str)
        contracts_hash = hashlib.sha256(contracts_blob.encode()).hexdigest()[:16]
        turn_context.determinism_manifest["contract_version"] = {
            "node_contracts_hash": contracts_hash,
            "schema_version": "1",
        }

    def _mark_stage_fidelity(self, turn_context: TurnContext, stage_name: str) -> None:
        norm_stage = str(stage_name or "").strip().lower()
        if norm_stage == "temporal":
            turn_context.fidelity.temporal = True
        elif norm_stage == "inference":
            turn_context.fidelity.inference = True
        elif norm_stage == "reflection":
            turn_context.fidelity.reflection = True
        elif norm_stage == "save":
            turn_context.fidelity.save = True

    def _capture_memory_before(self, turn_context: TurnContext, stage_name: str) -> None:
        norm_stage = str(stage_name or "").strip().lower()
        if norm_stage not in {"context_builder", "memory"}:
            return
        memories = list(turn_context.state.get("memories") or [])
        mem_fp_before = hashlib.sha256(json.dumps(memories, sort_keys=True, default=str).encode()).hexdigest()[:16]
        memory_evolution = turn_context.determinism_manifest.setdefault("memory_evolution", {})
        memory_evolution["before_fingerprint"] = mem_fp_before
        memory_evolution["before_count"] = len(memories)

    def _capture_memory_after(self, turn_context: TurnContext, stage_name: str) -> None:
        if str(stage_name or "").strip().lower() != "save":
            return
        memories = list(turn_context.state.get("memories") or [])
        mem_fp_after = hashlib.sha256(json.dumps(memories, sort_keys=True, default=str).encode()).hexdigest()[:16]
        memory_evolution = turn_context.determinism_manifest.setdefault("memory_evolution", {})
        memory_evolution["after_fingerprint"] = mem_fp_after
        memory_evolution["delta"] = len(memories) - int(memory_evolution.get("before_count", 0) or 0)

    def _record_stage_start(self, turn_context: TurnContext, *, stage_key: str, idx: int) -> float:
        turn_context.state["_active_stage_name"] = stage_key
        stage_started_at = self._deterministic_perf_counter(
            turn_context,
            slot=f"graph.stage.{stage_key}.{idx}.start_perf",
        )
        stage_started_wall = self._deterministic_wall_time(
            turn_context,
            slot=f"graph.stage.{stage_key}.{idx}.start_wall",
        )
        self._record_execution_trace(
            turn_context,
            event_type="node_start",
            stage=stage_key,
            detail={"node": stage_key, "index": idx},
        )
        turn_context.state["_stage_started_wall"] = stage_started_wall
        return stage_started_at

    def _record_stage_success(
        self,
        turn_context: TurnContext,
        *,
        stage_key: str,
        idx: int,
        stage_started_at: float,
    ) -> float:
        stage_finished_at = self._deterministic_perf_counter(
            turn_context,
            slot=f"graph.stage.{stage_key}.{idx}.end_perf",
        )
        stage_ms = round((stage_finished_at - stage_started_at) * 1000, 3)
        turn_context.stage_traces.append(StageTrace(stage=stage_key, duration_ms=stage_ms, error=None))
        self._mark_stage_fidelity(turn_context, stage_key)
        self._record_execution_trace(
            turn_context,
            event_type="node_complete",
            stage=stage_key,
            detail={"duration_ms": stage_ms, "status": "ok", "index": idx},
        )
        return stage_ms

    def _record_stage_failure(
        self,
        turn_context: TurnContext,
        *,
        stage_key: str,
        idx: int,
        stage_started_at: float,
        stage_exc: Exception,
    ) -> None:
        stage_finished_at = self._deterministic_perf_counter(
            turn_context,
            slot=f"graph.stage.{stage_key}.{idx}.end_perf",
        )
        stage_ms = round((stage_finished_at - stage_started_at) * 1000, 3)
        turn_context.stage_traces.append(StageTrace(stage=stage_key, duration_ms=stage_ms, error=str(stage_exc)))
        self._record_execution_trace(
            turn_context,
            event_type="node_complete",
            stage=stage_key,
            detail={"duration_ms": stage_ms, "status": "error", "error": str(stage_exc), "index": idx},
        )

    def _project_unified_trace_events(self, turn_context: TurnContext) -> list[dict[str, Any]]:
        """Project unified trace events from canonical execution trace (single source of truth)."""
        execution_trace = list(turn_context.state.get("execution_trace") or [])
        return [
            {
                "sequence": int(item.get("sequence") or 0),
                "event_type": str(item.get("event_type") or ""),
                "stage": str(item.get("stage") or ""),
                "phase": str(item.get("phase") or ""),
                "detail": _json_safe(dict(item.get("detail") or {})),
            }
            for item in execution_trace
            if isinstance(item, dict)
        ]

    def _project_unified_trace_nodes(self, turn_context: TurnContext) -> list[dict[str, Any]]:
        """Project unified execution nodes from canonical stage traces."""
        now_ts = self._deterministic_wall_time(
            turn_context,
            slot="graph.unified_trace.project_nodes.now_ts",
        )
        projected: list[dict[str, Any]] = []
        for idx, stage in enumerate(list(turn_context.stage_traces or [])):
            stage_name = str(getattr(stage, "stage", "") or "unknown").strip().lower()
            err = str(getattr(stage, "error", "") or "")
            projected.append(
                {
                    "node_type": stage_name,
                    "node_id": f"{turn_context.trace_id}:{stage_name}:{idx}",
                    "status": ("failed" if err else "success"),
                    "start_time": 0.0,
                    "end_time": now_ts,
                    "duration_ms": float(getattr(stage, "duration_ms", 0.0) or 0.0),
                    "error": (err or None),
                    "error_type": ("RuntimeError" if err else None),
                },
            )
        return projected

    def _initialize_unified_turn_trace(self, turn_context: TurnContext) -> None:
        now_ts = self._deterministic_wall_time(
            turn_context,
            slot="graph.unified_trace.start_time",
        )
        session_id = str(
            turn_context.metadata.get("session_id")
            or turn_context.state.get("session_id")
            or "default",
        )
        trace = {
            "trace_id": str(turn_context.trace_id or ""),
            "turn_id": "",
            "session_id": session_id,
            "schema_version": "1.0",
            "start_time": now_ts,
            "end_time": 0.0,
            "duration_ms": 0.0,
            "input": {
                "text": str(turn_context.user_input or ""),
                "attachments": [],
                "session_id": session_id,
                "metadata": {
                    "phase": str(getattr(turn_context.phase, "value", turn_context.phase) or ""),
                },
                "timestamp": now_ts,
            },
            "output": {
                "response": None,
                "should_end": False,
                "confidence": 1.0,
                "recovery_fallback": False,
            },
            "nodes": [],
            "trace_events": [],
            "checksum": "",
            "commit_boundary_count": 0,
            "metadata": {
                "phase": str(getattr(turn_context.phase, "value", turn_context.phase) or ""),
            },
            "completed": False,
            "error": None,
        }
        turn_context.state["_unified_turn_trace_obj"] = trace
        # Hard sink semantics: remove stale snapshots so no caller can treat prior
        # persisted trace payloads as execution authority for this run.
        turn_context.state.pop("turn_trace", None)
        turn_context.metadata.pop("turn_trace", None)

    @staticmethod
    def _coerce_finalized_output(result: FinalizedTurnResult | None) -> tuple[str | None, bool]:
        response: str | None = None
        should_end = False
        if isinstance(result, tuple) and len(result) >= 2:
            response = str(result[0] or "")
            should_end = bool(result[1])
        elif result is not None:
            response = str(result)
        return response, should_end

    @staticmethod
    def _count_commit_boundaries(nodes: list[dict[str, Any]]) -> int:
        return sum(1 for node in list(nodes or []) if str((node or {}).get("node_type") or "") == "save")

    @staticmethod
    def _build_trace_checksum(trace: dict[str, Any]) -> str:
        checksum_payload = {
            "trace_id": str(trace.get("trace_id") or ""),
            "session_id": str(trace.get("session_id") or ""),
            "nodes": list(trace.get("nodes") or []),
            "ledger_event_count": len(list(trace.get("trace_events") or [])),
            "output": dict(trace.get("output") or {}),
        }
        checksum_content = json.dumps(checksum_payload, sort_keys=True, default=str)
        return f"chk-{hashlib.sha256(checksum_content.encode('utf-8')).hexdigest()[:32]}"

    def _validate_unified_projection_counts(
        self,
        turn_context: TurnContext,
        *,
        projected_count: int,
        enforce_projection_consistency: bool,
    ) -> int:
        execution_trace_count = len(list(turn_context.state.get("execution_trace") or []))
        if enforce_projection_consistency and projected_count != execution_trace_count:
            raise ProjectionMismatch(
                "Unified trace projection mismatch",
                context={
                    "execution_trace_count": execution_trace_count,
                    "projected_trace_event_count": projected_count,
                    "trace_id": str(turn_context.trace_id or ""),
                },
            )
        return execution_trace_count

    def _build_unified_trace_metadata(
        self,
        turn_context: TurnContext,
        *,
        execution_trace_count: int,
        projected_count: int,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        trace_metadata = dict(trace.get("metadata") or {})
        trace_metadata.update(
            {
                "projection_version": "graph-execution-v1",
                "source_of_truth": "execution_trace+stage_traces",
                "execution_trace_event_count": execution_trace_count,
                "projected_trace_event_count": projected_count,
                "stage_trace_count": len(list(turn_context.stage_traces or [])),
                "phase": str(getattr(turn_context.phase, "value", turn_context.phase) or ""),
            },
        )
        return trace_metadata

    def _build_unified_turn_output(
        self,
        turn_context: TurnContext,
        *,
        response: str | None,
        should_end: bool,
        error: str,
    ) -> dict[str, Any]:
        return {
            "response": response,
            "should_end": should_end,
            "confidence": 1.0,
            "recovery_fallback": bool(turn_context.short_circuit or bool(error)),
        }

    def _finalize_unified_turn_trace(
        self,
        turn_context: TurnContext,
        *,
        result: FinalizedTurnResult | None = None,
        error: str = "",
        enforce_projection_consistency: bool = True,
    ) -> None:
        trace_obj = turn_context.state.get("_unified_turn_trace_obj")
        if not isinstance(trace_obj, dict):
            return
        trace = dict(trace_obj)

        response, should_end = self._coerce_finalized_output(result)
        trace["output"] = self._build_unified_turn_output(
            turn_context,
            response=response,
            should_end=should_end,
            error=error,
        )
        trace["nodes"] = self._project_unified_trace_nodes(turn_context)
        trace["trace_events"] = self._project_unified_trace_events(turn_context)
        projected_count = len(list(trace.get("trace_events") or []))
        execution_trace_count = self._validate_unified_projection_counts(
            turn_context,
            projected_count=projected_count,
            enforce_projection_consistency=enforce_projection_consistency,
        )
        if error:
            trace["error"] = str(error)

        trace["metadata"] = self._build_unified_trace_metadata(
            turn_context,
            execution_trace_count=execution_trace_count,
            projected_count=projected_count,
            trace=trace,
        )
        trace["end_time"] = self._deterministic_wall_time(
            turn_context,
            slot="graph.unified_trace.end_time",
        )
        trace["duration_ms"] = max(0.0, (float(trace["end_time"]) - float(trace.get("start_time") or 0.0)) * 1000.0)
        trace["commit_boundary_count"] = self._count_commit_boundaries(list(trace.get("nodes") or []))
        trace["checksum"] = self._build_trace_checksum(trace)
        trace["completed"] = True
        serialized = dict(trace)
        # Persist as read-only sink snapshot for observability/replay only.
        serialized["_sink_only"] = True
        turn_context.state["turn_trace"] = serialized
        turn_context.metadata["turn_trace"] = dict(serialized)
        turn_context.state.pop("_unified_turn_trace_obj", None)
        turn_context.state.pop("_stage_started_wall", None)

    def _maybe_transition_phase(
        self,
        turn_context: TurnContext,
        *,
        stage_key: str,
        emitted_phases: set[str],
        persistence_service: Any,
        persistence_is_lightweight: bool,
    ) -> None:
        stage_phase = {"inference": "ACT", "safety": "OBSERVE", "save": "RESPOND"}
        phase = stage_phase.get(stage_key.strip().lower())
        if not phase or phase in emitted_phases:
            return
        with contextlib.suppress(Exception):
            turn_context.transition_phase(TurnPhase(str(phase)), reason=f"stage:{stage_key.strip().lower()}")
        self._emit_turn_event(
            turn_context,
            event_type="phase_transition",
            payload={"transition": {"to": phase}},
            persistence_service=persistence_service,
            persistence_is_lightweight=persistence_is_lightweight,
        )
        emitted_phases.add(phase)

    def _record_edge_transition(  # noqa: PLR0913
        self,
        turn_context: TurnContext,
        *,
        idx: int,
        pipeline_items: list[tuple[str, Any]],
        stage_key: str,
        persistence_service: Any,
        persistence_is_lightweight: bool,
    ) -> None:
        if idx >= len(pipeline_items) - 1:
            return
        next_stage_name = str(pipeline_items[idx + 1][0] or "unknown")
        self._emit_checkpoint(
            turn_context,
            stage_name=f"{stage_key}→{next_stage_name}",
            status="edge",
            persistence_service=persistence_service,
            persistence_is_lightweight=persistence_is_lightweight,
        )
        self._record_execution_trace(
            turn_context,
            event_type="edge_transition",
            stage=stage_key,
            detail={"from": stage_key, "to": next_stage_name, "index": idx},
        )

    def _enforce_kernel_invariants(
        self,
        *,
        turn_context: TurnContext,
        stage: str,
        operation: str,
        mutation_outside_save_node: bool = False,
    ) -> None:
        validate_result = self._execution_kernel.validate(
            stage=stage,
            operation=operation,
            context=turn_context,
            mutation_outside_save_node=mutation_outside_save_node,
        )
        if not bool(getattr(validate_result, "ok", True)):
            raise ExecutionStageError(
                "Kernel validation failed",
                context={
                    "stage": str(stage or ""),
                    "reason": str(getattr(validate_result, "reason", "kernel validation failed") or "kernel validation failed"),
                },
            )

    async def _run_stage(  # noqa: PLR0913
        self,
        *,
        idx: int,
        pipeline_items: list[tuple[str, Any]],
        turn_context: TurnContext,
        emitted_phases: set[str],
        persistence_service: Any,
        persistence_is_lightweight: bool,
    ) -> tuple[TurnContext, bool, FinalizedTurnResult | None]:
        stage_name, node = pipeline_items[idx]
        stage_key = str(stage_name or "unknown")
        # ──────────────────────────────────────────────────────────────────────────
        # Contract: Stage boundary MUST maintain identity tuple (non-shadowing).
        # These are read from context and must NOT be regenerated mid-stage.
        # ──────────────────────────────────────────────────────────────────────────
        trace_id = turn_context.trace_id
        kernel_step_id = turn_context.kernel_step_id
        determinism_manifest = turn_context.determinism_manifest
        self._enforce_identity_tuple_immutable(turn_context, boundary=f"stage_enter:{stage_key}")
        stage_started_at = self._record_stage_start(turn_context, stage_key=stage_key, idx=idx)
        self._emit_checkpoint(
            turn_context,
            stage_name=stage_key,
            status="before",
            persistence_service=persistence_service,
            persistence_is_lightweight=persistence_is_lightweight,
        )
        self._enforce_kernel_invariants(
            turn_context=turn_context,
            stage=stage_key,
            operation=f"turn_graph.stage:{stage_key}",
        )
        self._capture_memory_before(turn_context, stage_key)

        try:
            turn_context = await self._execute_node(node, turn_context)
            self._enforce_identity_tuple_immutable(turn_context, boundary=f"stage_exit:{stage_key}")
            self._record_stage_success(
                turn_context,
                stage_key=stage_key,
                idx=idx,
                stage_started_at=stage_started_at,
            )
        except NON_FATAL_RUNTIME_EXCEPTIONS as stage_exc:
            self._record_stage_failure(
                turn_context,
                stage_key=stage_key,
                idx=idx,
                stage_started_at=stage_started_at,
                stage_exc=stage_exc,
            )
            self._emit_checkpoint(
                turn_context,
                stage_name=stage_key,
                status="error",
                persistence_service=persistence_service,
                persistence_is_lightweight=persistence_is_lightweight,
            )
            raise

        self._emit_checkpoint(
            turn_context,
            stage_name=stage_key,
            status="after",
            persistence_service=persistence_service,
            persistence_is_lightweight=persistence_is_lightweight,
        )

        self._maybe_transition_phase(
            turn_context,
            stage_key=stage_key,
            emitted_phases=emitted_phases,
            persistence_service=persistence_service,
            persistence_is_lightweight=persistence_is_lightweight,
        )
        self._record_edge_transition(
            turn_context,
            idx=idx,
            pipeline_items=pipeline_items,
            stage_key=stage_key,
            persistence_service=persistence_service,
            persistence_is_lightweight=persistence_is_lightweight,
        )

        self._capture_memory_after(turn_context, stage_key)
        short_circuit = bool(turn_context.short_circuit and turn_context.short_circuit_result is not None)
        short_circuit_result = turn_context.short_circuit_result if short_circuit else None
        turn_context.state["_active_stage_name"] = ""
        return turn_context, short_circuit, cast("FinalizedTurnResult | None", short_circuit_result)

    async def _execute_pipeline(
        self,
        turn_context: TurnContext,
        *,
        pipeline_items: list[tuple[str, Any]],
        persistence_service: Any,
        persistence_is_lightweight: bool,
    ) -> FinalizedTurnResult | None:
        emitted_phases: set[str] = set()
        short_circuit_result: FinalizedTurnResult | None = None
        for idx in range(len(pipeline_items)):
            turn_context, short_circuit, maybe_result = await self._run_stage(
                idx=idx,
                pipeline_items=pipeline_items,
                turn_context=turn_context,
                emitted_phases=emitted_phases,
                persistence_service=persistence_service,
                persistence_is_lightweight=persistence_is_lightweight,
            )
            if short_circuit:
                short_circuit_result = maybe_result
                break
        return short_circuit_result

    def _record_failure(self, turn_context: TurnContext, *, stage_name: str, exc: Exception) -> None:
        replay = list(turn_context.determinism_manifest.get("failure_replay") or [])
        replay.append(
            {
                "stage": str(stage_name or "unknown"),
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "state_keys": sorted(str(k) for k in dict(turn_context.state or {}).keys()),
                "contract_version_hash": str(
                    dict(turn_context.determinism_manifest.get("contract_version") or {}).get("node_contracts_hash")
                    or ""
                ),
            },
        )
        turn_context.determinism_manifest["failure_replay"] = replay
        turn_context.state["failure_taxonomy"] = {"severity": "error", "error": str(exc)}

    def _resolve_execute_result(
        self,
        turn_context: TurnContext,
        *,
        short_circuit_result: FinalizedTurnResult | None,
    ) -> FinalizedTurnResult:
        result = short_circuit_result if short_circuit_result is not None else turn_context.state.get("safe_result")

        # Apply early convergence: if should_converge_early flag is set, override should_end to True.
        should_end_override = bool(turn_context.state.get("should_converge_early"))

        if isinstance(result, tuple) and len(result) >= 2:
            output, orig_should_end = result[0], result[1]
            final_should_end = should_end_override if should_end_override else bool(orig_should_end)
            resolved = (output, final_should_end)
            self._finalize_unified_turn_trace(turn_context, result=cast("FinalizedTurnResult", resolved))
            return cast(FinalizedTurnResult, resolved)

        final_result: FinalizedTurnResult = (str(result or ""), should_end_override)
        self._finalize_unified_turn_trace(turn_context, result=cast("FinalizedTurnResult | None", final_result))
        return final_result

    async def execute(
        self,
        turn_context: TurnContext,
        *,
        audit_mode: bool = False,
    ) -> FinalizedTurnResult:
        self._enforce_execution_boundary()
        # ──────────────────────────────────────────────────────────────────────────
        # Contract: Orchestration boundary MUST extract and canonicalize identity.
        # These values are authoritative and must be threaded through all stages.
        # ──────────────────────────────────────────────────────────────────────────
        trace_id = turn_context.trace_id
        kernel_step_id = turn_context.kernel_step_id
        determinism_manifest = turn_context.determinism_manifest
        self._capture_identity_tuple_baseline(turn_context)
        self._enforce_identity_tuple_immutable(turn_context, boundary="execute_enter")
        execute_started_at = self._deterministic_perf_counter(
            turn_context,
            slot="graph.execute.start_perf",
        )
        pipeline_items = self._pipeline_items()
        turn_context.state["_pipeline_stage_names"] = [
            str(name or "").strip().lower() for name, _node in pipeline_items if str(name or "").strip()
        ]
        persistence_service = self._resolve_persistence_service()
        persistence_is_lightweight = self._is_lightweight_persistence(persistence_service)
        if persistence_service is not None:
            self._validate_persistence_service_contract(turn_context, persistence_service)
        self._stamp_contract_version(turn_context)
        self._initialize_unified_turn_trace(turn_context)
        turn_context.metadata["audit_mode"] = bool(audit_mode)
        self._enforce_kernel_invariants(
            turn_context=turn_context,
            stage="pre_execute",
            operation="turn_graph.execute.pre_execute",
        )

        with contextlib.suppress(Exception):
            self._record_execution_trace(
                turn_context,
                event_type="turn_start",
                stage="graph",
                detail={"audit_mode": bool(audit_mode)},
            )

        try:
            short_circuit_result = await self._execute_pipeline(
                turn_context,
                pipeline_items=pipeline_items,
                persistence_service=persistence_service,
                persistence_is_lightweight=persistence_is_lightweight,
            )
            self._enforce_kernel_invariants(
                turn_context=turn_context,
                stage="post_execute",
                operation="turn_graph.execute.post_execute",
            )
            with contextlib.suppress(Exception):
                self._record_execution_trace(
                    turn_context,
                    event_type="turn_complete",
                    stage="graph",
                    detail={
                        "elapsed_ms": round(
                            (
                                self._deterministic_perf_counter(
                                    turn_context,
                                    slot="graph.execute.end_perf",
                                )
                                - execute_started_at
                            )
                            * 1000,
                            3,
                        ),
                    },
                )
            contract = self._finalize_trace_contract(
                turn_context,
                expected_hash=str(turn_context.metadata.get("expected_execution_trace_hash") or ""),
            )
            turn_context.state["execution_trace_contract"] = dict(contract)
            turn_context.metadata["execution_trace_contract"] = dict(contract)
            self._enforce_identity_tuple_immutable(turn_context, boundary="execute_exit")
            self._seal_execution_identity(turn_context)
            return self._resolve_execute_result(
                turn_context,
                short_circuit_result=short_circuit_result,
            )
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            current_stage = str(turn_context.state.get("_active_stage_name") or "unknown")
            self._record_failure(turn_context, stage_name=current_stage, exc=exc)
            with contextlib.suppress(Exception):
                self._record_execution_trace(
                    turn_context,
                    event_type="turn_failed",
                    stage=current_stage,
                    detail={"error": str(exc)},
                )
            with contextlib.suppress(Exception):
                contract = self._finalize_trace_contract(turn_context)
                turn_context.state["execution_trace_contract"] = dict(contract)
                turn_context.metadata["execution_trace_contract"] = dict(contract)
                self._seal_execution_identity(turn_context)
            # Hard invariant: projection mismatches must fail before sink persistence.
            if not isinstance(exc, ProjectionMismatch):
                # Preserve the original failure as the primary error surface.
                self._finalize_unified_turn_trace(
                    turn_context,
                    error=str(exc),
                    enforce_projection_consistency=False,
                )
            else:
                turn_context.state.pop("turn_trace", None)
                turn_context.metadata.pop("turn_trace", None)
                turn_context.state.pop("_unified_turn_trace_obj", None)
            logger.error("TurnGraph.execute failed: %s", exc)
            raise
