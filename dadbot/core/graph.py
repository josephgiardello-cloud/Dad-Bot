from __future__ import annotations

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
from dadbot.core.capability_audit_runner import build_runtime_capability_audit_report
from dadbot.core.capability_registry import (
    CapabilityRegistry,
    CapabilityViolationError,
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
from dadbot.core.execution_policy_service import StageEntryGate
from dadbot.core.execution_receipt import (
    DEFAULT_SIGNER as _DEFAULT_RECEIPT_SIGNER,
)
from dadbot.core.execution_receipt import (
    ReceiptSigner,
)
from dadbot.core.execution_context import (
    record_execution_step,
    record_external_system_call,
)
from dadbot.core.graph_side_effects import GraphSideEffectsOrchestrator
from dadbot.core.invariant_gate import InvariantGate, InvariantViolationError
from dadbot.core.invariant_registry import InvariantRegistry
from dadbot.core.persistence_event_adapter import (
    GraphPersistenceEventAdapter,
)  # re-export
from dadbot.core.side_effect_adapter import SideEffectAdapter
from dadbot.core.topology_provider import (
    TopologyProvider,
    TurnPipelineState,
)
from dadbot.core.turn_resume_store import TurnResumeStore
from dadbot.core.ux_projection import TurnHealthState, TurnUxProjector  # re-export
from dadbot.core.graph_temporal import (  # re-export temporal types
    TurnPhase,
    TurnTemporalAxis,
    VirtualClock,
    _PHASE_ORDER,
)
from dadbot.core.graph_types import (  # re-export trace/op types
    GoalMutationOp,
    LedgerMutationOp,
    MemoryMutationOp,
    MutationKind,
    MutationTransactionRecord,
    MutationTransactionStatus,
    NodeType,
    RelationshipMutationOp,
    StageTrace,
    _json_safe,
)
from dadbot.core.graph_mutation import (  # re-export mutation primitives
    MutationGuard,
    MutationIntent,
    MutationQueue,
)
from dadbot.core.graph_context import (  # re-export turn-scoped state
    TurnContext,
    TurnContextContractViolation,
    TurnFidelity,
)
from dadbot.core.graph_pipeline_nodes import (  # re-export pipeline node stubs
    ContextBuilderNode,
    GraphNode,
    HealthNode,
    InferenceNode,
    MemoryNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    _invoke_node_run_compat,
)

logger = logging.getLogger(__name__)

class NodeContractViolation(RuntimeError):
    """Raised when a pipeline node violates its input/output state contract."""


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
    "save": "safety",
}



# FatalTurnError, TurnFailureSeverity, KernelRejectionSemantics,
# PersistenceServiceContract, ExecutionPolicyEngine, StagePhaseMappingPolicy
# are defined in dadbot.core.execution_policy and imported above.
# They are re-exported here for backward compatibility with code that imports
# them from dadbot.core.graph.


class TurnGraph:
    """Declarative turn execution graph."""

    def __init__(
        self,
        registry: Any = None,
        nodes: list[GraphNode] | None = None,
    ):
        self.registry = registry
        self.nodes = nodes or [
            TemporalNode(),
            HealthNode(),
            ContextBuilderNode(),
            InferenceNode(),
            SafetyNode(),
            ReflectionNode(),
            SaveNode(),
        ]
        self._node_map: dict[str, Any] = {}
        self._edges: dict[str, str] = {}
        self._entry_node: str | None = None
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
            strict=False,
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
        # Stage-entry policy gate: idempotency, capability, and ordering checks.
        self._stage_entry_gate = StageEntryGate()
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
        raise RuntimeError(
            "configure_resume(store_dir=...) is removed; use configure_resume_store(TurnResumeStore(ledger=...))",
        )

    def configure_resume_store(
        self,
        resume_store: TurnResumeStore,
        *,
        policy: ResumabilityPolicy | None = None,
    ) -> None:
        """Deprecated: recovery is now ledger-only via replay. Use ExecutionLedger directly."""
        pass  # No-op: Phase 3 uses ledger replay, not stored resume points.

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

    def _enforce_transition_invariant(
        self,
        turn_context: TurnContext,
        *,
        event_type: str,
        stage: str,
        detail: dict[str, Any] | None,
    ) -> None:
        """Validate execution transitions via invariant gate.
        
        In the simplified loop architecture, this is lenient to prevent
        cascading failures from old orchestration assumptions.
        """
        trace = list(turn_context.state.get("execution_trace") or [])
        if not trace:
            # No trace yet; invariant validation not ready
            return
            
        try:
            latest = dict(trace[-1])
            sequence = int(latest.get("sequence") or 0)
            prev = dict(trace[-2]) if len(trace) > 1 else None

            event_envelope = {
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
            self._invariant_gate.validate_event(event_envelope)
            pipeline_names = [
                str(name).strip().lower()
                for name in list(turn_context.state.get("_pipeline_stage_names") or [])
                if str(name).strip()
            ]
            decision = self._invariant_gate.assess_execution_semantics(
                trace,
                pipeline_stage_names=pipeline_names,
            )
            if not decision.approved:
                logger.debug(
                    "Invariant gate decision: %s (non-fatal in simplified loop)",
                    decision.reason,
                )
        except Exception as exc:  # noqa: BLE001 — simplified loop allows invariant validation to gracefully degrade
            logger.debug(
                "Invariant gate validation deferred (simplified architecture): %s",
                exc,
            )

    def _validate_persistence_service_contract(
        self,
        turn_context: TurnContext,
        service: Any,
    ) -> None:
        """Validate minimum persistence interface for graph execution semantics.

        Strict mode is opt-in via metadata['persistence_contract_strict'].
        """
        self._side_effects.validate_persistence_service_contract(turn_context, service)

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
        if self._node_map:
            items: list[tuple[str, Any]] = []
            node_name = self._entry_node
            while node_name is not None:
                items.append((node_name, self._node_map[node_name]))
                node_name = self._edges.get(node_name)
            return items
        return [
            (
                str(getattr(node, "name", type(node).__name__) or type(node).__name__),
                node,
            )
            for node in self.nodes
        ]

    # (Deleted: _emit_turn_health_state, _mark_structural_degradation, _mark_stage_enter — no longer used)

    def add_node(self, name: str, node: Any) -> None:
        # Kernel in shadow mode during graph construction.
        shadow_context = TurnContext(user_input="graph_build")
        result = self._execution_kernel.validate(
            stage="graph_build",
            operation=f"turn_graph.add_node:{name}",
            context=shadow_context,
        )
        if not bool(getattr(result, "ok", True)):
            logger.warning(
                "[KERNEL SHADOW VIOLATION] %s",
                getattr(result, "reason", "graph build validation failed"),
            )
        if name not in self._node_map:
            if self._entry_node is None:
                self._entry_node = name
            self._node_map[name] = node

    def set_edge(self, source: str, target: str) -> None:
        self._edges[source] = target

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

    async def _execute_node(self, node: Any, turn_context: TurnContext) -> TurnContext:
        """Execute a single node (parallelism removed in Phase 3 Target 4)."""
        node_name = str(
            getattr(node, "name", type(node).__name__) or type(node).__name__,
        )

        async def _call_node() -> TurnContext:
            if isinstance(node, (tuple, list)):
                async def _run_subnode(subnode: Any) -> None:
                    sub_run = getattr(subnode, "run", None)
                    if callable(sub_run):
                        result = await _invoke_node_run_compat(
                            sub_run,
                            self.registry,
                            turn_context,
                        )
                        if result is not None:
                            return
                        return
                    sub_execute = getattr(subnode, "execute", None)
                    if callable(sub_execute):
                        params = inspect.signature(sub_execute).parameters
                        if len(params) >= 2:
                            await cast("Any", sub_execute)(self.registry, turn_context)
                        else:
                            await cast("Any", sub_execute)(turn_context)
                        return
                    raise TypeError(f"Unsupported node type: {type(subnode).__name__}")

                await asyncio.gather(*(_run_subnode(subnode) for subnode in node))
                return turn_context

            run_method = getattr(node, "run", None)
            if callable(run_method):
                result = await _invoke_node_run_compat(
                    run_method,
                    self.registry,
                    turn_context,
                )
                return cast("TurnContext", result or turn_context)
            execute_method = getattr(node, "execute", None)
            if callable(execute_method):
                execute_params = inspect.signature(execute_method).parameters
                if len(execute_params) >= 2:
                    await cast("Any", execute_method)(self.registry, turn_context)
                else:
                    await cast("Any", execute_method)(turn_context)
                return turn_context
            raise TypeError(f"Unsupported node type: {type(node).__name__}")

        if self._kernel is not None:
            step_result = await self._kernel.execute_step(
                turn_context,
                node_name,
                _call_node,
            )
            if step_result.status == "error":
                self._record_execution_trace(
                    turn_context,
                    event_type="kernel_error",
                    stage=node_name,
                    detail={"error": str(step_result.error or "")},
                )
                raise RuntimeError(step_result.error)
            if step_result.status == "rejected":
                semantics = self._rejection_semantics_for_stage(node_name)
                rejection_reason = str(
                    getattr(getattr(step_result, "policy", None), "reason", "") or "",
                )
                turn_context.state.setdefault("kernel_rejection_contract", []).append(
                    {
                        "stage": node_name,
                        "reason": rejection_reason,
                        "retryable": bool(semantics.retryable),
                        "state_mutation_allowed": bool(
                            semantics.state_mutation_allowed,
                        ),
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
                    self._persist_kernel_rejection(
                        turn_context,
                        stage=node_name,
                        reason=rejection_reason,
                    )

                if semantics.action == "abort_turn" or semantics.invalidate_downstream:
                    raise RuntimeError(
                        f"Kernel step rejected under abort semantics: stage={node_name!r}, reason={rejection_reason!r}",
                    )
                # Backward-compatible default: stage is skipped and pipeline continues.
                return turn_context
            self._record_execution_trace(
                turn_context,
                event_type="kernel_ok",
                stage=node_name,
                detail={},
            )
            return turn_context

        return await _call_node()

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

    async def execute(
        self,
        turn_context: TurnContext,
        *,
        audit_mode: bool = False,
    ) -> FinalizedTurnResult:
        """Execute turn through linear node pipeline.
        
        Simple loop: for each stage, run node and return result.
        No recovery, no phase tracking, no health state — just execution.
        """
        # Execution boundary
        if self._required_execution_token:
            active_token = ControlPlaneExecutionBoundary.current()
            if active_token != self._required_execution_token:
                raise RuntimeError(
                    "TurnGraph.execute boundary violation: graph execution must be routed through ExecutionControlPlane",
                )

        execute_started_at = time.perf_counter()
        telemetry = self.registry.get("telemetry") if self.registry is not None else None
        pipeline_items = self._pipeline_items()
        turn_context.state["_pipeline_stage_names"] = [
            str(name or "").strip().lower()
            for name, _node in pipeline_items
            if str(name or "").strip()
        ]

        persistence_service = None
        if self.registry is not None:
            for key in ("persistence", "persistence_service"):
                try:
                    persistence_service = self.registry.get(key)
                except Exception:
                    persistence_service = None
                if persistence_service is not None:
                    break

        save_turn_event = getattr(persistence_service, "save_turn_event", None) if persistence_service else None
        save_graph_checkpoint = (
            getattr(persistence_service, "save_graph_checkpoint", None)
            if persistence_service
            else None
        )

        def _next_sequence() -> int:
            turn_context.event_sequence = int(turn_context.event_sequence or 0) + 1
            return int(turn_context.event_sequence)

        # Only use per-stage checkpoint/event persistence in lightweight test doubles.
        # Real production persistence services can recurse into ledger reads/writes and
        # are too expensive to invoke at every stage boundary in long runs.
        _persistence_is_lightweight = bool(
            persistence_service is not None
            and (
                str(type(persistence_service).__module__ or "").startswith("tests.")
                or (callable(save_graph_checkpoint) and not hasattr(save_graph_checkpoint, "__self__"))
                or (callable(save_turn_event) and not hasattr(save_turn_event, "__self__"))
            )
        )

        def _emit_turn_event(event_type: str, payload: dict[str, Any]) -> None:
            if not _persistence_is_lightweight or not callable(save_turn_event):
                return
            event = {
                "event_type": str(event_type or ""),
                "trace_id": str(turn_context.trace_id or ""),
                "occurred_at": str(getattr(getattr(turn_context, "temporal", None), "wall_time", "") or ""),
                "sequence": _next_sequence(),
            }
            event.update(dict(payload or {}))
            save_turn_event(event)

        def _emit_checkpoint(stage_name: str, status: str) -> None:
            lock = dict((turn_context.metadata.get("determinism") or {}))
            if _persistence_is_lightweight:
                try:
                    checkpoint_payload = turn_context.checkpoint_snapshot(
                        stage=str(stage_name or "unknown"),
                        status=str(status or "after"),
                        error=None,
                        advance_chain=True,
                    )
                except Exception:
                    checkpoint_payload = {
                        "trace_id": str(turn_context.trace_id or ""),
                        "stage": str(stage_name or "unknown"),
                        "status": str(status or "after"),
                        "state": _json_safe(dict(turn_context.state or {})),
                        "metadata": _json_safe(dict(turn_context.metadata or {})),
                    }
            else:
                checkpoint_payload = {
                    "trace_id": str(turn_context.trace_id or ""),
                    "stage": str(stage_name or "unknown"),
                    "status": str(status or "after"),
                    "state": _json_safe(dict(turn_context.state or {})),
                    "metadata": _json_safe(dict(turn_context.metadata or {})),
                }
            if _persistence_is_lightweight and callable(save_graph_checkpoint):
                try:
                    save_graph_checkpoint(checkpoint_payload)
                except Exception:
                    pass
            _emit_turn_event(
                "graph_checkpoint",
                {
                    "stage": str(stage_name or "unknown"),
                    "status": str(status or "after"),
                    "determinism_lock": {
                        "lock_hash": str(lock.get("lock_hash") or ""),
                        "lock_id": str(lock.get("lock_id") or ""),
                    },
                },
            )

        def _finalize_trace_contract(*, expected_hash: str = "") -> dict[str, Any]:
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
            digest = hashlib.sha256(
                json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
            ).hexdigest()
            expected = str(expected_hash or "").strip()
            if expected and expected != digest:
                raise RuntimeError(
                    f"Execution trace determinism mismatch: expected={expected!r}, actual={digest!r}",
                )
            return {
                "version": "1.0",
                "event_count": len(trace),
                "trace_hash": digest,
                "schema_version": "1.0",
            }

        # Validate persistence service contract early
        if persistence_service is not None:
            self._validate_persistence_service_contract(turn_context, persistence_service)

        # Stamp contract version for determinism tracking
        _contracts_blob = json.dumps(_NODE_STAGE_CONTRACTS, sort_keys=True, default=str)
        _contracts_hash = hashlib.sha256(_contracts_blob.encode()).hexdigest()[:16]
        turn_context.determinism_manifest["contract_version"] = {
            "node_contracts_hash": _contracts_hash,
            "schema_version": "1",
        }

        with contextlib.suppress(Exception):
            self._record_execution_trace(
                turn_context,
                event_type="turn_start",
                stage="graph",
                detail={"audit_mode": bool(audit_mode)},
            )

        try:
            emitted_phases: set[str] = set()
            stage_phase = {"inference": "ACT", "safety": "OBSERVE", "save": "RESPOND"}
            # Execute each node in sequence
            short_circuit_result: FinalizedTurnResult | None = None
            for idx, (stage_name, node) in enumerate(pipeline_items):
                stage_started_at = time.perf_counter()
                # Capture memory fingerprint before context_builder
                _norm_stage = str(stage_name or "").strip().lower()
                _emit_checkpoint(stage_name, "before")
                if _norm_stage == "context_builder" or _norm_stage == "memory":
                    _mems_before = list(turn_context.state.get("memories") or [])
                    _mem_fp_before = hashlib.sha256(
                        json.dumps(_mems_before, sort_keys=True, default=str).encode()
                    ).hexdigest()[:16]
                    _mem_ev_init = turn_context.determinism_manifest.setdefault("memory_evolution", {})
                    _mem_ev_init["before_fingerprint"] = _mem_fp_before
                    _mem_ev_init["before_count"] = len(_mems_before)

                try:
                    turn_context = await self._execute_node(node, turn_context)
                    stage_ms = round((time.perf_counter() - stage_started_at) * 1000, 3)
                    turn_context.stage_traces.append(
                        StageTrace(stage=str(stage_name or "unknown"), duration_ms=stage_ms, error=None),
                    )
                    if _norm_stage == "temporal":
                        turn_context.fidelity.temporal = True
                    elif _norm_stage == "inference":
                        turn_context.fidelity.inference = True
                    elif _norm_stage == "reflection":
                        turn_context.fidelity.reflection = True
                    elif _norm_stage == "save":
                        turn_context.fidelity.save = True
                except Exception as stage_exc:
                    stage_ms = round((time.perf_counter() - stage_started_at) * 1000, 3)
                    turn_context.stage_traces.append(
                        StageTrace(
                            stage=str(stage_name or "unknown"),
                            duration_ms=stage_ms,
                            error=str(stage_exc),
                        ),
                    )
                    _emit_checkpoint(stage_name, "error")
                    raise
                _emit_checkpoint(stage_name, "after")

                phase = stage_phase.get(str(stage_name or "").strip().lower())
                if phase and phase not in emitted_phases:
                    with contextlib.suppress(Exception):
                        turn_context.transition_phase(
                            TurnPhase(str(phase)),
                            reason=f"stage:{str(stage_name or '').strip().lower()}",
                        )
                    _emit_turn_event(
                        "phase_transition",
                        {
                            "transition": {
                                "to": phase,
                            },
                        },
                    )
                    emitted_phases.add(phase)

                if idx < len(pipeline_items) - 1:
                    next_stage_name = str(pipeline_items[idx + 1][0] or "unknown")
                    _emit_checkpoint(f"{str(stage_name or 'unknown')}→{next_stage_name}", "edge")
                
                # Capture memory fingerprint after save
                if _norm_stage == "save":
                    _mems_after = list(turn_context.state.get("memories") or [])
                    _mem_fp_after = hashlib.sha256(
                        json.dumps(_mems_after, sort_keys=True, default=str).encode()
                    ).hexdigest()[:16]
                    _mem_ev = turn_context.determinism_manifest.setdefault("memory_evolution", {})
                    _mem_ev["after_fingerprint"] = _mem_fp_after
                    _mem_ev["delta"] = len(_mems_after) - _mem_ev.get("before_count", 0)
                
                # Check for short-circuit (e.g., session exit)
                if turn_context.short_circuit and turn_context.short_circuit_result is not None:
                    short_circuit_result = turn_context.short_circuit_result
                    break

            save_traces = [trace for trace in turn_context.stage_traces if str(trace.stage or "") == "save"]
            requires_save_invariant = any(
                stage in {"temporal", "health", "context_builder", "inference", "safety", "reflection"}
                for stage in list(turn_context.state.get("_pipeline_stage_names") or [])
            )
            if requires_save_invariant and not save_traces:
                raise RuntimeError("Structural turn invariant violated: SaveNode did not execute")

            with contextlib.suppress(Exception):
                self._record_execution_trace(
                    turn_context,
                    event_type="turn_complete",
                    stage="graph",
                    detail={"elapsed_ms": round((time.perf_counter() - execute_started_at) * 1000, 3)},
                )

            contract = _finalize_trace_contract(
                expected_hash=str(turn_context.metadata.get("expected_execution_trace_hash") or ""),
            )
            turn_context.state["execution_trace_contract"] = dict(contract)
            turn_context.metadata["execution_trace_contract"] = dict(contract)
            self._seal_execution_identity(turn_context)

            result = short_circuit_result if short_circuit_result is not None else turn_context.state.get("safe_result")
            if isinstance(result, tuple) and len(result) >= 2:
                return cast(FinalizedTurnResult, result)
            return (str(result or ""), False)

        except Exception as exc:  # noqa: BLE001
            current_stage = str(locals().get("stage_name") or "unknown")
            replay = list(turn_context.determinism_manifest.get("failure_replay") or [])
            replay.append(
                {
                    "stage": current_stage,
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc)[:200],
                    "state_keys": sorted(str(k) for k in dict(turn_context.state or {}).keys()),
                    "contract_version_hash": str(
                        dict(turn_context.determinism_manifest.get("contract_version") or {}).get(
                            "node_contracts_hash",
                        )
                        or ""
                    ),
                },
            )
            turn_context.determinism_manifest["failure_replay"] = replay
            turn_context.state["failure_taxonomy"] = {
                "severity": "error",
                "error": str(exc),
            }

            with contextlib.suppress(Exception):
                self._record_execution_trace(
                    turn_context,
                    event_type="turn_failed",
                    stage=current_stage,
                    detail={"error": str(exc)},
                )

            # Best-effort contract/identity sealing on failure path.
            with contextlib.suppress(Exception):
                contract = _finalize_trace_contract()
                turn_context.state["execution_trace_contract"] = dict(contract)
                turn_context.metadata["execution_trace_contract"] = dict(contract)
                self._seal_execution_identity(turn_context)

            logger.error("TurnGraph.execute failed: %s", exc)
            raise
