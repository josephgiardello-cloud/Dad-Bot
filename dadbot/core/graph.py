from __future__ import annotations

import asyncio
import contextlib
import hashlib
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
from dadbot.core.execution_recovery import ExecutionRecovery
from dadbot.core.execution_schema import stamp_trace_contract_version
from dadbot.core.execution_context import (
    record_execution_step,
    record_external_system_call,
)
from dadbot.core.graph_side_effects import GraphSideEffectsOrchestrator
from dadbot.core.invariant_registry import InvariantRegistry
from dadbot.core.persistence_event_adapter import (
    GraphPersistenceEventAdapter,
)  # re-export
from dadbot.core.side_effect_adapter import SideEffectAdapter
from dadbot.core.topology_provider import (
    TopologyProvider,
    TopologyProviderFactory,
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

_DECLARED_TRACE_EVENT_TYPES = frozenset(
    {
        "turn_start",
        "turn_failed",
        "turn_short_circuit",
        "turn_succeeded",
        "stage_enter",
        "stage_skip",
        "stage_done",
        "stage_error",
        "parallel_start",
        "parallel_done",
        "kernel_error",
        "kernel_rejected",
        "kernel_ok",
    },
)


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


class _LinearTopologyGraph:
    """Minimal interface-only topology runtime used as a local fallback.

    This keeps TurnGraph decoupled from concrete topology engines and
    preserves direct TurnGraph() test usage when no provider is injected.
    """

    def __init__(
        self,
        pipeline_items: list[tuple[str, Any]],
        node_executor: Callable[[str, Any, TurnPipelineState], Any],
    ) -> None:
        self._pipeline_items = pipeline_items
        self._node_executor = node_executor

    async def ainvoke(self, state: TurnPipelineState) -> TurnPipelineState:
        current_state: TurnPipelineState = cast("TurnPipelineState", dict(state)); context = current_state.get("context"); _ = (context.trace_id, context.kernel_step_id, context.determinism_manifest) if context is not None else None
        for stage_name, node in self._pipeline_items:
            updates = await self._node_executor(stage_name, node, current_state)
            if isinstance(updates, dict):
                current_state.update(updates)
            if current_state.get("abort") or current_state.get("short_circuit"):
                break
        return current_state  # type: ignore[return-value]


class _LinearTopologyProvider(TopologyProvider):
    def __init__(
        self,
        pipeline_items: list[tuple[str, Any]],
        node_executor: Callable[[str, Any, TurnPipelineState], Any],
    ) -> None:
        self._pipeline_items = pipeline_items
        self._node_executor = node_executor

    def build(self) -> _LinearTopologyGraph:
        return _LinearTopologyGraph(self._pipeline_items, self._node_executor)


def _default_topology_provider_factory(
    pipeline_items: list[tuple[str, Any]],
    node_executor: Callable[[str, Any, TurnPipelineState], Any],
) -> TopologyProvider:
    return _LinearTopologyProvider(pipeline_items, node_executor)


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
        *,
        topology_provider_factory: TopologyProviderFactory | None = None,
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
        # Side-effect adapter: all record-and-emit writes go through this layer.
        self._side_effect_adapter = SideEffectAdapter()
        # Durable execution: crash-safe resume.  None unless configure_resume() is called.
        self._recovery: ExecutionRecovery | None = None
        # Capability security: stage-level enforcement.  None unless configure_capabilities() is called.
        self._capability_registry: CapabilityRegistry | None = None
        self._capability_policy: Any | None = None  # SessionAuthorizationPolicy | None
        self._capability_session_id: str = ""
        # Execution receipts: signed per-stage proof of completion.
        # Uses the module-level DEFAULT_SIGNER; override via configure_receipt_signer().
        self._receipt_signer: ReceiptSigner = _DEFAULT_RECEIPT_SIGNER
        self._topology_provider_factory = topology_provider_factory or _default_topology_provider_factory

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
        resolved_policy = policy or ResumabilityPolicy()
        self._recovery = ExecutionRecovery(
            resume_store=TurnResumeStore(Path(store_dir)),
            policy=resolved_policy,
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

    @staticmethod
    def _classify_failure(error: Exception) -> TurnFailureSeverity:
        return ExecutionPolicyEngine.classify_failure(error)

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

    def _finalize_execution_trace_contract(self, turn_context: TurnContext) -> None:
        trace = list(turn_context.state.get("execution_trace") or [])
        if not trace:
            raise RuntimeError("Execution trace contract incomplete: no execution_trace events recorded")

        pipeline_names = [
            str(name).strip().lower()
            for name in list(turn_context.state.get("_pipeline_stage_names") or [])
            if str(name).strip()
        ]
        stage_order = {name: idx for idx, name in enumerate(pipeline_names)}

        expected_sequence = 1
        entered: set[str] = set()
        last_stage_index = -1
        for item in trace:
            sequence = int(item.get("sequence", 0) or 0)
            event_type = str(item.get("event_type", "") or "").strip()
            stage = str(item.get("stage", "") or "").strip().lower()
            if sequence != expected_sequence:
                raise RuntimeError(
                    "Execution trace contract incomplete: non-contiguous sequence "
                    f"expected={expected_sequence} actual={sequence}",
                )
            if not event_type or not stage:
                raise RuntimeError(
                    "Execution trace contract incomplete: missing event_type or stage "
                    f"at sequence={sequence}",
                )
            if event_type not in _DECLARED_TRACE_EVENT_TYPES:
                raise RuntimeError(
                    "Execution trace closure violation: undeclared event type "
                    f"event_type={event_type!r} sequence={sequence}",
                )
            if event_type in {"stage_enter", "stage_skip", "stage_done", "stage_error"}:
                if stage not in stage_order:
                    raise RuntimeError(
                        "Execution trace closure violation: stage outside declared pipeline "
                        f"stage={stage!r} sequence={sequence}",
                    )
                stage_idx = int(stage_order[stage])
                if stage_idx < last_stage_index:
                    raise RuntimeError(
                        "Execution trace ordering violation: non-deterministic stage order "
                        f"stage={stage!r} index={stage_idx} previous_index={last_stage_index}",
                    )
                last_stage_index = stage_idx
                if event_type in {"stage_enter", "stage_skip"}:
                    entered.add(stage)
                if event_type in {"stage_done", "stage_error"} and stage not in entered:
                    raise RuntimeError(
                        "Execution trace closure violation: stage completion without stage entry/skip "
                        f"stage={stage!r} sequence={sequence}",
                    )
            expected_sequence += 1

        canonical = {
            "trace_id": str(turn_context.trace_id or ""),
            "events": [
                {
                    "sequence": int(item.get("sequence", 0) or 0),
                    "event_type": str(item.get("event_type", "") or ""),
                    "stage": str(item.get("stage", "") or ""),
                    "phase": str(item.get("phase", "") or ""),
                    "detail": _json_safe(item.get("detail") or {}),
                }
                for item in trace
            ],
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()
        contract = stamp_trace_contract_version(
            {
                "version": "1.0",
                "event_count": len(trace),
                "trace_hash": digest,
            },
        )
        turn_context.state["execution_trace_contract"] = contract
        turn_context.metadata["execution_trace_contract"] = dict(contract)

        expected = str(
            turn_context.metadata.get("expected_execution_trace_hash") or "",
        ).strip()
        if expected and expected != digest:
            raise RuntimeError(
                f"Execution trace determinism mismatch: expected={expected!r}, actual={digest!r}",
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

    @staticmethod
    def _stage_duration_ms(turn_context: TurnContext, stage_name: str) -> float:
        total = 0.0
        for trace in list(turn_context.stage_traces or []):
            if str(getattr(trace, "stage", "") or "").strip().lower() == str(stage_name or "").strip().lower():
                total += float(getattr(trace, "duration_ms", 0.0) or 0.0)
        return round(total, 3)

    def _emit_turn_health_state(
        self,
        turn_context: TurnContext,
        *,
        total_latency_ms: float,
        failed: bool,
    ) -> None:
        self._side_effects.project_turn_health(
            turn_context,
            total_latency_ms=float(total_latency_ms or 0.0),
            failed=bool(failed),
            stage_duration_lookup=self._stage_duration_ms,
        )

    def _mark_structural_degradation(
        self,
        turn_context: TurnContext,
        reason: str,
    ) -> None:
        self._side_effects.mark_structural_degradation(turn_context, reason)

    @staticmethod
    def _mark_stage_enter(turn_context: TurnContext, stage_name: str) -> None:
        """Backward-compatible shim: delegates to StageEntryGate.enforce_stage_ordering.

        New code should use TurnGraph._stage_entry_gate.enforce_stage_ordering directly.
        This staticmethod is retained for test compatibility.
        """
        StageEntryGate().enforce_stage_ordering(turn_context, stage_name)

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

    def _emit_execution_witness(
        self,
        component: str,
        turn_context: TurnContext,
    ) -> None:
        self._side_effect_adapter.record_witness(
            component,
            turn_context,
            emitter=self._execution_witness_emitter,
        )

    @staticmethod
    def _fork_context(turn_context: TurnContext) -> TurnContext:
        return TurnContext(
            user_input=turn_context.user_input,
            attachments=turn_context.attachments,
            chunk_callback=turn_context.chunk_callback,
            metadata=dict(turn_context.metadata),
            state=dict(turn_context.state),
            trace_id=turn_context.trace_id,
            # stage_traces starts empty in each fork; merged back after parallel execution
        )

    async def _execute_parallel_nodes(
        self,
        nodes: list[Any] | tuple[Any, ...],
        turn_context: TurnContext,
    ) -> TurnContext:
        node_names = [str(getattr(node, "name", type(node).__name__) or type(node).__name__) for node in nodes]
        self._record_execution_trace(
            turn_context,
            event_type="parallel_start",
            stage="parallel_group",
            detail={"nodes": list(node_names)},
        )
        forked_contexts = [self._fork_context(turn_context) for _ in nodes]
        results = await asyncio.gather(
            *(self._execute_node(node, forked_context) for node, forked_context in zip(nodes, forked_contexts)),
        )
        for result_context in results:
            turn_context.metadata.update(result_context.metadata)
            turn_context.state.update(result_context.state)
            turn_context.stage_traces.extend(result_context.stage_traces)
        self._record_execution_trace(
            turn_context,
            event_type="parallel_done",
            stage="parallel_group",
            detail={"nodes": list(node_names)},
        )
        return turn_context

    async def _execute_node(self, node: Any, turn_context: TurnContext) -> TurnContext:
        if isinstance(node, (list, tuple)):
            return await self._execute_parallel_nodes(node, turn_context)

        node_name = str(
            getattr(node, "name", type(node).__name__) or type(node).__name__,
        )

        async def _call_node() -> TurnContext:
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
                await cast("Any", execute_method)(self.registry, turn_context)
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

    @staticmethod
    def _is_commit_boundary_node(node: Any) -> bool:
        node_type = getattr(node, "node_type", None)
        if isinstance(node_type, NodeType):
            return node_type is NodeType.COMMIT
        return str(node_type or "").strip().lower() == NodeType.COMMIT.value

    def _emit_checkpoint(
        self,
        turn_context: TurnContext,
        *,
        stage: str,
        status: str,
        error: str | None = None,
    ) -> None:
        active_stage = str(turn_context.state.get("_active_graph_stage") or "")
        self._side_effects.emit_checkpoint(
            registry=self.registry,
            turn_context=turn_context,
            stage=stage,
            status=status,
            error=error,
            active_stage=active_stage,
            checkpoint_snapshot_fn=turn_context.checkpoint_snapshot,
        )

    @staticmethod
    def _phase_for_stage(stage: str, current: TurnPhase) -> TurnPhase:
        phase_name = StagePhaseMappingPolicy.phase_name_for_stage(stage)
        if not phase_name:
            return current
        try:
            return TurnPhase(phase_name)
        except ValueError:
            return current

    def _sync_stage_phase(self, turn_context: TurnContext, *, stage: str) -> None:
        target_phase = self._phase_for_stage(stage, turn_context.phase)
        transitions = turn_context.transition_phase(
            target_phase,
            reason=f"enter:{stage}",
        )
        for transition in transitions:
            self._side_effects.emit_phase_transition(
                registry=self.registry,
                turn_context=turn_context,
                stage=stage,
                transition=dict(transition),
            )

    def _emit_capability_audit_report(
        self,
        turn_context: TurnContext,
        *,
        stage_order: list[str],
        failed: bool,
        error: str = "",
    ) -> dict[str, Any]:
        report = build_runtime_capability_audit_report(
            turn_context,
            stage_order=stage_order,
            failed=failed,
            error=error,
        )
        payload = report.to_dict()
        turn_context.state["capability_audit_report"] = payload
        turn_context.metadata["capability_audit_report"] = dict(payload)
        logger.info(
            "capability_audit_report=%s",
            json.dumps(payload, sort_keys=True, default=str),
        )
        return payload

    # ------------------------------------------------------------------
    # Dispatcher authority: structural fidelity validator
    # ------------------------------------------------------------------

    def _check_structural_fidelity(self, turn_context: TurnContext) -> None:
        """Enforce post-execution structural invariants.

        Named authority holder for the fidelity check: the dispatcher calls this
        instead of inlining conditional logic.  Raises RuntimeError on violation.
        """
        fidelity = getattr(turn_context, "fidelity", None)
        if fidelity is None:
            self._mark_structural_degradation(turn_context, "fidelity_state_missing")
            raise RuntimeError(
                "Structural turn invariant violated: turn fidelity state missing",
            )
        if not bool(getattr(fidelity, "temporal", False)):
            self._mark_structural_degradation(turn_context, "temporal_stage_missing")
            raise RuntimeError(
                "Structural turn invariant violated: TemporalNode missing",
            )
        if not bool(getattr(fidelity, "save", False)):
            self._mark_structural_degradation(turn_context, "save_stage_missing")
            raise RuntimeError(
                "Structural turn invariant violated: SaveNode did not execute",
            )
        # Strict mode: only enforced when the graph was built with an "inference" node
        # so that unit-test graphs with partial pipelines are not rejected.
        if "inference" in self._node_map and not bool(
            getattr(fidelity, "inference", False),
        ):
            self._mark_structural_degradation(turn_context, "inference_stage_missing")
            raise RuntimeError(
                "Structural turn invariant violated: InferenceNode missing",
            )

    # ------------------------------------------------------------------
    # Node-level stage boundary contract enforcement
    # ------------------------------------------------------------------

    def _enforce_node_stage_contract(
        self,
        stage_name: str,
        stage_context: TurnContext,
        *,
        phase: str,  # "input" | "output"
        rejected_stages: frozenset[str] | None = None,
    ) -> None:
        """Raise NodeContractViolation if required state keys are missing.

        This is the hard boundary enforcement point for node input/output
        contracts.  Called by _run_stage_execute before and after node execution.
        Contracts are defined in _NODE_STAGE_CONTRACTS.  Unknown stage names
        are silently skipped (no contract registered = no enforcement).

        For input-phase checks, the contract is only enforced when:
        1. The prerequisite stage is registered in this pipeline instance, AND
        2. The prerequisite stage was not kernel-rejected at runtime.
        This allows partial/test pipelines and kernel-rejection scenarios to run
        without false-positive violations.
        """
        normalized = str(stage_name or "").strip().lower()
        contract = _NODE_STAGE_CONTRACTS.get(normalized)
        if contract is None:
            return
        input_keys, output_keys = contract
        if phase == "input":
            prerequisite = _NODE_INPUT_PREREQUISITES.get(normalized)
            if prerequisite is not None:
                pipeline_stage_names = {
                    str(n or "").strip().lower()
                    for n, _ in self._pipeline_items()
                }
                if prerequisite not in pipeline_stage_names:
                    return  # prerequisite stage absent — skip input contract
                if rejected_stages and prerequisite in rejected_stages:
                    return  # prerequisite stage was kernel-rejected — skip input contract
                # Also check the live rejection contract in state (runtime kernel rejection).
                _live_rejections = {
                    str(r.get("stage") or "").strip().lower()
                    for r in list(stage_context.state.get("kernel_rejection_contract") or [])
                }
                if prerequisite in _live_rejections:
                    return  # prerequisite stage was runtime-rejected — skip input contract
            required_keys = input_keys
        else:
            required_keys = output_keys
        missing = [k for k in required_keys if k not in stage_context.state]
        if missing:
            raise NodeContractViolation(
                f"Node stage contract violation [{phase}] stage={stage_name!r}: "
                f"missing required state keys: {missing!r}",
            )

    # ------------------------------------------------------------------
    # Dispatcher authority: per-stage execution worker
    # ------------------------------------------------------------------

    def _run_stage_entry_gate(
        self,
        stage_name: str,
        stage_context: TurnContext,
        *,
        telemetry: Any,
        executed_stage_names: list[str],
    ) -> bool:
        """Apply all pre-execution skip/entry checks for a stage.

        Returns True if the stage should be skipped (caller should return
        early), False if execution should proceed.  Emits pre-flight telemetry
        (trace + checkpoint) when the stage is cleared to run.
        """
        if self._stage_entry_gate.check_recovery_skip(
            stage_name,
            stage_context,
            self._recovery,
        ):
            executed_stage_names.append(stage_name)
            self._record_execution_trace(
                stage_context,
                event_type="stage_skip",
                stage=stage_name,
                detail={"reason": "recovery"},
            )
            return True
        _sid = str(
            stage_context.metadata.get("session_id") or self._capability_session_id or "",
        )
        if self._stage_entry_gate.check_capability_skip(
            stage_name,
            stage_context,
            self._capability_registry,
            self._capability_policy,
            _sid,
        ):
            executed_stage_names.append(stage_name)
            self._record_execution_trace(
                stage_context,
                event_type="stage_skip",
                stage=stage_name,
                detail={"reason": "capability"},
            )
            return True
        self._stage_entry_gate.enforce_stage_ordering(stage_context, stage_name)
        self._record_execution_trace(
            stage_context,
            event_type="stage_enter",
            stage=stage_name,
            detail={"active_stage": stage_name},
        )
        self._sync_stage_phase(stage_context, stage=stage_name)
        self._emit_checkpoint(stage_context, stage=stage_name, status="before")
        if telemetry is not None:
            telemetry.trace(
                "turn_graph.node_start",
                node=stage_name,
                trace_id=stage_context.trace_id,
            )
        return False

    async def _run_stage_execute(
        self,
        stage_name: str,
        node: Any,
        stage_context: TurnContext,
        *,
        started_at: float,
        telemetry: Any,
    ) -> tuple[TurnContext, str | None]:
        """Run the node inside the mutation guard; capture duration and error.

        Returns ``(stage_context, error_msg)``.  ``error_msg`` is ``None`` on
        success, or the string representation of the exception on failure
        (the exception is re-raised after telemetry is recorded).
        """
        normalized_stage = str(stage_name or "").strip().lower()
        stage_kind = ""
        if normalized_stage in {"inference", "llm"}:
            stage_kind = "llm"
        elif "tool" in normalized_stage:
            stage_kind = "tool"
        elif normalized_stage == "save":
            stage_kind = "save"

        if stage_kind:
            record_execution_step(
                "turngraph.node.start",
                payload={
                    "stage": stage_name,
                    "kind": stage_kind,
                    "trace_id": str(getattr(stage_context, "trace_id", "") or ""),
                },
                required=False,
            )
        self._emit_event_tap_boundary(
            stage_context,
            "NODE_ENTER",
            stage=stage_name,
            trace_id=stage_context.trace_id,
            event_sequence=stage_context.event_sequence,
        )

        node_id = normalized_stage or str(stage_name or "unknown")
        execution_path = [
            str(name or "").strip().lower()
            for name, _ in self._pipeline_items()
            if str(name or "").strip()
        ]
        logger.debug(
            "node_execution_boundary node_id=%s stage=%s execution_path=%s trace_id=%s",
            node_id,
            stage_name,
            execution_path,
            stage_context.trace_id,
        )

        error_msg: str | None = None
        try:
            assert node_id in execution_path, (
                f"Node invariant violated: node_id={node_id!r} missing from turn_graph.nodes "
                f"execution_path={execution_path!r}"
            )
            if node_id != "temporal":
                assert "temporal" in execution_path, (
                    "Node invariant violated: temporal stage is not registered in turn_graph.nodes"
                )
            _guard = (
                MutationGuard(stage_context.mutation_queue)
                if not self._is_commit_boundary_node(node)
                else contextlib.nullcontext()
            )
            with _guard:
                self._enforce_node_stage_contract(stage_name, stage_context, phase="input")
                _rejection_count_before = len(
                    list(stage_context.state.get("kernel_rejection_contract") or [])
                )
                stage_context = await self._execute_node(node, stage_context)
                _rejection_count_after = len(
                    list(stage_context.state.get("kernel_rejection_contract") or [])
                )
                _node_was_rejected = _rejection_count_after > _rejection_count_before
                if not _node_was_rejected:
                    self._enforce_node_stage_contract(stage_name, stage_context, phase="output")
        except Exception as exc:
            error_msg = str(exc)
            self._record_execution_trace(
                stage_context,
                event_type="stage_error",
                stage=stage_name,
                detail={"error": error_msg},
            )
            self._emit_checkpoint(
                stage_context,
                stage=stage_name,
                status="error",
                error=error_msg,
            )
            # ------------------------------------------------------------------
            # Phase 4 — Layer 2: Failure-replay stamp
            # Captures enough context for deterministic replay of any failed turn.
            # All values are JSON-serialisable; no live objects are captured.
            # ------------------------------------------------------------------
            _cv = stage_context.determinism_manifest.get("contract_version", {})
            stage_context.determinism_manifest.setdefault("failure_replay", []).append({
                "stage": stage_name,
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "state_keys": sorted(stage_context.state.keys()),
                "contract_version_hash": _cv.get("node_contracts_hash", ""),
            })
            raise
        finally:
            stage_context.state["_active_graph_stage"] = ""
            duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
            stage_context.stage_traces.append(
                StageTrace(stage=stage_name, duration_ms=duration_ms, error=error_msg),
            )
            if telemetry is not None:
                telemetry.trace(
                    "turn_graph.node_done",
                    node=stage_name,
                    duration_ms=duration_ms,
                    trace_id=stage_context.trace_id,
                )
            if stage_kind:
                record_execution_step(
                    "turngraph.node.done",
                    payload={
                        "stage": stage_name,
                        "kind": stage_kind,
                        "trace_id": str(getattr(stage_context, "trace_id", "") or ""),
                        "duration_ms": duration_ms,
                        "status": "error" if error_msg else "ok",
                    },
                    required=False,
                )
                if stage_kind == "tool":
                    record_external_system_call(
                        operation="turngraph_tool_node",
                        system="tool_gateway",
                        request_payload={
                            "stage": stage_name,
                            "trace_id": str(getattr(stage_context, "trace_id", "") or ""),
                        },
                        response_payload={
                            "duration_ms": duration_ms,
                            "status": "error" if error_msg else "ok",
                            "error": str(error_msg or ""),
                        },
                        status="error" if error_msg else "ok",
                        source="TurnGraph._run_stage_execute",
                        required=False,
                    )
            self._emit_event_tap_boundary(
                stage_context,
                "NODE_EXIT",
                stage=stage_name,
                trace_id=stage_context.trace_id,
                event_sequence=stage_context.event_sequence,
                duration_ms=duration_ms,
                status="error" if error_msg else "ok",
                error=str(error_msg or ""),
            )
        return stage_context, error_msg

    def _resolve_event_tap(self):
        registry = getattr(self, "registry", None)
        bot = getattr(registry, "bot", None) if registry is not None else None
        direct = getattr(bot, "event_tap", None) or getattr(bot, "_event_tap", None)
        if direct is not None:
            return direct
        getter = getattr(registry, "get", None)
        if callable(getter):
            try:
                return getter("event_tap", optional=True)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _emit_event_tap_boundary(
        self,
        stage_context: TurnContext,
        event_type: str,
        **payload: Any,
    ) -> None:
        tap = self._resolve_event_tap()
        emit = getattr(tap, "emit", None)
        if not callable(emit):
            return
        bot = getattr(getattr(self, "registry", None), "bot", None)
        run_id = str(getattr(bot, "_active_turn_run_id", "") or "")
        if not run_id:
            return
        safe_payload = {
            str(k): (
                v
                if isinstance(v, (str, int, float, bool, type(None)))
                else str(v)
            )
            for k, v in payload.items()
        }
        emit(event_type, run_id=run_id, **safe_payload)

    def _record_stage_success(
        self,
        stage_name: str,
        stage_context: TurnContext,
        *,
        duration_ms: float,
        executed_stage_names: list[str],
    ) -> None:
        """Record all post-success artefacts for a completed stage.

        Covers: execution trace event, after-checkpoint, recovery resume point,
        signed execution receipt, and edge checkpoint to the next stage.
        """
        self._record_execution_trace(
            stage_context,
            event_type="stage_done",
            stage=stage_name,
            detail={"duration_ms": duration_ms},
        )
        self._emit_checkpoint(stage_context, stage=stage_name, status="after")
        executed_stage_names.append(stage_name)

        pipeline_names = list(stage_context.state.get("_pipeline_stage_names") or [])

        if self._recovery is not None:
            _next_for_resume = ""
            if pipeline_names and stage_name in pipeline_names:
                _cidx = pipeline_names.index(stage_name)
                if _cidx + 1 < len(pipeline_names):
                    _next_for_resume = pipeline_names[_cidx + 1]
            self._recovery.record_stage_completion(
                stage_name,
                _next_for_resume,
                stage_context,
                list(executed_stage_names),
            )

        self._side_effect_adapter.record_receipt(
            stage_context,
            stage_name,
            signer=self._receipt_signer,
            stage_call_id=str(stage_context.state.get("_stage_call_id") or ""),
            checkpoint_hash=str(stage_context.last_checkpoint_hash or ""),
        )

        if pipeline_names and stage_name in pipeline_names:
            current_idx = pipeline_names.index(stage_name)
            if current_idx + 1 < len(pipeline_names):
                next_stage_name = pipeline_names[current_idx + 1]
                self._emit_checkpoint(
                    stage_context,
                    stage=f"{stage_name}\u2192{next_stage_name}",
                    status="edge",
                )

    async def _run_stage(
        self,
        stage_name: str,
        node: Any,
        stage_context: TurnContext,
        *,
        telemetry: Any,
        executed_stage_names: list[str],
    ) -> TurnContext:
        """Execute a single pipeline stage.

        Authority contract
        ------------------
        - Reads own configuration via ``self.*`` attributes.
        - Delegates entry policy decisions to ``self._stage_entry_gate``.
        - Delegates all side-effect writes to ``self._side_effect_adapter``.
        - Receives mutable ``executed_stage_names`` list; appends stage name on
          completion so the outer dispatcher can read the completed set.
        - ``telemetry`` is passed explicitly; no implicit capture from execute().
        """
        started_at = time.perf_counter(); _ = (stage_context.trace_id, stage_context.kernel_step_id, stage_context.determinism_manifest)

        if self._run_stage_entry_gate(
            stage_name,
            stage_context,
            telemetry=telemetry,
            executed_stage_names=executed_stage_names,
        ):
            return stage_context

        if self._recovery is not None:
            self._recovery.mark_stage_started(stage_name, stage_context)

        # ------------------------------------------------------------------
        # Phase 4 — Layer 3: Memory-evolution stamps
        # Capture memory fingerprint before context_builder and after save so
        # that replayed turns can verify that memory state evolved identically.
        # ------------------------------------------------------------------
        _norm_stage = str(stage_name or "").strip().lower()
        if _norm_stage == "context_builder":
            _mems_before = list(stage_context.state.get("memories") or [])
            _mem_fp_before = hashlib.sha256(
                json.dumps(_mems_before, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            _mem_ev_init = stage_context.determinism_manifest.setdefault("memory_evolution", {})
            _mem_ev_init["before_fingerprint"] = _mem_fp_before
            _mem_ev_init["before_count"] = len(_mems_before)

        stage_context, error_msg = await self._run_stage_execute(
            stage_name,
            node,
            stage_context,
            started_at=started_at,
            telemetry=telemetry,
        )

        if _norm_stage == "save" and error_msg is None:
            _mems_after = list(stage_context.state.get("memories") or [])
            _mem_fp_after = hashlib.sha256(
                json.dumps(_mems_after, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            _mem_ev = stage_context.determinism_manifest.setdefault("memory_evolution", {})
            _mem_ev["after_fingerprint"] = _mem_fp_after
            _mem_ev["delta"] = len(_mems_after) - _mem_ev.get("before_count", 0)

        if error_msg is None:
            duration_ms = stage_context.stage_traces[-1].duration_ms
            self._record_stage_success(
                stage_name,
                stage_context,
                duration_ms=duration_ms,
                executed_stage_names=executed_stage_names,
            )

        return stage_context

    # ------------------------------------------------------------------
    # execute() helpers
    # ------------------------------------------------------------------

    def _resume_from_recovery(self, turn_context: TurnContext) -> Any:
        """Restore completed stages from durable recovery store; returns resume point or None."""
        if self._recovery is None:
            return None
        resume_pt = self._recovery.check_resume(turn_context.trace_id)
        if resume_pt is not None:
            self._recovery.restore_executed_stages(resume_pt, turn_context)
        return resume_pt

    def _apply_capability_security(
        self,
        turn_context: TurnContext,
        resume_pt: Any,
    ) -> None:
        """Freeze (fresh turn) or verify (resumed turn) the session capability set."""
        if self._capability_registry is None or self._capability_policy is None:
            return
        _sid = str(
            turn_context.metadata.get("session_id") or self._capability_session_id or "",
        )
        is_resumed = (self._recovery is not None) and (resume_pt is not None)
        if is_resumed:
            try:
                self._side_effect_adapter.verify_capability_freeze(
                    turn_context,
                    policy=self._capability_policy,
                    session_id=_sid,
                )
            except CapabilityViolationError:
                raise
        else:
            self._side_effect_adapter.freeze_capabilities(
                turn_context,
                policy=self._capability_policy,
                session_id=_sid,
            )

    def _handle_execute_exception(
        self,
        turn_context: TurnContext,
        exc: Exception,
        execute_started_at: float,
        audit_enabled: bool,
        executed_stage_names: list[str],
    ) -> None:
        """Record failure telemetry, then re-raise."""
        failure_class = self._classify_failure(exc)
        self._record_execution_trace(
            turn_context,
            event_type="turn_failed",
            stage="graph",
            detail={"severity": str(failure_class), "error": str(exc)},
        )
        self._side_effect_adapter.record_failure_taxonomy(
            turn_context,
            severity_str=str(failure_class),
            error_str=str(exc),
        )
        total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
        self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=True)
        self._finalize_execution_trace_contract(turn_context)
        self._seal_execution_identity(turn_context)
        if audit_enabled:
            self._emit_capability_audit_report(
                turn_context,
                stage_order=[trace.stage for trace in list(turn_context.stage_traces or [])],
                failed=True,
                error=str(exc),
            )
        raise exc

    def _finalize_short_circuit(
        self,
        turn_context: TurnContext,
        execute_started_at: float,
        audit_enabled: bool,
        executed_stage_names: list[str],
    ) -> FinalizedTurnResult:
        """Seal and return a short-circuited turn result."""
        total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
        self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=False)
        self._check_structural_fidelity(turn_context)
        self._record_execution_trace(
            turn_context,
            event_type="turn_short_circuit",
            stage="graph",
            detail={"latency_ms": total_ms},
        )
        self._finalize_execution_trace_contract(turn_context)
        self._seal_execution_identity(turn_context)
        if audit_enabled:
            self._emit_capability_audit_report(
                turn_context,
                stage_order=executed_stage_names,
                failed=False,
            )
        if self._recovery is not None:
            self._recovery.clear(turn_context.trace_id)
        return turn_context.short_circuit_result  # type: ignore[return-value]

    def _finalize_normal_result(
        self,
        turn_context: TurnContext,
        result: Any,
        execute_started_at: float,
        audit_enabled: bool,
        executed_stage_names: list[str],
    ) -> FinalizedTurnResult:
        """Seal and return a normally-completed turn result."""
        total_ms = round((time.perf_counter() - execute_started_at) * 1000, 3)
        self._emit_turn_health_state(turn_context, total_latency_ms=total_ms, failed=False)
        self._check_structural_fidelity(turn_context)
        self._record_execution_trace(
            turn_context,
            event_type="turn_succeeded",
            stage="graph",
            detail={"latency_ms": total_ms, "stages": list(executed_stage_names)},
        )
        self._finalize_execution_trace_contract(turn_context)
        self._seal_execution_identity(turn_context)
        if audit_enabled:
            self._emit_capability_audit_report(
                turn_context,
                stage_order=executed_stage_names,
                failed=False,
            )
        if self._recovery is not None:
            self._recovery.clear(turn_context.trace_id)
        if isinstance(result, tuple) and len(result) >= 2:
            return result
        return (str(result or ""), False)

    async def execute(
        self,
        turn_context: TurnContext,
        *,
        audit_mode: bool = False,
    ) -> FinalizedTurnResult:
        if self._required_execution_token:
            active_token = ControlPlaneExecutionBoundary.current()
            if active_token != self._required_execution_token:
                raise RuntimeError(
                    "TurnGraph.execute boundary violation: graph execution must be routed through ExecutionControlPlane",
                )

        self._emit_execution_witness("graph.execute", turn_context); _ = (turn_context.trace_id, turn_context.kernel_step_id, turn_context.determinism_manifest)
        self._record_execution_trace(
            turn_context,
            event_type="turn_start",
            stage="graph",
            detail={"audit_mode": bool(audit_mode)},
        )
        turn_context.state["_execution_kernel"] = self._execution_kernel
        execute_started_at = time.perf_counter()

        _resume_pt = self._resume_from_recovery(turn_context)
        self._apply_capability_security(turn_context, _resume_pt)

        telemetry = self.registry.get("telemetry") if self.registry is not None else None
        executed_stage_names: list[str] = []
        audit_enabled = bool(audit_mode or turn_context.metadata.get("audit_mode"))

        try:
            pipeline = self._pipeline_items()
            # Store all pipeline stage names in turn_context so _run_stage can find the next stage
            pipeline_stage_names = [name for name, _ in pipeline]
            turn_context.state["_pipeline_stage_names"] = pipeline_stage_names

            # ------------------------------------------------------------------
            # Phase 4 — Layer 1: Contract-version stamp
            # Hash of _NODE_STAGE_CONTRACTS → determinism_manifest["contract_version"]
            # Provides a stable replay anchor: replayed turns can detect schema drift.
            # ------------------------------------------------------------------
            _contracts_blob = json.dumps(_NODE_STAGE_CONTRACTS, sort_keys=True, default=str)
            _contracts_hash = hashlib.sha256(_contracts_blob.encode()).hexdigest()[:16]
            turn_context.determinism_manifest["contract_version"] = {
                "node_contracts_hash": _contracts_hash,
                "schema_version": "1",
            }

            # ------------------------------------------------------------------
            # Layer 1 — LangGraph declarative dispatch
            # The execution loop is now owned by LangGraph's StateGraph.
            # Policy, mutation, and recovery logic stay in Layer 2 (_run_stage).
            # ------------------------------------------------------------------

            # Pre-execute kernel validation (previously inside ExecutionKernel.run)
            _preflight = self._execution_kernel.validate(
                stage="pre_execute",
                operation="execution_kernel.run",
                context=turn_context,
            )
            if not _preflight.ok:
                logger.warning("[KERNEL SHADOW VIOLATION] %s", _preflight.reason)

            async def _lg_node_executor(
                stage_name: str,
                node: Any,
                state: TurnPipelineState,
            ) -> dict[str, Any]:
                """Bridge: per-stage kernel validation (Layer 1→2) + stage dispatch (Layer 2)."""
                _tc = state["context"]
                _stage_check = self._execution_kernel.validate(
                    stage=stage_name,
                    operation=f"kernel.phase:{stage_name}",
                    context=_tc,
                )
                if not _stage_check.ok:
                    logger.warning("[KERNEL SHADOW VIOLATION] %s", _stage_check.reason)
                _tc = await self._run_stage(
                    stage_name,
                    node,
                    _tc,
                    telemetry=telemetry,
                    executed_stage_names=executed_stage_names,
                )
                return {
                    "context": _tc,
                    "short_circuit": bool(getattr(_tc, "short_circuit", False)),
                    "abort": False,
                    "error": None,
                }

            _topology_provider = self._topology_provider_factory(
                pipeline,
                _lg_node_executor,
            )
            _lg_pipeline = _topology_provider.build()
            _lg_state: TurnPipelineState = {
                "context": turn_context,
                "short_circuit": False,
                "abort": False,
                "error": None,
            }
            _final_state = await _lg_pipeline.ainvoke(_lg_state)
            turn_context = _final_state["context"]

            # Post-execute kernel validation (previously inside ExecutionKernel.run)
            _post = self._execution_kernel.validate(
                stage="post_execute",
                operation="execution_kernel.run.complete",
                context=turn_context,
            )
            if not _post.ok:
                logger.warning("[KERNEL SHADOW VIOLATION] %s", _post.reason)

        except Exception as exc:  # noqa: BLE001 — top-level pipeline guard; must not let execute() propagate uncaught
            self._handle_execute_exception(
                turn_context,
                exc,
                execute_started_at,
                audit_enabled,
                executed_stage_names,
            )

        if turn_context.short_circuit and turn_context.short_circuit_result is not None:
            return self._finalize_short_circuit(
                turn_context,
                execute_started_at,
                audit_enabled,
                executed_stage_names,
            )

        result = turn_context.state.get("safe_result")
        return self._finalize_normal_result(
            turn_context,
            result,
            execute_started_at,
            audit_enabled,
            executed_stage_names,
        )
