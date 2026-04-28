from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import time

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.execution_commitment import ExecutionCommitment
from dadbot.core.graph import TurnContext, TurnGraph, TurnTemporalAxis
from dadbot.core.invariance_contract import (
    build_boundary_compliance,
    evaluation_contract_hash,
    evaluation_contract_payload,
    get_evaluation_contract,
    resolve_boundary_declaration,
    serialize_boundary_declarations,
)
from dadbot.core.kernel import TurnKernel, bayesian_policy_gate
from dadbot.core.system_identity import (
    SYSTEM_SNAPSHOT_V0_HASH,
    compute_component_hashes,
    turn_graph_structure,
)
from dadbot.core.interfaces import (
    HealthService,
    InferenceService,
    MemoryService,
    PersistenceService,
    SafetyPolicyService,
    validate_pipeline_services,
)
from dadbot.core.nodes import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
    ToolExecutorNode,
    ToolRouterNode,
)
from dadbot.core.planner import PlannerNode
from dadbot.core.critic import CritiqueEngine
from dadbot.core.goal_scorer import GoalAwareRanker
from dadbot.core.observability import CorrelationContext, configure_exporter
from dadbot.core.persistence import CheckpointIntegrityError, CheckpointNotFoundError
from dadbot.core.contracts_adapter import ContractViolationError, FallbackEvent, FallbackRegistration, FallbackRegistry
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_trace_context import (
    ExecutionTraceRecorder,
    bind_execution_trace,
    build_execution_trace_context,
    record_execution_step,
)
from dadbot.core.execution_terminal_state import build_execution_terminal_state
from dadbot.core.truth_system import enforce_authoritative_truth_system
from dadbot.registry import ServiceRegistry, boot_registry

logger = logging.getLogger(__name__)

EXECUTION_ROLE = "production_kernel"


# ---------------------------------------------------------------------------
# Phase 4.1 — Orchestrator-scoped fallback registry
# All implicit attribute injections must be declared here.
# ---------------------------------------------------------------------------

_ORCHESTRATOR_FALLBACK_REGISTRY = FallbackRegistry()
_ORCHESTRATOR_FALLBACK_REGISTRY.register(FallbackRegistration(
    name="set_checkpointer",
    version="1.0.0",
    fallback_callable=lambda storage, checkpointer: setattr(storage, "checkpointer", checkpointer),
    contract_description=(
        "Persistence service does not implement set_checkpointer(); "
        "direct attribute injection is the safe fallback for legacy adapters."
    ),
    substituted_signature="set_checkpointer(self, checkpointer: Any) -> None",
))


# Rollback switch: keep baseline graph as default until tool-native path is promoted.
TOOL_SYSTEM_V2_ENABLED: bool = False


class DeterminismViolation(RuntimeError):
    """Raised when environment drift is detected between strict-mode replay turns."""


def _build_determinism_manifest() -> dict:
    """Compute a portable environment fingerprint for cross-machine drift detection.

    Hashes only environment variable *keys* (not values) to detect structural
    drift without exposing secrets in logs or trace metadata.
    """
    env_keys = sorted(os.environ.keys())
    env_hash = hashlib.sha256(
        json.dumps({"env_keys": env_keys}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dep_sample = ["pydantic", "ollama", "orjson", "python-dateutil"]
    dep_versions: dict[str, str] = {}
    for pkg in dep_sample:
        try:
            dep_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            dep_versions[pkg] = "unknown"
    return {
        "python_version": sys.version,
        "env_hash": env_hash,
        "dependency_versions": dep_versions,
        "timezone": list(time.tzname),
    }


def _stable_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _tool_trace_hash(context: TurnContext) -> str:
    tool_ir = dict(context.state.get("tool_ir") or {})
    execution_plan = list(tool_ir.get("execution_plan") or [])
    executions = list(tool_ir.get("executions") or [])
    results = list(context.state.get("tool_results") or [])
    payload = {
        "execution_plan": [
            {
                "sequence": int(item.get("sequence") or 0),
                "tool_name": str(item.get("tool_name") or ""),
                "intent": str(item.get("intent") or ""),
                "priority": int(item.get("priority") or 0),
                "deterministic_id": str(item.get("deterministic_id") or ""),
            }
            for item in execution_plan
        ],
        "executions": [
            {
                "sequence": int(item.get("sequence") or 0),
                "tool_name": str(item.get("tool_name") or ""),
                "input_hash": str(item.get("input_hash") or ""),
                "status": str(item.get("status") or ""),
                "output": item.get("output"),
                "deterministic_id": str(item.get("deterministic_id") or ""),
            }
            for item in executions
        ],
        "results": [
            {
                "sequence": int(item.get("sequence") or 0),
                "tool_name": str(item.get("tool_name") or ""),
                "status": str(item.get("status") or ""),
                "output": item.get("output"),
                "deterministic_id": str(item.get("deterministic_id") or ""),
            }
            for item in results
        ],
    }
    return _stable_sha256(payload)


class DadBotOrchestrator:
    """Refactored orchestrator that replaces monolithic turn orchestration."""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        *,
        config_path: str = "config.yaml",
        bot=None,
        strict: bool = False,
        enable_observability: bool = True,
        checkpointer=None,
        checkpoint_every_node: bool = False,
        tool_system_v2_enabled: bool = TOOL_SYSTEM_V2_ENABLED,
    ):
        self.bot = bot
        self._strict = strict
        self.checkpointer = checkpointer
        self.checkpoint_every_node = bool(checkpoint_every_node)
        self.tool_system_v2_enabled = bool(tool_system_v2_enabled)
        self.registry = registry or boot_registry(config_path=config_path, bot=bot)
        self.evaluation_contract = get_evaluation_contract()
        registry_declaration = getattr(self.registry, "boundary_contracts", {}).get("registry")
        boot_declaration = getattr(self.registry, "boundary_contracts", {}).get("boot")
        if boot_declaration is None and bot is not None:
            boot_declaration = getattr(bot, "_boundary_contracts", {}).get("boot")
        self.boundary_contracts = {
            "boot": resolve_boundary_declaration("boot", boot_declaration),
            "registry": resolve_boundary_declaration("registry", registry_declaration),
            "orchestrator": resolve_boundary_declaration(
                "orchestrator",
                build_boundary_compliance("orchestrator"),
            ),
        }
        if hasattr(self.registry, "declare_boundary_compliance"):
            for declaration in self.boundary_contracts.values():
                self.registry.declare_boundary_compliance(declaration)
        boundary_issues = [
            f"{name}: {'; '.join(declaration.notes)}"
            for name, declaration in self.boundary_contracts.items()
            if not declaration.compliant
        ]
        if boundary_issues:
            message = "Invariance contract boundary mismatch: " + " | ".join(boundary_issues)
            if self._strict:
                raise DeterminismViolation(message)
            logger.warning(message)
        if enable_observability:
            configure_exporter(enabled=True)
        self.graph = self._build_turn_graph()
        self.session_registry = SessionRegistry()
        self.control_plane = ExecutionControlPlane(
            registry=self.session_registry,
            kernel_executor=self._execute_job,
        )
        self.scheduler = self.control_plane.scheduler
        set_required_execution_token = getattr(self.graph, "set_required_execution_token", None)
        if callable(set_required_execution_token):
            set_required_execution_token(self.control_plane.execution_token)
        set_execution_witness_emitter = getattr(self.graph, "set_execution_witness_emitter", None)
        if callable(set_execution_witness_emitter):
            set_execution_witness_emitter(self._emit_graph_execution_witness)

    def _emit_graph_execution_witness(self, component: str, context: TurnContext) -> None:
        trace_metadata = dict(context.metadata.get("trace") or {})
        control_plane_metadata = dict(context.metadata.get("control_plane") or {})
        self.control_plane.ledger_writer.append_execution_witness(
            component=component,
            session_id=str(control_plane_metadata.get("session_id") or "default"),
            trace_id=str(context.trace_id or ""),
            correlation_id=str(trace_metadata.get("correlation_id") or ""),
            payload={
                "phase": str(getattr(context.phase, "value", context.phase) or ""),
            },
        )

    def _build_turn_graph(self) -> TurnGraph:
        """Wires pipeline: TemporalNode → (Health || Memory) → Inference → Safety → Reflection → Save.

        TemporalNode is enforced as a sequential first stage. All subsequent stages
        (including the parallel preflight group) depend on temporal context being populated.
        """
        health = self.registry.get("health")
        memory = self.registry.get("memory")
        llm = self.registry.get("llm")
        safety = self.registry.get("safety")
        storage = self.registry.get("storage")
        reflection = self.registry.get("reflection")

        # Wire optional durable checkpointer into persistence service.
        if self.checkpointer is not None:
            set_checkpointer = getattr(storage, "set_checkpointer", None)
            if callable(set_checkpointer):
                set_checkpointer(self.checkpointer)
            else:
                # Phase 4.1: Declared fallback — emit event instead of silently mutating.
                _ORCHESTRATOR_FALLBACK_REGISTRY.use(
                    "set_checkpointer",
                    source="DadBotOrchestrator._build_turn_graph",
                    reason=f"storage type '{type(storage).__name__}' has no callable set_checkpointer; "
                           "falling back to direct attribute injection.",
                    strict=bool(self._strict),
                )
                setattr(storage, "checkpointer", self.checkpointer)
            setattr(storage, "strict_mode", bool(self._strict))

        # Validate service contracts; raises in strict mode, warns otherwise.
        validate_pipeline_services({
            "health": (health, HealthService),
            "memory": (memory, MemoryService),
            "llm": (llm, InferenceService),
            "safety": (safety, SafetyPolicyService),
            "storage": (storage, PersistenceService),
        }, raise_on_failure=self._strict)

        graph = TurnGraph(registry=self.registry)
        # Stage 1: TemporalNode runs ALONE, sequentially, before anything else.
        # This is the absolute first stage. No fallback to system time is permitted.
        graph.add_node("temporal", TemporalNode())
        # Stage 2: HealthNode and ContextBuilderNode run in parallel AFTER temporal is resolved.
        graph.add_node(
            "preflight",
            (
                HealthNode(health),
                ContextBuilderNode(memory, goal_ranker=GoalAwareRanker()),
            ),
        )
        # Stage 3: PlannerNode decomposes intent and detects new goals.
        graph.add_node("planner", PlannerNode())
        if self.tool_system_v2_enabled:
            graph.add_node("tool_router", ToolRouterNode())
            graph.add_node("tool_executor", ToolExecutorNode())
        graph.add_node("inference", InferenceNode(llm, critique_engine=CritiqueEngine()))
        graph.add_node("safety", SafetyNode(safety))
        graph.add_node("reflection", ReflectionNode(reflection))
        graph.add_node("save", SaveNode(storage))

        graph.set_edge("temporal", "preflight")
        graph.set_edge("preflight", "planner")
        if self.tool_system_v2_enabled:
            graph.set_edge("planner", "tool_router")
            graph.set_edge("tool_router", "tool_executor")
            graph.set_edge("tool_executor", "inference")
        else:
            graph.set_edge("planner", "inference")
        graph.set_edge("inference", "safety")
        graph.set_edge("safety", "reflection")
        graph.set_edge("reflection", "save")

        runtime_bot = getattr(llm, 'bot', None)
        if runtime_bot is not None:
            kernel = TurnKernel(policy_gate=bayesian_policy_gate(runtime_bot))
        else:
            kernel = TurnKernel()
        graph.set_kernel(kernel)

        return graph

    def _build_turn_context(self, user_input: str, attachments: AttachmentList | None = None) -> TurnContext:
        """Build a TurnContext and seal the lock-hash execution primitive.

        Lock-hash primitive contract:
        - The ``ExecutionCommitment`` payload is the single source of truth for per-turn
          determinism identity (input + model identity + memory/tool/env fingerprints).
        - Identical payloads MUST yield identical ``lock_hash`` values.
        - Any payload drift MUST change ``lock_hash`` and therefore replay identity.
        - Model adapters consume this lock via ``determinism_context`` and normalize
          generated output under the same commitment boundary.
        """
        context = TurnContext(user_input=user_input, attachments=attachments)
        context.metadata.setdefault("temporal", context.temporal_snapshot())
        previous_health = dict(getattr(self.bot, "_last_turn_health_state", {}) or {}) if self.bot is not None else {}
        previous_status = str(previous_health.get("status") or "").strip().lower()
        if previous_status:
            context.metadata["turn_health_previous_status"] = previous_status
        llm_service = self.registry.get("llm")
        runtime_bot = getattr(llm_service, "bot", None)
        llm_provider = str(getattr(runtime_bot, "LLM_PROVIDER", "ollama") or "ollama")
        llm_model = str(getattr(runtime_bot, "LLM_MODEL", "") or "")
        memory_fingerprint = str(getattr(self.bot, "_last_memory_fingerprint", "") or "")

        # Determinism-safe, per-turn blackboard seed and fingerprint.
        blackboard_seed: dict[str, str] = {}
        blackboard_seed_fingerprint = _stable_sha256(blackboard_seed)
        context.metadata.setdefault("agent_blackboard_seed", dict(blackboard_seed))
        context.state.setdefault("agent_blackboard", dict(blackboard_seed))
        context.metadata["truth_system"] = enforce_authoritative_truth_system(
            metadata=context.metadata,
            state=context.state,
        )
        context.state.setdefault(
            "tool_ir",
            {
                "requests": [],
                "execution_plan": [],
                "executions": [],
            },
        )
        context.state.setdefault("tool_results", [])
        blackboard = dict(context.state.get("agent_blackboard") or context.state.get("blackboard") or {})
        blackboard_fingerprint = _stable_sha256(blackboard)[:16]

        component_hashes = compute_component_hashes(
            tool_system_v2_enabled=bool(self.tool_system_v2_enabled)
        )
        context.metadata["system_identity"] = {
            **component_hashes,
            "system_snapshot_v0_hash": SYSTEM_SNAPSHOT_V0_HASH,
            "tool_system_v2_enabled": bool(self.tool_system_v2_enabled),
            "evaluation_contract_hash": evaluation_contract_hash(),
            "turn_graph": turn_graph_structure(tool_system_v2_enabled=bool(self.tool_system_v2_enabled)),
        }

        context.metadata["evaluation_contract"] = evaluation_contract_payload()
        context.metadata["boundary_contracts"] = serialize_boundary_declarations(self.boundary_contracts)

        # Environment manifest: fingerprints Python version, env vars, and dependency
        # versions so cross-machine replay drift is detected at the envelope level.
        manifest = _build_determinism_manifest()
        manifest_hash = _stable_sha256(manifest)
        context.metadata["determinism_manifest"] = manifest

        base_tool_trace_hash = _tool_trace_hash(context)
        commitment = ExecutionCommitment(
            user_input=str(user_input or ""),
            attachments=list(attachments or []),
            llm_provider=llm_provider,
            llm_model=llm_model,
            state_machine="PLAN_ACT_OBSERVE_RESPOND",
            agent_blackboard_seed=blackboard_seed,
            agent_blackboard_seed_fingerprint=blackboard_seed_fingerprint,
            agent_blackboard=blackboard,
            agent_blackboard_fingerprint=blackboard_fingerprint,
            memory_fingerprint=memory_fingerprint,
            determinism_manifest_hash=manifest_hash,
            tool_trace_hash=base_tool_trace_hash,
            lock_version=3,
        )
        lock_hash = commitment.lock_hash
        if self._strict:
            determinism = {
                "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
                "enforced": True,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "seed_policy": "fixed_seed",
                "temperature_policy": "0.0",
                "lock_version": int(commitment.lock_version),
                "lock_hash": lock_hash,
                "lock_id": commitment.lock_id,
                "memory_layers_included": True,
                "memory_fingerprint": memory_fingerprint,
                "blackboard_fingerprint": blackboard_fingerprint,
                "agent_blackboard_seed_fingerprint": blackboard_seed_fingerprint,
                "tool_trace_hash": base_tool_trace_hash,
                "manifest": manifest,
                "manifest_hash": manifest_hash,
                "execution_commitment": commitment.payload(),
            }
        else:
            determinism = {
                "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
                "enforced": False,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "seed_policy": "unseeded",
                "temperature_policy": "runtime_default",
                "lock_version": int(commitment.lock_version),
                "lock_hash": lock_hash,
                "lock_id": commitment.lock_id,
                "memory_layers_included": True,
                "memory_fingerprint": memory_fingerprint,
                "blackboard_fingerprint": blackboard_fingerprint,
                "agent_blackboard_seed_fingerprint": blackboard_seed_fingerprint,
                "tool_trace_hash": base_tool_trace_hash,
                "manifest": manifest,
                "manifest_hash": manifest_hash,
                "execution_commitment": commitment.payload(),
            }
        context.metadata.setdefault("determinism", determinism)
        context.metadata["tool_system_v2_enabled"] = bool(self.tool_system_v2_enabled)
        context.metadata["system_snapshot_v0_hash"] = SYSTEM_SNAPSHOT_V0_HASH
        context.metadata["checkpoint_every_node"] = bool(self.checkpoint_every_node)
        if self._strict:
            # Deterministic replay envelope: same lock hash -> same temporal axis.
            context.temporal = TurnTemporalAxis.from_lock_hash(lock_hash)
            context.metadata["temporal"] = context.temporal_snapshot()
            if runtime_bot is not None:
                setattr(runtime_bot, "LLM_TEMPERATURE", 0.0)
                setattr(runtime_bot, "LLM_SEED", 42)
                set_deterministic = getattr(runtime_bot, "set_deterministic", None)
                if callable(set_deterministic):
                    set_deterministic(True)
        return context

    def _update_bot_last_state_safely(self, context: TurnContext) -> None:
        """Thread-safe update of legacy bot state mirrored from TurnContext."""
        if self.bot is None:
            return
        lock = getattr(self.bot, "_state_lock", None) or getattr(self.bot, "_session_lock", None)
        if lock is None:
            lock = contextlib.nullcontext()
        with lock:
            self.bot._last_turn_health_state = dict(context.state.get("turn_health_state") or {})
            self.bot._last_turn_ux_feedback = dict(context.state.get("ux_feedback") or {})
            self.bot._last_turn_health_evidence = dict(context.state.get("turn_health_evidence") or {})
            self.bot._last_capability_audit_report = dict(context.state.get("capability_audit_report") or {})
            self.bot._last_commit_id = context.state.get("last_commit_id")

    @staticmethod
    def _classify_tool_failure(execution: dict) -> str:
        status = str(execution.get("status") or "").strip().lower()
        if status == "timeout":
            return "timeout"
        message = str(execution.get("error") or execution.get("failure_reason") or "").strip().lower()
        if "context" in message or "missing" in message:
            return "missing_context"
        if "input" in message or "argument" in message or "param" in message:
            return "bad_input"
        if "wrong tool" in message or "unsupported" in message:
            return "wrong_tool"
        if status in {"failed", "error"}:
            return "runtime_exception"
        return "unknown"

    def _emit_causal_trace_fields(self, context: TurnContext) -> None:
        """Populate causal trace fields required by strict evaluation mode."""
        state = dict(context.state or {})
        user_input = str(getattr(context, "user_input", "") or "").strip().lower()

        # UX causal trace.
        if "ux_trace" not in state:
            correction_markers = [
                "ignore my last",
                "i meant something else",
                "redo",
                "different",
                "start over",
                "actually",
            ]
            intent_shift = any(marker in user_input for marker in correction_markers)
            replan_count = int(state.get("replan_count") or 0)
            ux_feedback = dict(state.get("ux_feedback") or {})
            user_confusion = bool(intent_shift and not replan_count)
            state["ux_trace"] = {
                "intent_shift_detected": bool(intent_shift),
                "clarification_requested": bool(state.get("clarification_requested", False)),
                "repair_event_emitted": bool(intent_shift),
                "user_confusion_detected": bool(user_confusion),
                "replan_triggered": bool(replan_count > 0),
                "memory_correction_written": bool(state.get("memory_structured")),
                "status": str(ux_feedback.get("status") or ""),
            }

        # Planner causal trace.
        if "planner_causal_trace" not in state:
            old_goal_count = len(list(state.get("session_goals") or []))
            new_goal_count = len(list(state.get("new_goals") or []))
            replan_reason = ""
            if int(state.get("replan_count") or 0) > 0:
                replan_reason = "planner_replan_count_incremented"
            elif old_goal_count != new_goal_count:
                replan_reason = "goal_set_changed"
            elif bool(state.get("ux_trace", {}).get("intent_shift_detected", False)):
                replan_reason = "user_intent_shift"
            state["planner_causal_trace"] = {
                "planner_replan_reason": replan_reason,
                "intent_delta_vector": (
                    ["user_intent_shift"]
                    if bool(state.get("ux_trace", {}).get("intent_shift_detected", False))
                    else []
                ),
                "dependency_graph_diff": list(
                    state.get("dependency_graph_diff")
                    or state.get("task_decomposition", {}).get("dependencies")
                    or []
                ),
            }

        # Memory causal trace.
        if "memory_causal_trace" not in state:
            memory_structured = dict(state.get("memory_structured") or {})
            state["memory_causal_trace"] = {
                "trigger": "user_correction" if bool(state.get("ux_trace", {}).get("intent_shift_detected", False)) else "context_retrieval",
                "read_link_id": str(state.get("memory_read_link_id") or ""),
                "write_link_id": str(state.get("memory_write_link_id") or ""),
                "influenced_final_response": bool(memory_structured),
                "overridden": bool(state.get("memory_override", False)),
            }

        # Tool failure semantics.
        if "tool_failure_semantics" not in state:
            semantics = []
            tool_ir = dict(state.get("tool_ir") or {})
            for execution in list(tool_ir.get("executions") or []):
                status = str(execution.get("status") or "").strip().lower()
                if status not in {"failed", "timeout", "error"}:
                    continue
                semantics.append(
                    {
                        "tool_name": str(execution.get("tool_name") or ""),
                        "failure_class": self._classify_tool_failure(execution),
                        "reason": str(execution.get("error") or execution.get("failure_reason") or "").strip(),
                    }
                )
            state["tool_failure_semantics"] = semantics

        context.state.update(state)

    async def _execute_job(self, session: dict, job) -> FinalizedTurnResult:
        correlation_id = str(
            getattr(job, "metadata", {}).get("correlation_id")
            or CorrelationContext.current()
            or CorrelationContext.ensure()
            or ""
        ).strip()
        context = self._build_turn_context(job.user_input, attachments=job.attachments)
        
        # Load checkpoint if persister is available (verify hash-chain and manifest)
        session_id = str(session.get("session_id") or job.session_id)
        if self.checkpointer:
            try:
                current_manifest = dict(context.metadata.get("determinism_manifest") or {})
                prev_checkpoint = self.checkpointer.load_checkpoint(
                    session_id,
                    current_manifest=current_manifest,
                    strict=bool(self._strict),
                )
                prev_manifest = dict(prev_checkpoint.get("manifest") or {})
                loaded_checkpoint_hash = str(prev_checkpoint.get("checkpoint_hash") or "")
                loaded_prev_hash = str(prev_checkpoint.get("prev_checkpoint_hash") or "")
                if loaded_checkpoint_hash:
                    context.last_checkpoint_hash = loaded_checkpoint_hash
                    context.prev_checkpoint_hash = loaded_prev_hash
                current_manifest = dict(context.metadata.get("determinism_manifest") or {})
                if prev_manifest.get("env_hash") != current_manifest.get("env_hash"):
                    message = (
                        "Env drift after checkpoint load: "
                        f"{prev_manifest.get('env_hash')!r} -> {current_manifest.get('env_hash')!r}"
                    )
                    if self._strict:
                        raise DeterminismViolation(message)
                    logger.warning(message)
                if prev_manifest.get("python_version") != current_manifest.get("python_version"):
                    message = (
                        "Python version drift after checkpoint load: "
                        f"{prev_manifest.get('python_version')!r} -> {current_manifest.get('python_version')!r}"
                    )
                    if self._strict:
                        raise DeterminismViolation(message)
                    logger.warning(message)
                logger.debug(f"Loaded checkpoint for session={session_id}")
            except CheckpointNotFoundError:
                logger.debug(f"No previous checkpoint for session={session_id} (first turn or new session)")
            except CheckpointIntegrityError as exc:
                if self._strict:
                    raise DeterminismViolation(str(exc)) from exc
                logger.error("Checkpoint load integrity failure (non-fatal): %s", exc)
            except Exception as e:
                if self._strict:
                    raise
                logger.error(f"Checkpoint load failed (non-fatal): {e}")
        
        # Load persisted goals into context so PlannerNode and ContextBuilderNode
        # can use goal-aware ranking and intent matching.
        context.state["session_goals"] = list(
            (session.get("state") or {}).get("goals") or []
        )
        # Strict-mode environment drift check: compare current manifest against the
        # manifest stored from the previous turn in this session.  Any change in
        # env-var keys, Python version, or timezone raises DeterminismViolation.
        if self._strict:
            stored_manifest = dict(
                (session.get("state") or {}).get("last_determinism_manifest") or {}
            )
            current_manifest = dict(context.metadata.get("determinism_manifest") or {})
            if stored_manifest:
                if stored_manifest.get("env_hash") != current_manifest.get("env_hash"):
                    raise DeterminismViolation(
                        "Environment drift detected: env_hash changed from "
                        f"{stored_manifest.get('env_hash')!r} to "
                        f"{current_manifest.get('env_hash')!r}"
                    )
                if stored_manifest.get("python_version") != current_manifest.get("python_version"):
                    raise DeterminismViolation(
                        "Python version drift: "
                        f"{stored_manifest.get('python_version')!r} -> "
                        f"{current_manifest.get('python_version')!r}"
                    )
        # Executor boundary: trace must exist and correlation must be propagated.
        job_trace_id = str(
            getattr(job, "metadata", {}).get("trace_id")
            or correlation_id
            or ""
        ).strip()
        if job_trace_id:
            context.trace_id = job_trace_id
        assert context.trace_id, "Tracing not initialized: TurnContext.trace_id is empty"
        context.metadata.setdefault("trace", {})["trace_id"] = context.trace_id
        context.metadata.setdefault("trace", {})["correlation_id"] = correlation_id
        context.metadata.setdefault(
            "control_plane",
            {
                "session_id": str(session.get("session_id") or job.session_id),
                "job_id": str(getattr(job, "job_id", "")),
                "status": str(session.get("status") or "active"),
                "enforced": True,
            },
        )
        trace_recorder = ExecutionTraceRecorder(
            trace_id=str(context.trace_id or ""),
            prompt=str(context.user_input or ""),
            metadata={
                "session_id": str(session.get("session_id") or job.session_id),
                "job_id": str(getattr(job, "job_id", "")),
            },
        )
        context.metadata["execution_trace_required"] = True
        with CorrelationContext.bind(correlation_id):
            with bind_execution_trace(trace_recorder, required=True):
                record_execution_step(
                    "kernel_turn_start",
                    payload={
                        "session_id": str(session.get("session_id") or job.session_id),
                        "job_id": str(getattr(job, "job_id", "")),
                    },
                    required=True,
                )
                try:
                    context.metadata["audit_mode"] = bool(getattr(job, "metadata", {}).get("audit_mode", False))
                    with ControlPlaneExecutionBoundary.bind(self.control_plane.execution_token):
                        result = await self.graph.execute(
                            context,
                            audit_mode=bool(context.metadata.get("audit_mode")),
                        )
                except Exception as exc:
                    logger.exception("Turn execution failed")
                    context.state["error"] = str(exc)
                    result = ("", False)
                    # In strict mode, propagate failures to catch determinism issues
                    if self._strict:
                        raise
                finally:
                    record_execution_step(
                        "kernel_turn_end",
                        payload={"has_error": bool(context.state.get("error"))},
                        required=True,
                    )
                    self._update_bot_last_state_safely(context)
                    self._last_turn_context = context
                context.metadata["execution_trace_context"] = build_execution_trace_context(
                    context=context,
                    result=result,
                    recorder=trace_recorder,
                )
        self._emit_causal_trace_fields(context)
        final_blackboard = dict(context.state.get("agent_blackboard") or {})
        final_blackboard_fingerprint = _stable_sha256(final_blackboard)
        determinism_meta = dict(context.metadata.get("determinism") or {})
        determinism_meta["agent_blackboard_final_fingerprint"] = final_blackboard_fingerprint

        final_tool_trace_hash = str(context.metadata.get("tool_execution_graph_hash") or _tool_trace_hash(context))
        determinism_meta["tool_trace_hash"] = final_tool_trace_hash
        lock_hash = str(determinism_meta.get("lock_hash") or "")
        lock_with_tools = _stable_sha256(
            {
                "lock_hash": lock_hash,
                "tool_trace_hash": final_tool_trace_hash,
                "tool_system_v2_enabled": bool(self.tool_system_v2_enabled),
            }
        )
        determinism_meta["lock_hash_with_tools"] = lock_with_tools
        determinism_meta["lock_id_with_tools"] = f"detx-{lock_with_tools[:16]}"

        context.metadata["determinism"] = determinism_meta
        context.metadata["determinism_hash_with_tools"] = lock_with_tools
        terminal_state = build_execution_terminal_state(context, finalized_result=result)
        context.metadata["terminal_state"] = terminal_state.to_dict()
        # Runtime hot path stays minimal: replay/oracle verification is post-hoc.
        context.metadata["terminal_state_replay_equivalence"] = {
            "mode": "deferred",
            "equivalent": None,
            "violations": [],
        }
        session_state = session.setdefault("state", {})
        session_state["last_result"] = result
        # Persist new goals detected this turn into session state.
        new_goals = list(context.state.get("new_goals") or [])
        if new_goals:
            current_goals: list = list(session_state.get("goals") or [])
            existing_ids = {g["id"] for g in current_goals if isinstance(g, dict) and "id" in g}
            for goal_dict in new_goals:
                if isinstance(goal_dict, dict) and goal_dict.get("id") not in existing_ids:
                    current_goals.append(goal_dict)
                    existing_ids.add(goal_dict.get("id"))
            session_state["goals"] = current_goals
        session_state["last_trace_id"] = context.trace_id
        session_state["last_turn_health_state"] = dict(context.state.get("turn_health_state") or {})
        session_state["last_turn_ux_feedback"] = dict(context.state.get("ux_feedback") or {})
        session_state["last_memory_full_history_id"] = str(context.state.get("memory_full_history_id") or "")
        session_state["last_memory_structured"] = dict(context.state.get("memory_structured") or {})
        session_state["last_agent_blackboard"] = final_blackboard
        session_state["last_arbitration_log"] = list(context.state.get("delegation_arbitration_log") or [])
        session_state["last_determinism"] = dict(context.metadata.get("determinism") or {})
        session_state["last_determinism_manifest"] = dict(context.metadata.get("determinism_manifest") or {})
        session_state["last_execution_trace_context"] = dict(context.metadata.get("execution_trace_context") or {})
        session_state["last_terminal_state"] = dict(context.metadata.get("terminal_state") or {})
        session_state["last_terminal_state_replay_equivalence"] = dict(
            context.metadata.get("terminal_state_replay_equivalence") or {}
        )
        session_state["last_evaluation_contract"] = dict(context.metadata.get("evaluation_contract") or {})
        session_state["last_boundary_contracts"] = dict(context.metadata.get("boundary_contracts") or {})

        return result

    async def handle_turn(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        """Single turn entry point routed through the unified execution control plane."""
        return await self.control_plane.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _run_coro_in_thread(coro):
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def _run_graph_turn_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        return await self.handle_turn(user_input, attachments=attachments)

    def _run_graph_turn_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._run_graph_turn_async(user_input, attachments=attachments)
        if loop is not None and loop.is_running():
            return self._run_coro_in_thread(coro)
        else:
            return asyncio.run(coro)

    def run(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        return self._run_graph_turn_sync(user_input, attachments=attachments)

    async def run_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        return await self._run_graph_turn_async(user_input, attachments=attachments)
