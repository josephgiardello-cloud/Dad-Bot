from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from dadbot.core.causal_dependency_graph import CausalDepGraph
from dadbot.core.cognitive_policy_engine import CognitivePolicyEngine
from dadbot.core.determinism import _content_hash
from dadbot.core.execution_contract import TurnRequest
from dadbot.core.memory_set_invariants import (
    MemorySetInvariantViolation,
    assert_memory_set_invariants,
    validate_lifecycle_transition,
)
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.system_health_scorer import SystemHealthReport, SystemHealthScorer
from dadbot.core.system_state_model import SystemStateBuilder, SystemStateSnapshot
from dadbot.core.tool_memory_causal_contract import CausalMemoryEntry
from dadbot.core.uncertainty_model import ConfidenceVector, UncertaintyPropagator
from dadbot.core.graph_temporal import TurnTemporalAxis
from dadbot.models import MemoryEntry


def _stable_digest(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class SideEffectRecord:
    effect_id: str
    effect_type: str
    subject: str
    trace_id: str
    timestamp_ms: int
    payload: dict[str, Any]


class SideEffectRegistry:
    """Canonical side-effect registry used by kernel closure checks."""

    def __init__(self) -> None:
        self._records: list[SideEffectRecord] = []

    def register(
        self,
        *,
        effect_type: str,
        subject: str,
        payload: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> SideEffectRecord:
        body = dict(payload or {})
        timestamp_ms = int(time.time() * 1000)
        digest = _stable_digest(
            {
                "effect_type": str(effect_type or ""),
                "subject": str(subject or ""),
                "trace_id": str(trace_id or ""),
                "payload": body,
                "timestamp_ms": timestamp_ms,
            }
        )
        record = SideEffectRecord(
            effect_id=f"fx:{digest[:20]}",
            effect_type=str(effect_type or "generic"),
            subject=str(subject or "unknown"),
            trace_id=str(trace_id or ""),
            timestamp_ms=timestamp_ms,
            payload=body,
        )
        self._records.append(record)
        return record

    def records(self, *, trace_id: str | None = None) -> list[SideEffectRecord]:
        if not trace_id:
            return list(self._records)
        token = str(trace_id)
        return [record for record in self._records if record.trace_id == token]

    def digest(self) -> str:
        payload = [record.__dict__ for record in self._records]
        return _stable_digest(payload)


class DriftDetector:
    """Slot-based drift detector using stable content hashes."""

    def __init__(self) -> None:
        self._slot_hashes: dict[str, str] = {}
        self._semantic_baselines: dict[str, dict[str, float]] = {}

    def observe(self, *, slot: str, value: Any) -> dict[str, Any]:
        key = str(slot or "")
        current = _content_hash(value)
        previous = self._slot_hashes.get(key)
        drifted = bool(previous is not None and previous != current)
        self._slot_hashes[key] = current
        return {
            "slot": key,
            "drifted": drifted,
            "previous_hash": previous,
            "current_hash": current,
        }

    def set_semantic_baseline(self, *, domain: str, metrics: dict[str, float]) -> dict[str, Any]:
        key = str(domain or "").strip().lower()
        baseline = {
            str(name): float(value)
            for name, value in dict(metrics or {}).items()
            if str(name).strip()
        }
        self._semantic_baselines[key] = baseline
        return {
            "domain": key,
            "baseline": dict(baseline),
        }

    def observe_semantic(
        self,
        *,
        domain: str,
        metrics: dict[str, float],
        relative_tolerance: float = 0.2,
    ) -> dict[str, Any]:
        key = str(domain or "").strip().lower()
        current = {
            str(name): float(value)
            for name, value in dict(metrics or {}).items()
            if str(name).strip()
        }
        baseline = dict(self._semantic_baselines.get(key) or {})
        if not baseline:
            self._semantic_baselines[key] = dict(current)
            return {
                "domain": key,
                "drifted": False,
                "reason": "baseline_initialized",
                "score": 0.0,
                "deltas": {},
            }

        tolerance = max(0.0, float(relative_tolerance))
        deltas: dict[str, float] = {}
        aggregate = 0.0
        counted = 0
        for name, base_value in baseline.items():
            now_value = float(current.get(name, base_value))
            scale = max(1e-9, abs(base_value))
            relative_delta = abs(now_value - base_value) / scale
            deltas[name] = float(relative_delta)
            aggregate += relative_delta
            counted += 1
        score = float(aggregate / float(max(1, counted)))
        drifted = bool(score > tolerance)
        self._semantic_baselines[key] = dict(current)
        return {
            "domain": key,
            "drifted": drifted,
            "score": round(score, 6),
            "relative_tolerance": tolerance,
            "deltas": {name: round(value, 6) for name, value in deltas.items()},
        }


class KernelClosureRuntime:
    """Phase 1 runtime: correctness boundaries and deterministic drift checks."""

    REQUIRED_STATE_KEYS: tuple[str, ...] = (
        "system_state_algebra",
        "execution_lifecycle",
        "turn_truth",
        "control_plane_invariant_gate",
        "memory_hierarchy",
    )

    def __init__(self) -> None:
        self.side_effect_registry = SideEffectRegistry()
        self.drift_detector = DriftDetector()
        self._contract_schema_hash: str | None = None
        self._uncertainty = UncertaintyPropagator()

    def ensure_global_canonical_state_schema(self, state: dict[str, Any] | None) -> dict[str, Any]:
        canonical = dict(state or {})
        for key in self.REQUIRED_STATE_KEYS:
            canonical.setdefault(key, {})
        canonical.setdefault("schema_version", "global-state-v1")
        return canonical

    def lock_execution_contract(self, request_payload: dict[str, Any] | TurnRequest) -> TurnRequest:
        try:
            request = request_payload if isinstance(request_payload, TurnRequest) else TurnRequest.model_validate(request_payload)
        except ValidationError as exc:
            raise RuntimeError(f"Execution Contract Lock rejected payload: {exc}") from exc

        schema_hash = _stable_digest(TurnRequest.model_json_schema())
        if self._contract_schema_hash is None:
            self._contract_schema_hash = schema_hash
        elif self._contract_schema_hash != schema_hash:
            raise RuntimeError("Execution Contract Lock detected schema drift in TurnRequest")
        return request

    def register_side_effect(
        self,
        *,
        effect_type: str,
        subject: str,
        payload: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> SideEffectRecord:
        return self.side_effect_registry.register(
            effect_type=effect_type,
            subject=subject,
            payload=payload,
            trace_id=trace_id,
        )

    def model_tool_uncertainty(
        self,
        *,
        tool_name: str,
        status: str,
        partial_confidence: float = 1.0,
        historical_reliability: float = 0.9,
        data_age_seconds: float = 0.0,
    ) -> dict[str, Any]:
        vector = ConfidenceVector.from_tool_result(
            tool_name=tool_name,
            status=status,
            partial_confidence=partial_confidence,
            historical_reliability=historical_reliability,
            data_age_seconds=data_age_seconds,
        )
        planner = self._uncertainty.planner_hint(vector)
        critic = self._uncertainty.critic_penalty(vector)
        return {
            "confidence": vector.to_dict(),
            "planner": {
                "mode": planner.weight_mode.value,
                "weight": planner.planning_weight,
                "reason": planner.reason,
            },
            "critic": {
                "penalty_factor": critic.penalty_factor,
                "reason": critic.reason,
            },
        }

    def detect_drift(self, *, slot: str, value: Any) -> dict[str, Any]:
        return self.drift_detector.observe(slot=slot, value=value)

    def detect_semantic_drift(
        self,
        *,
        domain: str,
        metrics: dict[str, float],
        relative_tolerance: float = 0.2,
    ) -> dict[str, Any]:
        return self.drift_detector.observe_semantic(
            domain=domain,
            metrics=metrics,
            relative_tolerance=relative_tolerance,
        )


class MemoryCompletionRuntime:
    """Phase 2 runtime: memory field model, lifecycle transitions, and consistency checks."""

    def normalize_memory_field_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = dict(payload or {})
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        raw.setdefault("created_at", timestamp)
        raw.setdefault("updated_at", timestamp)
        entry = MemoryEntry.model_validate(raw)
        return entry.model_dump(mode="json")

    def enforce_memory_lifecycle(self, *, from_state: str, to_state: str, context: str = "") -> None:
        validate_lifecycle_transition(from_state, to_state, context=context)

    def check_memory_consistency(
        self,
        *,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        decay_entries: list[dict[str, Any]] | None = None,
        context: str = "",
    ) -> None:
        assert_memory_set_invariants(
            before,
            after,
            decay_entries=decay_entries,
            context=context,
        )


@dataclass
class WorldModelSnapshot:
    system: SystemStateSnapshot
    temporal_axis: dict[str, Any]
    causal_nodes: int
    causal_edges: int
    snapshot_timestamp_ms: int
    execution_timestamp_ms: int
    max_staleness_ms: int

    @property
    def staleness_ms(self) -> int:
        return int(max(0, self.snapshot_timestamp_ms - self.execution_timestamp_ms))

    @property
    def is_fresh(self) -> bool:
        return bool(self.staleness_ms <= self.max_staleness_ms)


class WorldModelRuntime:
    """Phase 3 runtime: unified world model with causal and temporal state."""

    DEFAULT_MAX_STALENESS_MS = 2000

    def __init__(self) -> None:
        self._builder = SystemStateBuilder()
        self._graph = CausalDepGraph()

    @property
    def causal_graph(self) -> CausalDepGraph:
        return self._graph

    def ingest_causal_entries(
        self,
        *,
        entries: list[CausalMemoryEntry],
        edges: list[tuple[str, str, str]] | None = None,
    ) -> None:
        for entry in entries:
            self._graph.add_node(entry)
        for source_key, target_key, reason in list(edges or []):
            if source_key in self._graph and target_key in self._graph:
                self._graph.add_edge(source_key, target_key, reason=reason)

    def snapshot(
        self,
        *,
        tool_profiles: dict[str, Any],
        coherent_memory: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorldModelSnapshot:
        meta = dict(metadata or {})
        execution_timestamp_ms = int(meta.get("execution_timestamp_ms") or int(time.time() * 1000))
        max_staleness_ms = int(meta.get("max_staleness_ms") or self.DEFAULT_MAX_STALENESS_MS)
        system_snapshot = self._builder.build(
            tool_profiles=tool_profiles,
            coherent_memory=coherent_memory,
            causal_graph=self._graph,
            metadata=meta,
        )
        self.validate_memory_bindings(
            memory_entries=meta.get("memory_entries") or [],
            entity_bindings=meta.get("entity_bindings") or [],
        )
        temporal_axis = TurnTemporalAxis.from_now().to_dict()
        snapshot_timestamp_ms = int(time.time() * 1000)
        if snapshot_timestamp_ms < execution_timestamp_ms:
            snapshot_timestamp_ms = execution_timestamp_ms
        staleness_ms = int(snapshot_timestamp_ms - execution_timestamp_ms)
        if staleness_ms > max_staleness_ms:
            raise RuntimeError(
                "World Model Freshness Contract violation: "
                f"staleness_ms={staleness_ms} exceeds max_staleness_ms={max_staleness_ms}",
            )
        return WorldModelSnapshot(
            system=system_snapshot,
            temporal_axis=temporal_axis,
            causal_nodes=len(self._graph.nodes),
            causal_edges=len(self._graph.edges),
            snapshot_timestamp_ms=snapshot_timestamp_ms,
            execution_timestamp_ms=execution_timestamp_ms,
            max_staleness_ms=max_staleness_ms,
        )

    def validate_memory_bindings(
        self,
        *,
        memory_entries: list[dict[str, Any]],
        entity_bindings: list[dict[str, Any]],
    ) -> None:
        valid_memory_ids = {
            str(entry.get("id") or "").strip()
            for entry in list(memory_entries or [])
            if isinstance(entry, dict)
        }
        for binding in list(entity_bindings or []):
            if not isinstance(binding, dict):
                continue
            source = str(binding.get("source") or "memory").strip().lower()
            memory_id = str(binding.get("memory_id") or "").strip()
            if source == "derived":
                continue
            if not memory_id:
                raise RuntimeError("World model entity binding missing memory_id for memory source")
            if memory_id not in valid_memory_ids:
                raise RuntimeError(
                    "World model entity binding references non-existent memory entry: "
                    f"memory_id={memory_id}",
                )


class SelfMaintenanceRuntime:
    """Phase 4 runtime: self-repair orchestration and health scoring."""

    ALLOWED_REPAIR_ACTIONS: frozenset[str] = frozenset(
        {
            "cache_reset",
            "memory_reindex",
            "transient_state_cleanup",
        }
    )

    FORBIDDEN_REPAIR_ACTIONS: frozenset[str] = frozenset(
        {
            "schema_mutation",
            "contract_change",
            "identity_core_modification",
        }
    )

    def __init__(self) -> None:
        self._health = SystemHealthScorer()

    def run_self_repair(
        self,
        *,
        ledger: Any,
        session_store: Any = None,
        requested_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        actions = [str(item).strip().lower() for item in list(requested_actions or []) if str(item).strip()]
        self.enforce_repair_scope(actions=actions)
        manager = RecoveryManager(ledger=ledger)
        report = manager.recover(session_store=session_store)
        report["repair_status"] = "reconciled"
        report["self_repair_scope"] = {
            "allowed": sorted(self.ALLOWED_REPAIR_ACTIONS),
            "forbidden": sorted(self.FORBIDDEN_REPAIR_ACTIONS),
            "requested_actions": list(actions),
        }
        return report

    def enforce_repair_scope(self, *, actions: list[str]) -> None:
        requested = {str(item).strip().lower() for item in list(actions or []) if str(item).strip()}
        forbidden_hits = sorted(item for item in requested if item in self.FORBIDDEN_REPAIR_ACTIONS)
        if forbidden_hits:
            raise RuntimeError(
                "SelfRepairScope violation: forbidden action(s): " + ", ".join(forbidden_hits),
            )
        unknown = sorted(item for item in requested if item not in self.ALLOWED_REPAIR_ACTIONS)
        if unknown:
            raise RuntimeError(
                "SelfRepairScope violation: unsupported action(s): " + ", ".join(unknown),
            )

    def score_health(
        self,
        *,
        invariant_gate: Any = None,
        reconcile_metrics: dict[str, Any] | None = None,
        persistence_telemetry: dict[str, Any] | None = None,
        tool_execution_stats: dict[str, Any] | None = None,
    ) -> SystemHealthReport:
        return self._health.score(
            invariant_gate=invariant_gate,
            reconcile_metrics=reconcile_metrics,
            persistence_telemetry=persistence_telemetry,
            tool_execution_stats=tool_execution_stats,
        )


class UxLayerRuntime:
    """Phase 5 runtime: dialogue policy, persona continuity, and interaction pacing."""

    NON_INTERFERENCE_FORBIDDEN_KEYS: frozenset[str] = frozenset(
        {
            "strategy",
            "intent_type",
            "tool_routing_plan",
            "compositional_tool_plan",
            "reasoning_hypotheses",
            "decision_outcome",
            "planning_depth",
            "planner_tool",
        }
    )

    def __init__(self) -> None:
        self._policy = CognitivePolicyEngine()

    def dialogue_policy(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_input: str,
        memory_hits: int = 0,
        tool_candidates: int = 0,
    ) -> dict[str, Any]:
        return self._policy.build_plan(
            session_id=session_id,
            trace_id=trace_id,
            user_input=user_input,
            memory_hits=memory_hits,
            tool_candidates=tool_candidates,
        )

    def persona_continuity(
        self,
        *,
        persona_history: list[dict[str, Any]],
        relationship_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ordered = sorted(
            [dict(item) for item in persona_history if isinstance(item, dict)],
            key=lambda item: str(item.get("applied_at") or ""),
        )
        active_traits = [
            str(item.get("trait") or "").strip()
            for item in ordered
            if str(item.get("trait") or "").strip()
        ]
        trust = float((relationship_state or {}).get("trust_score") or 0.0)
        return {
            "active_traits": active_traits[-5:],
            "trait_count": len(active_traits),
            "relationship_trust_score": trust,
            "continuity_score": round(min(1.0, (len(active_traits[-5:]) / 5.0) * 0.7 + min(1.0, trust) * 0.3), 3),
        }

    def interaction_pacing(self, *, uncertainty_score: float, intent_type: str) -> dict[str, Any]:
        score = max(0.0, min(1.0, float(uncertainty_score or 0.0)))
        intent = str(intent_type or "statement").strip().lower()
        base_delay = 0.35
        if intent in {"emotional", "request"}:
            base_delay += 0.2
        delay = round(base_delay + (score * 0.9), 3)
        chunk_size = 110 if score >= 0.6 else 170
        return {
            "response_delay_seconds": delay,
            "stream_chunk_size": chunk_size,
            "pacing_mode": "careful" if score >= 0.6 else "normal",
        }

    def apply_output_policy(
        self,
        *,
        core_decision: dict[str, Any],
        output_projection: dict[str, Any],
    ) -> dict[str, Any]:
        self.enforce_non_interference(
            core_decision=core_decision,
            proposed_ux_overrides=output_projection,
        )
        response_text = str(output_projection.get("response") or core_decision.get("response") or "")
        style = str(output_projection.get("style") or "default")
        if style == "concise":
            response_text = response_text[: max(0, min(280, len(response_text)))]
        return {
            "response": response_text,
            "style": style,
            "pacing": dict(output_projection.get("pacing") or {}),
        }

    def enforce_non_interference(
        self,
        *,
        core_decision: dict[str, Any],
        proposed_ux_overrides: dict[str, Any],
    ) -> None:
        core = dict(core_decision or {})
        ux = dict(proposed_ux_overrides or {})
        forbidden_attempts = sorted(key for key in ux if key in self.NON_INTERFERENCE_FORBIDDEN_KEYS)
        if forbidden_attempts:
            raise RuntimeError(
                "UX non-interference violation: forbidden override key(s): " + ", ".join(forbidden_attempts),
            )
        for key in self.NON_INTERFERENCE_FORBIDDEN_KEYS:
            if key in ux and str(ux.get(key)) != str(core.get(key)):
                raise RuntimeError(f"UX non-interference violation: attempted to alter core decision field {key}")


@dataclass
class PhaseClosureRuntime:
    """Concrete runtime bundle for PHASE 1–5 closure capabilities."""

    kernel: KernelClosureRuntime = field(default_factory=KernelClosureRuntime)
    memory: MemoryCompletionRuntime = field(default_factory=MemoryCompletionRuntime)
    world: WorldModelRuntime = field(default_factory=WorldModelRuntime)
    maintenance: SelfMaintenanceRuntime = field(default_factory=SelfMaintenanceRuntime)
    ux: UxLayerRuntime = field(default_factory=UxLayerRuntime)


__all__ = [
    "DriftDetector",
    "KernelClosureRuntime",
    "MemoryCompletionRuntime",
    "PhaseClosureRuntime",
    "SideEffectRecord",
    "SideEffectRegistry",
    "SelfMaintenanceRuntime",
    "UxLayerRuntime",
    "WorldModelRuntime",
    "WorldModelSnapshot",
]