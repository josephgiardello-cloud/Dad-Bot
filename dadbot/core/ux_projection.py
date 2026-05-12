from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class HealthStatus(StrEnum):
    OK = "OK"
    DEGRADED_CAPABILITY = "DEGRADED_CAPABILITY"
    DEGRADED_PERFORMANCE = "DEGRADED_PERFORMANCE"
    DEGRADED_STRUCTURE = "DEGRADED_STRUCTURE"


@dataclass
class TurnHealthState:
    """User-facing per-turn health telemetry derived from stage timing."""

    status: str
    latency_ms: float
    memory_ops_time: float
    graph_sync_time: float
    inference_time: float
    fallback_used: bool = False
    fidelity: Any = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        fidelity = self.fidelity.to_dict() if hasattr(self.fidelity, "to_dict") else dict(self.fidelity or {})
        return {
            "status": str(self.status or HealthStatus.OK),
            "latency_ms": round(float(self.latency_ms or 0.0), 3),
            "memory_ops_time": round(float(self.memory_ops_time or 0.0), 3),
            "graph_sync_time": round(float(self.graph_sync_time or 0.0), 3),
            "inference_time": round(float(self.inference_time or 0.0), 3),
            "fallback_used": bool(self.fallback_used),
            "fidelity": fidelity,
        }


class TurnUxProjector:
    """Projects execution state into user-facing UX/health payloads."""

    def __init__(
        self,
        *,
        degraded_latency_threshold_ms: float = 2500.0,
        degraded_inference_threshold_ms: float = 2200.0,
        degraded_memory_threshold_ms: float = 1200.0,
        degraded_graph_sync_threshold_ms: float = 1200.0,
    ) -> None:
        self._degraded_latency_threshold_ms = float(degraded_latency_threshold_ms)
        self._degraded_inference_threshold_ms = float(degraded_inference_threshold_ms)
        self._degraded_memory_threshold_ms = float(degraded_memory_threshold_ms)
        self._degraded_graph_sync_threshold_ms = float(degraded_graph_sync_threshold_ms)

    @staticmethod
    def mark_structural_degradation(turn_context: Any, reason: str) -> None:
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return
        state["_structural_degradation"] = True
        health = dict(state.get("turn_health_state") or {})
        if health:
            health["status"] = str(HealthStatus.DEGRADED_STRUCTURE)
            state["turn_health_state"] = health
        evidence = dict(state.get("turn_health_evidence") or {})
        evidence["fidelity_degraded_reason"] = str(
            reason or "structural_invariant_violation",
        )
        evidence["health_status_tier"] = str(HealthStatus.DEGRADED_STRUCTURE)
        state["turn_health_evidence"] = evidence

    def _collect_timing_state(
        self,
        *,
        turn_context: Any,
        state: dict[str, Any],
        total_latency_ms: float,
        failed: bool,
        stage_duration_lookup: Callable[[Any, str], float],
    ) -> tuple[float, float, float, bool, bool, bool, HealthStatus]:
        memory_ops_ms = float(state.get("_timing_memory_ops_ms") or 0.0)
        graph_sync_ms = float(state.get("_timing_graph_sync_ms") or 0.0)
        inference_ms = float(stage_duration_lookup(turn_context, "inference") or 0.0)

        degraded_performance = any(
            [
                total_latency_ms >= self._degraded_latency_threshold_ms,
                inference_ms >= self._degraded_inference_threshold_ms,
                memory_ops_ms >= self._degraded_memory_threshold_ms,
                graph_sync_ms >= self._degraded_graph_sync_threshold_ms,
            ],
        )
        fallback_used = bool(state.get("fallback_used", False))
        degraded_capability = bool(
            failed or fallback_used or state.get("_capability_degraded", False),
        )
        degraded_structure = bool(state.get("_structural_degradation", False))

        if degraded_structure:
            status = HealthStatus.DEGRADED_STRUCTURE
        elif degraded_capability:
            status = HealthStatus.DEGRADED_CAPABILITY
        elif degraded_performance:
            status = HealthStatus.DEGRADED_PERFORMANCE
        else:
            status = HealthStatus.OK
        return (
            memory_ops_ms,
            graph_sync_ms,
            inference_ms,
            fallback_used,
            degraded_capability,
            degraded_structure,
            status,
        )

    @staticmethod
    def _stage_order(turn_context: Any) -> list[str]:
        return [
            str(getattr(trace, "stage", "") or "")
            for trace in list(getattr(turn_context, "stage_traces", []) or [])
        ]

    @staticmethod
    def _apply_fidelity_flags(
        *,
        fidelity: Any,
        stage_order: list[str],
        state: dict[str, Any],
    ) -> None:
        if fidelity is None:
            return
        fidelity.temporal = (
            "temporal" in stage_order
            or any("temporal" in str(s) for s in state.get("_graph_executed_stages") or set())
            or bool(state.get("temporal"))
        )
        fidelity.inference = "inference" in stage_order
        fidelity.reflection = "reflection" in stage_order
        fidelity.save = "save" in stage_order

    @staticmethod
    def _build_evidence_payload(
        *,
        turn_context: Any,
        stage_order: list[str],
        fidelity: Any,
    ) -> dict[str, Any]:
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        mutation_snapshot = mutation_queue.snapshot() if hasattr(mutation_queue, "snapshot") else {}
        return {
            "stage_order": stage_order,
            "save_node_executed": stage_order.count("save") == 1,
            "temporal_enforced": bool(getattr(fidelity, "temporal", False)),
            "pipeline_fidelity": fidelity.to_dict() if hasattr(fidelity, "to_dict") else {},
            "mutation_queue": mutation_snapshot,
            "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
        }

    @staticmethod
    def _evidence_digest_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
        reflection_summary = dict(state.get("reflection_summary") or {})
        evidence_graph = dict(reflection_summary.get("evidence_graph") or {})
        evidence_edges = list(evidence_graph.get("edges") or [])
        if not evidence_edges:
            return None

        top_edges, chain_preview = TurnUxProjector._top_evidence_edges(evidence_edges)
        if not top_edges:
            return None
        return {
            "node_count": int(evidence_graph.get("node_count") or 0),
            "edge_count": int(evidence_graph.get("edge_count") or len(evidence_edges)),
            "top_edges": top_edges,
            "chain_preview": chain_preview,
        }

    @staticmethod
    def _top_evidence_edges(evidence_edges: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
        top_edges: list[dict[str, Any]] = []
        chain_preview: list[str] = []
        for edge in evidence_edges[:3]:
            edge_entry, chain_entry = TurnUxProjector._evidence_edge_entry(edge)
            if edge_entry is None:
                continue
            top_edges.append(edge_entry)
            if chain_entry:
                chain_preview.append(chain_entry)
        return top_edges, chain_preview

    @staticmethod
    def _evidence_edge_entry(edge: Any) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(edge, dict):
            return None, None
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        entry = {
            "source": source,
            "target": target,
            "weight": round(float(edge.get("weight") or 0.0), 3),
            "observations": int(edge.get("observations") or 0),
        }
        chain = f"{source} -> {target}" if source and target else None
        return entry, chain

    def _build_ux_feedback(
        self,
        *,
        state: dict[str, Any],
        status: HealthStatus,
        memory_ops_ms: float,
        graph_sync_ms: float,
        inference_ms: float,
        evidence_digest: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        thinking = any(
            [
                inference_ms >= self._degraded_inference_threshold_ms,
                memory_ops_ms >= self._degraded_memory_threshold_ms,
                graph_sync_ms >= self._degraded_graph_sync_threshold_ms,
            ],
        )
        checking_memory = bool(
            memory_ops_ms >= self._degraded_memory_threshold_ms
            or graph_sync_ms >= self._degraded_graph_sync_threshold_ms,
        )
        mood_hint = str(state.get("mood") or "neutral")
        ux_feedback = {
            "dad_is_thinking": bool(thinking),
            "message": "Dad is thinking..." if thinking else "",
            "checking_memory": checking_memory,
            "memory_message": "Checking memory..." if checking_memory else "",
            "mood_hint": mood_hint,
            "status": status,
        }

        turn_plan = dict(state.get("turn_plan") or {})
        critique_record = dict(state.get("critique_record") or {})
        clarification_requested = str(turn_plan.get("strategy") or "").strip().lower() == "clarify"
        replan_triggered = bool(int(critique_record.get("iteration") or 0) > 0)
        repair_event_emitted = bool(replan_triggered or not bool(critique_record.get("passed", True)))
        ux_trace = {
            "intent_shift_detected": bool(turn_plan.get("new_goal_detected", False)),
            "clarification_requested": clarification_requested,
            "repair_event_emitted": repair_event_emitted,
            "user_confusion_detected": bool(clarification_requested),
            "replan_triggered": replan_triggered,
            "memory_correction_written": bool(state.get("memory_correction_written", False)),
        }
        ux_feedback.update(ux_trace)
        if evidence_digest is not None:
            ux_feedback["evidence_graph_digest"] = evidence_digest
        return ux_feedback, ux_trace

    def project(
        self,
        turn_context: Any,
        *,
        total_latency_ms: float,
        failed: bool,
        stage_duration_lookup: Callable[[Any, str], float],
    ) -> None:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        if not isinstance(state, dict) or not isinstance(metadata, dict):
            return

        (
            memory_ops_ms,
            graph_sync_ms,
            inference_ms,
            fallback_used,
            _,
            _,
            status,
        ) = self._collect_timing_state(
            turn_context=turn_context,
            state=state,
            total_latency_ms=total_latency_ms,
            failed=failed,
            stage_duration_lookup=stage_duration_lookup,
        )

        stage_order = self._stage_order(turn_context)
        fidelity = getattr(turn_context, "fidelity", None)
        self._apply_fidelity_flags(fidelity=fidelity, stage_order=stage_order, state=state)

        health = TurnHealthState(
            status=status,
            latency_ms=total_latency_ms,
            memory_ops_time=memory_ops_ms,
            graph_sync_time=graph_sync_ms,
            inference_time=inference_ms,
            fallback_used=fallback_used,
            fidelity=fidelity,
        )
        health_payload = health.to_dict()

        evidence = self._build_evidence_payload(turn_context=turn_context, stage_order=stage_order, fidelity=fidelity)
        evidence["health_status_tier"] = str(status)
        evidence_digest = self._evidence_digest_from_state(state)
        ux_feedback, ux_trace = self._build_ux_feedback(
            state=state,
            status=status,
            memory_ops_ms=memory_ops_ms,
            graph_sync_ms=graph_sync_ms,
            inference_ms=inference_ms,
            evidence_digest=evidence_digest,
        )

        turn_context.turn_health = health
        state["turn_health_state"] = health_payload
        metadata["turn_health_state"] = dict(health_payload)
        metadata["total_turn_ms"] = round(float(total_latency_ms or 0.0), 3)
        state["turn_health_evidence"] = evidence
        metadata["turn_health_evidence"] = dict(evidence)
        state["ux_feedback"] = ux_feedback
        metadata["ux_feedback"] = dict(ux_feedback)
        state["ux_trace"] = ux_trace
        metadata["ux_trace"] = dict(ux_trace)
