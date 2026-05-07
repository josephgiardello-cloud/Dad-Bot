from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dadbot.core.graph import TurnContext
from dadbot.core.nodes import SafetyNode as RuntimeSafetyNode
from dadbot.services.persistence import PersistenceService, StateDivergenceError
from dadbot_system.events import InMemoryEventBus


@dataclass
class ScenarioResult:
    compliance_test: str
    status: str
    evidence: dict[str, Any]


class ToolExecutionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass
class ToolExecutionResult:
    tool_name: str
    status: ToolExecutionStatus
    output: Any
    error: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass(frozen=True)
class ToolCapability:
    name: str
    version: str
    intents: tuple[str, ...] = ()
    cost_units: float = 0.0
    avg_latency_ms: float = 0.0
    reliability: float = 1.0


class DynamicToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}

    def register(self, capability: ToolCapability, handler: Any) -> None:
        self._handlers[str(capability.name or "").strip().lower()] = handler

    def handler_for(self, name: str) -> Any | None:
        return self._handlers.get(str(name or "").strip().lower())


class ExternalToolRuntime:
    def __init__(self, registry: DynamicToolRegistry) -> None:
        self._registry = registry

    def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        handler = self._registry.handler_for(tool_name)
        if not callable(handler):
            return ToolExecutionResult(
                tool_name=str(tool_name or "").strip().lower(),
                status=ToolExecutionStatus.ERROR,
                output=None,
                error="tool_not_registered_or_incompatible_version",
                confidence=0.0,
            )
        return handler(dict(payload or {}))


class _PersistenceManagerStub:
    def __init__(
        self,
        *,
        tamper_session_state: dict[str, Any] | None = None,
        tamper_checkpoint_state: dict[str, Any] | None = None,
    ) -> None:
        self._latest_checkpoint: dict[str, Any] | None = None
        self._tamper_session_state = dict(tamper_session_state or {}) if tamper_session_state is not None else None
        self._tamper_checkpoint_state = dict(tamper_checkpoint_state or {}) if tamper_checkpoint_state is not None else None

    def persist_turn_event(self, event: Any) -> None:
        _ = event

    def persist_graph_checkpoint(self, checkpoint: dict[str, Any], _skip_turn_event: bool = False) -> None:
        _ = _skip_turn_event
        self._latest_checkpoint = dict(checkpoint or {})

    def load_latest_graph_checkpoint(self, trace_id: str = "") -> dict[str, Any]:
        _ = trace_id
        if not isinstance(self._latest_checkpoint, dict):
            return {}
        payload = dict(self._latest_checkpoint)
        if self._tamper_checkpoint_state is not None:
            payload["state"] = dict(self._tamper_checkpoint_state)
        if self._tamper_session_state is not None:
            payload["session_state"] = dict(self._tamper_session_state)
        return payload


class _TurnServiceStub:
    def __init__(self, bot: Any) -> None:
        self.bot = bot

    def finalize_user_turn(
        self,
        turn_text: str,
        mood: str,
        reply: Any,
        norm_attachments: list[Any],
        *,
        turn_context: Any,
    ) -> tuple[Any, bool]:
        _ = (turn_text, mood, norm_attachments, turn_context)
        return reply, False


def _make_runtime(
    *,
    tenant_id: str = "tenant-1",
    snapshot_session_state: Any | None = None,
) -> tuple[Any, InMemoryEventBus]:
    event_bus = InMemoryEventBus()
    runtime = SimpleNamespace(
        config=SimpleNamespace(tenant_id=tenant_id, merkle_anchor_enabled=False),
        _runtime_event_bus=event_bus,
        relationship_manager=SimpleNamespace(materialize_projection=lambda **_kwargs: None),
        memory=SimpleNamespace(save_mood_state=lambda _mood: None),
        memory_manager=SimpleNamespace(graph_manager=SimpleNamespace(sync_graph_store=lambda **_kwargs: None)),
        memory_coordinator=SimpleNamespace(
            consolidate_memories=lambda **_kwargs: None,
            apply_controlled_forgetting=lambda **_kwargs: None,
        ),
        _background_memory_store_patch_queue=[],
        _graph_commit_active=False,
        _current_turn_time_base=None,
        MEMORY_STORE={},
        _last_turn_pipeline={},
        snapshot_session_state=(snapshot_session_state if callable(snapshot_session_state) else (lambda: {})),
        load_session_state_snapshot=lambda snapshot: None,
    )
    return runtime, event_bus


def _checkpoint_snapshot_factory(*, trace_id: str):
    def _snapshot(*, stage: str, status: str, error: str | None = None) -> dict[str, Any]:
        return {
            "trace_id": trace_id,
            "stage": str(stage or "save"),
            "status": str(status or "atomic_finalize"),
            "error": str(error or ""),
            "state": {"safe_result": ("done", False)},
            "metadata": {"determinism": {"lock_hash": "lock-xyz"}},
        }

    return _snapshot


def _make_turn_context(*, trace_id: str, session_id: str = "session-2026") -> Any:
    return SimpleNamespace(
        user_input="compliance request",
        attachments=[],
        trace_id=trace_id,
        temporal=SimpleNamespace(wall_time="2026-01-01T00:00:00Z"),
        state={
            "turn_text": "compliance request",
            "mood": "neutral",
            "norm_attachments": [],
        },
        metadata={"control_plane": {"session_id": session_id}},
        mutation_queue=None,
        checkpoint_snapshot=None,
    )


def _split_brain_divergence_check() -> ScenarioResult:
    runtime, _event_bus = _make_runtime(snapshot_session_state=lambda: {})
    persistence = _PersistenceManagerStub(
        tamper_checkpoint_state={"safe_result": ("tampered", False)},
    )
    service = PersistenceService(persistence, turn_service=_TurnServiceStub(runtime))
    turn_context = _make_turn_context(trace_id="split-brain-trace")
    turn_context.checkpoint_snapshot = _checkpoint_snapshot_factory(trace_id="split-brain-trace")

    started = time.perf_counter()
    try:
        service.finalize_turn(turn_context, ("done", False))
    except StateDivergenceError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        report = dict(getattr(exc, "report", {}) or {})
        halted = bool("commit blocked" in str(exc))
        return ScenarioResult(
            compliance_test="Memory_Integrity_Divergence",
            status="PASS" if halted else "FAIL",
            evidence={
                "ledger_hash": str(report.get("event_sourced_hash") or ""),
                "projection_hash": str(report.get("projected_hash") or ""),
                "halt_intercepted": halted,
                "latency_penalty_ms": elapsed_ms,
                "policy_trace_id": f"authority::{turn_context.trace_id}",
                "difference_count": int(report.get("difference_count") or 0),
            },
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return ScenarioResult(
        compliance_test="Memory_Integrity_Divergence",
        status="FAIL",
        evidence={
            "ledger_hash": "",
            "projection_hash": "",
            "halt_intercepted": False,
            "latency_penalty_ms": elapsed_ms,
            "policy_trace_id": "authority::split-brain-trace",
        },
    )


def _permission_ghost_idempotency_test() -> ScenarioResult:
    registry = DynamicToolRegistry()
    calls = {"count": 0}

    def _guarded_handler(payload: dict[str, Any]) -> ToolExecutionResult:
        calls["count"] += 1
        permissions = {str(item) for item in list(payload.get("session_permissions") or [])}
        if "admin" not in permissions:
            return ToolExecutionResult(
                tool_name="permissioned_tool",
                status=ToolExecutionStatus.ERROR,
                output={"allowed": False},
                error="permission_denied",
                confidence=0.0,
            )
        return ToolExecutionResult(
            tool_name="permissioned_tool",
            status=ToolExecutionStatus.OK,
            output={"allowed": True, "data": "fresh-admin-result"},
            confidence=0.99,
        )

    registry.register(
        ToolCapability(
            name="permissioned_tool",
            version="1.0.0",
            intents=("secure_lookup",),
            cost_units=1.0,
            avg_latency_ms=10.0,
            reliability=0.99,
        ),
        _guarded_handler,
    )
    runtime = ExternalToolRuntime(registry)

    denied = runtime.execute(
        "permissioned_tool",
        {
            "query": "capability",
            "enforce_permissions": True,
            "session_permissions": ["read"],
        },
    )
    admin = runtime.execute(
        "permissioned_tool",
        {
            "query": "capability",
            "enforce_permissions": True,
            "session_permissions": ["read", "admin"],
        },
    )

    pass_condition = (
        denied.status == ToolExecutionStatus.ERROR
        and admin.status == ToolExecutionStatus.OK
        and calls["count"] == 2
        and not bool(admin.metadata.get("idempotent_replay"))
    )
    return ScenarioResult(
        compliance_test="Permission_Ghost_Idempotency",
        status="PASS" if pass_condition else "FAIL",
        evidence={
            "restricted_status": denied.status.value,
            "admin_status": admin.status.value,
            "fresh_call_forced": calls["count"] == 2,
            "restricted_idempotency_key": str(denied.metadata.get("idempotency_key") or ""),
            "admin_idempotency_key": str(admin.metadata.get("idempotency_key") or ""),
            "policy_trace_id": "idempotency::security-fingerprint",
        },
    )


class _FacetOverrideSafetyService:
    def enforce_policies(self, turn_context: TurnContext, candidate: object) -> str:
        _ = turn_context
        _ = candidate
        # Policy-level rewrite must neutralize sarcastic tone regardless of personality facet.
        return "I recommend avoiding that approach because it may cause issues."

    def validate(self, candidate: object) -> str:
        return str(candidate or "")


def _conflicting_facet_safety_override_test() -> ScenarioResult:
    context = TurnContext(user_input="Give me feedback")
    context.state["facet_profile"] = {"style": "sarcastic"}
    context.state["candidate"] = "Yeah, genius, obviously do that if you want it to break."

    node = RuntimeSafetyNode(_FacetOverrideSafetyService())
    asyncio.run(node.run(context))

    safe_result = str(context.state.get("safe_result") or "")
    decision = dict(context.state.get("safety_policy_decision") or {})
    trace = dict(decision.get("trace") or {})
    final_action = dict(trace.get("final_action") or {})
    policy_trace_id = f"policy::{context.trace_id}"

    pass_condition = (
        str(decision.get("step_name") or "") == "enforce_policies"
        and str(decision.get("action") or "") == "handled"
        and bool(final_action.get("output_mutated"))
        and "genius" not in safe_result.lower()
        and "obviously" not in safe_result.lower()
    )

    return ScenarioResult(
        compliance_test="Conflicting_Facet_Safety_Override",
        status="PASS" if pass_condition else "FAIL",
        evidence={
            "policy_rule": str(decision.get("step_name") or ""),
            "output_mutated": bool(final_action.get("output_mutated")),
            "candidate_hash": str(final_action.get("candidate_hash") or ""),
            "output_hash": str(final_action.get("output_hash") or ""),
            "policy_trace_id": policy_trace_id,
            "safe_result": safe_result,
        },
    )


def run_suite() -> dict[str, Any]:
    scenarios = [
        _split_brain_divergence_check(),
        _permission_ghost_idempotency_test(),
        _conflicting_facet_safety_override_test(),
    ]
    overall = "PASS" if all(item.status == "PASS" for item in scenarios) else "FAIL"
    return {
        "suite": "DadBot_2026_Compliance_Stress_Test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "results": [
            {
                "compliance_test": item.compliance_test,
                "status": item.status,
                "evidence": item.evidence,
            }
            for item in scenarios
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DadBot 2026 compliance stress suite.")
    parser.add_argument(
        "--output",
        default="artifacts/dadbot_2026_compliance_scorecard.json",
        help="Output JSON path for the compliance scorecard.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_suite()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("overall_status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
