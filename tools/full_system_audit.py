from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AuditPersistenceService:
    """Lightweight persistence surface used by TurnGraph for checkpoint auditing."""

    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def save_graph_checkpoint(self, payload: dict[str, Any], **_kwargs: Any) -> None:
        self.checkpoints.append(dict(payload or {}))

    def save_turn_event(self, payload: dict[str, Any]) -> None:
        self.events.append(dict(payload or {}))

    def finalize_turn(self, _ctx: Any, result: Any) -> Any:
        return result

    def save_turn(self, _ctx: Any, _result: Any) -> None:
        return None

    def load_latest(self) -> dict[str, Any] | None:
        if not self.checkpoints:
            return None
        return dict(self.checkpoints[-1])


class AuditRegistry:
    """Minimal service registry for end-to-end graph execution."""

    def __init__(self) -> None:
        self.persistence = AuditPersistenceService()

    def get(self, key: str, default: Any = None) -> Any:
        if key == "maintenance_service":
            return type("Maintenance", (), {"tick": lambda _self, _ctx: {"status": "ok"}})()

        if key == "context_service":
            return type(
                "ContextService",
                (),
                {"build_context": lambda _self, _ctx: {"memory": [], "source": "audit"}},
            )()

        if key == "agent_service":
            class Agent:
                async def run_agent(self, _ctx: Any, _rich_context: Any) -> tuple[str, bool]:
                    return ("candidate_from_inference", False)

            return Agent()

        if key == "safety_service":
            class Safety:
                def enforce_policies(self, ctx: Any, candidate: Any) -> Any:
                    text = str(getattr(ctx, "user_input", "") or "").lower()
                    if "unsafe" in text:
                        return ("blocked_by_policy", False)
                    return candidate

            return Safety()

        if key == "reflection":
            return type(
                "Reflection",
                (),
                {"reflect_after_turn": lambda _self, *_args, **_kwargs: {"reflected": True}},
            )()

        if key in {"persistence", "persistence_service", "storage"}:
            return self.persistence

        if key == "telemetry":
            return None

        return default


def _stable_state_slice(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "safe_result": state.get("safe_result"),
        "safety_policy_decision": state.get("safety_policy_decision"),
        "policy_trace_events": state.get("policy_trace_events"),
        "recovery_decision": state.get("recovery_decision"),
        "recovery_action": state.get("recovery_action"),
        "recovery_strategy": state.get("recovery_strategy"),
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


async def _run_system_audit_async() -> dict[str, Any]:
    from dadbot.core.graph import TurnContext, TurnGraph
    from dadbot.core.reasoning_ir import ReasoningEngine

    print("INITIATING FULL SYSTEM AUDIT (PHASES A-E)...")

    registry = AuditRegistry()
    graph = TurnGraph(registry=registry)

    recovery_nodes = [node for node in graph.nodes if getattr(node, "name", "") == "recovery"]
    if not recovery_nodes:
        raise AssertionError("Recovery stage missing from default TurnGraph pipeline")
    if not isinstance(getattr(recovery_nodes[0], "_reasoner", None), ReasoningEngine):
        raise AssertionError("RecoveryNode is not wired to ReasoningEngine")

    print("\nAudit 1: Reasoning alignment and policy handshake...")
    turn_context = TurnContext(
        user_input="Run unsafe tool",
        metadata={"user_permission": "low", "audit_mode": True},
    )

    started = time.perf_counter()
    _ = await graph.execute(turn_context)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    policy_events = list(turn_context.state.get("policy_trace_events") or [])
    assert any(event.get("event_type") == "policy_decision" for event in policy_events), (
        "PolicyCompiler/Safety policy decision event was not emitted"
    )
    assert "recovery_action" in turn_context.state, "Phase E did not emit recovery_action"
    print(f"PASS: Logic handshake verified. Turn latency: {elapsed_ms:.2f} ms")

    print("\nAudit 2: State consistency aftermath check...")
    latest_checkpoint = registry.persistence.load_latest()
    if not isinstance(latest_checkpoint, dict):
        raise AssertionError("No checkpoint available from persistence adapter")

    persisted_state = dict(latest_checkpoint.get("state") or {})
    in_memory_slice = _stable_state_slice(dict(turn_context.state or {}))
    persisted_slice = _stable_state_slice(persisted_state)

    state_consistent = _fingerprint(in_memory_slice) == _fingerprint(persisted_slice)
    if state_consistent:
        print("PASS: State integrity verified (no drift after RecoveryNode execution).")
    else:
        print("FAIL: Drift detected between in-memory turn state and persisted checkpoint state.")

    return {
        "latency_ms": round(elapsed_ms, 3),
        "policy_events": len(policy_events),
        "recovery_strategy": str(turn_context.state.get("recovery_strategy") or ""),
        "recovery_action": dict(turn_context.state.get("recovery_action") or {}),
        "state_integrity": bool(state_consistent),
        "checkpoint_stage": str(latest_checkpoint.get("stage") or ""),
        "checkpoint_status": str(latest_checkpoint.get("status") or ""),
    }


def run_system_audit() -> dict[str, Any]:
    return asyncio.run(_run_system_audit_async())


if __name__ == "__main__":
    report = run_system_audit()
    print("\nAUDIT REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))