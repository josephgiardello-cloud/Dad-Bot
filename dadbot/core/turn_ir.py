from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnIntent:
    intent_type: str
    strategy: str
    tool_request_count: int = 0


@dataclass(frozen=True)
class ExecutionContext:
    session_id: str = "default"
    tenant_id: str = "default"
    trace_id: str = ""
    mode: str = "live"


@dataclass(frozen=True)
class PolicyInput:
    policy_name: str
    intent: TurnIntent
    execution: ExecutionContext
    candidate: Any
    runtime_turn_context: Any | None = None

    @property
    def candidate_kind(self) -> str:
        return type(self.candidate).__name__

    @property
    def candidate_present(self) -> bool:
        return self.candidate is not None

    def summary(self) -> dict[str, Any]:
        return {
            "intent_type": self.intent.intent_type,
            "strategy": self.intent.strategy,
            "tool_request_count": int(self.intent.tool_request_count),
            "candidate_kind": self.candidate_kind,
            "candidate_present": self.candidate_present,
            "session_id": self.execution.session_id,
            "tenant_id": self.execution.tenant_id,
            "mode": self.execution.mode,
        }


def _obj_to_dict(obj: Any) -> dict[str, Any]:
    """Convert object to dict, supporting dicts, Pydantic, dataclasses, etc.
    
    FIX: Handles Pydantic models, dataclasses, and other objects that aren't
    plain dicts. Prevents silent data loss when turn_context.state is an object.
    """
    if obj is None:
        return {}
    
    if isinstance(obj, dict):
        return dict(obj)
    
    # Try Pydantic model_dump() (v2)
    if hasattr(obj, "model_dump"):
        try:
            return dict(obj.model_dump())
        except Exception:
            pass
    
    # Try Pydantic dict() (v1)
    if hasattr(obj, "dict"):
        try:
            return dict(obj.dict())
        except Exception:
            pass
    
    # Try dataclass conversion
    if hasattr(obj, "__dataclass_fields__"):
        try:
            from dataclasses import asdict
            return dict(asdict(obj))
        except Exception:
            pass
    
    # Try __dict__ (standard Python objects)
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    
    # Fallback: return empty dict and log warning
    import logging
    logging.warning(
        "_obj_to_dict: could not convert %s to dict. Falling back to {}. "
        "Consider implementing __dict__ or model_dump().",
        type(obj).__name__,
    )
    return {}


def build_policy_input(policy_name: str, turn_context: Any, candidate: Any) -> PolicyInput:
    """Build PolicyInput IR with Pydantic/Dataclass support.
    
    FIX: Safely handles Pydantic models, dataclasses, and plain dicts.
    Prevents silent data loss from duck-typing failures.
    """
    state = getattr(turn_context, "state", None)
    state_mapping = _obj_to_dict(state)
    
    turn_plan = state_mapping.get("turn_plan", {})
    if not isinstance(turn_plan, dict):
        turn_plan = _obj_to_dict(turn_plan)
    
    tool_ir = state_mapping.get("tool_ir", {})
    if not isinstance(tool_ir, dict):
        tool_ir = _obj_to_dict(tool_ir)
    
    requests = tool_ir.get("requests", [])
    if not isinstance(requests, list):
        requests = []

    intent = TurnIntent(
        intent_type=str(turn_plan.get("intent_type") or ""),
        strategy=str(turn_plan.get("strategy") or ""),
        tool_request_count=len(requests),
    )
    execution = ExecutionContext(
        session_id=str(getattr(turn_context, "session_id", "") or "default"),
        tenant_id=str(getattr(turn_context, "tenant_id", "") or "default"),
        trace_id=str(getattr(turn_context, "trace_id", "") or ""),
        mode=str(getattr(turn_context, "mode", "") or "live"),
    )
    return PolicyInput(
        policy_name=str(policy_name or "safety"),
        intent=intent,
        execution=execution,
        candidate=candidate,
        runtime_turn_context=turn_context,
    )


__all__ = [
    "ExecutionContext",
    "PolicyInput",
    "TurnIntent",
    "build_policy_input",
]
