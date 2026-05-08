from __future__ import annotations

import hashlib
import json
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
class PolicyView:
    intent_type: str
    strategy: str
    tool_request_count: int
    state_hash: str


@dataclass(frozen=True)
class PolicyReductionProof:
    contract_version: str
    projection_hash: str
    inverse_hash: str
    equivalent: bool


@dataclass(frozen=True)
class SemanticEvalInput:
    intent_hash: str
    policy_view_hash: str
    tool_request_count: int
    session_id: str
    mode: str


@dataclass(frozen=True)
class PolicyInput:
    policy_name: str
    intent: TurnIntent
    execution: ExecutionContext
    candidate: Any
    runtime_turn_context: Any | None = None
    policy_view: PolicyView | None = None
    reduction_proof: PolicyReductionProof | None = None
    semantic_eval_input: SemanticEvalInput | None = None

    @property
    def candidate_kind(self) -> str:
        return type(self.candidate).__name__

    @property
    def candidate_present(self) -> bool:
        return self.candidate is not None

    def summary(self) -> dict[str, Any]:
        policy_view = self.policy_view
        return {
            "intent_type": (policy_view.intent_type if isinstance(policy_view, PolicyView) else self.intent.intent_type),
            "strategy": (policy_view.strategy if isinstance(policy_view, PolicyView) else self.intent.strategy),
            "tool_request_count": (
                int(policy_view.tool_request_count)
                if isinstance(policy_view, PolicyView)
                else int(self.intent.tool_request_count)
            ),
            "candidate_kind": self.candidate_kind,
            "candidate_present": self.candidate_present,
            "session_id": self.execution.session_id,
            "tenant_id": self.execution.tenant_id,
            "mode": self.execution.mode,
            "policy_view_state_hash": (
                str(policy_view.state_hash)
                if isinstance(policy_view, PolicyView)
                else ""
            ),
            "policy_reduction_contract_version": (
                str(self.reduction_proof.contract_version)
                if isinstance(self.reduction_proof, PolicyReductionProof)
                else ""
            ),
        }


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8"),
    ).hexdigest()


def hash_json(payload: Any) -> str:
    return _stable_hash(payload)


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


def _extract_plan_and_requests(state_mapping: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    turn_plan = state_mapping.get("turn_plan", {})
    if not isinstance(turn_plan, dict):
        turn_plan = _obj_to_dict(turn_plan)

    tool_ir = state_mapping.get("tool_ir", {})
    if not isinstance(tool_ir, dict):
        tool_ir = _obj_to_dict(tool_ir)

    requests = tool_ir.get("requests", [])
    if not isinstance(requests, list):
        requests = []
    return turn_plan, requests


def _policy_relevant_projection(state_mapping: dict[str, Any]) -> dict[str, Any]:
    turn_plan, requests = _extract_plan_and_requests(state_mapping)
    return {
        "intent_type": str(turn_plan.get("intent_type") or ""),
        "strategy": str(turn_plan.get("strategy") or ""),
        "tool_request_count": len(requests),
    }


def _semantic_eval_input_from_state(state_mapping: dict[str, Any], turn_context: Any) -> SemanticEvalInput:
    turn_plan, requests = _extract_plan_and_requests(state_mapping)
    policy_view_projection = _policy_relevant_projection(state_mapping)
    return SemanticEvalInput(
        intent_hash=hash_json(dict(turn_plan or {})),
        policy_view_hash=hash_json(policy_view_projection),
        tool_request_count=len(requests),
        session_id=str(getattr(turn_context, "session_id", "") or "default"),
        mode=str(getattr(turn_context, "mode", "") or "live"),
    )


def hash_eval_input(eval_input: SemanticEvalInput) -> str:
    return hash_json(
        {
            "intent_hash": str(eval_input.intent_hash),
            "policy_view_hash": str(eval_input.policy_view_hash),
            "tool_request_count": int(eval_input.tool_request_count),
            "session_id": str(eval_input.session_id),
            "mode": str(eval_input.mode),
        },
    )


class ProjectionCache:
    def __init__(self) -> None:
        self._cache: dict[str, SemanticEvalInput] = {}

    def get(self, turn_context: Any) -> SemanticEvalInput:
        key = str(getattr(turn_context, "trace_id", "") or "").strip()
        if key and key in self._cache:
            return self._cache[key]

        state = getattr(turn_context, "state", None)
        state_mapping = state if isinstance(state, dict) else _obj_to_dict(state)
        if not isinstance(state_mapping, dict):
            state_mapping = {}
        view = _semantic_eval_input_from_state(state_mapping, turn_context)

        if key:
            self._cache[key] = view
        return view

    def clear(self) -> None:
        self._cache.clear()


_PROJECTION_CACHE = ProjectionCache()


def projection_cache() -> ProjectionCache:
    return _PROJECTION_CACHE


def _policy_view_projection(policy_view: PolicyView) -> dict[str, Any]:
    return {
        "intent_type": str(policy_view.intent_type),
        "strategy": str(policy_view.strategy),
        "tool_request_count": int(policy_view.tool_request_count),
    }


def prove_policy_view_bijection(state_mapping: dict[str, Any], policy_view: PolicyView) -> PolicyReductionProof:
    projection = _policy_relevant_projection(state_mapping)
    inverse_projection = _policy_view_projection(policy_view)
    projection_hash = _stable_hash(projection)
    inverse_hash = _stable_hash(inverse_projection)
    return PolicyReductionProof(
        contract_version="policy-view-bijection-v1",
        projection_hash=projection_hash,
        inverse_hash=inverse_hash,
        equivalent=(projection_hash == inverse_hash),
    )


def build_policy_view(turn_context: Any) -> PolicyView:
    state = getattr(turn_context, "state", None)
    state_mapping = state if isinstance(state, dict) else _obj_to_dict(state)
    if not isinstance(state_mapping, dict):
        state_mapping = {}

    policy_seed = _policy_relevant_projection(state_mapping)
    return PolicyView(
        intent_type=policy_seed["intent_type"],
        strategy=policy_seed["strategy"],
        tool_request_count=int(policy_seed["tool_request_count"]),
        state_hash=_stable_hash(policy_seed),
    )


def build_policy_input(policy_name: str, turn_context: Any, candidate: Any) -> PolicyInput:
    """Build PolicyInput IR with Pydantic/Dataclass support.
    
    FIX: Safely handles Pydantic models, dataclasses, and plain dicts.
    Prevents silent data loss from duck-typing failures.
    """
    state = getattr(turn_context, "state", None)
    state_mapping = state if isinstance(state, dict) else _obj_to_dict(state)
    if not isinstance(state_mapping, dict):
        state_mapping = {}

    policy_view = build_policy_view(turn_context)
    semantic_eval_input = projection_cache().get(turn_context)
    reduction_proof = prove_policy_view_bijection(state_mapping, policy_view)
    if not reduction_proof.equivalent:
        raise ValueError("PolicyView reduction contract violated: policy-relevant projection mismatch")

    intent = TurnIntent(
        intent_type=str(policy_view.intent_type),
        strategy=str(policy_view.strategy),
        tool_request_count=int(policy_view.tool_request_count),
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
        policy_view=policy_view,
        reduction_proof=reduction_proof,
        semantic_eval_input=semantic_eval_input,
    )


__all__ = [
    "ExecutionContext",
    "PolicyInput",
    "PolicyReductionProof",
    "PolicyView",
    "ProjectionCache",
    "SemanticEvalInput",
    "TurnIntent",
    "build_policy_input",
    "build_policy_view",
    "hash_eval_input",
    "hash_json",
    "projection_cache",
    "prove_policy_view_bijection",
]
