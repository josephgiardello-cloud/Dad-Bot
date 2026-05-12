from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from typing import Any


def serialize(state: Any) -> str:
    """Canonical serializer for semantic state and payloads."""
    return json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)


def hash(state: Any) -> str:
    """Canonical semantic hash for serialized state."""
    return hashlib.sha256(serialize(state).encode("utf-8")).hexdigest()


def schedule(dag: Any) -> list[Any]:
    """Canonical DAG scheduling primitive."""
    from dadbot.core.tool_scheduler import ToolScheduler

    scheduler = ToolScheduler(seed=0)
    return scheduler.schedule(dag)


def _apply_single_mutation(result: dict, mutation: object) -> None:
    if not isinstance(mutation, dict):
        return
    op = str(mutation.get("op") or "set").strip().lower()
    key = str(mutation.get("key") or "").strip()
    value = mutation.get("value")
    if not key:
        return
    if op == "set":
        result[key] = value
    elif op == "unset":
        result.pop(key, None)
    elif op == "append":
        existing = result.get(key)
        if isinstance(existing, list):
            existing.append(value)
        elif existing is None:
            result[key] = [value]
    elif op == "update" and isinstance(value, dict):
        existing = result.get(key)
        if isinstance(existing, dict):
            existing.update(value)
        elif existing is None:
            result[key] = dict(value)


def apply_mutations(state: Any) -> Any:
    """Apply deterministic mutation intents declared on state['mutations']."""
    if not isinstance(state, dict):
        return state

    result = deepcopy(state)
    mutations = list(result.pop("mutations", []) or [])
    for mutation in mutations:
        _apply_single_mutation(result, mutation)
    return result


def _coerce_eval_input(input_value: Any) -> Any:
    from dadbot.core.turn_ir import SemanticEvalInput

    if isinstance(input_value, dict):
        return SemanticEvalInput(
            intent_hash=str(input_value.get("intent_hash") or ""),
            policy_view_hash=str(input_value.get("policy_view_hash") or ""),
            tool_request_count=int(input_value.get("tool_request_count") or 0),
            session_id=str(input_value.get("session_id") or "default"),
            mode=str(input_value.get("mode") or "live"),
        )
    return input_value


def _semantic_tool_budget() -> int:
    raw_budget = str(os.environ.get("DADBOT_SEMANTIC_TOOL_BUDGET", "8")).strip()
    return max(1, int(raw_budget) if raw_budget.isdigit() else 8)


def _semantic_policy_deny_set() -> set[str]:
    raw_deny_set = str(os.environ.get("DADBOT_POLICY_DENY_SET", "")).strip()
    return {item.strip().lower() for item in raw_deny_set.split(",") if item.strip()}


def evaluate_policy(input: Any) -> Any:
    """Canonical policy evaluation primitive."""
    from dadbot.core.policy_compiler import SemanticDecision
    from dadbot.core.turn_ir import SemanticEvalInput

    input = _coerce_eval_input(input)

    if not isinstance(input, SemanticEvalInput):
        raise TypeError(
            "evaluate_policy input must be SemanticEvalInput or dict-compatible payload",
        )

    tool_budget = _semantic_tool_budget()
    if int(input.tool_request_count) > tool_budget:
        return SemanticDecision.deny("tool_budget_exceeded")

    deny_set = _semantic_policy_deny_set()
    if str(input.intent_hash or "").strip().lower() in deny_set:
        return SemanticDecision.deny("policy_block")

    if str(input.mode or "").strip().lower() == "dry_run":
        return SemanticDecision.allow(degraded=True)

    return SemanticDecision.allow()


__all__ = [
    "apply_mutations",
    "evaluate_policy",
    "hash",
    "schedule",
    "serialize",
]
