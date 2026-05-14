from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from dadbot.contracts import AttachmentList


class RuntimePlanner:
    """Builds and evolves runtime planning metadata for a turn."""

    def __init__(self, now_fn: Callable[[], float] | None = None) -> None:
        self._now = now_fn or time.time

    def runtime_plan_intent(self, user_input: str) -> str:
        text = str(user_input or "").strip().lower()
        if not text:
            return "statement"
        if "?" in text:
            return "question"
        if text.startswith(("please", "can you", "could you", "help me", "show me")):
            return "request"
        if any(token in text for token in ("i feel", "i am anxious", "i am stressed", "i am worried", "i'm worried")):
            return "emotional"
        return "statement"

    def runtime_plan_strategy(self, intent_type: str) -> str:
        intent = str(intent_type or "").strip().lower()
        if intent == "emotional":
            return "empathy_first"
        if intent == "request":
            return "task_plan"
        if intent == "question":
            return "direct_answer"
        return "direct_answer"

    def build_runtime_plan(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        existing = dict(metadata.get("runtime_plan") or {})
        intent_type = str(
            existing.get("intent_type")
            or dict(metadata.get("turn_plan") or {}).get("intent_type")
            or self.runtime_plan_intent(user_input),
        )
        strategy = str(
            existing.get("strategy")
            or dict(metadata.get("turn_plan") or {}).get("strategy")
            or self.runtime_plan_strategy(intent_type),
        )
        revision = int(existing.get("revision") or 0)
        now = float(self._now())
        plan = {
            "plan_id": str(existing.get("plan_id") or f"plan:{session_id}:{trace_id}"),
            "revision": max(1, revision if revision > 0 else 1),
            "intent_type": intent_type,
            "strategy": strategy,
            "status": str(existing.get("status") or "active"),
            "created_at": float(existing.get("created_at") or now),
            "updated_at": now,
            "subgoals": list(existing.get("subgoals") or []),
            "tool_routing": {
                "latency_budget_ms": int(dict(existing.get("tool_routing") or {}).get("latency_budget_ms") or 2500),
                "cost_tier": str(dict(existing.get("tool_routing") or {}).get("cost_tier") or "balanced"),
                "mode": str(dict(existing.get("tool_routing") or {}).get("mode") or "adaptive"),
                "attachment_count": int(len(attachments or [])),
            },
            "branch_history": list(existing.get("branch_history") or []),
        }
        return plan

    def mutate_runtime_plan(
        self,
        *,
        metadata: dict[str, Any],
        reason: str,
        status: str = "active",
        strategy: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        plan = dict(metadata.get("runtime_plan") or {})
        if not plan:
            return plan
        plan["revision"] = int(plan.get("revision") or 1) + 1
        if strategy:
            plan["strategy"] = str(strategy)
        plan["status"] = str(status or "active")
        now = float(self._now())
        plan["updated_at"] = now
        history = list(plan.get("branch_history") or [])
        history.append(
            {
                "revision": int(plan.get("revision") or 1),
                "reason": str(reason or "replan"),
                "status": str(status or "active"),
                "note": str(note or ""),
                "timestamp": now,
            },
        )
        plan["branch_history"] = history[-32:]
        metadata["runtime_plan"] = plan
        return plan

    def build_semantic_memory_candidates(
        self,
        *,
        user_input: str,
        response: str,
        trace_id: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        now = float(self._now())
        candidates: list[dict[str, Any]] = []
        text = str(user_input or "").strip()
        if text and "?" not in text and len(text) >= 12:
            if "i " in text.lower() or "my " in text.lower():
                candidates.append(
                    {
                        "kind": "episodic_fact",
                        "text": text[:240],
                        "score": 0.72,
                        "trace_id": str(trace_id or ""),
                        "session_id": str(session_id or "default"),
                        "source": "user_input",
                        "created_at": now,
                    },
                )
        answer = str(response or "").strip()
        if answer:
            candidates.append(
                {
                    "kind": "assistant_summary",
                    "text": answer[:240],
                    "score": 0.51,
                    "trace_id": str(trace_id or ""),
                    "session_id": str(session_id or "default"),
                    "source": "assistant_response",
                    "created_at": now,
                },
            )
        return candidates


class ToolContractManager:
    """Normalizes and validates tool runtime contract metadata."""

    def normalize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        raw = dict(metadata.get("tool_runtime_contract") or metadata.get("tool_request") or {})
        tool_name = str(raw.get("tool_name") or raw.get("name") or "").strip()
        tool_version = str(raw.get("version") or "").strip() or "latest"
        required_permissions = [
            str(item).strip().lower()
            for item in list(raw.get("required_permissions") or [])
            if str(item).strip()
        ]
        return {
            "tool_name": tool_name,
            "version": tool_version,
            "required_permissions": sorted(set(required_permissions)),
            "timeout_seconds": float(raw.get("timeout_seconds") or 10.0),
            "side_effect_class": str(raw.get("side_effect_class") or "unknown"),
            "determinism": str(raw.get("determinism") or "unknown"),
            "contract_valid": bool(tool_name),
        }

    def validate(self, metadata: dict[str, Any]) -> tuple[bool, str]:
        contract = dict(metadata.get("tool_runtime_contract") or {})
        if not contract:
            return True, "ok"
        tool_name = str(contract.get("tool_name") or "").strip()
        if not tool_name:
            return True, "ok"
        timeout_seconds = float(contract.get("timeout_seconds") or 10.0)
        if timeout_seconds < 0.05 or timeout_seconds > 120.0:
            return False, "tool timeout_seconds out of accepted range [0.05, 120.0]"
        required_permissions = {
            str(item).strip().lower()
            for item in list(contract.get("required_permissions") or [])
            if str(item).strip()
        }
        granted_permissions = {
            str(item).strip().lower()
            for item in list(metadata.get("session_permissions") or [])
            if str(item).strip()
        }
        missing = sorted(required_permissions - granted_permissions)
        if missing:
            return False, f"tool permission denied: missing {', '.join(missing)}"
        side_effect_class = str(contract.get("side_effect_class") or "").strip().lower()
        if side_effect_class in {"stateful", "logged"} and not bool(metadata.get("approval_granted", False)):
            return False, "tool requires approval_granted for side effects"
        return True, "ok"


_default_planner = RuntimePlanner()
_default_contract_manager = ToolContractManager()


def runtime_plan_intent(user_input: str) -> str:
    return _default_planner.runtime_plan_intent(user_input)


def runtime_plan_strategy(intent_type: str) -> str:
    return _default_planner.runtime_plan_strategy(intent_type)


def build_runtime_plan(
    *,
    session_id: str,
    trace_id: str,
    user_input: str,
    attachments: AttachmentList | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return _default_planner.build_runtime_plan(
        session_id=session_id,
        trace_id=trace_id,
        user_input=user_input,
        attachments=attachments,
        metadata=metadata,
    )


def mutate_runtime_plan(
    *,
    metadata: dict[str, Any],
    reason: str,
    status: str = "active",
    strategy: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    return _default_planner.mutate_runtime_plan(
        metadata=metadata,
        reason=reason,
        status=status,
        strategy=strategy,
        note=note,
    )


def build_semantic_memory_candidates(
    *,
    user_input: str,
    response: str,
    trace_id: str,
    session_id: str,
) -> list[dict[str, Any]]:
    return _default_planner.build_semantic_memory_candidates(
        user_input=user_input,
        response=response,
        trace_id=trace_id,
        session_id=session_id,
    )


def normalize_tool_runtime_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    return _default_contract_manager.normalize(metadata)


def validate_tool_runtime_contract(metadata: dict[str, Any]) -> tuple[bool, str]:
    return _default_contract_manager.validate(metadata)
