from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def _now() -> float:
    return float(time.time())


def _tokenize(text: str) -> list[str]:
    return [token for token in str(text or "").lower().replace("\n", " ").split(" ") if token.strip()]


@dataclass(frozen=True)
class UncertaintyState:
    score: float
    reasons: tuple[str, ...]


class CognitivePolicyEngine:
    """Production-grade planning policy for long-horizon turns.

    This engine is intentionally deterministic and side-effect free.
    """

    _AMBIGUITY_TOKENS: frozenset[str] = frozenset({"maybe", "not sure", "unsure", "probably", "guess"})

    def classify_intent(self, user_input: str) -> str:
        text = str(user_input or "").strip().lower()
        if not text:
            return "statement"
        if "?" in text:
            return "question"
        if any(token in text for token in ("please", "can you", "could you", "help me", "show me")):
            return "request"
        if any(token in text for token in ("i feel", "i am anxious", "i am stressed", "i am worried", "i'm worried")):
            return "emotional"
        return "statement"

    def model_uncertainty(self, *, user_input: str, memory_hits: int = 0, tool_candidates: int = 0) -> UncertaintyState:
        reasons: list[str] = []
        score = 0.0
        text = str(user_input or "").strip().lower()
        if not text:
            return UncertaintyState(score=1.0, reasons=("empty_input",))
        if "?" in text:
            score += 0.2
            reasons.append("interrogative")
        if any(token in text for token in self._AMBIGUITY_TOKENS):
            score += 0.25
            reasons.append("ambiguous_language")
        if memory_hits <= 0:
            score += 0.25
            reasons.append("no_memory_grounding")
        if tool_candidates <= 0:
            score += 0.2
            reasons.append("no_tool_support")
        if len(text) < 12:
            score += 0.1
            reasons.append("short_context")
        return UncertaintyState(score=max(0.0, min(1.0, score)), reasons=tuple(reasons))

    def decompose_subgoals(self, user_input: str, *, max_subgoals: int = 6) -> list[dict[str, Any]]:
        text = str(user_input or "").strip()
        if not text:
            return []
        segments: list[str] = []
        for chunk in text.replace(" then ", " and ").split(" and "):
            item = chunk.strip(" ,.;")
            if item:
                segments.append(item)
        if not segments:
            segments = [text]
        subgoals: list[dict[str, Any]] = []
        for idx, segment in enumerate(segments[: max(1, int(max_subgoals))], start=1):
            subgoals.append(
                {
                    "id": f"sg-{idx}",
                    "text": segment,
                    "status": "pending",
                    "priority": idx,
                },
            )
        return subgoals

    def select_strategy(self, *, intent_type: str, uncertainty: UncertaintyState) -> str:
        if uncertainty.score >= 0.65:
            return "clarify_before_action"
        intent = str(intent_type or "").strip().lower()
        if intent == "emotional":
            return "empathy_first"
        if intent == "request":
            return "task_execution"
        if intent == "question":
            return "grounded_answer"
        return "direct_answer"

    def build_plan(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_input: str,
        existing_plan: dict[str, Any] | None = None,
        memory_hits: int = 0,
        tool_candidates: int = 0,
    ) -> dict[str, Any]:
        prior = dict(existing_plan or {})
        intent_type = str(prior.get("intent_type") or self.classify_intent(user_input))
        uncertainty = self.model_uncertainty(
            user_input=user_input,
            memory_hits=int(memory_hits),
            tool_candidates=int(tool_candidates),
        )
        strategy = str(prior.get("strategy") or self.select_strategy(intent_type=intent_type, uncertainty=uncertainty))
        revision = int(prior.get("revision") or 0)
        plan = {
            "plan_id": str(prior.get("plan_id") or f"plan:{session_id}:{trace_id}"),
            "revision": max(1, revision if revision > 0 else 1),
            "intent_type": intent_type,
            "strategy": strategy,
            "status": str(prior.get("status") or "active"),
            "subgoals": list(prior.get("subgoals") or self.decompose_subgoals(user_input)),
            "uncertainty": {
                "score": float(uncertainty.score),
                "reasons": list(uncertainty.reasons),
            },
            "created_at": float(prior.get("created_at") or _now()),
            "updated_at": _now(),
            "branch_history": list(prior.get("branch_history") or []),
        }
        return plan

    def revise_plan(
        self,
        *,
        plan: dict[str, Any],
        reason: str,
        status: str = "active",
        strategy: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        current = dict(plan or {})
        if not current:
            return current
        current["revision"] = int(current.get("revision") or 1) + 1
        current["status"] = str(status or "active")
        current["updated_at"] = _now()
        if strategy:
            current["strategy"] = str(strategy)
        history = list(current.get("branch_history") or [])
        history.append(
            {
                "revision": int(current.get("revision") or 1),
                "reason": str(reason or "replan"),
                "status": str(status or "active"),
                "note": str(note or ""),
                "timestamp": _now(),
            },
        )
        current["branch_history"] = history[-64:]
        return current
