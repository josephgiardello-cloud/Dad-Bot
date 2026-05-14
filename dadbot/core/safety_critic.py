from __future__ import annotations

import re
from typing import Any


class DadSafetyCritic:
    """Lightweight risk critic for reply preflight decisions."""

    def __init__(self) -> None:
        self.pii_patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        ]
        self.harmful_keywords = ("suicide", "self-harm", "illegal", "bomb")

    def _effective_harmful_keywords(self, context: dict[str, Any]) -> tuple[str, ...]:
        boundaries = dict(context.get("safety_boundaries") or {})
        configured = boundaries.get("harmful_keywords")
        if configured is None:
            configured = context.get("harmful_keywords")
        if isinstance(configured, list | tuple | set):
            normalized = tuple(str(item).strip().lower() for item in configured if str(item).strip())
            if normalized:
                return normalized
        return tuple(str(item).strip().lower() for item in self.harmful_keywords if str(item).strip())

    def evaluate(self, user_input: str, proposed_reply: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = dict(context or {})
        score = 0.0
        reasons: list[str] = []

        combined = f"{str(user_input or '')}\n{str(proposed_reply or '')}".lower()

        for pattern in self.pii_patterns:
            if re.search(pattern, str(user_input or ""), flags=re.IGNORECASE) or re.search(
                pattern,
                str(proposed_reply or ""),
                flags=re.IGNORECASE,
            ):
                score += 0.4
                reasons.append("pii_detected")
                break

        effective_harmful_keywords = self._effective_harmful_keywords(context)
        if any(keyword in combined for keyword in effective_harmful_keywords):
            score += 0.6
            reasons.append("harmful_content")

        trust = float(context.get("relationship_trust", 0.5) or 0.5)
        if trust < 0.3 and len(str(proposed_reply or "")) > 200:
            score += 0.2
            reasons.append("advice_too_long_for_low_trust")

        boundaries = dict(context.get("safety_boundaries") or {})
        blocked_terms = [str(item).strip().lower() for item in list(boundaries.get("blocked_terms") or []) if str(item).strip()]
        if blocked_terms and any(term in combined for term in blocked_terms):
            score += 0.7
            reasons.append("policy_blocked_term")

        score = min(max(score, 0.0), 1.0)
        decision = "BLOCK" if score >= 0.7 else "REVIEW" if score >= 0.4 else "PASS"
        return {
            "risk_score": round(score, 3),
            "decision": decision,
            "reasons": reasons,
            "allowed": decision == "PASS",
        }
