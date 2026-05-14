from __future__ import annotations

from typing import Any


def _tokenize(text: str) -> set[str]:
    return {
        token.strip().lower()
        for token in str(text or "").replace("\n", " ").split(" ")
        if len(token.strip()) >= 3
    }


class SemanticSafetyEngine:
    """Semantic safety classifier, intent drift detector, and grounding risk assessor."""

    _BLOCKLIST: frozenset[str] = frozenset(
        {
            "kill",
            "bomb",
            "exploit",
            "dox",
            "steal",
            "weapon",
            "malware",
            "phishing",
        },
    )

    def classify(self, *, user_input: str, runtime_plan: dict[str, Any], memory_context: list[dict[str, Any]]) -> dict[str, Any]:
        text = str(user_input or "")
        lowered = text.lower()
        blocked_hits = sorted(token for token in self._BLOCKLIST if token in lowered)
        semantic_risk = 1.0 if blocked_hits else 0.0

        drift = self._intent_drift(user_input=text, runtime_plan=runtime_plan)
        grounding = self._grounding_risk(user_input=text, memory_context=memory_context)

        decision = "allow"
        if semantic_risk >= 1.0:
            decision = "block"
        elif drift["score"] >= 0.75 or grounding["risk"] >= 0.75:
            decision = "review"

        return {
            "decision": decision,
            "semantic_risk": semantic_risk,
            "blocked_terms": blocked_hits,
            "intent_drift": drift,
            "grounding": grounding,
        }

    def _intent_drift(self, *, user_input: str, runtime_plan: dict[str, Any]) -> dict[str, Any]:
        plan_text = " ".join(
            [
                str(runtime_plan.get("intent_type") or ""),
                str(runtime_plan.get("strategy") or ""),
                " ".join(str(item.get("text") or "") for item in list(runtime_plan.get("subgoals") or [])),
            ],
        )
        query_tokens = _tokenize(user_input)
        plan_tokens = _tokenize(plan_text)
        if not query_tokens or not plan_tokens:
            return {"score": 0.0, "reason": "insufficient_tokens"}
        overlap = len(query_tokens.intersection(plan_tokens))
        ratio = float(overlap) / float(max(1, len(query_tokens)))
        drift_score = max(0.0, min(1.0, 1.0 - ratio))
        return {
            "score": float(round(drift_score, 6)),
            "overlap_ratio": float(round(ratio, 6)),
        }

    def _grounding_risk(self, *, user_input: str, memory_context: list[dict[str, Any]]) -> dict[str, Any]:
        query_tokens = _tokenize(user_input)
        if not query_tokens:
            return {"risk": 0.0, "reason": "empty_input"}
        evidence_tokens: set[str] = set()
        for row in list(memory_context or []):
            payload = dict(row.get("payload") or {}) if isinstance(row, dict) else {}
            evidence_tokens.update(_tokenize(str(payload.get("summary") or payload.get("text") or "")))
        overlap = len(query_tokens.intersection(evidence_tokens))
        overlap_ratio = float(overlap) / float(max(1, len(query_tokens)))
        risk = max(0.0, min(1.0, 1.0 - overlap_ratio))
        return {
            "risk": float(round(risk, 6)),
            "overlap_ratio": float(round(overlap_ratio, 6)),
            "evidence_terms": int(len(evidence_tokens)),
        }
