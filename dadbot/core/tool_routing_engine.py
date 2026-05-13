from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dadbot.core.conflict_resolution import ConflictResolver, ToolOutput
from dadbot.core.runtime_types import ToolSpec
from dadbot.core.uncertainty_model import ConfidenceVector


@dataclass(frozen=True)
class ToolCandidate:
    name: str
    version: str
    score: float
    reason: str


class ToolRoutingEngine:
    """Capability ranking, fallback routing, and parallel arbitration for tools."""

    def __init__(self) -> None:
        self._resolver = ConflictResolver()

    def rank_candidates(
        self,
        *,
        tool_request: dict[str, Any],
        available_specs: list[ToolSpec],
        uncertainty_score: float,
    ) -> list[ToolCandidate]:
        requested_name = str(tool_request.get("tool_name") or "").strip().lower()
        requested_caps = {
            str(item).strip().lower() for item in list(tool_request.get("required_permissions") or []) if str(item).strip()
        }

        ranked: list[ToolCandidate] = []
        for spec in available_specs:
            score = 0.0
            reasons: list[str] = []
            if requested_name and requested_name in str(spec.name or "").lower():
                score += 0.45
                reasons.append("name_match")
            capability_overlap = len(
                requested_caps.intersection({str(cap).strip().lower() for cap in spec.required_permissions}),
            )
            if capability_overlap > 0:
                score += min(0.35, 0.15 * capability_overlap)
                reasons.append("permission_overlap")
            if not spec.has_side_effects():
                score += 0.12
                reasons.append("pure_tool_bonus")
            if spec.is_idempotent():
                score += 0.08
                reasons.append("idempotent_bonus")
            score = max(0.0, min(1.0, score - (0.25 * float(max(0.0, uncertainty_score)))))
            if score > 0.0:
                ranked.append(
                    ToolCandidate(
                        name=spec.name,
                        version=spec.version,
                        score=round(float(score), 6),
                        reason=",".join(reasons) if reasons else "ranked",
                    ),
                )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def build_routing_plan(
        self,
        *,
        tool_request: dict[str, Any],
        available_specs: list[ToolSpec],
        uncertainty_score: float,
    ) -> dict[str, Any]:
        ranked = self.rank_candidates(
            tool_request=tool_request,
            available_specs=available_specs,
            uncertainty_score=uncertainty_score,
        )
        primary = ranked[0] if ranked else None
        fallback = ranked[1] if len(ranked) >= 2 else None
        return {
            "mode": "parallel_arbitration" if len(ranked) >= 2 else "single",
            "primary": None
            if primary is None
            else {"tool_name": primary.name, "version": primary.version, "score": primary.score, "reason": primary.reason},
            "fallback": None
            if fallback is None
            else {"tool_name": fallback.name, "version": fallback.version, "score": fallback.score, "reason": fallback.reason},
            "candidates": [
                {"tool_name": candidate.name, "version": candidate.version, "score": candidate.score, "reason": candidate.reason}
                for candidate in ranked[:8]
            ],
        }

    def arbitrate_parallel_results(
        self,
        *,
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        outputs: list[ToolOutput] = []
        for row in tool_results:
            payload = row.get("payload")
            cv = ConfidenceVector.from_tool_result(
                tool_name=str(row.get("tool_name") or ""),
                status=str(row.get("status") or "ok"),
                partial_confidence=float(row.get("partial_confidence") or 1.0),
                historical_reliability=float(row.get("historical_reliability") or 0.9),
                data_age_seconds=float(row.get("data_age_seconds") or 0.0),
            )
            outputs.append(
                ToolOutput(
                    tool_name=str(row.get("tool_name") or ""),
                    output=payload,
                    confidence_vector=cv,
                    status=str(row.get("status") or "ok"),
                    is_partial=bool(row.get("is_partial", False)),
                    raw_error=str(row.get("error") or ""),
                ),
            )

        if not outputs:
            return {
                "resolved": False,
                "reason": "no_tool_results",
                "output": None,
            }

        resolved = self._resolver.resolve(outputs)
        return {
            "resolved": True,
            "policy": str(resolved.policy_used.value),
            "winning_tool": str(resolved.winning_tool),
            "requires_reexecution": bool(resolved.requires_reexecution),
            "confidence": float(resolved.confidence_vector.aggregate),
            "output": resolved.resolved_output,
            "notes": list(resolved.resolution_notes),
        }
