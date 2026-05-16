"""Cross-subsystem coherence checks for strict evaluation mode."""

from __future__ import annotations

from evaluation.trace_schema import CrossSubsystemCoherenceScore


class CoherenceEngine:
    """Computes global consistency penalties across subsystem traces."""

    def score(self, raw_state: dict) -> CrossSubsystemCoherenceScore:
        penalties: list[str] = []

        planner = dict(raw_state.get("planner_causal_trace") or {})
        memory = dict(raw_state.get("memory_causal_trace") or {})
        ux = dict(raw_state.get("ux_trace") or raw_state.get("ux_feedback") or {})
        tool_failures = list(raw_state.get("tool_failure_semantics") or [])

        # If planner changed intent but no replan reason is logged, causality is weak.
        if (
            list(planner.get("intent_delta_vector") or [])
            and not str(planner.get("planner_replan_reason") or "").strip()
        ):
            penalties.append("planner_intent_delta_without_replan_reason")

        # If memory says contradiction resolved but UX still reports confusion.
        if (
            bool(memory.get("influenced_final_response", False))
            and bool(memory.get("overridden", False))
            and bool(ux.get("user_confusion_detected", False))
        ):
            penalties.append("memory_override_but_ux_confusion_persists")

        # If tool failed but planner did not adapt.
        if tool_failures and not list(planner.get("dependency_graph_diff") or []):
            penalties.append("tool_failure_without_planner_adaptation")

        # Score starts at 1.0 and drops by 0.2 per inconsistency.
        score = max(0.0, 1.0 - (0.2 * len(penalties)))
        return CrossSubsystemCoherenceScore(score=round(score, 4), penalties=penalties)
