from __future__ import annotations

import logging

from dadbot.contracts import DadBotContext, SupportsRelationshipRuntime


logger = logging.getLogger(__name__)


class RelationshipManager:
    """Owns projection-only relationship snapshots derived from the memory graph."""

    def __init__(self, bot: DadBotContext | SupportsRelationshipRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def current_state(self) -> dict:
        graph = self._memory_graph_snapshot()
        return self.get_relationship_view(graph)

    def get_relationship_view(self, graph_state: dict) -> dict:
        if not isinstance(graph_state, dict):
            raise RuntimeError("Relationship projection requires graph_state dict")
        return self._project_state_from_graph_state(graph_state)

    def materialize_projection(self, *, turn_context=None) -> dict:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError("TemporalNode required — execution invalid")
        if not bool(getattr(self.bot, "_graph_commit_active", False)):
            raise RuntimeError("Relationship projection is SaveNode-only in strict mode")
        projection = self.get_relationship_view(self._memory_graph_snapshot())
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["relationship_projection"] = dict(projection)
        return projection

    def _memory_graph_snapshot(self) -> dict:
        memory = getattr(self.bot, "memory", None)
        snapshot_fn = getattr(memory, "memory_graph_snapshot", None)
        if not callable(snapshot_fn):
            return {"nodes": [], "edges": []}
        graph = snapshot_fn() or {}
        if not isinstance(graph, dict):
            return {"nodes": [], "edges": []}
        return graph

    def _project_state_from_graph_state(self, graph: dict) -> dict:
        nodes = [dict(item) for item in list(graph.get("nodes") or []) if isinstance(item, dict)]
        edges = [dict(item) for item in list(graph.get("edges") or []) if isinstance(item, dict)]

        weighted_nodes = sorted(
            nodes,
            key=lambda item: (-float(item.get("weight", 0) or 0), str(item.get("label") or "")),
        )
        top_topics = []
        recurring_topics: dict[str, int] = {}
        for node in weighted_nodes:
            label = str(node.get("label") or "").strip().lower()
            if not label:
                continue
            weight = max(1, int(float(node.get("weight", 0) or 0)))
            recurring_topics[label] = recurring_topics.get(label, 0) + weight
            if label not in top_topics:
                top_topics.append(label)
            if len(top_topics) >= 6:
                break

        node_pressure = min(25, len(nodes) * 2)
        edge_pressure = min(20, len(edges) * 2)
        trust_level = self.bot.clamp_score(45 + node_pressure // 2 + edge_pressure // 2)
        openness_level = self.bot.clamp_score(45 + node_pressure // 2 + max(0, edge_pressure - 4))

        heavy_terms = {"sad", "stress", "stressed", "frustrated", "tired", "burnout", "overwhelmed"}
        warm_terms = {"gratitude", "progress", "wins", "family", "support", "calm", "exercise"}
        heavy_hits = sum(1 for topic in top_topics if any(term in topic for term in heavy_terms))
        warm_hits = sum(1 for topic in top_topics if any(term in topic for term in warm_terms))
        if heavy_hits > warm_hits and heavy_hits >= 2:
            emotional_momentum = "heavy"
        elif warm_hits > heavy_hits:
            emotional_momentum = "warming"
        else:
            emotional_momentum = "steady"

        profiles = self.bot.relationship_hypothesis_profiles()
        posteriors: dict[str, float] = {
            str(item["name"]): float(item.get("probability", 0.0) or 0.0)
            for item in self.bot.default_relationship_hypotheses()
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        if emotional_momentum == "heavy":
            posteriors["acute_stress"] = posteriors.get("acute_stress", 0.15) + 0.18
            posteriors["guarded_distance"] = posteriors.get("guarded_distance", 0.15) + 0.08
            posteriors["supportive_baseline"] = max(0.01, posteriors.get("supportive_baseline", 0.4) - 0.12)
        elif emotional_momentum == "warming":
            posteriors["positive_rebound"] = posteriors.get("positive_rebound", 0.25) + 0.2
            posteriors["supportive_baseline"] = posteriors.get("supportive_baseline", 0.4) + 0.08

        total = sum(max(0.01, value) for value in posteriors.values()) or 1.0
        hypotheses = []
        for name, value in posteriors.items():
            profile = profiles.get(name, {"label": name.replace("_", " ").title(), "summary": ""})
            hypotheses.append(
                {
                    "name": name,
                    "label": str(profile.get("label") or name),
                    "summary": str(profile.get("summary") or ""),
                    "probability": round(max(0.01, value) / total, 4),
                }
            )
        hypotheses.sort(key=lambda item: (-float(item.get("probability", 0.0) or 0.0), str(item.get("name") or "")))

        return {
            "trust_level": trust_level,
            "openness_level": openness_level,
            "recurring_topics": recurring_topics,
            "recent_checkins": [],
            "emotional_momentum": emotional_momentum,
            "hypotheses": hypotheses,
            "active_hypothesis": hypotheses[0]["name"] if hypotheses else "supportive_baseline",
            "last_hypothesis_updated": str(graph.get("updated_at") or self.bot.runtime_timestamp()),
            "last_reflection": "projection-only",
            "last_updated": str(graph.get("updated_at") or self.bot.runtime_timestamp()),
        }

    def _assert_graph_execution_context(self, turn_context=None) -> None:
        if turn_context is None:
            raise RuntimeError("Compatibility shim requires explicit turn_context")
        if getattr(turn_context, "temporal", None) is None:
            raise RuntimeError("Compatibility shim requires turn_context.temporal")
        if not bool(getattr(self.bot, "_graph_commit_active", False)):
            raise RuntimeError("Compatibility shim requires in-graph execution context")
        active_stage = str(getattr(turn_context, "state", {}).get("_active_graph_stage") or "").strip().lower()
        if active_stage not in {"save", ""}:
            raise RuntimeError(
                "Compatibility shim requires SaveNode graph execution context "
                f"(active_stage={active_stage!r})"
            )

    def _run_compat_mutation(self, callback, *, turn_context=None):
        # Relationship subsystem is projection-only; keep compat call sites alive
        # without requiring SaveNode mutation context.
        return callback(turn_context)

    def _require_turn_temporal(self, turn_context=None):
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        wall_time = str(getattr(temporal, "wall_time", "")).strip()
        wall_date = str(getattr(temporal, "wall_date", "")).strip()
        if not wall_time or not wall_date:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        return temporal

    def _assert_save_commit_boundary(self) -> None:
        if not bool(getattr(self.bot, "_graph_commit_active", False)):
            raise RuntimeError("Relationship mutation outside SaveNode commit boundary is forbidden in strict mode")

    def _turn_date(self, turn_context=None) -> str:
        temporal = self._require_turn_temporal(turn_context)
        wall_date = str(getattr(temporal, "wall_date", "")).strip()
        if not wall_date:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        return wall_date

    def _turn_timestamp(self, turn_context=None) -> str:
        temporal = self._require_turn_temporal(turn_context)
        wall_time = str(getattr(temporal, "wall_time", "")).strip()
        if not wall_time:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        return wall_time

    def emotional_momentum(self, recent_checkins: list[dict]) -> str:
        moods = [self.bot.normalize_mood(item.get("mood")) for item in recent_checkins[-6:]]
        latest_mood = moods[-1] if moods else "neutral"
        heavy_count = sum(1 for mood in moods if mood in {"sad", "stressed", "frustrated", "tired"})
        positive_count = sum(1 for mood in moods if mood == "positive")
        recent_pair = moods[-2:]

        if recent_pair and all(mood == "positive" for mood in recent_pair):
            return "warming"
        if latest_mood == "positive":
            return "steady" if heavy_count >= positive_count else "warming"

        if latest_mood in {"sad", "stressed", "frustrated", "tired"} and heavy_count >= 3 and heavy_count > positive_count:
            return "heavy"
        if positive_count >= 2 and positive_count >= heavy_count:
            return "warming"
        return "steady"

    def relationship_emotional_momentum(self, recent_checkins: list[dict]) -> str:
        return self.emotional_momentum(recent_checkins)

    def top_topics(self, state: dict | None = None, limit: int = 3) -> list[str]:
        snapshot = state or self.current_state()
        recurring_topics = {
            topic: count
            for topic, count in snapshot.get("recurring_topics", {}).items()
            if topic and topic != "general"
        }
        ranked = sorted(recurring_topics.items(), key=lambda item: (-item[1], item[0]))
        return [topic for topic, count in ranked[:limit] if count > 0]

    def top_relationship_topics(self, state: dict | None = None, limit: int = 3) -> list[str]:
        return self.top_topics(state=state, limit=limit)

    @staticmethod
    def level_label(score: int) -> str:
        if score >= 75:
            return "strong"
        if score >= 60:
            return "growing"
        if score >= 45:
            return "steady"
        return "guarded"

    def hypotheses(self, state: dict | None = None, limit: int | None = None) -> list[dict]:
        snapshot = state or self.current_state()
        hypotheses = [dict(entry) for entry in snapshot.get("hypotheses", []) if isinstance(entry, dict)]
        hypotheses.sort(key=lambda entry: (-float(entry.get("probability", 0.0) or 0.0), str(entry.get("name", ""))))
        if limit is None:
            return hypotheses
        return hypotheses[: max(1, int(limit or 1))]

    def relationship_hypotheses(self, state: dict | None = None, limit: int | None = None) -> list[dict]:
        return self.hypotheses(state=state, limit=limit)

    # ------------------------------------------------------------------
    # build_hypothesis_posteriors pipeline stages
    # ------------------------------------------------------------------

    def _load_hypothesis_priors(self, state: dict) -> tuple[dict, dict]:
        """Load hypothesis profiles and prior probabilities from current state."""
        profiles = self.bot.relationship_hypothesis_profiles()
        priors = {entry["name"]: float(entry.get("probability", 0.0) or 0.0) for entry in self.hypotheses(state)}
        if not priors:
            priors = {entry["name"]: float(entry["probability"]) for entry in self.bot.default_relationship_hypotheses()}
        return profiles, priors

    def _extract_turn_signals(self, state: dict, user_input: str, current_mood: str) -> dict:
        """Extract all observable signals from the current turn for evidence weighting."""
        _VULNERABILITY_MARKERS = {
            "feel", "feeling", "worried", "anxious", "overwhelmed", "sad", "lonely", "hurt",
            "struggling", "struggle", "afraid", "scared", "drained", "tired",
        }
        _CONNECTION_MARKERS = ["thank you", "thanks", "appreciate", "love you", "needed that", "that helped"]
        _GUARDED_MARKERS = ["leave me alone", "stop", "whatever", "fine", "forget it", "dont want to talk", "don't want to talk"]
        _HEAVY_MOODS = {"sad", "stressed", "frustrated", "tired"}

        lowered = str(user_input or "").strip().lower()
        tokens = self.bot.tokenize(user_input)
        recent_moods = [self.bot.normalize_mood(item.get("mood")) for item in list(state.get("recent_checkins", []))[-6:]]
        return {
            "lowered": lowered,
            "tokens": tokens,
            "has_vulnerability": bool(_VULNERABILITY_MARKERS & tokens),
            "has_connection": any(marker in lowered for marker in _CONNECTION_MARKERS),
            "has_guarded": any(marker in lowered for marker in _GUARDED_MARKERS),
            "is_heavy_mood": current_mood in _HEAVY_MOODS,
            "is_positive_mood": current_mood == "positive",
            "is_neutral_or_stressed": current_mood in {"neutral", "stressed"},
            "heavy_recent": sum(1 for mood in recent_moods if mood in _HEAVY_MOODS),
            "positive_recent": sum(1 for mood in recent_moods if mood == "positive"),
            "average_level": (int(state.get("trust_level", 50)) + int(state.get("openness_level", 50))) / 2.0,
            "is_heavy_momentum": str(state.get("emotional_momentum") or "steady") == "heavy",
        }

    def _build_evidence_weights(self, profiles: dict, signals: dict, state: dict) -> dict:
        """Compute per-hypothesis likelihood multipliers from extracted turn signals."""
        evidence = {name: 1.0 for name in profiles}
        evidence["supportive_baseline"] *= 1.05

        if signals["average_level"] >= 65:
            evidence["supportive_baseline"] *= 1.35
        elif signals["average_level"] <= 42:
            evidence["guarded_distance"] *= 1.40

        if signals["has_vulnerability"]:
            evidence["acute_stress"] *= 2.40
            evidence["supportive_baseline"] *= 1.15
        if signals["is_heavy_mood"]:
            evidence["acute_stress"] *= 2.00
        if signals["is_positive_mood"]:
            evidence["positive_rebound"] *= 1.90
            evidence["supportive_baseline"] *= 1.20
        if signals["has_connection"]:
            evidence["supportive_baseline"] *= 1.80
        if signals["has_guarded"]:
            evidence["guarded_distance"] *= 2.30
            evidence["supportive_baseline"] *= 0.75
        if signals["is_heavy_momentum"]:
            evidence["acute_stress"] *= 1.30
            evidence["guarded_distance"] *= 1.10
        if signals["is_positive_mood"] and signals["heavy_recent"] >= 2:
            evidence["positive_rebound"] *= 2.10
        if signals["heavy_recent"] >= 3 and signals["is_neutral_or_stressed"]:
            evidence["acute_stress"] *= 1.25
        if signals["positive_recent"] >= 2 and signals["is_positive_mood"]:
            evidence["positive_rebound"] *= 1.20

        # Continuity bias: nudge the currently-leading hypothesis
        current_hypotheses = self.hypotheses(state)
        if current_hypotheses:
            leading_name = str(current_hypotheses[0].get("name") or "").strip().lower()
            if leading_name in evidence:
                evidence[leading_name] *= 1.08

        return evidence

    def _compute_raw_posteriors(self, profiles: dict, priors: dict, evidence: dict) -> list[dict]:
        """Apply Bayes update with 25/75 smoothing toward the prior."""
        raw_total = sum(max(0.01, priors.get(name, 0.01)) * evidence[name] for name in profiles)
        if raw_total <= 0:
            raw_total = float(len(profiles))
        entries = []
        for name, profile in profiles.items():
            prior = max(0.01, priors.get(name, 0.01))
            posterior = (prior * evidence[name]) / raw_total
            smoothed = prior * 0.25 + posterior * 0.75
            entries.append({
                "name": name,
                "label": profile["label"],
                "summary": profile["summary"],
                "probability": smoothed,
            })
        return entries

    def _normalize_and_rank_posteriors(self, entries: list[dict]) -> list[dict]:
        """Normalize probabilities to sum to 1.0 and sort descending."""
        total = sum(entry["probability"] for entry in entries) or 1.0
        for entry in entries:
            entry["probability"] = round(entry["probability"] / total, 4)
        entries.sort(key=lambda entry: (-entry["probability"], entry["name"]))
        return entries

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def build_hypothesis_posteriors(self, state: dict, user_input: str, current_mood: str) -> list[dict]:
        profiles, priors = self._load_hypothesis_priors(state)
        signals = self._extract_turn_signals(state, user_input, current_mood)
        evidence = self._build_evidence_weights(profiles, signals, state)
        entries = self._compute_raw_posteriors(profiles, priors, evidence)
        return self._normalize_and_rank_posteriors(entries)

    def snapshot(self) -> dict:
        state = self.current_state()
        hypotheses = self.hypotheses(state, limit=3)
        active_hypothesis = hypotheses[0] if hypotheses else {"name": "supportive_baseline", "label": "Supportive Baseline", "probability": 1.0}
        return {
            "trust_level": state["trust_level"],
            "trust_label": self.level_label(state["trust_level"]),
            "openness_level": state["openness_level"],
            "openness_label": self.level_label(state["openness_level"]),
            "emotional_momentum": state["emotional_momentum"],
            "active_hypothesis": active_hypothesis.get("name", "supportive_baseline"),
            "active_hypothesis_label": active_hypothesis.get("label", "Supportive Baseline"),
            "active_hypothesis_probability": round(float(active_hypothesis.get("probability", 0.0) or 0.0), 3),
            "hypotheses": [
                {
                    "name": entry.get("name", ""),
                    "label": entry.get("label", ""),
                    "probability": round(float(entry.get("probability", 0.0) or 0.0), 3),
                }
                for entry in hypotheses
            ],
            "top_topics": self.top_topics(state),
            "last_reflection": state.get("last_reflection", ""),
            "last_updated": state.get("last_updated"),
        }

    def relationship_snapshot(self) -> dict:
        return self.snapshot()

    def record_history_point(self, *, trust_level, openness_level, source="turn", turn_context=None) -> dict:
        return {
            "recorded_at": self.bot.runtime_timestamp(),
            "trust_level": self.bot.clamp_score(trust_level),
            "openness_level": self.bot.clamp_score(openness_level),
            "source": str(source or "turn").strip().lower() or "turn",
            "projection_only": True,
        }

    def build_prompt_context(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "Relationship state with Tony:",
            f"- Trust is {snapshot['trust_label']} ({snapshot['trust_level']}/100).",
            f"- Openness is {snapshot['openness_label']} ({snapshot['openness_level']}/100).",
            f"- Active working theory: {snapshot['active_hypothesis_label']} ({snapshot['active_hypothesis_probability']:.2f}).",
        ]

        momentum = snapshot["emotional_momentum"]
        if momentum == "heavy":
            lines.append("- Recent conversations have carried heavier emotions, so lead with presence before solutions.")
        elif momentum == "warming":
            lines.append("- Tony has shown a bit more positive energy lately, so warmth and light pride can land well.")
        else:
            lines.append("- The relationship tone has been steady lately.")

        top_topics = snapshot.get("top_topics", [])
        if top_topics:
            lines.append(f"- Recurring themes lately: {self.bot.natural_list(top_topics)}.")

        if snapshot["openness_level"] >= 65:
            lines.append("- Tony has been opening up more lately; acknowledge that gently without making it sound clinical.")
        elif snapshot["openness_level"] <= 40:
            lines.append("- Give Tony room and keep the tone easy to approach.")

        return "\n".join(lines)

    def build_reflection_prompt(self, current_state: dict, recent_messages: list[dict]) -> str:
        transcript = self.bot.transcript_from_messages(recent_messages)
        summary_section = self.bot.session_summary or "No session summary yet."
        return f"""
You are reviewing how the relationship between Tony and Dad is evolving during this chat.

Current relationship state:
- trust_level: {current_state['trust_level']}
- openness_level: {current_state['openness_level']}
- emotional_momentum: {current_state['emotional_momentum']}

Rolling session summary:
{summary_section}

Recent conversation:
{transcript}

Return only JSON with these keys:
- trust_delta: integer from -2 to 2
- openness_delta: integer from -2 to 2
- emotional_momentum: one of steady, warming, heavy
- reflection: one short sentence

Rules:
- Make small adjustments only.
- Base them on Tony's openness, warmth, strain, or relief across the conversation.
- Do not overwrite with extreme values.
""".strip()

    def build_relationship_reflection_prompt(self, current_state: dict, recent_messages: list[dict]) -> str:
        return self.build_reflection_prompt(current_state, recent_messages)

    def reflect_read_only(self, turn_context=None) -> dict | None:
        _ = turn_context
        return self.current_state()


__all__ = ["RelationshipManager"]
