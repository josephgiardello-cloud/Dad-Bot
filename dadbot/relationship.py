from __future__ import annotations

import json
import logging
from datetime import date

from dadbot.contracts import DadBotContext, SupportsRelationshipRuntime


logger = logging.getLogger(__name__)


class RelationshipManager:
    """Owns relationship state, prompt-facing snapshots, and reflection/update logic."""

    def __init__(self, bot: DadBotContext | SupportsRelationshipRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def current_state(self) -> dict:
        return self.bot.memory_manager.relationship_state()

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

    def build_hypothesis_posteriors(self, state: dict, user_input: str, current_mood: str) -> list[dict]:
        profiles = self.bot.relationship_hypothesis_profiles()
        priors = {entry["name"]: float(entry.get("probability", 0.0) or 0.0) for entry in self.hypotheses(state)}
        if not priors:
            priors = {entry["name"]: float(entry["probability"]) for entry in self.bot.default_relationship_hypotheses()}

        lowered = str(user_input or "").strip().lower()
        tokens = self.bot.tokenize(user_input)
        vulnerability_markers = {
            "feel", "feeling", "worried", "anxious", "overwhelmed", "sad", "lonely", "hurt",
            "struggling", "struggle", "afraid", "scared", "drained", "tired",
        }
        connection_markers = ["thank you", "thanks", "appreciate", "love you", "needed that", "that helped"]
        guarded_markers = ["leave me alone", "stop", "whatever", "fine", "forget it", "dont want to talk", "don't want to talk"]
        heavy_moods = {"sad", "stressed", "frustrated", "tired"}
        recent_moods = [self.bot.normalize_mood(item.get("mood")) for item in list(state.get("recent_checkins", []))[-6:]]
        heavy_recent = sum(1 for mood in recent_moods if mood in heavy_moods)
        positive_recent = sum(1 for mood in recent_moods if mood == "positive")
        average_level = (int(state.get("trust_level", 50)) + int(state.get("openness_level", 50))) / 2.0

        evidence = {name: 1.0 for name in profiles}
        evidence["supportive_baseline"] *= 1.05
        if average_level >= 65:
            evidence["supportive_baseline"] *= 1.35
        elif average_level <= 42:
            evidence["guarded_distance"] *= 1.40

        if vulnerability_markers & tokens:
            evidence["acute_stress"] *= 2.40
            evidence["supportive_baseline"] *= 1.15

        if current_mood in heavy_moods:
            evidence["acute_stress"] *= 2.00
        if current_mood == "positive":
            evidence["positive_rebound"] *= 1.90
            evidence["supportive_baseline"] *= 1.20

        if any(marker in lowered for marker in connection_markers):
            evidence["supportive_baseline"] *= 1.80
        if any(marker in lowered for marker in guarded_markers):
            evidence["guarded_distance"] *= 2.30
            evidence["supportive_baseline"] *= 0.75

        if str(state.get("emotional_momentum") or "steady") == "heavy":
            evidence["acute_stress"] *= 1.30
            evidence["guarded_distance"] *= 1.10

        if current_mood == "positive" and heavy_recent >= 2:
            evidence["positive_rebound"] *= 2.10
        if heavy_recent >= 3 and current_mood in {"neutral", "stressed"}:
            evidence["acute_stress"] *= 1.25
        if positive_recent >= 2 and current_mood == "positive":
            evidence["positive_rebound"] *= 1.20

        current_hypotheses = self.hypotheses(state)
        if current_hypotheses:
            leading_name = str(current_hypotheses[0].get("name") or "").strip().lower()
            if leading_name in evidence:
                evidence[leading_name] *= 1.08

        posterior_entries = []
        raw_total = sum(max(0.01, priors.get(name, 0.01)) * evidence[name] for name in profiles)
        if raw_total <= 0:
            raw_total = float(len(profiles))

        for name, profile in profiles.items():
            prior = max(0.01, priors.get(name, 0.01))
            posterior = (prior * evidence[name]) / raw_total
            smoothed = prior * 0.25 + posterior * 0.75
            posterior_entries.append(
                {
                    "name": name,
                    "label": profile["label"],
                    "summary": profile["summary"],
                    "probability": smoothed,
                }
            )

        total_probability = sum(entry["probability"] for entry in posterior_entries) or 1.0
        for entry in posterior_entries:
            entry["probability"] = round(entry["probability"] / total_probability, 4)

        posterior_entries.sort(key=lambda entry: (-entry["probability"], entry["name"]))
        return posterior_entries

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

    def reflect(self, force: bool = False) -> dict | None:
        turn_count = self.bot.session_turn_count()
        if turn_count < 3:
            return None
        if not force and turn_count - self.bot.last_relationship_reflection_turn < self.bot.RELATIONSHIP_REFLECTION_INTERVAL:
            return None

        recent_messages = self.bot.prompt_history()
        if not recent_messages:
            return None

        current_state = self.current_state()
        prompt = self.build_reflection_prompt(current_state, recent_messages)

        try:
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
                response_format="json",
                purpose="relationship reflection",
            )
            content = self.bot.extract_ollama_message_content(response)
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue("relationship reflection", "keeping the previous relationship state", exc)
            return None

        try:
            reflection = self.bot.parse_model_json_content(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Relationship reflection returned invalid JSON: %s", exc)
            return None

        if not isinstance(reflection, dict):
            logger.warning("Relationship reflection returned non-dict payload: %r", reflection)
            return None

        try:
            trust_delta = max(-2, min(2, int(reflection.get("trust_delta", 0))))
            openness_delta = max(-2, min(2, int(reflection.get("openness_delta", 0))))
        except (TypeError, ValueError) as exc:
            logger.warning("Relationship reflection returned invalid deltas: %r (%s)", reflection, exc)
            return None

        momentum = str(reflection.get("emotional_momentum") or current_state["emotional_momentum"]).strip().lower()
        if momentum not in {"steady", "warming", "heavy"}:
            momentum = current_state["emotional_momentum"]

        current_state["trust_level"] = self.bot.clamp_score(current_state["trust_level"] + trust_delta)
        current_state["openness_level"] = self.bot.clamp_score(current_state["openness_level"] + openness_delta)
        current_state["emotional_momentum"] = momentum
        current_state["last_reflection"] = str(reflection.get("reflection") or "").strip()
        current_state["last_updated"] = date.today().isoformat()
        self.bot.last_relationship_reflection_turn = turn_count
        self.bot.mutate_memory_store(relationship_state=current_state)
        self.bot.record_relationship_history_point(
            trust_level=current_state["trust_level"],
            openness_level=current_state["openness_level"],
            source="reflection",
        )
        self.bot.update_trait_impact_from_relationship_feedback(trust_delta, openness_delta)
        return current_state

    def reflect_relationship_state(self, force: bool = False) -> dict | None:
        return self.reflect(force=force)

    def apply_feedback(self, feedback_kind: str) -> dict:
        state = self.current_state()
        key = str(feedback_kind or "").strip().lower()
        if key in {"supportive", "more_supportive", "warm"}:
            trust_delta, openness_delta = 2, 2
        elif key in {"distant", "less_supportive", "cold"}:
            trust_delta, openness_delta = -2, -2
        else:
            trust_delta, openness_delta = 0, 0

        state["trust_level"] = self.bot.clamp_score(state.get("trust_level", 50) + trust_delta)
        state["openness_level"] = self.bot.clamp_score(state.get("openness_level", 50) + openness_delta)
        state["last_updated"] = date.today().isoformat()
        self.bot.mutate_memory_store(relationship_state=state)
        self.bot.record_relationship_history_point(
            trust_level=state["trust_level"],
            openness_level=state["openness_level"],
            source=f"feedback:{key or 'unknown'}",
        )
        self.bot.update_trait_impact_from_relationship_feedback(trust_delta, openness_delta)
        return state

    def apply_relationship_feedback(self, feedback_kind: str) -> dict:
        return self.apply_feedback(feedback_kind)

    def update_relationship_state(self, user_input: str, current_mood: str) -> dict:
        return self.update(user_input, current_mood)

    def update(self, user_input: str, current_mood: str) -> dict:
        state = self.current_state()
        lowered = str(user_input or "").lower()
        tokens = self.bot.tokenize(user_input)

        vulnerability_markers = {
            "feel", "feeling", "worried", "anxious", "overwhelmed", "sad", "lonely", "hurt",
            "struggling", "struggle", "afraid", "scared", "drained", "tired",
        }
        connection_markers = ["thank you", "thanks", "appreciate", "love you", "needed that", "that helped"]

        openness_delta = 0
        trust_delta = 0

        if len(tokens) >= 12:
            openness_delta += 1
        if vulnerability_markers & tokens:
            openness_delta += 2
            trust_delta += 1
        if current_mood in {"sad", "stressed", "frustrated", "tired"}:
            openness_delta += 2
        if current_mood == "positive":
            trust_delta += 1
        if any(marker in lowered for marker in connection_markers):
            trust_delta += 2
            openness_delta += 1

        inferred_topic = self.bot.infer_memory_category(user_input)
        if inferred_topic == "general":
            topic_matches = self.bot.profile_context.matching_topics(user_input)
            if topic_matches:
                inferred_topic = topic_matches[0]["name"].replace("_", " ")

        recurring_topics = dict(state.get("recurring_topics", {}))
        recurring_topics[inferred_topic] = recurring_topics.get(inferred_topic, 0) + 1

        recent_checkins = list(state.get("recent_checkins", []))
        recent_checkins.append(
            {
                "date": date.today().isoformat(),
                "mood": self.bot.normalize_mood(current_mood),
                "topic": inferred_topic,
            }
        )

        state["trust_level"] = self.bot.clamp_score(state.get("trust_level", 50) + trust_delta)
        state["openness_level"] = self.bot.clamp_score(state.get("openness_level", 50) + openness_delta)
        state["recurring_topics"] = recurring_topics
        state["recent_checkins"] = recent_checkins[-24:]
        state["emotional_momentum"] = self.emotional_momentum(state["recent_checkins"])
        hypotheses = self.build_hypothesis_posteriors(state, user_input, current_mood)
        state["hypotheses"] = hypotheses
        state["active_hypothesis"] = hypotheses[0]["name"] if hypotheses else "supportive_baseline"
        state["last_hypothesis_updated"] = date.today().isoformat()
        state["last_updated"] = date.today().isoformat()
        self.bot.mutate_memory_store(relationship_state=state)
        self.bot.record_relationship_history_point(
            trust_level=state["trust_level"],
            openness_level=state["openness_level"],
            source="turn",
        )
        return state


__all__ = ["RelationshipManager"]