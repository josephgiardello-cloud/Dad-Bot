from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import date, datetime, timedelta

from dadbot.prompts import DadPrompts

logger = logging.getLogger(__name__)


class LongTermSignalsManager:
    """Owns persona evolution, wisdom generation, life-pattern detection, and family echo behavior."""

    def __init__(self, bot):
        self.bot = bot

    def should_evolve_persona(self, force=False):
        if force:
            return True

        cadence = self.bot.cadence_settings()
        archive_count = len(self.bot.session_archive())
        if archive_count < cadence["persona_evolution_min_sessions"]:
            return False

        history = self.bot.persona_evolution_history()
        last_session_count = history[-1].get("session_count", 0) if history else 0
        return archive_count - last_session_count >= cadence["persona_evolution_session_gap"]

    @staticmethod
    def trait_strength(entry):
        try:
            return max(0.25, min(3.0, float(entry.get("strength", 1.0))))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def trait_impact(entry):
        try:
            return float(entry.get("impact_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def critique_score(entry):
        try:
            return max(0, min(10, int(entry.get("critique_score", 0))))
        except (TypeError, ValueError):
            return 0

    def decayed_trait_strength(self, entry):
        strength = self.trait_strength(entry)
        anchor = str(
            entry.get("last_reinforced_at") or entry.get("applied_at") or "",
        ).strip()
        elapsed_days = self.bot.days_since_iso_date(anchor)
        if elapsed_days is None or elapsed_days <= 0:
            return round(strength, 2)
        decayed = strength * (0.985 ** min(elapsed_days, 180))
        return round(max(0.25, decayed), 2)

    def trait_similarity(self, left_trait, right_trait):
        left = self.bot.normalize_memory_text(left_trait)
        right = self.bot.normalize_memory_text(right_trait)
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        left_tokens = self.bot.tokenize(left)
        right_tokens = self.bot.tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

    def trait_priority(self, entry):
        return round(
            self.decayed_trait_strength(entry) + self.trait_impact(entry) * 0.35 + self.critique_score(entry) * 0.05,
            4,
        )

    def active_persona_trait_entries(self, limit=3):
        history = self.bot.persona_evolution_history()
        ranked = sorted(
            history,
            key=lambda entry: (
                -self.trait_priority(entry),
                -self.trait_impact(entry),
                str(entry.get("applied_at", "")),
            ),
        )
        return ranked[: max(1, int(limit or 3))]

    def most_positive_persona_traits(self, limit=3):
        positive_entries = [
            entry
            for entry in self.active_persona_trait_entries(limit=max(3, limit or 3) * 2)
            if self.trait_impact(entry) > 0
        ]
        return [entry.get("trait", "") for entry in positive_entries[: max(1, int(limit or 3))] if entry.get("trait")]

    def recent_mood_trend(self):
        if not self.bot.session_moods:
            return "neutral"
        return Counter(
            [mood for mood in self.bot.session_moods[-10:] if mood],
        ).most_common(1)[0][0]

    def build_persona_evolution_prompt(self):
        relationship_state = self.bot.relationship_state()
        top_topics = ", ".join(self.bot.relationship.top_topics(limit=5)) or "none"
        current_traits = ", ".join(self.bot.profile_runtime.evolved_persona_traits()) or "None"
        positive_traits = ", ".join(self.most_positive_persona_traits(limit=3)) or "None yet"
        return DadPrompts.persona_evolution(
            relationship_state=relationship_state,
            top_topics=top_topics,
            current_traits=current_traits,
            positive_traits=positive_traits,
            timeline=self.bot.relationship_timeline(),
            mood_trend=self.recent_mood_trend(),
        )

    def critique_evolved_trait(self, trait, reason):
        prompt = f"""You are reviewing a proposed permanent evolution for Dad's persona.

Proposed trait: {trait}
Reason: {reason}

Existing evolved traits: {", ".join(self.bot.profile_runtime.evolved_persona_traits()) or "None"}

Score this trait 1-10 on (be STRICT):
- Quality (specific, useful, in-character?)
- Novelty (different from existing traits?)
- Stability risk (will it cause drift?)

Good: "more patient with my mistakes", "uses my full name when proud", "asks about my day before advice"
Bad: "more encouraging", "slightly warmer", "more supportive"

Return only JSON:
{{
	"score": int,
	"approved": true/false,
	"feedback": "one short sentence",
	"suggested_refinement": "string or null"
}}
"""
        try:
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                response_format="json",
                purpose="trait critique",
            )
            critique = self.bot.parse_model_json_content(response["message"]["content"])
            if not isinstance(critique, dict):
                raise TypeError("Trait critique returned non-dict payload")
            return critique
        except Exception:
            return {
                "score": 5,
                "approved": True,
                "feedback": "critique failed",
                "suggested_refinement": None,
            }

    def consolidate_persona_evolution_history(self, history=None, persist=False):
        source_history = history if history is not None else self.bot.persona_evolution_history()
        normalized_history = []
        for entry in source_history:
            normalized = self.bot.normalize_persona_evolution_entry(entry)
            if normalized is None:
                continue
            normalized["strength"] = self.decayed_trait_strength(normalized)
            normalized_history.append(normalized)

        if not normalized_history:
            if persist:
                self.bot.mutate_memory_store(persona_evolution=[])
            return []

        ranked_history = sorted(
            normalized_history,
            key=lambda entry: (
                -self.trait_priority(entry),
                -self.trait_impact(entry),
                str(entry.get("applied_at", "")),
            ),
        )

        ranked_history = [
            entry
            for entry in ranked_history
            if self.trait_impact(entry) >= -0.5 or self.bot.days_since_iso_date(entry.get("applied_at")) <= 30
        ]

        merged = []
        for entry in ranked_history:
            match = next(
                (
                    existing
                    for existing in merged
                    if self.trait_similarity(
                        existing.get("trait", ""),
                        entry.get("trait", ""),
                    )
                    >= 0.72
                ),
                None,
            )
            if match is None:
                merged.append(dict(entry))
                continue

            if len(str(entry.get("trait", ""))) > len(str(match.get("trait", ""))):
                match["trait"] = entry.get("trait", "")
            if len(str(entry.get("reason", ""))) > len(str(match.get("reason", ""))):
                match["reason"] = entry.get("reason", "")
            if not match.get("announcement") and entry.get("announcement"):
                match["announcement"] = entry["announcement"]
            if len(str(entry.get("critique_feedback", ""))) > len(
                str(match.get("critique_feedback", "")),
            ):
                match["critique_feedback"] = entry.get("critique_feedback", "")
            match["session_count"] = max(
                int(match.get("session_count", 0)),
                int(entry.get("session_count", 0)),
            )
            match["strength"] = round(
                min(
                    3.0,
                    max(self.trait_strength(match), self.trait_strength(entry)) + 0.1,
                ),
                2,
            )
            match["impact_score"] = round(
                max(self.trait_impact(match), self.trait_impact(entry)),
                2,
            )
            match["critique_score"] = max(
                self.critique_score(match),
                self.critique_score(entry),
            )
            match["applied_at"] = max(
                str(match.get("applied_at", "")),
                str(entry.get("applied_at", "")),
            )
            match["last_reinforced_at"] = max(
                str(match.get("last_reinforced_at", "")),
                str(entry.get("last_reinforced_at", "")),
            )

        max_traits = min(
            8,
            int(self.bot.runtime_config.store_limits.get("persona_evolution", 8)),
        )
        trimmed = sorted(
            merged,
            key=lambda entry: (
                -self.trait_priority(entry),
                -self.trait_impact(entry),
                str(entry.get("applied_at", "")),
            ),
        )[:max_traits]
        trimmed.sort(key=lambda entry: str(entry.get("applied_at", "")))

        if persist:
            self.bot.mutate_memory_store(persona_evolution=trimmed)
        return trimmed

    @staticmethod
    def persona_announcement(trait, reason):
        trait_text = str(trait or "").strip()
        reason_text = str(reason or "").strip().rstrip(".")
        if not trait_text:
            return ""
        if reason_text:
            return f"You know, buddy... you've been opening up more lately. I'm finding myself a little more {trait_text} with you. {reason_text}."
        return f"You know, buddy... over time I'm finding myself a little more {trait_text} with you."

    def evolve_persona(self, force=False):
        if not self.should_evolve_persona(force=force):
            return None

        try:
            response = self.bot.call_ollama_chat(
                messages=[
                    {"role": "user", "content": self.build_persona_evolution_prompt()},
                ],
                options={"temperature": 0.2},
                response_format="json",
                purpose="persona evolution",
            )
            content = response["message"]["content"]
        except (RuntimeError, KeyError, TypeError) as exc:
            logger.warning("Persona evolution request failed: %s", exc)
            return None

        try:
            parsed = self.bot.parse_model_json_content(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Persona evolution returned invalid JSON: %s", exc)
            return None

        if not isinstance(parsed, dict):
            logger.warning("Persona evolution returned non-dict payload: %r", parsed)
            return None

        trait = str(parsed.get("new_trait") or "").strip()
        reason = str(parsed.get("reason") or "").strip()
        if not trait:
            return None

        critique = self.critique_evolved_trait(trait, reason)
        if not critique.get("approved", True) or int(critique.get("score", 0) or 0) < 7:
            logger.info("Rejected weak trait evolution: %s", trait)
            return None

        suggested_refinement = str(critique.get("suggested_refinement") or "").strip()
        if suggested_refinement:
            trait = suggested_refinement

        if any(
            self.trait_similarity(existing.get("trait", ""), trait) >= 0.72
            for existing in self.bot.persona_evolution_history()
        ):
            return None

        applied_at = datetime.now().isoformat(timespec="seconds")
        entry = self.bot.normalize_persona_evolution_entry(
            {
                "trait": trait,
                "reason": reason,
                "announcement": self.persona_announcement(trait, reason),
                "session_count": len(self.bot.session_archive()),
                "applied_at": applied_at,
                "last_reinforced_at": applied_at,
                "strength": 1.0,
                "impact_score": 0.0,
                "critique_score": int(critique.get("score", 0) or 0),
                "critique_feedback": str(critique.get("feedback") or "").strip(),
            },
        )
        if entry is None:
            return None

        history = self.bot.persona_evolution_history()
        history.append(entry)
        consolidated = self.consolidate_persona_evolution_history(history)
        self.bot.mutate_memory_store(persona_evolution=consolidated)
        self.bot.mark_memory_graph_dirty()
        if entry.get("announcement"):
            self.bot.queue_proactive_message(
                entry["announcement"],
                source="persona-evolution",
            )
        return entry

    def reject_persona_trait(self, trait_query=""):
        history = [dict(entry) for entry in self.bot.persona_evolution_history()]
        if not history:
            return None

        query = str(trait_query or "").strip()
        target_index = None
        if not query or query.lower() == "trait":
            target_index = len(history) - 1
        else:
            normalized_query = self.bot.normalize_memory_text(query)
            for index in range(len(history) - 1, -1, -1):
                trait = str(history[index].get("trait", ""))
                if (
                    normalized_query in self.bot.normalize_memory_text(trait)
                    or self.trait_similarity(trait, query) >= 0.72
                ):
                    target_index = index
                    break

        if target_index is None:
            return None

        removed = history.pop(target_index)
        self.bot.mutate_memory_store(
            persona_evolution=self.consolidate_persona_evolution_history(history),
        )
        self.bot.mark_memory_graph_dirty()
        return removed

    def update_trait_impact_from_relationship_feedback(
        self,
        trust_delta,
        openness_delta,
    ):
        history = [dict(entry) for entry in self.bot.persona_evolution_history()]
        if not history:
            return []

        total_delta = int(trust_delta or 0) + int(openness_delta or 0)
        refreshed = []
        for entry in history:
            normalized = self.bot.normalize_persona_evolution_entry(entry)
            if normalized is None:
                continue
            normalized["strength"] = self.decayed_trait_strength(normalized)
            refreshed.append(normalized)

        if not refreshed:
            return []

        latest_entry = max(
            refreshed,
            key=lambda entry: str(entry.get("applied_at", "")),
        )
        latest_entry["impact_score"] = round(
            self.trait_impact(latest_entry) + total_delta,
            2,
        )
        if total_delta > 0:
            latest_entry["strength"] = round(
                min(3.0, self.trait_strength(latest_entry) + 0.12 * total_delta),
                2,
            )
            latest_entry["last_reinforced_at"] = datetime.now().isoformat(
                timespec="seconds",
            )
        elif total_delta < 0:
            latest_entry["strength"] = round(
                max(0.25, self.trait_strength(latest_entry) + 0.08 * total_delta),
                2,
            )

        consolidated = self.consolidate_persona_evolution_history(refreshed)
        self.bot.mutate_memory_store(persona_evolution=consolidated)
        self.bot.mark_memory_graph_dirty()
        return consolidated

    @staticmethod
    def accumulate_memory_graph_node(
        node_weights,
        node_types,
        label,
        node_type,
        weight=1,
    ):
        normalized_label = str(label).strip().lower()
        if not normalized_label:
            return
        node_weights[normalized_label] = node_weights.get(normalized_label, 0) + weight
        node_types[normalized_label] = node_type

    @staticmethod
    def accumulate_memory_graph_edge(edge_weights, left, right, weight=1):
        left_label = str(left).strip().lower()
        right_label = str(right).strip().lower()
        if not left_label or not right_label or left_label == right_label:
            return
        edge_key = tuple(sorted((left_label, right_label)))
        edge_weights[edge_key] = edge_weights.get(edge_key, 0) + weight

    def add_memory_nodes_to_graph(self, node_weights, node_types, edge_weights):
        for memory in self.bot.MEMORY_STORE.get("memories", []):
            category = str(memory.get("category", "general")).strip().lower()
            mood = self.bot.normalize_mood(memory.get("mood"))
            weight = self.bot.recency_weight(
                memory.get("updated_at") or memory.get("created_at"),
            )
            if category and category != "general":
                self.accumulate_memory_graph_node(
                    node_weights,
                    node_types,
                    category,
                    "category",
                    weight,
                )
            if mood != "neutral":
                self.accumulate_memory_graph_node(
                    node_weights,
                    node_types,
                    mood,
                    "mood",
                    weight,
                )
            if category and category != "general" and mood != "neutral":
                self.accumulate_memory_graph_edge(edge_weights, category, mood, weight)

    def add_relationship_topics_to_graph(self, node_weights, node_types):
        for topic, count in self.bot.relationship_state().get("recurring_topics", {}).items():
            topic_name = str(topic).strip().lower()
            if topic_name and topic_name != "general":
                self.accumulate_memory_graph_node(
                    node_weights,
                    node_types,
                    topic_name,
                    "topic",
                    max(1, int(count)),
                )

    def add_archive_nodes_to_graph(self, node_weights, node_types, edge_weights):
        for entry in self.bot.session_archive():
            mood = self.bot.normalize_mood(entry.get("dominant_mood"))
            weight = self.bot.recency_weight(entry.get("created_at"))
            if mood != "neutral":
                self.accumulate_memory_graph_node(
                    node_weights,
                    node_types,
                    mood,
                    "mood",
                    weight,
                )
            for topic in entry.get("topics", []):
                self.accumulate_memory_graph_node(
                    node_weights,
                    node_types,
                    topic,
                    "topic",
                    weight,
                )
                if mood != "neutral":
                    self.accumulate_memory_graph_edge(edge_weights, topic, mood, weight)

    @staticmethod
    def build_memory_graph_nodes(node_weights, node_types):
        return [
            {
                "id": f"{node_types[label]}:{label}",
                "label": label,
                "type": node_types[label],
                "weight": weight,
            }
            for label, weight in sorted(
                node_weights.items(),
                key=lambda item: (-item[1], item[0]),
            )[:18]
        ]

    @staticmethod
    def build_memory_graph_edges(edge_weights):
        return [
            {
                "source": left,
                "target": right,
                "weight": weight,
            }
            for (left, right), weight in sorted(
                edge_weights.items(),
                key=lambda item: (-item[1], item[0]),
            )[:18]
        ]

    def mark_memory_graph_dirty(self):
        self.bot._memory_graph_dirty = True

    def refresh_memory_graph(self, force=False):
        """Sync the durable graph store, then refresh the lightweight preview used by UI and diagnostics."""
        current_graph = self.bot.memory.memory_graph_snapshot()
        temporal_missing_msg = "temporalnode required"
        if not force:
            if not getattr(self.bot, "_memory_graph_dirty", False):
                return current_graph
            debounce_seconds = max(
                0,
                int(getattr(self.bot, "GRAPH_REFRESH_DEBOUNCE_SECONDS", 0) or 0),
            )
            last_refresh = float(
                getattr(self.bot, "_last_memory_graph_refresh_monotonic", 0.0) or 0.0,
            )
            if debounce_seconds and last_refresh and time.monotonic() - last_refresh < debounce_seconds:
                return current_graph

        with self.bot._graph_refresh_lock:
            current_graph = self.bot.memory.memory_graph_snapshot()
            if not force:
                if not getattr(self.bot, "_memory_graph_dirty", False):
                    return current_graph
                debounce_seconds = max(
                    0,
                    int(getattr(self.bot, "GRAPH_REFRESH_DEBOUNCE_SECONDS", 0) or 0),
                )
                last_refresh = float(
                    getattr(self.bot, "_last_memory_graph_refresh_monotonic", 0.0) or 0.0,
                )
                if debounce_seconds and last_refresh and time.monotonic() - last_refresh < debounce_seconds:
                    return current_graph

            try:
                snapshot = self.bot.sync_graph_store()
                graph = self.bot.memory_manager.preview_memory_graph(snapshot)
                self.bot.mutate_memory_store(memory_graph=graph)
            except Exception as exc:
                exc_message = str(exc or "").strip().lower()
                if temporal_missing_msg in exc_message:
                    # Background refreshes can run outside a turn context; skip noisy
                    # degradation logging when strict temporal context is unavailable.
                    return current_graph
                self.bot.record_runtime_issue(
                    "memory graph refresh",
                    "keeping the previous graph snapshot",
                    exc,
                )
                return current_graph

            self.bot._memory_graph_dirty = False
            self.bot._last_memory_graph_refresh_monotonic = time.monotonic()
            return graph

    def summarize_memory_graph(self):
        graph = self.bot.memory.memory_graph_snapshot()
        if not graph.get("nodes") and not graph.get("edges"):
            return "No strong graph links yet."

        node_text = ", ".join(f"{node['label']} ({node['weight']})" for node in graph.get("nodes", [])[:5]) or "none"
        edge_text = (
            "; ".join(
                f"{edge['source']} <-> {edge['target']} ({edge['weight']})" for edge in graph.get("edges", [])[:5]
            )
            or "none"
        )
        return f"Top nodes: {node_text}. Strongest links: {edge_text}."

    def should_generate_wisdom_insight(self, user_input, force=False):
        if force:
            return True

        cadence = self.bot.cadence_settings()
        if len(self.bot.session_archive()) < cadence["wisdom_min_archived_sessions"]:
            return False
        if not self.bot.memory.memory_graph_snapshot().get("edges"):
            return False

        turn_interval = cadence["wisdom_turn_interval"]
        return self.bot.session_turn_count() % turn_interval == 0 or bool(
            self.bot.significant_tokens(user_input) & set(self.bot.relationship.top_topics()),
        )

    def build_wisdom_prompt(self, user_input):
        consolidated_lines = (
            "\n".join(f"- {entry.get('summary', '')}" for entry in self.bot.consolidated_memories()[-5:])
            or "- None yet."
        )
        return DadPrompts.wisdom_prompt(
            self.summarize_memory_graph(),
            consolidated_lines,
            user_input,
        )

    def generate_wisdom_insight(self, user_input, force=False):
        if not self.should_generate_wisdom_insight(user_input, force=force):
            return None

        try:
            response = self.bot.call_ollama_chat(
                messages=[
                    {"role": "user", "content": self.build_wisdom_prompt(user_input)},
                ],
                options={"temperature": 0.3},
                response_format="json",
                purpose="wisdom insight",
            )
            content = self.bot.extract_ollama_message_content(response)
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue(
                "wisdom insight",
                "skipping wisdom generation for this turn",
                exc,
            )
            return None

        try:
            parsed = self.bot.parse_model_json_content(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Wisdom insight returned invalid JSON: %s", exc)
            return None

        if not isinstance(parsed, dict):
            logger.warning("Wisdom insight returned non-dict payload: %r", parsed)
            return None

        entry = self.bot.normalize_wisdom_entry(
            {
                "summary": parsed.get("summary"),
                "topic": parsed.get("topic"),
                "trigger": user_input,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        if entry is None:
            return None

        existing = self.bot.wisdom_catalog()
        recent_dedup_window = self.bot.runtime_config.window("recent_wisdom_dedup", 6)
        if any(
            self.bot.normalize_memory_text(item.get("summary", "")) == self.bot.normalize_memory_text(entry["summary"])
            for item in existing[-recent_dedup_window:]
        ):
            return next(
                (
                    item
                    for item in reversed(existing)
                    if self.bot.normalize_memory_text(item.get("summary", ""))
                    == self.bot.normalize_memory_text(entry["summary"])
                ),
                None,
            )

        existing.append(entry)
        self.bot.mutate_memory_store(
            wisdom_insights=self.bot.runtime_config.tail(existing, "wisdom_insights"),
        )
        return entry

    def relevant_wisdom_for_input(self, user_input):
        scored = []
        for entry in self.bot.wisdom_catalog():
            score = self.bot.memory_relevance_score(
                user_input,
                f"{entry.get('topic', '')} {entry.get('summary', '')}",
            )
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].get("created_at", "")))
        return [entry for _, entry in scored[:2]]

    def build_wisdom_context(self, user_input):
        wisdom_entries = self.relevant_wisdom_for_input(user_input)
        if not wisdom_entries:
            generated = self.generate_wisdom_insight(user_input)
            if generated is not None:
                wisdom_entries = [generated]

        if not wisdom_entries:
            return None

        lines = "\n".join(f"- [{entry.get('topic', 'general')}] {entry.get('summary', '')}" for entry in wisdom_entries)
        return (
            "Dad wisdom you've earned from the long view of this relationship:\n"
            f"{lines}\n"
            "If it fits naturally, weave one of these observations into the reply without sounding scripted."
        )

    def _derived_deep_pattern_documents(self):
        archive_window = self.bot.runtime_config.window(
            "deep_pattern_archive_window",
            18,
        )
        archive = self.bot.session_archive()[-archive_window:]
        if not archive:
            return []

        topic_mood_counter = Counter()
        latest_evidence = {}
        latest_timestamps = {}
        for entry in archive:
            topics = [topic for topic in entry.get("topics", []) if topic and topic != "general"]
            if not topics:
                continue
            mood = self.bot.normalize_mood(entry.get("dominant_mood"))
            summary = str(entry.get("summary") or "").strip()
            created_at = str(entry.get("created_at") or "").strip()
            for topic in topics:
                key = (topic, mood)
                topic_mood_counter[key] += 1
                if summary:
                    latest_evidence[key] = summary
                if created_at:
                    latest_timestamps[key] = created_at

        min_occurrences = max(
            2,
            int(
                self.bot.cadence_settings().get("deep_pattern_min_occurrences", 3) or 3,
            ),
        )
        documents = []
        for (topic, mood), count in topic_mood_counter.items():
            if count < min_occurrences:
                continue
            if mood == "neutral":
                summary = f"Across several chats, {topic} keeps surfacing as a steady theme in Tony's life."
            else:
                summary = f"Across several chats, {topic} keeps surfacing when Tony feels {mood}."
            documents.append(
                {
                    "source_type": "derived_arc",
                    "topic": topic,
                    "mood": mood,
                    "summary": summary,
                    "evidence": latest_evidence.get((topic, mood), ""),
                    "updated_at": latest_timestamps.get((topic, mood), ""),
                    "strength": round(min(1.0, 0.35 + count * 0.15), 2),
                },
            )
        return documents

    def deep_pattern_documents(self):
        documents = []
        for pattern in self.bot.life_patterns():
            documents.append(
                {
                    "source_type": "life_pattern",
                    "topic": str(pattern.get("topic") or "general").strip().lower() or "general",
                    "mood": self.bot.normalize_mood(pattern.get("mood")),
                    "summary": str(pattern.get("summary") or "").strip(),
                    "evidence": str(pattern.get("proactive_message") or "").strip(),
                    "updated_at": str(pattern.get("last_seen_at") or "").strip(),
                    "strength": round(
                        max(
                            0.2,
                            min(1.0, int(pattern.get("confidence", 0) or 0) / 100.0),
                        ),
                        2,
                    ),
                },
            )

        for entry in self.bot.wisdom_catalog():
            documents.append(
                {
                    "source_type": "wisdom",
                    "topic": str(entry.get("topic") or "general").strip().lower() or "general",
                    "mood": "neutral",
                    "summary": str(entry.get("summary") or "").strip(),
                    "evidence": str(entry.get("trigger") or "").strip(),
                    "updated_at": str(entry.get("created_at") or "").strip(),
                    "strength": 0.72,
                },
            )

        for entry in self.bot.consolidated_memories():
            try:
                source_count = max(1, int(entry.get("source_count", 1) or 1))
            except (TypeError, ValueError):
                source_count = 1
            if source_count < 2:
                continue
            documents.append(
                {
                    "source_type": "consolidated_memory",
                    "topic": str(entry.get("category") or "general").strip().lower() or "general",
                    "mood": "neutral",
                    "summary": str(entry.get("summary") or "").strip(),
                    "evidence": "; ".join(entry.get("supporting_summaries", [])[:2]),
                    "updated_at": str(entry.get("updated_at") or "").strip(),
                    "strength": round(
                        max(0.25, min(1.0, float(entry.get("confidence", 0.5) or 0.5))),
                        2,
                    ),
                },
            )

        for insight in list(self.bot.MEMORY_STORE.get("longitudinal_insights") or []):
            if not isinstance(insight, dict):
                continue
            summary = str(
                insight.get("insight") or insight.get("summary") or "",
            ).strip()
            if not summary:
                continue
            confidence = float(insight.get("confidence", 0.55) or 0.55)
            documents.append(
                {
                    "source_type": "longitudinal_insight",
                    "topic": str(insight.get("topic") or "general").strip().lower() or "general",
                    "mood": self.bot.normalize_mood(
                        insight.get("dominant_mood") or "neutral",
                    ),
                    "summary": summary,
                    "evidence": str(insight.get("evidence") or "").strip(),
                    "updated_at": str(
                        insight.get("updated_at") or insight.get("created_at") or "",
                    ).strip(),
                    "strength": round(max(0.25, min(1.0, confidence)), 2),
                },
            )

        documents.extend(self._derived_deep_pattern_documents())
        return [document for document in documents if document.get("summary")]

    def _topic_mood_clusters(self, archives):
        clusters = {}
        for entry in archives:
            if not isinstance(entry, dict):
                continue
            topics = [str(topic).strip().lower() for topic in (entry.get("topics") or []) if str(topic).strip()]
            if not topics:
                topics = ["general"]
            mood = self.bot.normalize_mood(entry.get("dominant_mood"))
            summary = str(entry.get("summary") or "").strip()
            created_at = str(entry.get("created_at") or "").strip()
            for topic in topics[:2]:
                key = (topic, mood)
                bucket = clusters.setdefault(
                    key,
                    {
                        "count": 0,
                        "summaries": [],
                        "latest_at": "",
                    },
                )
                bucket["count"] += 1
                if summary and len(bucket["summaries"]) < 5:
                    bucket["summaries"].append(summary)
                if created_at and created_at > str(bucket.get("latest_at") or ""):
                    bucket["latest_at"] = created_at
        return clusters

    @staticmethod
    def _insight_confidence(count, mood):
        base = 0.42 + min(0.38, max(0, int(count or 0)) * 0.06)
        if mood in {"sad", "stressed", "tired", "frustrated"}:
            base += 0.08
        return round(max(0.35, min(0.92, base)), 2)

    def synthesize_longitudinal_insights(
        self,
        force=False,
        reference_time=None,
        max_items=12,
    ):
        now = (
            self.bot.maintenance_scheduler._coerce_reference_time(reference_time)
            if hasattr(self.bot, "maintenance_scheduler")
            else datetime.now().replace(second=0, microsecond=0)
        )
        last_run = str(
            self.bot.MEMORY_STORE.get("last_insight_synthesis_at") or "",
        ).strip()
        if not force and last_run:
            try:
                if now - datetime.fromisoformat(last_run) < timedelta(hours=18):
                    return list(
                        self.bot.MEMORY_STORE.get("longitudinal_insights") or [],
                    )
            except ValueError:
                pass

        archive = list(self.bot.session_archive())
        if len(archive) < 4:
            return list(self.bot.MEMORY_STORE.get("longitudinal_insights") or [])

        clusters = self._topic_mood_clusters(archive[-36:])
        insights = []
        for (topic, mood), payload in sorted(
            clusters.items(),
            key=lambda item: item[1].get("count", 0),
            reverse=True,
        ):
            count = int(payload.get("count", 0) or 0)
            if count < 2 or topic == "general":
                continue
            summaries = [str(item).strip() for item in payload.get("summaries", []) if str(item).strip()]
            evidence = "; ".join(summaries[:2])
            if mood in {"sad", "stressed", "tired", "frustrated"}:
                insight_text = f"When {topic} is in play, emotional load tends to spike ({mood}) and calls for slower, steadier coaching."
            else:
                insight_text = f"{topic} tends to be a high-momentum lane where confidence can be reinforced with concrete next steps."
            insights.append(
                {
                    "topic": topic,
                    "dominant_mood": mood,
                    "insight": insight_text,
                    "evidence": evidence,
                    "evidence_count": count,
                    "confidence": self._insight_confidence(count, mood),
                    "updated_at": now.isoformat(timespec="seconds"),
                },
            )

        if not insights:
            return list(self.bot.MEMORY_STORE.get("longitudinal_insights") or [])

        insights = insights[: max(3, int(max_items or 12))]
        self.bot.mutate_memory_store(
            longitudinal_insights=insights,
            last_insight_synthesis_at=now.isoformat(timespec="seconds"),
        )
        return insights

    def deep_pattern_matches(self, user_input, limit=3):
        query = str(user_input or "").strip()
        if not query:
            return []

        query_tokens = self.bot.significant_tokens(query)
        recent_topics = set(self.bot.relationship.top_topics(limit=5))
        mood_anchor = self.bot.last_saved_mood()
        candidate_limit = max(
            limit,
            self.bot.runtime_config.window("deep_pattern_candidates", 14),
        )
        scored = []
        seen = set()
        for document in self.deep_pattern_documents():
            document_key = (
                document.get("source_type", ""),
                document.get("topic", ""),
                document.get("summary", ""),
            )
            if document_key in seen:
                continue
            seen.add(document_key)

            searchable = " ".join(
                item
                for item in [
                    str(document.get("topic") or "").strip(),
                    str(document.get("mood") or "").strip(),
                    str(document.get("summary") or "").strip(),
                    str(document.get("evidence") or "").strip(),
                ]
                if item
            )
            base_score = float(self.bot.memory_relevance_score(query, searchable))
            overlap = query_tokens & self.bot.significant_tokens(searchable)
            if base_score <= 0 and not overlap:
                continue

            score = base_score
            if overlap:
                score += min(2.0, len(overlap) * 0.4)
            topic = str(document.get("topic") or "").strip().lower()
            if topic and topic in recent_topics:
                score += 0.85
            if mood_anchor != "neutral" and self.bot.normalize_mood(document.get("mood")) == mood_anchor:
                score += 0.65
            score += float(document.get("strength", 0.0) or 0.0) * 2.0
            age_days = self.bot.days_since_iso_date(
                document.get("updated_at") or document.get("created_at"),
            )
            if age_days is not None:
                score *= max(0.55, 1.0 - min(age_days, 180) / 360)
            if score > 0.8:
                scored.append((round(score, 4), document))

        ranked = sorted(
            scored,
            key=lambda item: (
                item[0],
                item[1].get("strength", 0.0),
                item[1].get("updated_at", ""),
                item[1].get("summary", ""),
            ),
            reverse=True,
        )

        selected = []
        selected_topics = set()
        for _, document in ranked[:candidate_limit]:
            topic = str(document.get("topic") or "").strip().lower()
            if topic and topic in selected_topics and len(selected) >= max(1, int(limit or 3)):
                continue
            selected.append(document)
            if topic:
                selected_topics.add(topic)
            if len(selected) >= max(1, int(limit or 3)):
                break
        return selected

    def build_deep_pattern_context(self, user_input, limit=None):
        matches = self.deep_pattern_matches(
            user_input,
            limit=max(
                1,
                int(
                    limit or self.bot.runtime_config.window("deep_pattern_context_limit", 3) or 3,
                ),
            ),
        )
        if not matches:
            return None

        source_labels = {
            "life_pattern": "Recurring pattern",
            "wisdom": "Dad wisdom",
            "consolidated_memory": "Durable insight",
            "derived_arc": "Longitudinal arc",
            "longitudinal_insight": "Synthesis insight",
        }
        lines = []
        for document in matches:
            source_label = source_labels.get(
                document.get("source_type"),
                "Long-term signal",
            )
            topic = str(document.get("topic") or "general").strip().lower() or "general"
            mood = self.bot.normalize_mood(document.get("mood"))
            line = f"- [{source_label}; topic={topic}; mood={mood}] {document.get('summary', '')}"
            evidence = str(document.get("evidence") or "").strip()
            if evidence and evidence.lower() not in line.lower() and document.get("source_type") == "derived_arc":
                line += f" Evidence: {evidence}"
            lines.append(line)

        return (
            "Long-horizon patterns Dad has noticed across time:\n"
            + "\n".join(lines)
            + "\nTreat these as slow-moving arcs in Tony's life, not just one-off facts."
        )

    @staticmethod
    def build_pattern_message(pattern):
        proactive = str(pattern.get("proactive_message") or "").strip()
        if proactive:
            return proactive
        return str(pattern.get("summary") or "").strip()

    @staticmethod
    def pattern_identity(pattern):
        if not isinstance(pattern, dict):
            return None
        return (
            str(pattern.get("topic") or "general").strip().lower() or "general",
            str(pattern.get("day_hint") or "").strip().lower(),
            str(pattern.get("mood") or "neutral").strip().lower() or "neutral",
        )

    def detect_life_patterns(self, force=False):
        today_stamp = date.today().isoformat()
        if not force and str(self.bot.MEMORY_STORE.get("last_pattern_detection_at") or "") == today_stamp:
            return []

        cadence = self.bot.cadence_settings()
        archive = self.bot.session_archive()
        if len(archive) < cadence["life_pattern_min_archived_sessions"]:
            if force:
                self.bot.mutate_memory_store(last_pattern_detection_at=today_stamp)
            return []

        combinations = {}
        for entry in archive[-cadence["life_pattern_window"] :]:
            created_at = str(entry.get("created_at") or "")
            try:
                created_dt = datetime.fromisoformat(created_at)
            except ValueError:
                continue
            mood = self.bot.normalize_mood(entry.get("dominant_mood"))
            for topic in entry.get("topics", []) or ["general"]:
                key = (
                    created_dt.strftime("%A"),
                    str(topic).strip().lower() or "general",
                    mood,
                )
                combinations[key] = combinations.get(key, 0) + 1

        detected = []
        existing_summaries = {
            self.bot.normalize_memory_text(item.get("summary", "")) for item in self.bot.life_patterns()
        }
        existing_pattern_keys = {
            key for key in (self.pattern_identity(item) for item in self.bot.life_patterns()) if key is not None
        }
        for (day_name, topic, mood), count in sorted(
            combinations.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if count < cadence["life_pattern_min_occurrences"] or topic == "general":
                continue

            confidence = min(
                95,
                40 + count * 15 + (10 if mood in {"stressed", "sad", "frustrated", "tired"} else 0),
            )
            if confidence <= cadence["life_pattern_confidence_threshold"]:
                continue

            if mood in {"stressed", "sad", "frustrated", "tired"}:
                summary = f"Tony often carries {topic} {mood} on {day_name}s."
                proactive = f"I've noticed {day_name}s seem to carry extra {topic} weight for you lately. Want to talk about it, or should I just sit with you?"
            else:
                summary = f"Tony often brings up {topic} on {day_name}s."
                proactive = f"I've noticed {day_name}s are when {topic} comes up a lot for you. Anything on your mind there today?"

            pattern_key = (topic, day_name.strip().lower(), mood)
            if pattern_key in existing_pattern_keys or self.bot.normalize_memory_text(summary) in existing_summaries:
                continue

            pattern = self.bot.normalize_life_pattern_entry(
                {
                    "summary": summary,
                    "topic": topic,
                    "mood": mood,
                    "day_hint": day_name,
                    "confidence": confidence,
                    "last_seen_at": archive[-1].get("created_at"),
                    "proactive_message": proactive,
                },
            )
            if pattern is None:
                continue

            detected.append(pattern)
            existing_summaries.add(
                self.bot.normalize_memory_text(pattern.get("summary", "")),
            )
            existing_pattern_keys.add(self.pattern_identity(pattern))

        if detected:
            patterns = self.bot.life_patterns()
            patterns.extend(detected)
            self.bot.mutate_memory_store(
                life_patterns=self.bot.runtime_config.tail(patterns, "life_patterns"),
                last_pattern_detection_at=today_stamp,
            )
            self.bot.mark_memory_graph_dirty()
            for pattern in detected[: cadence["life_pattern_queue_limit"]]:
                self.bot.queue_proactive_message(
                    self.build_pattern_message(pattern),
                    source="life-pattern",
                )
            return detected

        self.bot.mutate_memory_store(last_pattern_detection_at=today_stamp)
        return []

    def build_family_echo_prompt(self, user_input, current_mood):
        return DadPrompts.family_echo(user_input, self.bot.normalize_mood(current_mood))

    def should_offer_family_echo(self, user_input, current_mood):
        lowered = str(user_input or "").lower()
        if "carrie" in lowered or "mom" in lowered:
            return False
        return self.bot.normalize_mood(current_mood) in {"sad", "stressed", "positive"}

    def family_echo(self, user_input, current_mood):
        if not self.should_offer_family_echo(user_input, current_mood):
            return None

        try:
            response = self.bot.call_ollama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": self.build_family_echo_prompt(
                            user_input,
                            current_mood,
                        ),
                    },
                ],
                options={"temperature": 0.3},
                purpose="family echo",
            )
            line = str(response["message"]["content"] or "").strip()
        except (RuntimeError, KeyError, TypeError) as exc:
            logger.warning("Family echo request failed: %s", exc)
            return None

        if not line:
            return None
        return line.strip().strip('"')

    def maybe_add_family_echo(self, reply, user_input, current_mood):
        cadence = self.bot.cadence_settings()
        if not user_input or self.bot.session_turn_count() % cadence["family_echo_turn_interval"] != 0:
            return reply
        if "carrie" in str(reply).lower() or "mom" in str(reply).lower():
            return reply

        echo = self.bot.family_echo(user_input, current_mood)
        if not echo:
            return reply
        return f"{reply.strip()} {echo}"

    # â”€â”€ Continuous learning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def should_run_continuous_learning(self, force: bool = False) -> bool:
        """Return True when a learning cycle is due (every 8 turns) or forced."""
        if force:
            return True
        turns = self.bot.session_turn_count()
        if not self.bot.MEMORY_STORE.get("last_continuous_learning_at"):
            return turns >= 4
        return turns - int(self.bot.MEMORY_STORE.get("last_learning_turn", 0) or 0) >= 8

    def _aggregate_learning_signals(self) -> dict:
        """Collect available feedback signals for the learning cycle."""
        return {
            "relationship_feedback_count": len(
                (self.bot.MEMORY_STORE.get("relationship_history") or [])[-10:],
            ),
            "consolidated_votes": len(
                [m for m in (self.bot.consolidated_memories() or []) if m.get("feedback_score")],
            ),
            "recent_moods": len(self.bot.recent_mood_history() or []),
        }

    def perform_continuous_learning_cycle(self) -> dict:
        """RLHF-style reflection cycle. Called in background via schedule_continuous_learning."""
        turn_count = self.bot.session_turn_count()
        recent_history = self.bot.conversation_history()[-12:]

        feedback_summary = self._aggregate_learning_signals()
        self.bot.relationship_manager.reflect(force=True)

        if self.should_evolve_persona():
            self.evolve_persona()

        _apply_feedback = getattr(
            self.bot.memory_coordinator,
            "apply_consolidated_feedback_from_recent_turns",
            None,
        )
        if callable(_apply_feedback):
            _apply_feedback()

        _last_user = recent_history[-1]["content"] if recent_history else ""
        _last_dad = recent_history[-2]["content"] if len(recent_history) > 1 else ""
        self.bot.internal_state_manager.reflect_after_turn(
            user_input=_last_user,
            current_mood=self.bot.last_saved_mood(),
            dad_reply=_last_dad,
        )

        self.bot.mutate_memory_store(
            last_continuous_learning_at=self.bot.runtime_timestamp(),
            last_learning_turn=turn_count,
            learning_cycle_count=int(
                self.bot.MEMORY_STORE.get("learning_cycle_count", 0) or 0,
            )
            + 1,
            save=True,
        )

        result = {
            "cycle": int(self.bot.MEMORY_STORE.get("learning_cycle_count", 0) or 0),
            "turns_analyzed": len(recent_history),
            "hypotheses_updated": True,
            "persona_evolved": self.should_evolve_persona(),
            "feedback_signals": len(feedback_summary),
        }

        self.bot.runtime_state_container.record_state_update(
            "continuous_learning_completed",
            result,
        )
        return result

    def schedule_continuous_learning(self):
        """Enqueue a learning cycle in background if one is due. Returns task or None."""
        if not self.should_run_continuous_learning():
            return None
        return self.bot.submit_background_task(
            self.perform_continuous_learning_cycle,
            task_kind="continuous-learning",
            metadata={"turns": self.bot.session_turn_count()},
        )


__all__ = ["LongTermSignalsManager"]
