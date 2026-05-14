"""Memory normalisation and quality policy sub-component.

Extracted from MemoryManager so that the data-shape / curation concern lives in
its own focused class.  MemoryManager keeps delegation shims so all existing
call-sites continue to work unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date, datetime

from dateutil import parser as dateutil_parser
from pydantic import ValidationError

from dadbot.models import (
    ConsolidatedMemory,
    MemoryEntry,
    MemoryStore,
    PersonaTrait,
    RelationshipState,
    RuntimeHealthTrendPoint,
    WisdomInsight,
)


class MemoryNormalizer:
    """Owns all entry-level normalisation, store normalisation, and quality
    policy for the DadBot memory store.

    Receives a *bot* reference (``DadBotContext.bot``) for the handful of
    bot-level helpers it needs (``normalize_mood``, ``runtime_config``, etc.).
    No storage I/O or catalog access lives here.
    """

    def __init__(self, bot) -> None:
        self.bot = bot

    def _turn_date_fallback(self) -> str:
        temporal = getattr(self.bot, "_current_turn_time_base", None)
        wall_date = getattr(temporal, "wall_date", None)
        if wall_date:
            return str(wall_date)
        return date.today().isoformat()

    def _turn_timestamp_fallback(self) -> str:
        temporal = getattr(self.bot, "_current_turn_time_base", None)
        turn_started_at = getattr(temporal, "turn_started_at", None)
        if turn_started_at:
            return str(turn_started_at)
        return datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------ static coerce helpers

    @staticmethod
    def _coerce_memory_confidence(value: object, *, default: float = 0.5) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _coerce_impact_score(value: object, *, default: float = 1.0) -> float:
        try:
            impact_score = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, impact_score)

    @staticmethod
    def _coerce_unit_float(value: object, *, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _normalize_memory_contradictions(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _normalize_optional_timestamp(value: object) -> str | None:
        raw_value = str(value or "").strip()
        if not raw_value:
            return None
        try:
            parsed = dateutil_parser.parse(raw_value)
        except (TypeError, ValueError, OverflowError):
            return None
        if "T" in raw_value or " " in raw_value:
            return parsed.replace(microsecond=0).isoformat(timespec="seconds")
        return parsed.date().isoformat()

    @staticmethod
    def _normalize_memory_timestamp(value: object, *, fallback: str) -> str:
        raw_value = str(value or fallback).strip() or fallback
        try:
            parsed = dateutil_parser.parse(raw_value)
        except (TypeError, ValueError, OverflowError):
            return fallback
        if "T" in raw_value or " " in raw_value:
            return parsed.replace(microsecond=0).isoformat(timespec="seconds")
        return parsed.date().isoformat()

    @staticmethod
    def _coerce_persona_strength(value: object, *, default: float = 1.0) -> float:
        try:
            strength = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.25, min(3.0, strength))

    @staticmethod
    def _coerce_persona_impact(value: object, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def memory_sort_key(memory):
        created_at = str(memory.get("created_at", "") or "")
        updated_at = str(memory.get("updated_at", "") or "")
        summary = str(memory.get("summary", "") or "")
        category = str(memory.get("category", "") or "")
        mood = str(memory.get("mood", "") or "")
        # Stable final tiebreaker so equal timestamps/summary/category/mood still
        # produce deterministic ordering independent of input sequence.
        payload_hash = hashlib.sha256(
            json.dumps(dict(memory or {}), sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()
        return (updated_at, created_at, summary, category, mood, payload_hash)

    @staticmethod
    def infer_memory_category(summary):
        lowered = summary.lower()
        category_keywords = {
            "work": ["work", "job", "career", "boss", "coworker", "office"],
            "health": [
                "exercise",
                "work out",
                "workout",
                "gym",
                "health",
                "sleep",
                "diet",
                "stress",
                "anxiety",
            ],
            "finance": [
                "money",
                "saving",
                "save",
                "budget",
                "debt",
                "spending",
                "finance",
                "financial",
            ],
            "relationships": [
                "friend",
                "girlfriend",
                "boyfriend",
                "partner",
                "wife",
                "marriage",
                "relationship",
            ],
            "family": ["mom", "dad", "carrie", "tony", "family", "son"],
            "school": [
                "school",
                "college",
                "class",
                "teacher",
                "study",
                "exam",
                "homework",
            ],
            "goals": ["goal", "trying to", "want to", "plan", "hope to", "working on"],
        }
        for category, keywords in category_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                return category
        return "general"

    # ------------------------------------------------------------------ entry normalizers

    @staticmethod
    def _normalize_reminder_status(raw_status):
        status = str(raw_status or "open").strip().lower()
        if status not in {"open", "done"}:
            return "open"
        return status

    @staticmethod
    def _parse_datetime_minute_precision(raw_value):
        text = str(raw_value or "").strip()
        if not text:
            return None
        try:
            parsed = dateutil_parser.parse(text)
            return parsed.replace(second=0, microsecond=0).isoformat(timespec="seconds")
        except (ValueError, TypeError, OverflowError):
            return None

    @staticmethod
    def _parse_due_at(raw_due_at, due_text):
        parsed_raw_due_at = MemoryNormalizer._parse_datetime_minute_precision(raw_due_at)
        if parsed_raw_due_at is not None:
            return parsed_raw_due_at
        if not str(due_text or "").strip():
            return None
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(due_text or "").strip()):
                return f"{str(due_text).strip()}T09:00:00"
            parsed_due_at = dateutil_parser.parse(
                str(due_text or "").strip(),
                default=datetime.now().replace(
                    hour=9,
                    minute=0,
                    second=0,
                    microsecond=0,
                ),
            )
            return parsed_due_at.replace(second=0, microsecond=0).isoformat(timespec="seconds")
        except (ValueError, TypeError, OverflowError):
            return None

    @staticmethod
    def _normalize_notification_count(value):
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def normalize_reminder_entry(self, reminder):
        if not isinstance(reminder, dict):
            return None

        title = str(reminder.get("title") or "").strip()
        if not title:
            return None

        created_at = str(
            reminder.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        )
        updated_at = str(reminder.get("updated_at") or created_at)
        status = self._normalize_reminder_status(reminder.get("status"))

        reminder_id = str(reminder.get("id") or uuid.uuid4().hex[:12])
        due_text = str(reminder.get("due_text") or "").strip()
        due_at = self._parse_due_at(reminder.get("due_at"), due_text)
        last_notified_at = self._parse_datetime_minute_precision(reminder.get("last_notified_at"))
        notification_count = self._normalize_notification_count(reminder.get("notification_count", 0))
        return {
            "id": reminder_id,
            "title": title,
            "due_text": due_text,
            "due_at": due_at,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "last_notified_at": last_notified_at,
            "notification_count": notification_count,
        }

    def normalize_session_archive_entry(self, entry):
        if not isinstance(entry, dict):
            return None

        summary = str(entry.get("summary") or "").strip()
        if not summary:
            return None

        created_at = str(
            entry.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        )
        topics = []
        for topic in entry.get("topics", [])[:5]:
            topic_name = str(topic).strip().lower()
            if topic_name and topic_name not in topics:
                topics.append(topic_name)

        try:
            turn_count = max(0, int(entry.get("turn_count", 0)))
        except (TypeError, ValueError):
            turn_count = 0

        return {
            "id": str(
                entry.get("id") or hashlib.sha1(f"{summary}|{created_at}".encode()).hexdigest()[:12],
            ),
            "created_at": created_at,
            "summary": summary,
            "topics": topics,
            "dominant_mood": self.bot.normalize_mood(entry.get("dominant_mood")),
            "turn_count": turn_count,
        }

    def _normalize_consolidated_source_count(self, entry: dict) -> int:
        try:
            return max(1, int(entry.get("source_count", 1)))
        except (TypeError, ValueError):
            return 1

    def _normalize_consolidated_supporting_summaries(self, entry: dict) -> list[str]:
        supporting_summaries: list[str] = []
        supporting_limit = self.bot.runtime_config.window("supporting_summaries", 4)
        for item in entry.get("supporting_summaries", []):
            cleaned = self.bot.naturalize_memory_summary(str(item or ""))
            if cleaned and cleaned not in supporting_summaries:
                supporting_summaries.append(cleaned)
            if len(supporting_summaries) >= supporting_limit:
                break
        return supporting_summaries

    def _normalize_consolidated_contradictions(self, entry: dict) -> list[str]:
        contradictions: list[str] = []
        contradiction_limit = self.bot.runtime_config.window("contradictions", 4)
        for item in entry.get("contradictions", []):
            cleaned = str(item or "").strip()
            if cleaned and cleaned not in contradictions:
                contradictions.append(cleaned)
            if len(contradictions) >= contradiction_limit:
                break
        return contradictions

    @staticmethod
    def _normalize_consolidated_version(entry: dict) -> int:
        try:
            return max(1, int(entry.get("version", 1)))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _normalize_consolidated_importance(entry: dict) -> float:
        try:
            return max(0.0, min(1.0, float(entry.get("importance_score", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def _build_consolidated_payload(
        self,
        *,
        entry: dict,
        summary: str,
        source_count: int,
        contradictions: list[str],
        supporting_summaries: list[str],
        updated_at: str,
        superseded_at: str | None,
        last_reinforced_at: str | None,
    ) -> dict:
        return {
            "summary": summary,
            "category": str(
                entry.get("category") or self.bot.infer_memory_category(summary),
            )
            .strip()
            .lower()
            or "general",
            "source_count": source_count,
            "confidence": self.bot.normalize_confidence(
                entry.get("confidence"),
                source_count=source_count,
                contradiction_count=len(contradictions),
                updated_at=updated_at,
            ),
            "importance_score": self._normalize_consolidated_importance(entry),
            "version": self._normalize_consolidated_version(entry),
            "superseded": bool(entry.get("superseded", False)),
            "superseded_by": str(entry.get("superseded_by") or "").strip(),
            "superseded_reason": str(entry.get("superseded_reason") or "").strip(),
            "superseded_at": superseded_at,
            "last_reinforced_at": last_reinforced_at,
            "supporting_summaries": supporting_summaries,
            "contradictions": contradictions,
            "updated_at": updated_at,
        }

    @staticmethod
    def _dump_consolidated_entry(dumped: dict, *, updated_at: str) -> dict:
        return {
            "summary": dumped["summary"],
            "category": dumped["category"],
            "source_count": dumped["source_count"],
            "confidence": dumped["confidence"],
            "importance_score": round(float(dumped.get("importance_score", 0.0)), 3),
            "version": int(dumped.get("version", 1)),
            "superseded": bool(dumped.get("superseded", False)),
            "superseded_by": str(dumped.get("superseded_by") or ""),
            "superseded_reason": str(dumped.get("superseded_reason") or ""),
            "superseded_at": dumped.get("superseded_at"),
            "last_reinforced_at": dumped.get("last_reinforced_at"),
            "supporting_summaries": dumped["supporting_summaries"],
            "contradictions": dumped["contradictions"],
            "updated_at": updated_at,
        }

    def normalize_consolidated_memory_entry(self, entry):
        if isinstance(entry, str):
            entry = {"summary": entry}

        if not isinstance(entry, dict):
            return None

        summary = self.bot.naturalize_memory_summary(entry.get("summary", ""))
        if not summary:
            return None

        source_count = self._normalize_consolidated_source_count(entry)
        supporting_summaries = self._normalize_consolidated_supporting_summaries(entry)
        contradictions = self._normalize_consolidated_contradictions(entry)

        updated_at = self._normalize_memory_timestamp(
            entry.get("updated_at"),
            fallback=self._turn_date_fallback(),
        )
        superseded_at = self._normalize_optional_timestamp(entry.get("superseded_at"))
        last_reinforced_at = self._normalize_optional_timestamp(
            entry.get("last_reinforced_at"),
        )
        payload = self._build_consolidated_payload(
            entry=entry,
            summary=summary,
            source_count=source_count,
            contradictions=contradictions,
            supporting_summaries=supporting_summaries,
            updated_at=updated_at,
            superseded_at=superseded_at,
            last_reinforced_at=last_reinforced_at,
        )
        try:
            validated = ConsolidatedMemory.model_validate(payload)
        except ValidationError:
            return None

        dumped = validated.model_dump(mode="json")
        return self._dump_consolidated_entry(dumped, updated_at=updated_at)

    def normalize_persona_evolution_entry(self, entry):
        if not isinstance(entry, dict):
            return None

        trait = str(entry.get("trait") or entry.get("new_trait") or "").strip()
        if not trait:
            return None

        try:
            session_count = max(0, int(entry.get("session_count", 0)))
        except (TypeError, ValueError):
            session_count = 0

        applied_at = self._normalize_memory_timestamp(
            entry.get("applied_at"),
            fallback=self._turn_timestamp_fallback(),
        )
        last_reinforced_at = self._normalize_memory_timestamp(
            entry.get("last_reinforced_at"),
            fallback=applied_at,
        )
        strength = self._coerce_persona_strength(entry.get("strength"))
        impact_score = self._coerce_persona_impact(entry.get("impact_score"))
        try:
            critique_score = max(0, min(10, int(entry.get("critique_score", 0))))
        except (TypeError, ValueError):
            critique_score = 0

        payload = {
            "trait": trait,
            "reason": str(entry.get("reason") or "").strip(),
            "announcement": str(entry.get("announcement") or "").strip(),
            "session_count": session_count,
            "applied_at": applied_at,
            "last_reinforced_at": last_reinforced_at,
            "strength": round(strength, 2),
            "impact_score": round(impact_score, 2),
            "critique_score": critique_score,
            "critique_feedback": str(entry.get("critique_feedback") or "").strip(),
        }
        try:
            validated = PersonaTrait.model_validate(payload)
        except ValidationError:
            return None

        dumped = validated.model_dump(mode="json")
        return {
            "trait": dumped["trait"],
            "reason": dumped["reason"],
            "announcement": dumped["announcement"],
            "session_count": dumped["session_count"],
            "applied_at": applied_at,
            "last_reinforced_at": last_reinforced_at,
            "strength": round(float(dumped["strength"]), 2),
            "impact_score": round(float(dumped["impact_score"]), 2),
            "critique_score": dumped["critique_score"],
            "critique_feedback": dumped["critique_feedback"],
        }

    def normalize_wisdom_entry(self, entry):
        if not isinstance(entry, dict):
            return None

        summary = str(entry.get("summary") or "").strip()
        if not summary:
            return None

        created_at = self._normalize_memory_timestamp(
            entry.get("created_at"),
            fallback=self._turn_timestamp_fallback(),
        )
        payload = {
            "summary": summary,
            "topic": str(entry.get("topic") or self.bot.infer_memory_category(summary)).strip().lower() or "general",
            "trigger": str(entry.get("trigger") or "").strip(),
            "created_at": created_at,
        }
        try:
            validated = WisdomInsight.model_validate(payload)
        except ValidationError:
            return None

        dumped = validated.model_dump(mode="json")
        return {
            "summary": dumped["summary"],
            "topic": dumped["topic"],
            "trigger": dumped["trigger"],
            "created_at": created_at,
        }

    def normalize_life_pattern_entry(self, entry):
        if not isinstance(entry, dict):
            return None

        summary = str(entry.get("summary") or "").strip()
        if not summary:
            return None

        try:
            confidence = max(0, min(100, int(entry.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0

        return {
            "summary": summary,
            "topic": str(entry.get("topic") or "general").strip().lower() or "general",
            "mood": self.bot.normalize_mood(entry.get("mood")),
            "day_hint": str(entry.get("day_hint") or "").strip(),
            "confidence": confidence,
            "last_seen_at": str(
                entry.get("last_seen_at") or self._turn_timestamp_fallback(),
            ),
            "proactive_message": str(entry.get("proactive_message") or "").strip(),
            "last_proactive_at": str(entry.get("last_proactive_at") or "").strip() or None,
        }

    def normalize_proactive_message_entry(self, entry):
        if not isinstance(entry, dict):
            return None

        message = str(entry.get("message") or "").strip()
        if not message:
            return None

        return {
            "message": message,
            "source": str(entry.get("source") or "general").strip().lower() or "general",
            "created_at": str(
                entry.get("created_at") or self._turn_timestamp_fallback(),
            ),
        }
    @staticmethod
    def _normalize_graph_node(node: object) -> dict | None:
        if not isinstance(node, dict):
            return None
        label = str(node.get("label") or "").strip().lower()
        node_type = str(node.get("type") or "topic").strip().lower()
        if not label or node_type not in {"topic", "category", "mood"}:
            return None
        try:
            weight = max(1, int(node.get("weight", 1)))
        except (TypeError, ValueError):
            weight = 1
        return {
            "id": str(node.get("id") or f"{node_type}:{label}"),
            "label": label,
            "type": node_type,
            "weight": weight,
        }

    @staticmethod
    def _normalize_graph_edge(edge: object) -> dict | None:
        if not isinstance(edge, dict):
            return None
        source = str(edge.get("source") or "").strip().lower()
        target = str(edge.get("target") or "").strip().lower()
        if not source or not target:
            return None
        try:
            weight = max(1, int(edge.get("weight", 1)))
        except (TypeError, ValueError):
            weight = 1
        return {"source": source, "target": target, "weight": weight}

    @staticmethod
    def _normalize_dict_list(raw: object, *, tail: int | None = None) -> list[dict]:
        items = list(raw or [])
        if tail is not None:
            items = items[-tail:]
        return [dict(item) for item in items if isinstance(item, dict)]

    def normalize_memory_graph(self, graph):
        default_graph = self.bot.default_memory_graph()
        if not isinstance(graph, dict):
            return default_graph
        nodes = [n for node in graph.get("nodes", [])[:24] if (n := self._normalize_graph_node(node)) is not None]
        edges = [e for edge in graph.get("edges", [])[:24] if (e := self._normalize_graph_edge(edge)) is not None]
        return {
            "nodes": nodes,
            "edges": edges,
            "updated_at": graph.get("updated_at") or default_graph["updated_at"],
        }

    @staticmethod
    def _normalize_emotional_momentum(raw_momentum: object) -> str:
        momentum = str(raw_momentum)
        return momentum if momentum in {"steady", "warming", "heavy"} else "steady"

    @staticmethod
    def _normalize_recurring_topics(recurring_topics: object) -> dict[str, int]:
        cleaned_topics: dict[str, int] = {}
        if not isinstance(recurring_topics, dict):
            return cleaned_topics
        for topic, count in recurring_topics.items():
            topic_name = str(topic).strip().lower()
            if not topic_name:
                continue
            try:
                cleaned_topics[topic_name] = max(0, int(count))
            except (TypeError, ValueError):
                continue
        return cleaned_topics

    def _normalize_recent_checkins(self, raw_checkins: object) -> list[dict[str, str]]:
        recent_checkins: list[dict[str, str]] = []
        for item in self.bot.runtime_config.tail(raw_checkins if isinstance(raw_checkins, list) else [], "recent_checkins"):
            if not isinstance(item, dict):
                continue
            recent_checkins.append(
                {
                    "date": item.get("date") or self._turn_date_fallback(),
                    "mood": self.bot.normalize_mood(item.get("mood")),
                    "topic": str(item.get("topic") or "general").strip().lower() or "general",
                },
            )
        return recent_checkins

    def _normalize_relationship_hypotheses(self, state: dict, profiles: dict) -> list[dict[str, object]]:
        raw_hypotheses = state.get("hypotheses", [])
        candidate_hypotheses: list[dict[str, object]] = []
        if isinstance(raw_hypotheses, dict):
            raw_hypotheses = [{"name": name, "probability": probability} for name, probability in raw_hypotheses.items()]
        for item in raw_hypotheses:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower()
            if name not in profiles:
                continue
            try:
                probability = max(0.01, float(item.get("probability", 0.0)))
            except (TypeError, ValueError):
                probability = 0.01
            candidate_hypotheses.append(
                {
                    "name": name,
                    "label": profiles[name]["label"],
                    "summary": profiles[name]["summary"],
                    "probability": probability,
                },
            )
        if not candidate_hypotheses:
            candidate_hypotheses = self.bot.default_relationship_hypotheses()
        total_probability = sum(item.get("probability", 0.0) for item in candidate_hypotheses) or 1.0
        for item in candidate_hypotheses:
            item["probability"] = round(float(item.get("probability", 0.0)) / total_probability, 4)
        candidate_hypotheses.sort(key=lambda item: (-item.get("probability", 0.0), item.get("name", "")))
        return candidate_hypotheses[: len(profiles)]

    def normalize_relationship_state(self, state):
        default_state = self.bot.default_relationship_state()
        if not isinstance(state, dict):
            return default_state

        last_updated = self._normalize_memory_timestamp(
            state.get("last_updated"),
            fallback=default_state["last_updated"],
        )
        trust_level = self.bot.decay_relationship_level(
            state.get("trust_level", default_state["trust_level"]),
            last_updated,
        )
        openness_level = self.bot.decay_relationship_level(
            state.get("openness_level", default_state["openness_level"]),
            last_updated,
        )
        emotional_momentum = self._normalize_emotional_momentum(
            state.get("emotional_momentum", default_state["emotional_momentum"]),
        )
        cleaned_topics = self._normalize_recurring_topics(state.get("recurring_topics", {}))
        recent_checkins = self._normalize_recent_checkins(state.get("recent_checkins", []))
        profiles = self.bot.relationship_hypothesis_profiles()
        hypotheses = self._normalize_relationship_hypotheses(state, profiles)
        active_hypothesis = hypotheses[0]["name"] if hypotheses else default_state["active_hypothesis"]
        last_hypothesis_updated = self._normalize_memory_timestamp(
            state.get("last_hypothesis_updated"),
            fallback=last_updated,
        )
        last_reflection = str(state.get("last_reflection") or "").strip()

        payload = {
            "trust_level": trust_level,
            "openness_level": openness_level,
            "emotional_momentum": emotional_momentum,
            "recurring_topics": cleaned_topics,
            "recent_checkins": recent_checkins,
            "hypotheses": hypotheses,
            "active_hypothesis": active_hypothesis,
            "last_hypothesis_updated": last_hypothesis_updated,
            "last_reflection": last_reflection,
            "last_updated": last_updated,
        }
        try:
            validated = RelationshipState.model_validate(payload)
        except ValidationError:
            return default_state

        dumped = validated.model_dump(mode="json")
        return {
            "trust_level": dumped["trust_level"],
            "openness_level": dumped["openness_level"],
            "emotional_momentum": dumped["emotional_momentum"],
            "recurring_topics": cleaned_topics,
            "recent_checkins": recent_checkins,
            "hypotheses": hypotheses,
            "active_hypothesis": dumped["active_hypothesis"],
            "last_hypothesis_updated": last_hypothesis_updated,
            "last_reflection": dumped["last_reflection"],
            "last_updated": last_updated,
        }

    # ------------------------------------------------------------------ memory entry normalizers

    def _memory_entry_payload(self, memory, *, summary, category, created_at, updated_at):
        return {
            "summary": summary,
            "category": category,
            "mood": self.bot.normalize_mood(memory.get("mood")) if "mood" in memory else "neutral",
            "confidence": self._coerce_memory_confidence(memory.get("confidence")),
            "impact_score": self._coerce_impact_score(memory.get("impact_score")),
            "importance_score": self._coerce_unit_float(
                memory.get("importance_score", 0.0),
                default=0.0,
            ),
            "emotional_intensity": self._coerce_unit_float(
                memory.get("emotional_intensity", 0.25),
                default=0.25,
            ),
            "relationship_impact": self._coerce_unit_float(
                memory.get("relationship_impact", 0.5),
                default=0.5,
            ),
            "pinned": bool(memory.get("pinned", False)),
            "created_at": created_at,
            "updated_at": updated_at,
            "contradictions": self._normalize_memory_contradictions(
                memory.get("contradictions"),
            ),
        }

    @staticmethod
    def _project_memory_entry_common(memory, dumped, *, created_at, updated_at):
        normalized = {
            "summary": dumped["summary"],
            "category": dumped["category"],
        }
        if "mood" in memory:
            normalized["mood"] = dumped["mood"]
        if "created_at" in memory:
            normalized["created_at"] = created_at
        if "updated_at" in memory:
            normalized["updated_at"] = updated_at
        if "confidence" in memory:
            normalized["confidence"] = dumped["confidence"]
        if "impact_score" in memory:
            normalized["impact_score"] = dumped["impact_score"]
        if "importance_score" in memory:
            normalized["importance_score"] = round(float(dumped["importance_score"]), 3)
        if "emotional_intensity" in memory:
            normalized["emotional_intensity"] = round(
                float(dumped["emotional_intensity"]),
                3,
            )
        if "relationship_impact" in memory:
            normalized["relationship_impact"] = round(
                float(dumped["relationship_impact"]),
                3,
            )
        if "pinned" in memory:
            normalized["pinned"] = bool(dumped["pinned"])
        if "contradictions" in memory:
            normalized["contradictions"] = dumped["contradictions"]
        return normalized

    @staticmethod
    def _project_memory_entry_counters(memory, normalized):
        if "access_count" in memory:
            try:
                normalized["access_count"] = max(0, int(memory.get("access_count", 0) or 0))
            except (TypeError, ValueError):
                normalized["access_count"] = 0
        if "high_confidence_hits" in memory:
            try:
                normalized["high_confidence_hits"] = max(0, int(memory.get("high_confidence_hits", 0) or 0))
            except (TypeError, ValueError):
                normalized["high_confidence_hits"] = 0
        if "confidence_history" in memory:
            raw = memory.get("confidence_history") or {}
            if not isinstance(raw, dict):
                raw = {}
            normalized["confidence_history"] = {
                "high": max(0, int(raw.get("high", 0) or 0)),
                "medium": max(0, int(raw.get("medium", 0) or 0)),
                "low": max(0, int(raw.get("low", 0) or 0)),
            }

    def normalize_memory_entry(self, memory):
        if not isinstance(memory, dict):
            return None

        summary = self.bot.naturalize_memory_summary(memory.get("summary", ""))
        if not summary:
            return None

        category = memory.get("category")
        if category in {None, "summary", "category"}:
            category = self.infer_memory_category(summary)

        created_at = self._normalize_memory_timestamp(
            memory.get("created_at"),
            fallback=self._turn_date_fallback(),
        )
        updated_at = self._normalize_memory_timestamp(
            memory.get("updated_at"),
            fallback=created_at,
        )
        payload = self._memory_entry_payload(
            memory,
            summary=summary,
            category=category,
            created_at=created_at,
            updated_at=updated_at,
        )
        try:
            validated = MemoryEntry.model_validate(payload)
        except ValidationError:
            return None

        dumped = validated.model_dump(mode="json")
        normalized = self._project_memory_entry_common(
            memory,
            dumped,
            created_at=created_at,
            updated_at=updated_at,
        )
        self._project_memory_entry_counters(memory, normalized)
        return normalized

    def normalize_persisted_memory_entry(self, memory):
        if not isinstance(memory, dict):
            return None

        summary = self.bot.naturalize_memory_summary(memory.get("summary", ""))
        if not summary:
            return None

        category = memory.get("category")
        if category in {None, "summary", "category"}:
            category = self.infer_memory_category(summary)

        created_at = self._normalize_memory_timestamp(
            memory.get("created_at"),
            fallback=self._turn_date_fallback(),
        )
        updated_at = self._normalize_memory_timestamp(
            memory.get("updated_at"),
            fallback=created_at,
        )
        payload = {
            "summary": summary,
            "category": category,
            "mood": self.bot.normalize_mood(memory.get("mood")),
            "confidence": self._coerce_memory_confidence(memory.get("confidence")),
            "impact_score": self._coerce_impact_score(memory.get("impact_score")),
            "importance_score": self._coerce_unit_float(
                memory.get("importance_score", 0.0),
                default=0.0,
            ),
            "emotional_intensity": self._coerce_unit_float(
                memory.get("emotional_intensity", 0.25),
                default=0.25,
            ),
            "relationship_impact": self._coerce_unit_float(
                memory.get("relationship_impact", 0.5),
                default=0.5,
            ),
            "pinned": bool(memory.get("pinned", False)),
            "created_at": created_at,
            "updated_at": updated_at,
            "contradictions": self._normalize_memory_contradictions(
                memory.get("contradictions"),
            ),
        }
        try:
            validated = MemoryEntry.model_validate(payload)
        except ValidationError:
            return None

        dumped = validated.model_dump(mode="json")
        normalized = {
            "summary": dumped["summary"],
            "category": dumped["category"],
            "mood": dumped["mood"],
            "confidence": dumped["confidence"],
            "impact_score": dumped["impact_score"],
            "importance_score": round(float(dumped["importance_score"]), 3),
            "emotional_intensity": round(float(dumped["emotional_intensity"]), 3),
            "relationship_impact": round(float(dumped["relationship_impact"]), 3),
            "pinned": bool(dumped["pinned"]),
            "created_at": created_at,
            "updated_at": updated_at,
            "contradictions": dumped["contradictions"],
        }
        try:
            normalized["access_count"] = max(0, int(memory.get("access_count", 0) or 0))
        except (TypeError, ValueError):
            normalized["access_count"] = 0
        try:
            normalized["high_confidence_hits"] = max(0, int(memory.get("high_confidence_hits", 0) or 0))
        except (TypeError, ValueError):
            normalized["high_confidence_hits"] = 0
        raw = memory.get("confidence_history") or {}
        if not isinstance(raw, dict):
            raw = {}
        normalized["confidence_history"] = {
            "high": max(0, int(raw.get("high", 0) or 0)),
            "medium": max(0, int(raw.get("medium", 0) or 0)),
            "low": max(0, int(raw.get("low", 0) or 0)),
        }
        return normalized

    # ------------------------------------------------------------------ quality policy

    def memory_quality_score(self, memory):
        summary = self.bot.normalize_memory_text(memory.get("summary", ""))
        if not summary:
            return -100

        score = 0
        tokens = self.bot.tokenize(summary)
        word_count = len(summary.split())
        score += min(len(summary), 120)
        if summary.startswith("tony shared that "):
            score -= 10
        weak_phrases = [
            "personal struggles",
            "personal history",
            "personal concerns",
            "emotional state",
            "mental health",
            "stressed",
            "sad",
            "asked about age at marriage",
        ]
        if any(phrase == summary or phrase in summary for phrase in weak_phrases):
            score -= 40

        generic_tokens = {
            "thing",
            "things",
            "stuff",
            "issue",
            "issues",
            "problem",
            "problems",
            "situation",
            "life",
            "anything",
            "everything",
            "something",
        }
        filler_tokens = {
            "tony",
            "shared",
            "that",
            "has",
            "been",
            "is",
            "was",
            "feels",
            "feeling",
        }
        if len(tokens) <= 6 and tokens & generic_tokens:
            score -= 35

        meaningful_tokens = tokens - generic_tokens - filler_tokens
        if len(meaningful_tokens) <= 2:
            score -= 25

        strong_keywords = [
            "work",
            "exercise",
            "budget",
            "saving",
            "anxious",
            "overwhelmed",
            "sad",
            "stress",
        ]
        if any(keyword in summary for keyword in strong_keywords):
            score += 20
        if word_count < 3:
            score -= 20
        return score

    def is_high_quality_memory(self, memory):
        summary = self.bot.normalize_memory_text(memory.get("summary", ""))
        category = str(memory.get("category", "")).strip().lower()
        if not summary:
            return False
        if category in {"summary", "category"}:
            return False

        low_signal_patterns = [
            r"^tony shared that personal [a-z ]+\.?$",
            r"^tony shared that emotional state\.?$",
            r"^tony shared that mental health\.?$",
            r"^tony shared that stressed\.?$",
            r"^tony shared that sad\.?$",
            r"^tony shared that asked about age at marriage\.?$",
            r"^tony shared that (something|things|stuff|life|issues|problems).*$",
            r"^tony (is|was|feels|has been) dealing with (things|stuff|issues|problems).*$",
        ]
        if any(re.match(pattern, summary) for pattern in low_signal_patterns):
            return False
        return self.memory_quality_score(memory) >= 5

    def memory_dedup_key(self, memory):
        summary = self.bot.normalize_memory_text(memory.get("summary", ""))
        summary = re.sub(r"^tony shared that ", "", summary)
        summary = re.sub(
            r"\b(has been|is|was|wants to|wants|needs to|needs|is trying to)\b",
            "",
            summary,
        )
        summary = re.sub(r"\s+", " ", summary).strip(" .")
        category = str(memory.get("category", "general")).strip().lower()
        if "saving" in summary or "budget" in summary or "money" in summary:
            return ("finance", "money-goals")
        if "work" in summary and any(term in summary for term in ["stress", "stressed", "overwhelmed", "anxious"]):
            return ("work", "work-stress")
        if "exercise" in summary or "workout" in summary or "gym" in summary:
            return ("health", "exercise")
        return (category, summary)

    def clean_memory_entries(self, memories):
        cleaned = []
        for memory in memories:
            normalized = self.normalize_memory_entry(memory)
            if normalized is None or not self.is_high_quality_memory(normalized):
                continue
            cleaned.append(normalized)

        deduped = {}
        for memory in cleaned:
            key = self.memory_dedup_key(memory)
            existing = deduped.get(key)
            if existing is None or self.memory_quality_score(
                memory,
            ) >= self.memory_quality_score(existing):
                deduped[key] = memory
        return sorted(deduped.values(), key=self.memory_sort_key)

    # ------------------------------------------------------------------ store normalizer

    def normalize_memory_store(self, store):
        default_store = self.bot.default_memory_store()
        if not isinstance(store, dict):
            return default_store

        normalized = dict(default_store)
        normalized["memories"] = (
            sorted(
                [
                    entry
                    for entry in (self.normalize_persisted_memory_entry(item) for item in store.get("memories", []))
                    if entry is not None
                ],
                key=self.memory_sort_key,
            )
            if isinstance(store.get("memories"), list)
            else []
        )

        self._normalize_catalog_list_fields(store, normalized)
        self._normalize_health_fields(store, normalized)
        self._normalize_timestamp_fields(store, normalized, default_store)
        normalized["recent_moods"] = self._normalize_recent_moods(store)
        normalized["relationship_state"] = self.normalize_relationship_state(
            store.get("relationship_state"),
        )
        self._normalize_internal_state(store, normalized, default_store)
        normalized["relationship_history"] = self._normalize_relationship_history(store)
        self._normalize_bounded_list_fields(store, normalized)

        try:
            validated = MemoryStore.model_validate(normalized)
        except ValidationError:
            return default_store
        return validated.model_dump(mode="json")

    # -- Private field-group helpers for normalize_memory_store -----------------

    def _normalize_catalog_entries(self, store: dict, *, source_key: str, tail_key: str, normalizer) -> list[dict]:
        return self.bot.runtime_config.tail(
            [
                entry
                for entry in (normalizer(item) for item in store.get(source_key, []))
                if entry is not None
            ],
            tail_key,
        )

    def _normalize_catalog_list_fields(self, store: dict, normalized: dict) -> None:
        normalized["consolidated_memories"] = self._normalize_catalog_entries(
            store,
            source_key="consolidated_memories",
            tail_key="consolidated_memories",
            normalizer=self.normalize_consolidated_memory_entry,
        )
        normalized["persona_evolution"] = self._normalize_catalog_entries(
            store,
            source_key="persona_evolution",
            tail_key="persona_evolution",
            normalizer=self.normalize_persona_evolution_entry,
        )
        normalized["wisdom_insights"] = self._normalize_catalog_entries(
            store,
            source_key="wisdom_insights",
            tail_key="wisdom_insights",
            normalizer=self.normalize_wisdom_entry,
        )
        normalized["life_patterns"] = self._normalize_catalog_entries(
            store,
            source_key="life_patterns",
            tail_key="life_patterns",
            normalizer=self.normalize_life_pattern_entry,
        )
        normalized["pending_proactive_messages"] = self._normalize_catalog_entries(
            store,
            source_key="pending_proactive_messages",
            tail_key="pending_proactive_messages",
            normalizer=self.normalize_proactive_message_entry,
        )
        normalized["reminders"] = self._normalize_catalog_entries(
            store,
            source_key="reminders",
            tail_key="reminders",
            normalizer=self.normalize_reminder_entry,
        )
        normalized["session_archive"] = self._normalize_catalog_entries(
            store,
            source_key="session_archive",
            tail_key="session_archive",
            normalizer=self.normalize_session_archive_entry,
        )

    def _normalize_health_fields(self, store: dict, normalized: dict) -> None:
        normalized["health_history"] = self.bot.runtime_config.tail(
            [
                RuntimeHealthTrendPoint.model_validate(item).model_dump(mode="python")
                for item in store.get("health_history", [])
                if isinstance(item, dict)
            ],
            "health_history",
        )
        normalized["health_quiet_mode"] = bool(store.get("health_quiet_mode", False))
        runtime_optimization = store.get("runtime_optimization", {})
        normalized["runtime_optimization"] = (
            dict(runtime_optimization) if isinstance(runtime_optimization, dict) else {}
        )

    def _normalize_timestamp_fields(
        self,
        store: dict,
        normalized: dict,
        default_store: dict,
    ) -> None:
        normalized["last_consolidated_at"] = self._normalize_optional_timestamp(
            store.get("last_consolidated_at"),
        )
        normalized["last_pattern_detection_at"] = self._normalize_optional_timestamp(
            store.get("last_pattern_detection_at"),
        )
        normalized["last_mood"] = self.bot.normalize_mood(store.get("last_mood"))
        normalized["last_mood_updated_at"] = self._normalize_memory_timestamp(
            store.get("last_mood_updated_at"),
            fallback=default_store["last_mood_updated_at"],
        )
        normalized["last_background_synthesis_at"] = self._normalize_optional_timestamp(
            store.get("last_background_synthesis_at"),
        )
        try:
            normalized["last_background_synthesis_turn"] = max(
                0,
                int(store.get("last_background_synthesis_turn", 0)),
            )
        except (TypeError, ValueError):
            normalized["last_background_synthesis_turn"] = 0
        normalized["last_memory_compaction_at"] = self._normalize_optional_timestamp(
            store.get("last_memory_compaction_at"),
        )
        normalized["last_memory_compaction_summary"] = str(
            store.get("last_memory_compaction_summary") or "",
        ).strip()
        normalized["last_scheduled_proactive_at"] = self._normalize_optional_timestamp(
            store.get("last_scheduled_proactive_at"),
        )
        normalized["last_daily_checkin_at"] = self._normalize_optional_timestamp(
            store.get("last_daily_checkin_at"),
        )

    def _normalize_recent_moods(self, store: dict) -> list:
        result = []
        if isinstance(store.get("recent_moods"), list):
            for item in store.get("recent_moods", []):
                if isinstance(item, dict):
                    result.append(
                        {
                            "mood": self.bot.normalize_mood(item.get("mood")),
                            "date": self._normalize_memory_timestamp(
                                item.get("date"),
                                fallback=date.today().isoformat(),
                            ),
                        },
                    )
                elif isinstance(item, str):
                    result.append(
                        {
                            "mood": self.bot.normalize_mood(item),
                            "date": date.today().isoformat(),
                        },
                    )
        return result

    def _normalize_internal_state(
        self,
        store: dict,
        normalized: dict,
        default_store: dict,
    ) -> None:
        internal_state = store.get("internal_state")
        default_internal_state = default_store.get("internal_state", {})
        if isinstance(default_internal_state, dict):
            normalized_internal_state = dict(default_internal_state)
        else:
            normalized_internal_state = {}
        if isinstance(internal_state, dict):
            normalized_internal_state.update(internal_state)
        normalized["internal_state"] = normalized_internal_state

    def _normalize_relationship_history(self, store: dict) -> list:
        relationship_history = []
        for item in store.get("relationship_history", []):
            if not isinstance(item, dict):
                continue
            try:
                trust_level = max(0, min(100, int(item.get("trust_level", 50) or 50)))
            except (TypeError, ValueError):
                trust_level = 50
            try:
                openness_level = max(
                    0,
                    min(100, int(item.get("openness_level", 50) or 50)),
                )
            except (TypeError, ValueError):
                openness_level = 50
            relationship_history.append(
                {
                    "recorded_at": self._normalize_memory_timestamp(
                        item.get("recorded_at"),
                        fallback=date.today().isoformat(),
                    ),
                    "trust_level": trust_level,
                    "openness_level": openness_level,
                    "source": str(item.get("source") or "turn").strip().lower() or "turn",
                },
            )
        return self.bot.runtime_config.tail(
            relationship_history,
            "relationship_history",
        )

    def _normalize_bounded_list_fields(self, store: dict, normalized: dict) -> None:
        normalized["mcp_local_store"] = dict(store.get("mcp_local_store") or {})
        normalized["narrative_memories"] = self._normalize_dict_list(store.get("narrative_memories"))
        normalized["heritage_cross_links"] = self._normalize_dict_list(store.get("heritage_cross_links"))
        normalized["advice_audits"] = self._normalize_dict_list(store.get("advice_audits"), tail=160)
        normalized["environmental_cues_history"] = self._normalize_dict_list(store.get("environmental_cues_history"), tail=200)
        normalized["longitudinal_insights"] = self._normalize_dict_list(store.get("longitudinal_insights"), tail=40)
        normalized["relationship_timeline"] = str(store.get("relationship_timeline") or "").strip()
        normalized["memory_graph"] = self.normalize_memory_graph(store.get("memory_graph"))


__all__ = ["MemoryNormalizer"]
