from __future__ import annotations

import re


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


class GraphEntityResolver:
    """Entity canonicalization and relation-type vocabulary for the memory graph.

    Centralises entity aliases, type mappings, and relation vocab so the
    graph manager is not responsible for domain knowledge about named entities.
    """

    ENTITY_ALIASES: dict[str, set[str]] = {
        "work_boss": {"boss", "manager", "supervisor"},
        "work_deadlines": {"deadline", "deadlines"},
        "budgeting": {"budget", "budgeting", "budget spreadsheet"},
        "saving_money": {"saving money", "save money", "savings"},
        "emergency_fund": {"emergency fund", "rainy day fund"},
        "work_stress": {"work stress", "job stress", "career stress"},
        "next_steps": {"next step", "next steps"},
    }
    ENTITY_TYPES: dict[str, str] = {
        "work_boss": "person",
        "work_deadlines": "stressor",
        "budgeting": "habit",
        "saving_money": "goal",
        "emergency_fund": "goal",
        "work_stress": "emotion",
        "next_steps": "advice_pattern",
    }
    RELATION_VOCAB: frozenset[str] = frozenset(
        {
            "categorized_as",
            "mentions",
            "concerns",
            "struggles_with",
            "plans_for",
            "feels",
            "mentioned_on",
            "recurs_on",
            "reflects_mood",
            "covers_topic",
            "pattern_topic",
            "pattern_mood",
            "expresses_trait",
            "reinforced_by",
            "responds_to",
            "contradicted_by",
        }
    )

    def __init__(self, bot) -> None:
        self._bot = bot

    # --- Entity canonicalization ---

    def canonical_entity(self, label, semantic_type=None) -> tuple[str, str]:
        lowered = str(label or "").strip().lower()
        if not lowered:
            return semantic_type or "topic", ""
        if semantic_type in {"mood", "emotion"}:
            return "emotion", self._bot.normalize_mood(lowered)
        if semantic_type == "day":
            return "day", lowered
        if semantic_type == "category":
            return "category", lowered
        if semantic_type == "trait":
            return "trait", _slug(lowered)
        for canonical, aliases in self.ENTITY_ALIASES.items():
            if lowered == canonical or lowered in aliases or any(alias in lowered for alias in aliases):
                return self.ENTITY_TYPES.get(canonical, semantic_type or "topic"), canonical
        return semantic_type or "topic", _slug(lowered)

    # --- Relation type resolution ---

    def normalize_relation_type(self, relation_type: str) -> str:
        normalized = _slug(relation_type)
        if normalized in self.RELATION_VOCAB:
            return normalized
        relation_map = {
            "categorized_as": "categorized_as",
            "covers_topic": "covers_topic",
            "reflects_mood": "reflects_mood",
            "pattern_topic": "pattern_topic",
            "pattern_mood": "pattern_mood",
            "recurs_on": "recurs_on",
            "mentioned_on": "mentioned_on",
            "expresses_trait": "expresses_trait",
            "reinforced_by": "reinforced_by",
            "responds_to": "responds_to",
        }
        return relation_map.get(normalized, "mentions")

    def relation_type_for_entity(
        self,
        source_type: str,
        entity_type: str,
        text: str,
        fallback: str = "mentions",
    ) -> str:
        lowered = str(text or "").lower()
        if entity_type in {"emotion", "mood"}:
            return "feels"
        if source_type == "life_pattern" and entity_type == "day":
            return "recurs_on"
        if source_type == "archive_session" and entity_type == "day":
            return "mentioned_on"
        if entity_type == "goal" or any(
            token in lowered for token in ["plan", "goal", "saving", "save", "budget", "trying to", "working on"]
        ):
            return "plans_for"
        if entity_type in {"stressor", "person"} and any(
            token in lowered for token in ["stress", "stressed", "overwhelmed", "anxious", "worried", "heavy", "pressure"]
        ):
            return "struggles_with"
        if source_type == "persona_trait" and entity_type in {"topic", "advice_pattern"}:
            return "responds_to"
        return self.normalize_relation_type(fallback)


__all__ = ["GraphEntityResolver"]
