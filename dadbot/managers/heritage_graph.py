"""Heritage Graph â€” Semantic Cross-Linking for Narrative Memories.

Scans the distilled ``narrative_memories`` and ``consolidated_memories``
archives to find resonant connections across different life topics and
time periods, then generates "parental wisdom" style bridge phrases.

This is purely in-process keyword/topic overlap scoring â€” no LLM call.
The output is injected by ContextBuilder so Dad can say things like:
  "The persistence you showed in math today reminds me of how you
   handled that woodworking project back in March."
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


# â”€â”€ Topic synonym groups for broader matching â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_TOPIC_SYNONYMS: dict[str, list[str]] = {
    "math": ["math", "division", "arithmetic", "calculus", "algebra", "geometry", "numbers"],
    "woodworking": ["woodworking", "workshop", "carpentry", "build", "craft", "wood", "shop"],
    "fitness": ["fitness", "exercise", "gym", "running", "workout", "health", "training"],
    "work": ["work", "job", "career", "office", "boss", "promotion", "project"],
    "relationships": ["relationship", "friend", "girlfriend", "boyfriend", "partner", "social"],
    "stress": ["stress", "anxiety", "pressure", "overwhelm", "worried", "nervous"],
    "confidence": ["confidence", "courage", "brave", "try", "attempt", "risk"],
    "persistence": ["persist", "keep going", "keep trying", "never give up", "don't quit", "resilience"],
    "creativity": ["creative", "art", "design", "music", "writing", "poetry", "draw", "paint"],
    "learning": ["learn", "study", "school", "practice", "improve", "progress", "grow"],
}

# Bridge phrase templates â€” keyed by the shared emotional/behavioral theme
_BRIDGE_TEMPLATES: list[tuple[list[str], str]] = [
    (
        ["persist", "keep", "try", "give up", "resilience", "never", "don't quit"],
        "The same persistence you're showing with {current} is exactly how you handled {past} back in {period} â€” "
        "you have a real track record of not backing down.",
    ),
    (
        ["progress", "improve", "better", "growth", "learn"],
        "You made steady progress on {past} in {period}, and that same growth instinct is showing up in {current} right now.",
    ),
    (
        ["stress", "anxiety", "overwhelm", "pressure", "nervous"],
        "Remember how you navigated that tough stretch with {past} in {period}? "
        "You came through it â€” and the same steady approach will carry you through {current}.",
    ),
    (
        ["confidence", "courage", "brave", "risk", "try"],
        "Taking on {current} takes the same kind of guts you showed when you tackled {past} back in {period}.",
    ),
    (
        ["creative", "art", "design", "build", "craft"],
        "There's a creative thread running from your {past} work in {period} right into what you're doing with {current} today.",
    ),
    (
        ["happy", "proud", "excited", "achieve", "success", "win"],
        "You had that same excited energy about {past} in {period} â€” seeing it light you up again with {current} never gets old.",
    ),
]

_DEFAULT_BRIDGE = (
    "Your work on {past} back in {period} keeps coming back to me when we talk about {current} â€” "
    "there's a real thread connecting these two parts of your story."
)


def _tokenize(text: str) -> set[str]:
    """Lower-case word-level tokens, no stopwords."""
    _STOP = {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do", "for",
        "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
        "me", "my", "no", "not", "of", "on", "or", "our", "so", "that", "the",
        "their", "them", "they", "this", "to", "up", "us", "was", "we", "were",
        "will", "with", "you", "your",
    }
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _expand_synonyms(tokens: set[str]) -> set[str]:
    """Expand tokens with known synonym groups."""
    expanded = set(tokens)
    for _group_label, synonyms in _TOPIC_SYNONYMS.items():
        if any(s in tokens for s in synonyms):
            expanded.update(synonyms)
    return expanded


def _overlap_score(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard-like overlap score."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def _pick_bridge_template(past_summary: str, current_text: str) -> str:
    combined = (past_summary + " " + current_text).lower()
    best_score = 0
    best_template = _DEFAULT_BRIDGE
    for keywords, template in _BRIDGE_TEMPLATES:
        score = sum(1 for kw in keywords if kw in combined)
        if score > best_score:
            best_score = score
            best_template = template
    return best_template


class HeritageGraphManager:
    """Semantic cross-linker over narrative and consolidated memories."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def cross_link_query(
        self,
        current_text: str,
        *,
        max_links: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the top ``max_links`` heritage cross-links for ``current_text``.

        Each link is a dict with keys:
        - ``from_topic``: the past topic
        - ``from_period``: month or date string
        - ``past_summary``: the compressed past narrative
        - ``bridge_phrase``: Dad-voice sentence connecting past and present
        - ``score``: float relevance score
        """
        narratives: list[dict[str, Any]] = list(self.bot.narrative_memories() or [])
        consolidated: list[dict[str, Any]] = list(self.bot.consolidated_memories() or [])

        current_tokens = _expand_synonyms(_tokenize(current_text))
        if not current_tokens:
            return []

        candidates: list[tuple[float, dict[str, Any]]] = []

        # Score narrative memories
        for entry in narratives:
            topic = str(entry.get("topic") or "general")
            summary = str(entry.get("summary") or "")
            evidence = str(entry.get("evidence") or "")
            period = str(entry.get("period") or entry.get("period_start") or "")[:7]
            if not summary:
                continue
            entry_tokens = _expand_synonyms(_tokenize(summary + " " + evidence + " " + topic))
            score = _overlap_score(current_tokens, entry_tokens)
            if score > 0.04:  # minimum relevance threshold
                candidates.append((score, {"source": "narrative", "topic": topic, "period": period, "summary": summary}))

        # Score consolidated memories
        for entry in consolidated[-20:]:  # only recent consolidated slice
            summary = str(entry.get("summary") or "")
            category = str(entry.get("category") or "general")
            created_at = str(entry.get("created_at") or "")[:7]
            if not summary:
                continue
            entry_tokens = _expand_synonyms(_tokenize(summary + " " + category))
            score = _overlap_score(current_tokens, entry_tokens)
            if score > 0.06:
                candidates.append((score, {"source": "consolidated", "topic": category, "period": created_at, "summary": summary}))

        if not candidates:
            return []

        # Sort by score, deduplicate by topic to ensure variety
        candidates.sort(key=lambda x: x[0], reverse=True)
        seen_topics: set[str] = set()

        # Detect the primary topic of current context to cross-link away from it
        current_primary = self._primary_topic_of(current_text)

        links: list[dict[str, Any]] = []
        for score, entry in candidates:
            topic = entry["topic"]
            # Cross-link means different topic from what Tony is talking about now
            if topic == current_primary:
                continue
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            past_summary = entry["summary"]
            period = entry["period"] or "recently"
            template = _pick_bridge_template(past_summary, current_text)
            bridge = template.format(
                current=current_primary or "what you're working on",
                past=topic,
                period=_friendly_period(period),
            )
            links.append(
                {
                    "from_topic": topic,
                    "from_period": period,
                    "past_summary": past_summary,
                    "bridge_phrase": bridge,
                    "score": round(score, 4),
                }
            )
            if len(links) >= int(max_links or 3):
                break

        return links

    def heritage_context_block(self, current_text: str) -> str | None:
        """Return a formatted context string for injection into the system prompt."""
        links = self.cross_link_query(current_text, max_links=2)
        if not links:
            return None
        lines = ["Dad's Heritage Memory â€” cross-linked life lessons Dad can draw on:"]
        for link in links:
            lines.append(f'- {link["bridge_phrase"]}')
        return "\n".join(lines)

    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _primary_topic_of(text: str) -> str:
        """Very lightweight: return the most topic-label-like word from text."""
        tokens = _tokenize(text)
        for _label, synonyms in _TOPIC_SYNONYMS.items():
            if any(s in tokens for s in synonyms):
                return _label
        # Fallback: longest non-trivial token
        sorted_tokens = sorted(tokens, key=len, reverse=True)
        return sorted_tokens[0] if sorted_tokens else "general"


def _friendly_period(period: str) -> str:
    """Turn '2026-03' into 'March 2026', leave other strings as-is."""
    if not period or len(period) < 7:
        return period or "recently"
    try:
        year = int(period[:4])
        month = int(period[5:7])
        months = [
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        return f"{months[month]} {year}" if 1 <= month <= 12 else period
    except (ValueError, IndexError):
        return period
