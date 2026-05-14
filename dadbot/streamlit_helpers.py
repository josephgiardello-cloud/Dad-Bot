"""Streamlit-specific helpers for memory management and UX enhancements.

Includes "Save This Moment" functionality and relationship visualization.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

IMPORTANT_MEMORIES_PATH = Path("runtime") / "important_memories.jsonl"


def save_important_memory(
    *,
    content: str,
    context: str | None = None,
    tags: list[str] | None = None,
    importance: str = "high",
) -> bool:
    """Save a moment as an important memory.

    Args:
        content: The main memory text (typically dad's response)
        context: User's input or context for this moment
        tags: Optional tags: ["family", "goal", "emotional", "advice", "wisdom"]
        importance: Importance level: "high" (default), "medium", "low"

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        memory = {
            "content": str(content or "").strip(),
            "context": str(context or "").strip(),
            "tags": list(tags or []),
            "importance": str(importance or "high"),
            "timestamp": datetime.now().isoformat(),
            "type": "user_saved",
        }

        # Validate
        if not memory["content"]:
            return False

        # Append to JSONL log
        IMPORTANT_MEMORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(IMPORTANT_MEMORIES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(memory) + "\n")

        return True
    except Exception as e:
        print(f"Failed to save important memory: {e}")
        return False


def load_important_memories(limit: int = 20) -> list[dict[str, Any]]:
    """Load recently saved important memories.

    Args:
        limit: Max number of memories to return (most recent first)

    Returns:
        List of memory dicts
    """
    if not IMPORTANT_MEMORIES_PATH.exists():
        return []

    try:
        memories = []
        with open(IMPORTANT_MEMORIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    memories.append(json.loads(line.strip()))
                except (json.JSONDecodeError, ValueError):
                    continue

        # Reverse to get most recent first
        return list(reversed(memories))[:max(0, int(limit))]
    except Exception as e:
        print(f"Failed to load important memories: {e}")
        return []


def get_relationship_health_stats(world_model: dict[str, Any] | None) -> dict[str, Any]:
    """Extract relationship health metrics from world model.

    Returns a dict with:
        - trust_level: 0-100
        - openness: 0-100
        - recent_mood: current mood
        - relationship_trend: "improving", "stable", "declining"
    """
    if not world_model:
        return {
            "trust_level": 50,
            "openness": 50,
            "recent_mood": "neutral",
            "relationship_trend": "stable",
        }

    # Extract from world model
    trust_level = int(world_model.get("trust_metric") or 50)
    openness = int(world_model.get("openness_level") or 50)
    recent_mood = str(world_model.get("recent_mood") or "neutral")

    # Simple trend detection (could be enhanced)
    mood_history = list(world_model.get("mood_history") or [])
    relationship_trend = "stable"
    if len(mood_history) >= 2:
        recent = mood_history[-1]
        older = mood_history[0]
        if recent.get("valence", 0) > older.get("valence", 0):
            relationship_trend = "improving"
        elif recent.get("valence", 0) < older.get("valence", 0):
            relationship_trend = "declining"

    return {
        "trust_level": max(0, min(100, trust_level)),
        "openness": max(0, min(100, openness)),
        "recent_mood": recent_mood,
        "relationship_trend": relationship_trend,
    }


def format_relationship_card(health_stats: dict[str, Any]) -> str:
    """Format relationship health stats as markdown/HTML card."""
    trust = health_stats.get("trust_level", 50)
    openness = health_stats.get("openness", 50)
    mood = health_stats.get("recent_mood", "neutral")
    trend = health_stats.get("relationship_trend", "stable")

    # Color based on trust level
    if trust >= 75:
        trust_color = "🟢"
    elif trust >= 50:
        trust_color = "🟡"
    else:
        trust_color = "🔴"

    # Trend emoji
    trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(trend, "➡️")

    card = f"""
**Relationship Health**

{trust_color} **Trust**: {trust}%  
😊 **Openness**: {openness}%  
🎭 **Current Mood**: {mood}  
{trend_emoji} **Trend**: {trend}
"""
    return card.strip()
