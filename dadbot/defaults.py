"""
Static configuration and initial state profiles for the DadBot ecosystem.
Moving these here keeps Dadbot.py focused on orchestration logic.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import date, datetime


# The 'Soul' of the bot's initial memory
DEFAULT_MEMORY_STORE = {
    "user_metadata": {
        "name": "Tony",
        "relationship": "son",
        "preferences": {},
        "significant_events": [],
    },
    "interaction_stats": {
        "total_turns": 0,
        "last_interaction": None,
        "average_sentiment": 0.0,
    },
    "long_term_summary": "Initial state: Start of a new mentorship journey.",
    "last_scheduled_proactive_at": None,
}


# The TONY Engine Personality Profiles
RELATIONSHIP_HYPOTHESIS_PROFILES = {
    "nurturing": {
        "trust_threshold": 0.8,
        "firmness_level": 0.2,
        "humor_frequency": 0.9,
        "description": "High warmth, high trust. Focuses on support and dad jokes.",
    },
    "guiding": {
        "trust_threshold": 0.5,
        "firmness_level": 0.5,
        "humor_frequency": 0.6,
        "description": "Balanced approach. Supportive but sets clear boundaries.",
    },
    "authoritative": {
        "trust_threshold": 0.3,
        "firmness_level": 0.8,
        "humor_frequency": 0.3,
        "description": "Low trust/High friction state. Focuses on discipline and directness.",
    },
}


# Initial state for the TONY reputation engine
DEFAULT_RELATIONSHIP_STATE = {
    "tony_score": 0.5,
    "volatility": 0.1,
    "current_hypothesis": "guiding",
    "history": [],
}


# Debugging and Planner defaults
PLANNER_DEBUG_STATE = {
    "active_goals": [],
    "last_critique": None,
    "internal_monologue": deque(maxlen=5),
}


# Runtime-compatible defaults used by the current service layer.
RUNTIME_RELATIONSHIP_HYPOTHESIS_PROFILES = {
    "supportive_baseline": {
        "label": "Supportive Baseline",
        "summary": "Tony is fundamentally trusting and wants steady support, even if a turn is rough.",
    },
    "acute_stress": {
        "label": "Acute Stress Spike",
        "summary": "Tony is reacting to current pressure or overload more than to the relationship itself.",
    },
    "guarded_distance": {
        "label": "Guarded Distance",
        "summary": "Tony may be holding back or testing safety before opening up further.",
    },
    "positive_rebound": {
        "label": "Positive Rebound",
        "summary": "Tony may be emerging from a heavy stretch and reconnecting with relief or momentum.",
    },
}


def base_memory_store() -> dict:
    return deepcopy(DEFAULT_MEMORY_STORE)


def planner_debug_state() -> dict:
    return deepcopy(PLANNER_DEBUG_STATE)


def relationship_hypothesis_profiles() -> dict:
    return deepcopy(RUNTIME_RELATIONSHIP_HYPOTHESIS_PROFILES)


def default_relationship_hypotheses() -> list:
    profiles = relationship_hypothesis_profiles()
    defaults = [
        ("supportive_baseline", 0.52),
        ("acute_stress", 0.23),
        ("guarded_distance", 0.15),
        ("positive_rebound", 0.10),
    ]
    return [
        {
            "name": name,
            "label": profiles[name]["label"],
            "summary": profiles[name]["summary"],
            "probability": probability,
        }
        for name, probability in defaults
    ]


def default_relationship_state() -> dict:
    return {
        "trust_level": 50,
        "openness_level": 50,
        "emotional_momentum": "steady",
        "recurring_topics": {},
        "recent_checkins": [],
        "hypotheses": default_relationship_hypotheses(),
        "active_hypothesis": "supportive_baseline",
        "last_hypothesis_updated": date.today().isoformat(),
        "last_reflection": "",
        "last_updated": date.today().isoformat(),
    }


def default_memory_graph() -> dict:
    return {
        "nodes": [],
        "edges": [],
        "updated_at": None,
    }


def default_memory_store() -> dict:
    from dadbot.managers.internal_state import InternalStateManager

    store = base_memory_store()
    store.update(
        {
            "memories": [],
            "consolidated_memories": [],
            "persona_evolution": [],
            "wisdom_insights": [],
            "life_patterns": [],
            "pending_proactive_messages": [],
            "health_history": [],
            "health_quiet_mode": False,
            "runtime_optimization": {},
            "last_consolidated_at": None,
            "last_pattern_detection_at": None,
            "last_mood": "neutral",
            "last_mood_updated_at": date.today().isoformat(),
            "recent_moods": [],
            "relationship_state": default_relationship_state(),
            "internal_state": InternalStateManager.default_state(),
            "relationship_history": [],
            "reminders": [],
            "session_archive": [],
            "last_background_synthesis_at": None,
            "last_background_synthesis_turn": 0,
            "last_memory_compaction_at": None,
            "last_memory_compaction_summary": "",
            "last_daily_checkin_at": None,
            "relationship_timeline": "",
            "memory_graph": default_memory_graph(),
            "mcp_local_store": {},
            "narrative_memories": [],
            "heritage_cross_links": [],
            "advice_audits": [],
            "environmental_cues_history": [],
            "longitudinal_insights": [],
            "learning_cycle_count": 0,
            "last_continuous_learning_at": None,
            "last_learning_turn": 0,
            "last_ical_sync": None,
        }
    )
    interaction_stats = dict(store.get("interaction_stats") or {})
    interaction_stats.setdefault("last_interaction", None)
    store["interaction_stats"] = interaction_stats
    return store


def default_planner_debug_state() -> dict:
    state = planner_debug_state()
    state["internal_monologue"] = list(state.get("internal_monologue") or [])
    state.update(
        {
            "updated_at": None,
            "user_input": "",
            "current_mood": "neutral",
            "planner_status": "idle",
            "planner_reason": "",
            "planner_tool": "",
            "planner_parameters": {},
            "planner_observation": "",
            "fallback_status": "idle",
            "fallback_reason": "",
            "fallback_tool": "",
            "fallback_observation": "",
            "final_path": "idle",
        }
    )
    return state


__all__ = [
    "DEFAULT_MEMORY_STORE",
    "DEFAULT_RELATIONSHIP_STATE",
    "PLANNER_DEBUG_STATE",
    "RELATIONSHIP_HYPOTHESIS_PROFILES",
    "RUNTIME_RELATIONSHIP_HYPOTHESIS_PROFILES",
    "base_memory_store",
    "default_memory_graph",
    "default_memory_store",
    "default_planner_debug_state",
    "default_relationship_hypotheses",
    "default_relationship_state",
    "planner_debug_state",
    "relationship_hypothesis_profiles",
    "datetime",
]
