"""Thin memory events stub."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryEventType(str, Enum):
    """Memory event type enum."""

    WRITE = "write"
    READ = "read"
    UPDATE = "update"
    REWARD = "reward"
    POLICY_UPDATE = "policy_update"


@dataclass
class MemoryEvent:
    """Memory event stub."""

    event_type: MemoryEventType
    payload: dict
    agent_id: str = ""


class MemoryEventEmitter:
    """Memory event emitter stub."""

    def emit(self, *, event_type: MemoryEventType, payload: dict, agent_id: str = "") -> MemoryEvent:
        return MemoryEvent(event_type=event_type, payload=dict(payload or {}), agent_id=str(agent_id or ""))


class MemoryReducer:
    """Memory reducer stub."""

    def reduce(self, *, events: list[MemoryEvent], memory_state: dict) -> dict:
        updated = dict(memory_state or {})
        episodic = list(updated.get("episodic") or [])
        policy_bias = dict(updated.get("policy_bias") or {})

        for event in list(events or []):
            if event.event_type == MemoryEventType.REWARD:
                episodic.append(dict(event.payload or {}))
            elif event.event_type == MemoryEventType.POLICY_UPDATE:
                payload = dict(event.payload or {})
                for key in ("strategy_weights", "tool_preferences", "structural_patterns"):
                    current = dict(policy_bias.get(key) or {})
                    current.update(dict(payload.get(key) or {}))
                    policy_bias[key] = current
                if "decay_rate" in payload:
                    policy_bias["decay_rate"] = payload.get("decay_rate")

        updated["episodic"] = episodic
        updated["policy_bias"] = policy_bias
        return updated
