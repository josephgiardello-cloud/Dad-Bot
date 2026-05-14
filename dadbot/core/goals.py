"""Goal System: persistent long-term objective memory for DadBot.

Goals capture what the user is trying to achieve across multiple turns.
They are stored in session state and surface in the planner and memory
ranker to keep the conversation goal-aware rather than purely reactive.

Design:
- GoalRecord: immutable-ish value type (use update_status / add_note)
- GoalStore: in-memory registry, loaded/saved via session state
- GoalStatus: active → completed | abandoned (one-way transitions)
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class GoalStatus(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class GoalRecord:
    """A single named objective tracked across conversation turns."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    description: str = ""
    status: str = GoalStatus.ACTIVE
    priority: str = GoalPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress_notes: list[str] = field(default_factory=list)
    source_turn: str = ""
    completion_turn: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": str(self.status),
            "priority": str(self.priority),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress_notes": list(self.progress_notes),
            "source_turn": self.source_turn,
            "completion_turn": self.completion_turn,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoalRecord:
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex[:16]),
            description=str(d.get("description") or ""),
            status=str(d.get("status") or GoalStatus.ACTIVE),
            priority=str(d.get("priority") or GoalPriority.MEDIUM),
            created_at=float(d.get("created_at") or time.time()),
            updated_at=float(d.get("updated_at") or time.time()),
            progress_notes=list(d.get("progress_notes") or []),
            source_turn=str(d.get("source_turn") or ""),
            completion_turn=str(d.get("completion_turn") or ""),
            tags=list(d.get("tags") or []),
        )


# ---------------------------------------------------------------------------
# GoalStore
# ---------------------------------------------------------------------------


class GoalStore:
    """In-memory goal registry.  Persist with ``to_state()`` / restore with ``load_from_state()``."""

    def __init__(self) -> None:
        self._goals: dict[str, GoalRecord] = {}

    # -- Queries --

    def all_goals(self) -> list[GoalRecord]:
        return list(self._goals.values())

    def active_goals(self) -> list[GoalRecord]:
        return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    def pending_goals(self) -> list[GoalRecord]:
        return [g for g in self._goals.values() if g.status == GoalStatus.PENDING]

    def get(self, goal_id: str) -> GoalRecord | None:
        return self._goals.get(goal_id)

    def size(self) -> int:
        return len(self._goals)

    # -- Mutations --

    def upsert(self, goal: GoalRecord) -> None:
        """Insert or update a goal."""
        goal.updated_at = time.time()
        self._goals[goal.id] = goal

    def add_note(self, goal_id: str, note: str) -> bool:
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        goal.progress_notes.append(str(note)[:500])
        goal.updated_at = time.time()
        return True

    def complete(self, goal_id: str, *, turn_id: str = "") -> bool:
        goal = self._goals.get(goal_id)
        if goal is None or goal.status == GoalStatus.COMPLETED:
            return False
        goal.status = GoalStatus.COMPLETED
        goal.completion_turn = turn_id
        goal.updated_at = time.time()
        return True

    def abandon(self, goal_id: str) -> bool:
        goal = self._goals.get(goal_id)
        if goal is None or goal.status in (GoalStatus.COMPLETED, GoalStatus.ABANDONED):
            return False
        goal.status = GoalStatus.ABANDONED
        goal.updated_at = time.time()
        return True

    # -- Serialisation --

    def load_from_state(self, state: list[dict[str, Any]]) -> None:
        """Restore from a serialised goal list (e.g. session_state['goals'])."""
        self._goals.clear()
        for item in state or []:
            try:
                rec = GoalRecord.from_dict(item)
                self._goals[rec.id] = rec
            except Exception:  # noqa: BLE001
                pass

    def to_state(self) -> list[dict[str, Any]]:
        """Serialise all goals for persistence in session state."""
        return [g.to_dict() for g in self._goals.values()]

    def active_dicts(self) -> list[dict[str, Any]]:
        """Return active goals as plain dicts (safe to store in TurnContext.state)."""
        return [g.to_dict() for g in self.active_goals()]

    def snapshot(self) -> dict[str, Any]:
        total = len(self._goals)
        active = sum(1 for g in self._goals.values() if g.status == GoalStatus.ACTIVE)
        return {
            "total": total,
            "active": active,
            "completed": sum(1 for g in self._goals.values() if g.status == GoalStatus.COMPLETED),
            "abandoned": total - active - sum(1 for g in self._goals.values() if g.status == GoalStatus.COMPLETED),
        }


# ---------------------------------------------------------------------------
# Goal detection helper
# ---------------------------------------------------------------------------

# Patterns that indicate the user is expressing a persistent intent/goal.
_GOAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:i want|i'd like|i wish)\s+to\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:my goal is|my objective is|i aim to)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:i'm trying to|i need to|i have to)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:i plan to|i intend to|i hope to)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:help me|assist me|i need help)\s+(?:to\s+)?(.+)", re.IGNORECASE),
    re.compile(r"(?:i'm working on|i've been working on)\s+(.+)", re.IGNORECASE),
]

_MIN_GOAL_WORDS = 3
_MAX_GOAL_DESC_LEN = 200


def detect_goal_in_input(user_input: str) -> GoalRecord | None:
    """Return a GoalRecord if the user input expresses a recognisable objective.

    Returns ``None`` when no goal pattern is detected.  The caller decides
    whether to persist the new goal.
    """
    text = str(user_input or "").strip()
    for pattern in _GOAL_PATTERNS:
        match = pattern.search(text)
        if match:
            desc = match.group(1).strip().rstrip("!?.").strip()
            # Ignore trivially short matches (noise).
            if len(desc.split()) < _MIN_GOAL_WORDS:
                continue
            desc = desc[:_MAX_GOAL_DESC_LEN]
            return GoalRecord(description=desc)
    return None
