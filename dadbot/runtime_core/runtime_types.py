"""Runtime type definitions shared across dadbot.runtime_core."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanResult:
    """Outcome of the planner phase for a single turn."""

    candidates: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_hint: dict[str, Any] = field(default_factory=dict)


__all__ = ["PlanResult"]
