"""Unified System State Model — Global Truth Layer.

Gap 1 of the remaining architecture: each subsystem (tool memory, coherence
engine, causal graph, policy engine) has its own correct view, but no single
unified "what is the system right now" truth exists.

This module provides SystemStateSnapshot: one coherent aggregate of all
subsystem state at a point in time.  It is:

  Queryable:    answer "is the system healthy?" in one call
  Composable:   built from existing subsystem views, not recomputed from scratch
  Snapshotable: can be compared over time via SystemStateHistory

The canonical entry point is SystemStateBuilder.build(...) which derives
overall_health, active_fault_count, and policy_posture from the inputs.

Usage
-----
    builder = SystemStateBuilder()
    snapshot = builder.build(
        tool_profiles=profiles,
        coherent_memory=memory_view,
        causal_graph=graph,
    )
    print(snapshot.overall_health)    # SystemHealthStatus.HEALTHY
    print(snapshot.degraded_tools())  # ["slow_api_tool"]
    print(snapshot.policy_posture)    # "lenient"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dadbot.core.memory_feedback_policy import ToolMemoryProfile
from dadbot.core.memory_coherence_engine import CoherentMemoryView
from dadbot.core.causal_dependency_graph import CausalDepGraph


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------


class SystemHealthStatus(Enum):
    """Overall health classification of the system at a snapshot point."""

    HEALTHY = "healthy"
    """All tools operational; no active faults."""

    DEGRADED = "degraded"
    """Some tools are unhealthy but the system continues to operate."""

    CRITICAL = "critical"
    """Majority of tools are unhealthy OR fault count is severe."""

    UNKNOWN = "unknown"
    """No tool profile data is available to assess health."""


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class SystemStateSnapshot:
    """The unified truth state of the system at a specific moment.

    Build this via SystemStateBuilder.build(), not directly (unless writing tests).

    Attributes
    ----------
    timestamp_ms:
        Epoch milliseconds when this snapshot was taken.
    tool_profiles:
        Memory profiles for every known tool.
    coherent_memory:
        The coherent memory view at snapshot time (None if not available).
    causal_graph:
        The causal dependency graph at snapshot time (None if not available).
    overall_health:
        Computed health classification.
    active_fault_count:
        Total abort + escalation events across all tools — used as a fault signal.
    policy_posture:
        Recommended policy stance: "aggressive" | "moderate" | "lenient".
    metadata:
        Optional extra context (e.g., build version, node ID).
    """

    timestamp_ms: int
    tool_profiles: dict[str, ToolMemoryProfile]
    coherent_memory: CoherentMemoryView | None = None
    causal_graph: CausalDepGraph | None = None
    overall_health: SystemHealthStatus = SystemHealthStatus.UNKNOWN
    active_fault_count: int = 0
    policy_posture: str = "moderate"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        return self.overall_health == SystemHealthStatus.HEALTHY

    @property
    def is_operational(self) -> bool:
        """True when the system can still process requests (not CRITICAL)."""
        return self.overall_health != SystemHealthStatus.CRITICAL

    def degraded_tools(self) -> list[str]:
        """Names of tools whose memory profile marks them as unhealthy."""
        return [name for name, p in self.tool_profiles.items() if not p.is_healthy]

    def healthy_tool_count(self) -> int:
        return sum(1 for p in self.tool_profiles.values() if p.is_healthy)

    def total_tool_count(self) -> int:
        return len(self.tool_profiles)

    def memory_entry_count(self) -> int:
        return len(self.coherent_memory.entries) if self.coherent_memory else 0

    def causal_node_count(self) -> int:
        return len(self.causal_graph) if self.causal_graph else 0

    def snapshot_summary(self) -> dict[str, Any]:
        """Return a flat summary dict for logging / monitoring."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "overall_health": self.overall_health.value,
            "tool_count": self.total_tool_count(),
            "healthy_tools": self.healthy_tool_count(),
            "degraded_tools": self.degraded_tools(),
            "active_fault_count": self.active_fault_count,
            "policy_posture": self.policy_posture,
            "memory_entry_count": self.memory_entry_count(),
            "causal_node_count": self.causal_node_count(),
        }


# ---------------------------------------------------------------------------
# Derived-field computation helpers
# ---------------------------------------------------------------------------


def _compute_overall_health(
    profiles: dict[str, ToolMemoryProfile],
    active_fault_count: int,
) -> SystemHealthStatus:
    if not profiles:
        return SystemHealthStatus.UNKNOWN

    total = len(profiles)
    unhealthy = sum(1 for p in profiles.values() if not p.is_healthy)
    unhealthy_fraction = unhealthy / total

    if active_fault_count >= 10 or unhealthy_fraction >= 0.6:
        return SystemHealthStatus.CRITICAL
    if unhealthy_fraction > 0 or active_fault_count > 0:
        return SystemHealthStatus.DEGRADED
    return SystemHealthStatus.HEALTHY


def _compute_active_faults(profiles: dict[str, ToolMemoryProfile]) -> int:
    """Sum of abort + escalation counts across all tools."""
    return sum(p.abort_count + p.escalation_count for p in profiles.values())


def _compute_policy_posture(profiles: dict[str, ToolMemoryProfile]) -> str:
    """Derive recommended policy posture from aggregate tool profile stats.

    Returns "aggressive" | "moderate" | "lenient".
    """
    if not profiles:
        return "moderate"

    total_executions = sum(p.total_executions for p in profiles.values())
    if total_executions == 0:
        return "moderate"

    total_aborts = sum(p.abort_count for p in profiles.values())
    total_escalations = sum(p.escalation_count for p in profiles.values())

    abort_rate = total_aborts / total_executions
    escalation_rate = total_escalations / total_executions

    if abort_rate >= 0.3 or escalation_rate >= 0.4:
        return "aggressive"   # Reduce retries; system is struggling

    avg_reliability = sum(p.reliability_score for p in profiles.values()) / len(profiles)
    if avg_reliability >= 0.8:
        return "lenient"      # System healthy; allow more recovery attempts

    return "moderate"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class SystemStateBuilder:
    """Builds SystemStateSnapshot from subsystem views.

    Derived fields (overall_health, active_fault_count, policy_posture) are
    always computed here — callers do not set them directly.
    """

    def build(
        self,
        tool_profiles: dict[str, ToolMemoryProfile],
        *,
        coherent_memory: CoherentMemoryView | None = None,
        causal_graph: CausalDepGraph | None = None,
        now_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SystemStateSnapshot:
        """Produce a unified state snapshot from subsystem views."""
        ts = now_ms if now_ms is not None else int(time.time() * 1000)
        active_faults = _compute_active_faults(tool_profiles)
        overall_health = _compute_overall_health(tool_profiles, active_faults)
        policy_posture = _compute_policy_posture(tool_profiles)

        return SystemStateSnapshot(
            timestamp_ms=ts,
            tool_profiles=dict(tool_profiles),
            coherent_memory=coherent_memory,
            causal_graph=causal_graph,
            overall_health=overall_health,
            active_fault_count=active_faults,
            policy_posture=policy_posture,
            metadata=dict(metadata or {}),
        )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class SystemStateHistory:
    """Rolling window of SystemStateSnapshot objects for trend analysis.

    Parameters
    ----------
    max_snapshots:
        Maximum snapshots to retain.  Oldest is evicted once the cap is reached.
    """

    def __init__(self, max_snapshots: int = 100) -> None:
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")
        self._max = max_snapshots
        self._snapshots: list[SystemStateSnapshot] = []

    def push(self, snapshot: SystemStateSnapshot) -> None:
        """Record a snapshot, evicting the oldest if at capacity."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max:
            self._snapshots.pop(0)

    def latest(self) -> SystemStateSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def all_snapshots(self) -> list[SystemStateSnapshot]:
        return list(self._snapshots)

    def health_timeline(self) -> list[tuple[int, str]]:
        """Return [(timestamp_ms, health_value)] for all stored snapshots."""
        return [(s.timestamp_ms, s.overall_health.value) for s in self._snapshots]

    def fault_trend(self) -> list[int]:
        """Return active_fault_count for each stored snapshot."""
        return [s.active_fault_count for s in self._snapshots]

    def is_degrading(self) -> bool:
        """True if the last 3 snapshots show a monotonically worsening health."""
        if len(self._snapshots) < 3:
            return False
        rank = {
            SystemHealthStatus.HEALTHY: 3,
            SystemHealthStatus.DEGRADED: 2,
            SystemHealthStatus.CRITICAL: 1,
            SystemHealthStatus.UNKNOWN: 0,
        }
        recent = [rank[s.overall_health] for s in self._snapshots[-3:]]
        return recent[0] > recent[-1]   # started better than ended


__all__ = [
    "SystemHealthStatus",
    "SystemStateSnapshot",
    "SystemStateBuilder",
    "SystemStateHistory",
]
