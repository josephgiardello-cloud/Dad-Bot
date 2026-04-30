from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SignalCategory = str


@dataclass(frozen=True)
class RepoSignal:
    subsystem: str
    category: SignalCategory  # correctness | performance | architecture | determinism | observability | safety
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> RepoSignal:
        clamped = max(0.0, min(1.0, float(self.score)))
        return RepoSignal(
            subsystem=self.subsystem,
            category=self.category,
            score=clamped,
            metadata=dict(self.metadata),
        )


@dataclass
class RepoSignalBus:
    signals: list[RepoSignal] = field(default_factory=list)

    def add(self, signal: RepoSignal) -> None:
        self.signals.append(signal.normalized())

    def extend(self, signals: list[RepoSignal]) -> None:
        for signal in signals:
            self.add(signal)

    def by_category(self, category: SignalCategory) -> list[RepoSignal]:
        return [s for s in self.signals if s.category == category]

    def by_subsystem(self, subsystem: str) -> list[RepoSignal]:
        return [s for s in self.signals if s.subsystem == subsystem]

    def mean_for_category(self, category: SignalCategory) -> float:
        bucket = self.by_category(category)
        if not bucket:
            return 0.0
        return sum(s.score for s in bucket) / len(bucket)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": [
                {
                    "subsystem": s.subsystem,
                    "category": s.category,
                    "score": s.score,
                    "metadata": s.metadata,
                }
                for s in self.signals
            ],
        }


@dataclass
class SubsystemHealth:
    subsystem: str
    score: float
    coupling: float
    centrality: float
    blast_radius: float
    test_coverage_ratio: float
    runtime_criticality: float


@dataclass(frozen=True)
class BenchmarkProfile:
    planning: float
    tools: float
    memory: float
    determinism: float
    observability: float
    safety: float

    def to_dict(self) -> dict[str, float]:
        return {
            "planning": self.planning,
            "tools": self.tools,
            "memory": self.memory,
            "determinism": self.determinism,
            "observability": self.observability,
            "safety": self.safety,
        }


@dataclass
class RefactorSuggestion:
    target: str
    issue: str
    fix: list[str]
    impact: str
    risk: str


@dataclass
class UrailReport:
    phase4_completion: dict[str, float]
    subsystem_health: dict[str, float]
    benchmark_alignment: dict[str, Any]
    risk_heatmap: list[dict[str, Any]]
    upgrade_recommendations: list[dict[str, Any]]
    signal_bus: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase4_completion": self.phase4_completion,
            "subsystem_health": self.subsystem_health,
            "benchmark_alignment": self.benchmark_alignment,
            "risk_heatmap": self.risk_heatmap,
            "upgrade_recommendations": self.upgrade_recommendations,
            "signal_bus": self.signal_bus,
        }
