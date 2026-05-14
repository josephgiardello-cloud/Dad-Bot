from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dadbot.uril.models import BenchmarkProfile, RepoSignalBus, SubsystemHealth

TIER_A_OSS = BenchmarkProfile(
    planning=0.62,
    tools=0.65,
    memory=0.58,
    determinism=0.4,
    observability=0.45,
    safety=0.5,
)

TIER_B_PRODUCTION = BenchmarkProfile(
    planning=0.8,
    tools=0.82,
    memory=0.78,
    determinism=0.86,
    observability=0.82,
    safety=0.84,
)

TIER_C_RESEARCH = BenchmarkProfile(
    planning=0.9,
    tools=0.88,
    memory=0.86,
    determinism=0.94,
    observability=0.9,
    safety=0.9,
)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _health_lookup(rows: list[SubsystemHealth], key: str) -> float:
    for row in rows:
        if row.subsystem == key:
            return row.score
    return 0.0


def _signal_score(
    bus: RepoSignalBus,
    subsystem_prefix: str,
    category: str | None = None,
) -> float:
    matches = []
    for signal in bus.signals:
        if subsystem_prefix and not signal.subsystem.startswith(subsystem_prefix):
            continue
        if category and signal.category != category:
            continue
        matches.append(signal.score)
    return _mean(matches)


def build_system_profile(
    signal_bus: RepoSignalBus,
    subsystem_health: list[SubsystemHealth],
) -> BenchmarkProfile:
    planning = _mean(
        [
            _signal_score(signal_bus, "benchmark_planning"),
            _health_lookup(subsystem_health, "graph_engine"),
            _health_lookup(subsystem_health, "dadbot_core"),
        ],
    )
    tools = _mean(
        [
            _signal_score(signal_bus, "benchmark_tool"),
            _health_lookup(subsystem_health, "tool_registry"),
            _health_lookup(subsystem_health, "mcp_layer"),
        ],
    )
    memory = _mean(
        [
            _signal_score(signal_bus, "benchmark_memory"),
            _health_lookup(subsystem_health, "persistence"),
            _health_lookup(subsystem_health, "graph_engine"),
        ],
    )
    determinism = _mean(
        [
            signal_bus.mean_for_category("determinism"),
            _health_lookup(subsystem_health, "kernel"),
            _health_lookup(subsystem_health, "validator"),
        ],
    )
    observability = _mean(
        [
            signal_bus.mean_for_category("observability"),
            _health_lookup(subsystem_health, "observability"),
            _health_lookup(subsystem_health, "kernel"),
        ],
    )
    safety = _mean(
        [
            signal_bus.mean_for_category("architecture"),
            _health_lookup(subsystem_health, "validator"),
            _health_lookup(subsystem_health, "dadbot_core"),
        ],
    )

    return BenchmarkProfile(
        planning=max(0.0, min(1.0, planning)),
        tools=max(0.0, min(1.0, tools)),
        memory=max(0.0, min(1.0, memory)),
        determinism=max(0.0, min(1.0, determinism)),
        observability=max(0.0, min(1.0, observability)),
        safety=max(0.0, min(1.0, safety)),
    )


def _distance(system: BenchmarkProfile, target: BenchmarkProfile) -> dict[str, float]:
    s = asdict(system)
    t = asdict(target)
    return {k: abs(float(s[k]) - float(t[k])) for k in s}


def benchmark_alignment_report(system: BenchmarkProfile) -> dict[str, Any]:
    tiers = {
        "tier_a_oss": TIER_A_OSS,
        "tier_b_production": TIER_B_PRODUCTION,
        "tier_c_research": TIER_C_RESEARCH,
    }

    comparisons: dict[str, Any] = {}
    for name, profile in tiers.items():
        dist = _distance(system, profile)
        weighted_gap = _mean(list(dist.values()))
        comparisons[name] = {
            "distance": {k: round(v, 3) for k, v in dist.items()},
            "weighted_gap": round(weighted_gap, 3),
            "alignment_score": round(1.0 - weighted_gap, 3),
        }

    prod_dist = comparisons["tier_b_production"]["distance"]
    return {
        "system_profile": {k: round(v, 3) for k, v in asdict(system).items()},
        "tiers": comparisons,
        "production_gap_percent": {k: round(v * 100.0, 1) for k, v in prod_dist.items()},
    }
