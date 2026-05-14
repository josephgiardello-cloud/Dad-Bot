from __future__ import annotations

from dadbot.uril.models import RefactorSuggestion, RepoSignalBus, SubsystemHealth


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def generate_refactor_suggestions(
    signal_bus: RepoSignalBus,
    subsystem_health: list[SubsystemHealth],
) -> list[RefactorSuggestion]:
    suggestions: list[RefactorSuggestion] = []

    high_risk = [row for row in subsystem_health if ((row.coupling + row.centrality + row.blast_radius) / 3.0) >= 0.65]
    for row in sorted(
        high_risk,
        key=lambda r: r.coupling + r.centrality + r.blast_radius,
        reverse=True,
    ):
        suggestions.append(
            RefactorSuggestion(
                target=row.subsystem,
                issue="over-centralized orchestration risk",
                fix=[
                    "split stage responsibilities into narrower services",
                    "introduce explicit adapter boundary around side effects",
                    "reduce import fan-in by moving policy code into dedicated module",
                ],
                impact="high",
                risk="medium",
            ),
        )

    observability_signals = [s.score for s in signal_bus.by_category("observability")]
    if _mean(observability_signals) < 0.75:
        suggestions.append(
            RefactorSuggestion(
                target="observability",
                issue="trace/export coverage below production target",
                fix=[
                    "enforce default trace level in runtime bootstrap",
                    "add exporter health heartbeat checks",
                    "expand bridge tests for partial-backend failure modes",
                ],
                impact="high",
                risk="low",
            ),
        )

    correctness_mean = signal_bus.mean_for_category("correctness")
    if correctness_mean < 0.98:
        suggestions.append(
            RefactorSuggestion(
                target="test_harness",
                issue="correctness below strict gate",
                fix=[
                    "cluster failing modules by subsystem and add focused regression tests",
                    "promote flaky timeout-sensitive tests into deterministic stubs",
                ],
                impact="high",
                risk="low",
            ),
        )

    if not suggestions:
        suggestions.append(
            RefactorSuggestion(
                target="repo",
                issue="no critical architecture risks detected",
                fix=[
                    "maintain current layering",
                    "add golden snapshot drift checks for URIL outputs",
                ],
                impact="medium",
                risk="low",
            ),
        )

    return suggestions
