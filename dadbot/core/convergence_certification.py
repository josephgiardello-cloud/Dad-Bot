from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _extract_breach_count(run_summary: Any) -> int:
    if isinstance(run_summary, int):
        return max(0, int(run_summary))
    if not isinstance(run_summary, dict):
        return 0

    direct = run_summary.get("breach_count")
    if isinstance(direct, (int, float)):
        return max(0, int(direct))

    direct_total = run_summary.get("total_breaches")
    if isinstance(direct_total, (int, float)):
        return max(0, int(direct_total))

    metrics = run_summary.get("canonical_gate_metrics")
    if isinstance(metrics, dict):
        nested_total = metrics.get("total_breaches")
        if isinstance(nested_total, (int, float)):
            return max(0, int(nested_total))

    return 0


@dataclass(frozen=True)
class ConvergenceCertification:
    required_runs: int
    evaluated_runs: int
    zero_breach_runs: int
    total_breaches: int
    failure_runs: tuple[int, ...]
    converged: bool
    legacy_path_deletable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_runs": self.required_runs,
            "evaluated_runs": self.evaluated_runs,
            "zero_breach_runs": self.zero_breach_runs,
            "total_breaches": self.total_breaches,
            "failure_runs": list(self.failure_runs),
            "converged": self.converged,
            "legacy_path_deletable": self.legacy_path_deletable,
        }


def certify_canonical_convergence(
    run_summaries: Iterable[Any],
    *,
    required_runs: int,
) -> ConvergenceCertification:
    required = max(1, int(required_runs))
    items = list(run_summaries)
    window = items[-required:]

    failures: list[int] = []
    total_breaches = 0
    for index, item in enumerate(window, start=1):
        breaches = _extract_breach_count(item)
        total_breaches += breaches
        if breaches > 0:
            failures.append(index)

    evaluated = len(window)
    zero_breach_runs = max(0, evaluated - len(failures))
    converged = evaluated >= required and total_breaches == 0

    return ConvergenceCertification(
        required_runs=required,
        evaluated_runs=evaluated,
        zero_breach_runs=zero_breach_runs,
        total_breaches=total_breaches,
        failure_runs=tuple(failures),
        converged=converged,
        legacy_path_deletable=converged,
    )
