from __future__ import annotations

from dadbot.core.convergence_certification import certify_canonical_convergence


def test_convergence_certified_when_zero_breaches_across_required_runs() -> None:
    runs = [{"breach_count": 0} for _ in range(5)]

    report = certify_canonical_convergence(runs, required_runs=5)

    assert report.converged is True
    assert report.legacy_path_deletable is True
    assert report.total_breaches == 0
    assert report.failure_runs == ()


def test_convergence_not_certified_when_any_breach_exists() -> None:
    runs = [
        {"canonical_gate_metrics": {"total_breaches": 0}},
        {"canonical_gate_metrics": {"total_breaches": 2}},
        {"breach_count": 0},
    ]

    report = certify_canonical_convergence(runs, required_runs=3)

    assert report.converged is False
    assert report.legacy_path_deletable is False
    assert report.total_breaches == 2
    assert report.failure_runs == (2,)


def test_convergence_requires_full_window() -> None:
    runs = [{"breach_count": 0}, {"breach_count": 0}]

    report = certify_canonical_convergence(runs, required_runs=3)

    assert report.evaluated_runs == 2
    assert report.converged is False
    assert report.legacy_path_deletable is False
