from __future__ import annotations

import json

from tools.repo_filesystem_checker import build_repo_filesystem_report


def test_repo_filesystem_checker_reports_expected_gaps_and_optimizations() -> None:
    report = build_repo_filesystem_report()

    assert "status" in report
    assert "missing_expected_files" in report
    assert "missing_expected_tests" in report
    assert "underoptimized" in report

    # Hard failure only when required files/tests are missing.
    assert report["status"] in {"PASS", "WARN"}, json.dumps(report, indent=2, sort_keys=True)


def test_repo_filesystem_checker_report_shape_is_stable() -> None:
    report = build_repo_filesystem_report()
    counts = report.get("counts") or {}

    assert isinstance(counts.get("missing_expected_files"), int)
    assert isinstance(counts.get("missing_expected_tests"), int)
    assert isinstance(counts.get("underoptimized"), int)
