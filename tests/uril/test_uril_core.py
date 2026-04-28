from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dadbot.uril.report import build_uril_report
from dadbot.uril.signal_bus import SignalCollectionOptions


def _write_junit(path: Path, tests: int, failures: int, errors: int, skipped: int) -> None:
    path.write_text(
        (
            "<testsuites>"
            f"<testsuite name='repo' tests='{tests}' failures='{failures}' errors='{errors}' skipped='{skipped}'/>"
            "</testsuites>"
        ),
        encoding="utf-8",
    )


def _write_snapshot_index(snapshot_dir: Path, planning: float, tool: float, memory: float) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "entries": [
            {
                "snapshot_id": "bench-test-1",
                "created_at": "2026-01-01T00:00:00Z",
                "run_label": "unit",
                "execution_mode": "mock",
                "scenario_count": 15,
                "overall_average": 0.75,
                "scoring_engine_hash": "abc",
            }
        ]
    }
    (snapshot_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")

    snapshot = {
        "snapshot_id": "bench-test-1",
        "category_aggregates": {
            "planning": planning,
            "tool": tool,
            "memory": memory,
        },
    }
    (snapshot_dir / "bench-test-1.json").write_text(json.dumps(snapshot), encoding="utf-8")


def _write_phase4_json(path: Path, status: str = "PASS") -> None:
    payload = {
        "phase4_status": status,
        "coverage": {
            "core": 1.0,
            "observability": 1.0,
            "export": 1.0,
            "fuzzing": 1.0,
        },
        "critical_gaps": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_filesystem_json(path: Path, status: str = "PASS") -> None:
    payload = {
        "status": status,
        "counts": {
            "missing_expected_files": 0,
            "missing_expected_tests": 0,
            "underoptimized": 0,
        },
        "underoptimized": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _options(tmp_path: Path, failures: int = 0) -> SignalCollectionOptions:
    tmp_path.mkdir(parents=True, exist_ok=True)

    junit = tmp_path / "junit.xml"
    phase4 = tmp_path / "phase4.json"
    fs = tmp_path / "fs.json"
    snaps = tmp_path / "snaps"

    _write_junit(junit, tests=100, failures=failures, errors=0, skipped=0)
    _write_phase4_json(phase4, status="PASS")
    _write_filesystem_json(fs, status="PASS")
    _write_snapshot_index(snaps, planning=0.75, tool=0.7, memory=0.8)

    return SignalCollectionOptions(
        pytest_junit_path=junit,
        phase4_auditor_json=phase4,
        filesystem_json=fs,
        benchmark_snapshot_dir=snaps,
        stress_json=None,
        run_probes=False,
    )


def test_signal_consistency(tmp_path: Path) -> None:
    payload = build_uril_report(_options(tmp_path, failures=0))

    categories = [s["category"] for s in payload["signal_bus"]["signals"]]
    assert "correctness" in categories
    assert "architecture" in categories
    assert "determinism" in categories


def test_scoring_stability(tmp_path: Path) -> None:
    options = _options(tmp_path, failures=0)
    a = build_uril_report(options)
    b = build_uril_report(options)

    ah = hashlib.sha256(json.dumps(a, sort_keys=True).encode("utf-8")).hexdigest()
    bh = hashlib.sha256(json.dumps(b, sort_keys=True).encode("utf-8")).hexdigest()
    assert ah == bh


def test_mock_vs_real_divergence_signal(tmp_path: Path) -> None:
    clean = build_uril_report(_options(tmp_path / "clean", failures=0))
    degraded = build_uril_report(_options(tmp_path / "degraded", failures=20))

    assert degraded["phase4_completion"]["correctness"] < clean["phase4_completion"]["correctness"]


def test_truth_invariance_same_repo_same_score(tmp_path: Path) -> None:
    options = _options(tmp_path, failures=1)
    first = build_uril_report(options)
    second = build_uril_report(options)

    assert first["phase4_completion"] == second["phase4_completion"]
    assert first["subsystem_health"] == second["subsystem_health"]
