from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dadbot.uril.models import RepoSignal, RepoSignalBus


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SignalCollectionOptions:
    pytest_junit_path: Path | None = None
    benchmark_snapshot_dir: Path | None = None
    phase4_auditor_json: Path | None = None
    filesystem_json: Path | None = None
    stress_json: Path | None = None
    run_probes: bool = True


def _extract_json_payload(raw: str) -> dict[str, Any] | None:
    text = str(raw or "")
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] != "{":
            continue
        candidate = text[idx:].strip()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _run_repo_python(args: list[str]) -> dict[str, Any] | None:
    cmd = [sys.executable, *args]
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    payload = _extract_json_payload((completed.stdout or "") + "\n" + (completed.stderr or ""))
    return payload


def _collect_pytest_signals(junit_path: Path | None) -> list[RepoSignal]:
    default = Path(os.environ.get("TEMP", ".")) / "dadbot_full_noncloud_final.xml"
    xml_path = junit_path or default
    if not xml_path.exists():
        return [
            RepoSignal(
                subsystem="repo",
                category="correctness",
                score=0.0,
                metadata={"missing_junit": str(xml_path)},
            )
        ]

    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

    tests = failures = errors = skipped = 0
    per_suite: list[RepoSignal] = []
    for suite in suites:
        t = int(suite.attrib.get("tests", 0) or 0)
        f = int(suite.attrib.get("failures", 0) or 0)
        e = int(suite.attrib.get("errors", 0) or 0)
        s = int(suite.attrib.get("skipped", 0) or 0)
        tests += t
        failures += f
        errors += e
        skipped += s
        if t > 0:
            passed = max(0, t - f - e - s)
            per_suite.append(
                RepoSignal(
                    subsystem=str(suite.attrib.get("name") or "pytest_suite"),
                    category="correctness",
                    score=passed / t,
                    metadata={"tests": t, "failures": f, "errors": e, "skipped": s},
                )
            )

    passed_total = max(0, tests - failures - errors - skipped)
    summary = RepoSignal(
        subsystem="repo",
        category="correctness",
        score=(passed_total / tests) if tests else 0.0,
        metadata={
            "tests": tests,
            "passed": passed_total,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "source": str(xml_path),
        },
    )
    return [summary, *per_suite]


def _collect_phase4_auditor_signals(path: Path | None, run_probe: bool) -> list[RepoSignal]:
    payload: dict[str, Any] | None = None
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif run_probe:
        payload = _run_repo_python(["tools/repo_phase4_auditor.py"])

    if not isinstance(payload, dict):
        return [RepoSignal("phase4", "architecture", 0.0, {"missing": True})]

    status = str(payload.get("phase4_status") or "FAIL").upper()
    coverage = dict(payload.get("coverage") or {})
    score = sum(float(v or 0.0) for v in coverage.values()) / max(len(coverage), 1)
    if status == "PASS":
        score = max(score, 0.9)

    return [
        RepoSignal(
            subsystem="phase4",
            category="architecture",
            score=score,
            metadata={"status": status, "coverage": coverage, "critical_gaps": payload.get("critical_gaps", [])},
        )
    ]


def _collect_filesystem_signals(path: Path | None, run_probe: bool) -> list[RepoSignal]:
    payload: dict[str, Any] | None = None
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif run_probe:
        payload = _run_repo_python(["tools/repo_filesystem_checker.py"])

    if not isinstance(payload, dict):
        return [RepoSignal("filesystem", "architecture", 0.0, {"missing": True})]

    status = str(payload.get("status") or "FAIL").upper()
    counts = dict(payload.get("counts") or {})
    missing = int(counts.get("missing_expected_files", 0) or 0) + int(counts.get("missing_expected_tests", 0) or 0)
    underoptimized = int(counts.get("underoptimized", 0) or 0)

    base = 1.0
    base -= min(0.8, missing * 0.2)
    base -= min(0.3, underoptimized * 0.05)
    if status == "PASS":
        base = max(base, 0.95)
    elif status == "WARN":
        base = max(base, 0.75)

    return [
        RepoSignal(
            subsystem="filesystem",
            category="architecture",
            score=base,
            metadata={"status": status, "counts": counts, "underoptimized": payload.get("underoptimized", [])},
        )
    ]


def _collect_benchmark_signals(snapshot_dir: Path | None) -> list[RepoSignal]:
    folder = snapshot_dir or (ROOT / "evaluation" / "snapshots")
    index_path = folder / "index.json"
    if not index_path.exists():
        return [RepoSignal("benchmark", "performance", 0.0, {"missing_index": str(index_path)})]

    try:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return [RepoSignal("benchmark", "performance", 0.0, {"invalid_index": str(index_path)})]

    entries = list(index_payload.get("entries") or [])
    if not entries:
        return [RepoSignal("benchmark", "performance", 0.0, {"empty": True})]

    latest = entries[0]
    snapshot_id = str(latest.get("snapshot_id") or "")
    snapshot_file = folder / f"{snapshot_id}.json"
    if not snapshot_file.exists():
        return [RepoSignal("benchmark", "performance", 0.0, {"missing_snapshot": snapshot_id})]

    payload = json.loads(snapshot_file.read_text(encoding="utf-8"))
    category_aggregates = dict(payload.get("category_aggregates") or {})

    signals = []
    for category, value in category_aggregates.items():
        cat = str(category)
        score = float(value or 0.0)
        normalized_category = "performance" if cat in {"planning", "tool", "ux"} else "determinism"
        if cat == "memory":
            normalized_category = "performance"
        signals.append(
            RepoSignal(
                subsystem=f"benchmark_{cat}",
                category=normalized_category,
                score=score,
                metadata={"snapshot_id": snapshot_id, "raw_category": cat},
            )
        )

    if not signals:
        signals.append(RepoSignal("benchmark", "performance", 0.0, {"no_aggregates": True}))
    return signals


def _collect_stress_signals(path: Path | None) -> list[RepoSignal]:
    if path is None or not path.exists():
        return [RepoSignal("stress", "determinism", 0.5, {"missing": True})]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [RepoSignal("stress", "determinism", 0.0, {"invalid": True, "path": str(path)})]

    score_raw = float(payload.get("score", 0) or 0)
    normalized = score_raw / 100.0 if score_raw > 1 else score_raw
    return [
        RepoSignal(
            subsystem="stress",
            category="determinism",
            score=normalized,
            metadata={
                "phase4_certification": payload.get("phase4_certification"),
                "failures": payload.get("failures", []),
                "risk_flags": payload.get("risk_flags", []),
            },
        )
    ]


def collect_signal_bus(options: SignalCollectionOptions) -> RepoSignalBus:
    bus = RepoSignalBus()
    bus.extend(_collect_pytest_signals(options.pytest_junit_path))
    bus.extend(_collect_benchmark_signals(options.benchmark_snapshot_dir))
    bus.extend(_collect_phase4_auditor_signals(options.phase4_auditor_json, options.run_probes))
    bus.extend(_collect_filesystem_signals(options.filesystem_json, options.run_probes))
    bus.extend(_collect_stress_signals(options.stress_json))
    return bus
