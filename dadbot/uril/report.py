from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dadbot.uril.architecture import build_subsystem_health, subsystem_health_map, subsystem_risk_heatmap
from dadbot.uril.benchmark import benchmark_alignment_report, build_system_profile
from dadbot.uril.models import UrailReport
from dadbot.uril.oracle import generate_refactor_suggestions
from dadbot.uril.signal_bus import SignalCollectionOptions, collect_signal_bus


ROOT = Path(__file__).resolve().parents[2]


def _pct(value: float) -> float:
    return round(max(0.0, min(1.0, value)) * 100.0, 1)


def _phase4_completion(signal_bus, subsystem_rows, benchmark_alignment: dict[str, Any]) -> dict[str, float]:
    observability = signal_bus.mean_for_category("observability")
    if observability == 0.0:
        obs_row = [r for r in subsystem_rows if r.subsystem == "observability"]
        observability = obs_row[0].score if obs_row else 0.0

    return {
        "correctness": _pct(signal_bus.mean_for_category("correctness")),
        "architecture": _pct(signal_bus.mean_for_category("architecture")),
        "determinism": _pct(signal_bus.mean_for_category("determinism")),
        "observability": _pct(observability),
        "benchmark_alignment": _pct(benchmark_alignment["tiers"]["tier_b_production"]["alignment_score"]),
    }


def _progress_summary(signal_bus) -> dict[str, Any]:
    repo_correctness = [s for s in signal_bus.by_category("correctness") if s.subsystem == "repo"]
    if not repo_correctness:
        return {"tests": 0, "passed": 0, "failures": 0, "errors": 0, "skipped": 0, "pass_rate": 0.0}

    meta = dict(repo_correctness[0].metadata or {})
    tests = int(meta.get("tests", 0) or 0)
    passed = int(meta.get("passed", 0) or 0)
    failures = int(meta.get("failures", 0) or 0)
    errors = int(meta.get("errors", 0) or 0)
    skipped = int(meta.get("skipped", 0) or 0)
    pass_rate = (passed / tests) if tests else 0.0
    return {
        "tests": tests,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "pass_rate": round(pass_rate, 4),
    }


def _proven_aspects(signal_bus, subsystem_rows, benchmark_alignment: dict[str, Any]) -> list[str]:
    aspects: list[str] = []

    correctness = signal_bus.mean_for_category("correctness")
    if correctness >= 0.98:
        aspects.append("Full non-cloud correctness gate is effectively production-green")

    determinism = signal_bus.mean_for_category("determinism")
    if determinism >= 0.9:
        aspects.append("Determinism and replay-related checks are strongly validated")

    phase4_arch = [s for s in signal_bus.by_subsystem("phase4") if s.category == "architecture"]
    if phase4_arch and phase4_arch[0].score >= 0.9:
        aspects.append("Phase 4 architecture/compliance auditor reports no critical gaps")

    persistence_row = next((r for r in subsystem_rows if r.subsystem == "persistence"), None)
    if persistence_row and persistence_row.score >= 0.8:
        aspects.append("Persistence surfaces are stable under current structural health model")

    prod_alignment = benchmark_alignment["tiers"]["tier_b_production"]["alignment_score"]
    if prod_alignment >= 0.8:
        aspects.append("System profile is close to production agent-system baseline")

    if not aspects:
        aspects.append("No major capability can yet be certified as fully proven")

    return aspects


def build_uril_report(options: SignalCollectionOptions) -> dict[str, Any]:
    signal_bus = collect_signal_bus(options)
    subsystem_rows = build_subsystem_health(signal_bus)
    system_profile = build_system_profile(signal_bus, subsystem_rows)
    alignment = benchmark_alignment_report(system_profile)
    risk_heatmap = subsystem_risk_heatmap(signal_bus)
    suggestions = generate_refactor_suggestions(signal_bus, subsystem_rows)

    report = UrailReport(
        phase4_completion=_phase4_completion(signal_bus, subsystem_rows, alignment),
        subsystem_health=subsystem_health_map(signal_bus),
        benchmark_alignment=alignment,
        risk_heatmap=risk_heatmap,
        upgrade_recommendations=[
            {
                "target": s.target,
                "issue": s.issue,
                "fix": s.fix,
                "impact": s.impact,
                "risk": s.risk,
            }
            for s in suggestions
        ],
        signal_bus=signal_bus.to_dict(),
    )

    payload = report.to_dict()
    payload["progress_summary"] = _progress_summary(signal_bus)
    payload["proven_aspects"] = _proven_aspects(signal_bus, subsystem_rows, alignment)
    payload["uril_version"] = "0.1"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Repo Intelligence + Certification Layer (URIL)")
    parser.add_argument("--pytest-junit", default="", help="Path to pytest JUnit XML")
    parser.add_argument("--phase4-auditor-json", default="", help="Path to saved phase4 auditor JSON")
    parser.add_argument("--filesystem-json", default="", help="Path to saved filesystem checker JSON")
    parser.add_argument("--stress-json", default="", help="Path to stress certification JSON")
    parser.add_argument("--benchmark-snapshot-dir", default="", help="Path to evaluation snapshot directory")
    parser.add_argument("--no-probes", action="store_true", help="Do not run live probe scripts; only read provided files")
    parser.add_argument("--json-out", default="", help="Write full JSON report to file")
    return parser.parse_args()


def _path_or_none(value: str) -> Path | None:
    v = str(value or "").strip()
    return Path(v) if v else None


def main() -> None:
    args = parse_args()
    options = SignalCollectionOptions(
        pytest_junit_path=_path_or_none(args.pytest_junit),
        benchmark_snapshot_dir=_path_or_none(args.benchmark_snapshot_dir),
        phase4_auditor_json=_path_or_none(args.phase4_auditor_json),
        filesystem_json=_path_or_none(args.filesystem_json),
        stress_json=_path_or_none(args.stress_json),
        run_probes=not bool(args.no_probes),
    )
    payload = build_uril_report(options)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Snapshot delta comparator — ROI #6
# ---------------------------------------------------------------------------

def delta_compare(
    old_snapshot: dict[str, Any],
    new_snapshot: dict[str, Any],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare two URIL report snapshots and measure drift.

    Walks numeric scalar leaves in both snapshots and computes the absolute
    difference.  Returns a summary dict with:

    ``drifted_keys``
        List of dotted-key paths whose absolute delta exceeds *tolerance*.
    ``delta``
        Dict mapping dotted-key → absolute numeric delta (for all numeric keys).
    ``within_tolerance``
        True when no key has drifted beyond *tolerance*.
    ``max_drift``
        Highest absolute delta observed across all numeric keys.

    Non-numeric values are compared for equality and listed under
    ``changed_non_numeric`` when they differ.
    """

    def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                full = f"{prefix}.{k}" if prefix else k
                out.update(_flatten(v, full))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                full = f"{prefix}[{i}]"
                out.update(_flatten(v, full))
        else:
            out[prefix] = obj
        return out

    old_flat = _flatten(old_snapshot)
    new_flat = _flatten(new_snapshot)

    delta: dict[str, float] = {}
    drifted_keys: list[str] = []
    changed_non_numeric: list[str] = []

    all_keys = set(old_flat) | set(new_flat)
    for key in sorted(all_keys):
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            diff = abs(float(new_val) - float(old_val))
            delta[key] = round(diff, 6)
            if diff > tolerance:
                drifted_keys.append(key)
        elif old_val != new_val:
            changed_non_numeric.append(key)

    max_drift = max(delta.values(), default=0.0)

    return {
        "within_tolerance": len(drifted_keys) == 0,
        "max_drift": round(max_drift, 6),
        "drifted_keys": drifted_keys,
        "delta": delta,
        "changed_non_numeric": changed_non_numeric,
        "tolerance": tolerance,
    }


if __name__ == "__main__":
    main()
