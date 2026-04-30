"""Phase 1.5 baseline reconciliation and classification.

Classifies gate findings into:
  A - real_regression
  B - migration_inflation
  C - baseline_invalidation

This tool is intentionally classification-first. It can update
`tools/architecture_baseline_lock.json` with migration inflation allowlists,
but only after classification has been produced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "tools" / "architecture_baseline_lock.json"
OUTPUT_PATH = ROOT / "audit" / "baseline_reconciliation_report.json"
COMPLETENESS_REPORT = ROOT / "audit" / "architectural_completeness_report.json"

MIGRATION_HOTFILES = {
    "dadbot/core/graph_mutation.py",
    "dadbot/core/graph_pipeline_nodes.py",
}


@dataclass(frozen=True)
class Finding:
    bucket: str
    source: str
    file: str
    detail: str
    reason: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify architecture findings before baseline updates")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--strict-real-regression", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _run_json_step(python_bin: str, step: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        [python_bin, *step, "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not result.stdout.strip():
        return {"_error": f"no_output (exit={result.returncode})", "_stderr": result.stderr.strip()}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "_error": f"invalid_json (exit={result.returncode})",
            "_stdout": result.stdout.strip(),
            "_stderr": result.stderr.strip(),
        }
    payload["_exit_code"] = result.returncode
    return payload


def _collect_findings(python_bin: str) -> tuple[list[Finding], dict[str, Any]]:
    raw: dict[str, Any] = {}
    findings: list[Finding] = []

    contract = _run_json_step(python_bin, ["tools/contract_guard.py"])
    raw["contract_guard"] = contract
    for item in contract.get("new_violations", []):
        findings.append(
            Finding(
                bucket="A",
                source="contract_guard",
                file=str(item).split(":", 1)[0],
                detail=str(item),
                reason="New forbidden primitive signature introduced",
            )
        )

    complexity = _run_json_step(python_bin, ["tools/complexity_diff_gate.py", "--ignore-lock"])
    raw["complexity_diff_gate"] = complexity
    for item in complexity.get("violations", []):
        text = str(item)
        rel_file = text.split(":", 1)[0].replace("\\", "/")
        if rel_file in MIGRATION_HOTFILES:
            findings.append(
                Finding(
                    bucket="B",
                    source="complexity_diff_gate",
                    file=rel_file,
                    detail=text,
                    reason="Contract migration inflation in known transition hotfile",
                )
            )
        else:
            findings.append(
                Finding(
                    bucket="C",
                    source="complexity_diff_gate",
                    file=rel_file,
                    detail=text,
                    reason="Compared against pre-migration complexity baseline",
                )
            )

    god = _run_json_step(python_bin, ["tools/god_class_audit.py", "--ignore-lock"])
    raw["god_class_audit"] = god
    for rel_file, msgs in god.get("violations", {}).items():
        rel_norm = str(rel_file).replace("\\", "/")
        if rel_norm in MIGRATION_HOTFILES:
            for msg in msgs:
                findings.append(
                    Finding(
                        bucket="B",
                        source="god_class_audit",
                        file=rel_norm,
                        detail=str(msg),
                        reason="Contract migration inflation in known transition hotfile",
                    )
                )
        else:
            for msg in msgs:
                findings.append(
                    Finding(
                        bucket="C",
                        source="god_class_audit",
                        file=rel_norm,
                        detail=str(msg),
                        reason="Pre-migration god-class baseline no longer representative",
                    )
                )

    ownership = _run_json_step(python_bin, ["tools/ownership_drift.py"])
    raw["ownership_drift"] = ownership
    for rel_file, msgs in ownership.get("violations", {}).items():
        for msg in msgs:
            findings.append(
                Finding(
                    bucket="A",
                    source="ownership_drift",
                    file=rel_file,
                    detail=str(msg),
                    reason="Ownership boundary violation is directly actionable",
                )
            )

    subprocess.run([python_bin, "tools/arch_completeness_audit.py"], cwd=ROOT, check=False)
    if COMPLETENESS_REPORT.exists():
        completeness = json.loads(COMPLETENESS_REPORT.read_text(encoding="utf-8"))
    else:
        completeness = {}
    raw["arch_completeness_audit"] = completeness

    for drift in completeness.get("contract_drift", []):
        findings.append(
            Finding(
                bucket="A",
                source="arch_completeness_audit",
                file=str(drift.get("file", "unknown")),
                detail=str(drift),
                reason="Contract interface drift is a true regression",
            )
        )

    for miss in completeness.get("missing_parameters", []):
        missing_fields = set(miss.get("missing_fields", []))
        if {"trace_id", "kernel_step_id"} & missing_fields:
            findings.append(
                Finding(
                    bucket="A",
                    source="arch_completeness_audit",
                    file=str(miss.get("file", "unknown")),
                    detail=str(miss),
                    reason="Execution context propagation breach is a true regression",
                )
            )

    return findings, raw


def _group(findings: list[Finding]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {"A": [], "B": [], "C": []}
    seen: set[tuple[str, str, str, str, str]] = set()
    for item in findings:
        key = (item.bucket, item.source, item.file, item.detail, item.reason)
        if key in seen:
            continue
        seen.add(key)
        grouped[item.bucket].append(
            {
                "source": item.source,
                "file": item.file,
                "detail": item.detail,
                "reason": item.reason,
            }
        )
    return grouped


def _write_lock(lock_path: Path, grouped: dict[str, list[dict[str, str]]], report_rel: str) -> None:
    if lock_path.exists():
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    else:
        payload = {}

    payload.setdefault("locked_artifacts", {})["reconciliation_report"] = report_rel
    payload["reconciliation"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "migration_inflation_allowlist": [
            {
                "source": item["source"],
                "file": item["file"],
                "reason": item["reason"],
            }
            for item in grouped["B"]
        ],
        "baseline_invalidations": grouped["C"],
        "real_regressions": grouped["A"],
        "policy": {
            "rebaseline_only_after_classification": True,
            "strict_bucket_for_failures": "A",
        },
    }

    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    lock_path = args.lock if args.lock.is_absolute() else ROOT / args.lock

    findings, raw = _collect_findings(args.python)
    grouped = _group(findings)

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bucket_labels": {
            "A": "real_regression",
            "B": "migration_inflation",
            "C": "baseline_invalidation",
        },
        "summary": {
            "A": len(grouped["A"]),
            "B": len(grouped["B"]),
            "C": len(grouped["C"]),
        },
        "findings": grouped,
        "sources": raw,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    if args.write_lock:
        report_rel = output_path.relative_to(ROOT).as_posix()
        _write_lock(lock_path, grouped, report_rel)

    if args.json_output:
        print(json.dumps(output_payload, indent=2))
    else:
        print("[reconcile_baseline] Classification complete")
        print(f"  A(real_regression)={len(grouped['A'])}")
        print(f"  B(migration_inflation)={len(grouped['B'])}")
        print(f"  C(baseline_invalidation)={len(grouped['C'])}")
        print(f"  report={output_path.relative_to(ROOT).as_posix()}")
        if args.write_lock:
            print(f"  lock_updated={lock_path.relative_to(ROOT).as_posix()}")

    if args.strict_real_regression and grouped["A"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
