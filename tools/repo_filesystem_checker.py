from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CRITICAL_FILES = [
    "dadbot/core/dadbot.py",
    "dadbot/core/boot_mixin.py",
    "dadbot/core/turn_mixin.py",
    "dadbot/core/llm_mixin.py",
    "dadbot/core/mcp_mixin.py",
    "dadbot/core/health_mixin.py",
    "dadbot/core/capability_registry.py",
    "dadbot/core/execution_receipt.py",
    "dadbot/core/observability.py",
    "dadbot/core/otel_bridge.py",
    "dadbot/core/prometheus_bridge.py",
    "tools/phase4_certify.py",
    "tools/repo_phase4_auditor.py",
]

EXPECTED_TEST_FILES = [
    "tests/test_observability_control_plane.py",
    "tests/test_observability_bridges.py",
    "tests/adversarial/test_determinism_fuzzing.py",
]


def _line_count(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    except Exception:
        return 0


def build_repo_filesystem_report(root: Path | None = None) -> dict[str, Any]:
    repo_root = root or ROOT

    missing_expected_files = [rel for rel in EXPECTED_CRITICAL_FILES if not (repo_root / rel).exists()]
    missing_expected_tests = [rel for rel in EXPECTED_TEST_FILES if not (repo_root / rel).exists()]

    underoptimized: list[dict[str, Any]] = []

    # Rule 1: core facade should remain thin.
    facade_path = repo_root / "dadbot/core/dadbot.py"
    facade_lines = _line_count(facade_path)
    if facade_path.exists() and facade_lines > 800:
        underoptimized.append(
            {
                "kind": "facade_too_large",
                "path": "dadbot/core/dadbot.py",
                "line_count": facade_lines,
                "threshold": 800,
            }
        )

    # Rule 2: any very large python module in core is likely under-optimized.
    for py in sorted((repo_root / "dadbot/core").glob("*.py")):
        count = _line_count(py)
        if count > 1200:
            underoptimized.append(
                {
                    "kind": "module_too_large",
                    "path": str(py.relative_to(repo_root)).replace("\\", "/"),
                    "line_count": count,
                    "threshold": 1200,
                }
            )

    # Rule 3: required package markers for expected packages.
    expected_packages = [
        "dadbot",
        "dadbot/core",
        "dadbot/managers",
        "dadbot/runtime",
        "tests",
    ]
    missing_package_markers: list[str] = []
    for package in expected_packages:
        pkg_path = repo_root / package
        if pkg_path.exists() and not (pkg_path / "__init__.py").exists():
            missing_package_markers.append(str((pkg_path / "__init__.py").relative_to(repo_root)).replace("\\", "/"))

    if missing_package_markers:
        underoptimized.append(
            {
                "kind": "missing_package_marker",
                "paths": missing_package_markers,
            }
        )

    status = "PASS"
    if missing_expected_files or missing_expected_tests:
        status = "FAIL"
    elif underoptimized:
        status = "WARN"

    return {
        "status": status,
        "missing_expected_files": missing_expected_files,
        "missing_expected_tests": missing_expected_tests,
        "underoptimized": underoptimized,
        "counts": {
            "missing_expected_files": len(missing_expected_files),
            "missing_expected_tests": len(missing_expected_tests),
            "underoptimized": len(underoptimized),
        },
    }


def main() -> int:
    report = build_repo_filesystem_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
