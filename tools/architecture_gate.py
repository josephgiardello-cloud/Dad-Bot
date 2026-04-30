"""Architecture gate runner.

Runs core architecture hardening checks and fails on the first non-zero result.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_STEPS = [
    ["tools/reconcile_baseline.py", "--write-lock", "--strict-real-regression"],
    ["tools/enforce_no_bypass.py"],
    ["tools/contract_guard.py"],
    ["tools/complexity_diff_gate.py"],
    ["tools/complexity_gate.py"],
    ["tools/god_class_audit.py"],
    ["tools/ownership_drift.py"],
    ["tools/arch_completeness_audit.py"],
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run architecture quality gates")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    failures: list[tuple[str, int]] = []

    for step in DEFAULT_STEPS:
        label = " ".join(step)
        print(f"[architecture_gate] running: {label}")
        result = subprocess.run([args.python, *step], cwd=root, check=False)
        if result.returncode != 0:
            failures.append((label, int(result.returncode)))

    if failures:
        print("[architecture_gate] FAILED")
        for label, code in failures:
            print(f"  {label} -> exit {code}")
        return 1

    print("[architecture_gate] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
