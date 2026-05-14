"""CI gate entrypoint for contract and adversarial closure checks.

Usage examples:
  python _ci_gate_check.py --contract-gate
  python _ci_gate_check.py --contract-gate --fail-on-untested
  python _ci_gate_check.py --adversarial-closure-gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    return int(completed.returncode)


def _contract_gate(*, fail_on_untested: bool) -> int:
    cmd = [
        sys.executable,
        "tools/contract_test_compiler.py",
        "--check",
        "--validate-nodeids",
    ]
    if fail_on_untested:
        cmd.append("--fail-on-untested")
    return _run(cmd)


def _adversarial_closure_gate() -> int:
    return _run([sys.executable, "ci/adversarial_closure_gate.py"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CI contract and closure gates")
    parser.add_argument("--contract-gate", action="store_true", help="Run contract map -> tests gate")
    parser.add_argument("--fail-on-untested", action="store_true", help="Fail if any mapped contract has no tests")
    parser.add_argument(
        "--adversarial-closure-gate",
        action="store_true",
        help="Run adversarial closure governance checks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ran_any = False
    exit_codes: list[int] = []

    if args.contract_gate:
        ran_any = True
        exit_codes.append(_contract_gate(fail_on_untested=bool(args.fail_on_untested)))

    if args.adversarial_closure_gate:
        ran_any = True
        exit_codes.append(_adversarial_closure_gate())

    if not ran_any:
        return 0

    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
