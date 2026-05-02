from __future__ import annotations

import sys
from dataclasses import dataclass

if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.filesystem_enforcement import run_checks as run_filesystem_checks
from ci.import_graph_check import run_checks as run_import_graph_checks
from ci.mutation_tracking_check import run_checks as run_mutation_checks
from ci.ast_invariant_check import run_checks as run_ast_invariant_checks


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    detail: str


def _print_header() -> None:
    print("[KERNEL BOUNDARY VALIDATOR] enforcing hard layer separation")


def _print_violation(v: Violation) -> None:
    print(f"FAIL {v.rule}: {v.path}")
    print(f"  -> {v.detail}")


def run_kernel_boundary_checks() -> list[Violation]:
    violations: list[Violation] = []

    for item in run_import_graph_checks():
        violations.append(Violation(rule=item.rule, path=item.path, detail=item.detail))
    for item in run_filesystem_checks():
        violations.append(Violation(rule=item.rule, path=item.path, detail=item.detail))
    for item in run_mutation_checks():
        violations.append(Violation(rule=item.rule, path=item.path, detail=item.detail))
    for item in run_ast_invariant_checks():
        violations.append(Violation(rule=item.rule, path=item.path, detail=item.detail))

    return violations


def main() -> int:
    _print_header()
    violations = run_kernel_boundary_checks()
    if not violations:
        print("PASS kernel boundary checks")
        return 0

    print(f"FAIL detected {len(violations)} kernel-boundary violation(s)")
    for violation in violations:
        _print_violation(violation)
    print("STOP pipeline due to hard boundary violation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
