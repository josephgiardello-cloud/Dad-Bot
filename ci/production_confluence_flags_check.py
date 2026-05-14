from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ASSIGNMENTS: dict[str, str] = {
    "DADBOT_GLOBAL_CONFLUENCE_MODE": "enforce",
    "DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY": "0",
}


@dataclass(frozen=True)
class Violation:
    file: str
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _has_assignment(content: str, key: str, expected: str) -> bool:
    pattern = re.compile(rf"{re.escape(key)}\s*=\s*{re.escape(expected)}", re.IGNORECASE)
    return bool(pattern.search(content))


def _has_compose_env(content: str, key: str, expected: str) -> bool:
    # Supports list-style compose env entries: - KEY=value
    pattern = re.compile(rf"-\s*{re.escape(key)}\s*=\s*{re.escape(expected)}\b", re.IGNORECASE)
    return bool(pattern.search(content))


def run_checks() -> list[Violation]:
    violations: list[Violation] = []

    dockerfile = ROOT / "Dockerfile"
    compose = ROOT / "docker-compose.yml"

    dockerfile_text = _read(dockerfile)
    compose_text = _read(compose)

    for key, expected in REQUIRED_ASSIGNMENTS.items():
        if not _has_assignment(dockerfile_text, key, expected):
            violations.append(
                Violation(
                    file="Dockerfile",
                    detail=f"Missing strict assignment: {key}={expected}",
                ),
            )

    for key, expected in REQUIRED_ASSIGNMENTS.items():
        if not _has_compose_env(compose_text, key, expected):
            violations.append(
                Violation(
                    file="docker-compose.yml",
                    detail=f"Missing strict compose env entry: {key}={expected}",
                ),
            )

    return violations


def main() -> int:
    violations = run_checks()
    if not violations:
        print("PASS production confluence strict flags")
        return 0

    print(f"FAIL detected {len(violations)} production confluence flag violation(s)")
    for violation in violations:
        print(f"FAIL {violation.file}: {violation.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
