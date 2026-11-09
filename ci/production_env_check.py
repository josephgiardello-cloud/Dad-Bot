from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_KEYS = (
    "DADBOT_AUTHORITY_SQLITE_PATH",
    "DADBOT_DURABLE_EXECUTION_LEDGER",
    "DADBOT_EXECUTION_LEDGER_BACKEND",
)


def main() -> int:
    env_file = Path(__file__).resolve().parents[1] / ".env.production.example"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"Cannot read {env_file}: {exc}", file=sys.stderr)
        return 1

    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", stripped)
        if match is None:
            continue
        key, value = match.groups()
        if key in values:
            print(f"{env_file}:{line_number}: duplicate key {key}", file=sys.stderr)
            return 1
        values[key] = value.strip().strip("\"'")

    missing = [key for key in REQUIRED_KEYS if not values.get(key)]
    if missing:
        print(f"{env_file}: missing or empty required keys: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"{env_file}: production environment contract passed ({len(REQUIRED_KEYS)} required keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
