from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _append_malformed_line(path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_type": "PHANTOM", "payload": ')  # intentionally truncated


def _append_phantom_json(path: Path) -> None:
    payload = {
        "event_type": "PHANTOM_EVENT",
        "timestamp": time.time(),
        "payload": {"source": "live_audit_attack"},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a malicious out-of-band write to a ledger file.")
    parser.add_argument("--path", default="session_logs/relational_ledger.jsonl")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--mode", choices=("malformed", "phantom"), default="malformed")
    args = parser.parse_args()

    target = Path(args.path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8")

    time.sleep(max(0.0, float(args.delay)))
    if args.mode == "malformed":
        _append_malformed_line(target)
    else:
        _append_phantom_json(target)

    print(f"attack_written mode={args.mode} path={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
