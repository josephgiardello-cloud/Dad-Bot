#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def run_stressor(
    *,
    db_path: Path,
    event_count: int,
    batch_size: int = 5000,
    summary_out: Path | None = None,
) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    wal_path = Path(str(db_path) + "-wal")

    before_db = _file_size(db_path)
    before_wal = _file_size(wal_path)

    started = time.perf_counter()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synthetic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """,
        )
        conn.commit()

        remaining = int(max(0, event_count))
        idx = 0
        while remaining > 0:
            take = min(int(batch_size), remaining)
            rows: list[tuple[str, str, str, float]] = []
            now = time.time()
            for _ in range(take):
                payload = {
                    "sequence": idx,
                    "trace_id": f"stress-trace-{idx // 1000}",
                    "op": "synthetic_replay_probe",
                    "contract_version": "1.0",
                }
                rows.append(
                    (
                        "stress-session",
                        "SYNTHETIC_EVENT",
                        json.dumps(payload, sort_keys=True),
                        now,
                    )
                )
                idx += 1

            conn.executemany(
                "INSERT INTO synthetic_events(session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            remaining -= take

        snapshot_started = time.perf_counter()
        total_events = int(conn.execute("SELECT COUNT(*) FROM synthetic_events").fetchone()[0])
        last_event = conn.execute(
            "SELECT id, payload FROM synthetic_events ORDER BY id DESC LIMIT 1",
        ).fetchone()
        snapshot_elapsed = time.perf_counter() - snapshot_started

    elapsed = time.perf_counter() - started
    after_db = _file_size(db_path)
    after_wal = _file_size(wal_path)
    report = {
        "event_count_requested": int(event_count),
        "event_count_observed": int(total_events),
        "elapsed_seconds": round(float(elapsed), 4),
        "snapshot_summary_seconds": round(float(snapshot_elapsed), 4),
        "db_size_before_bytes": before_db,
        "db_size_after_bytes": after_db,
        "wal_size_before_bytes": before_wal,
        "wal_size_after_bytes": after_wal,
        "last_event_id": int(last_event[0]) if last_event else 0,
    }

    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Million-event replay stressor")
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--db", type=Path, default=Path("session_logs") / "stress_semantic.sqlite3")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("session_logs") / "million_event_replay_summary.json",
    )
    args = parser.parse_args()

    report = run_stressor(
        db_path=args.db,
        event_count=args.events,
        batch_size=args.batch_size,
        summary_out=args.summary,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
