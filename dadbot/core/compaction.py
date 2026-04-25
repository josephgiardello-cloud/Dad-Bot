"""Event log compaction and archive tier.

CompactionPolicy â€” decides when compaction should trigger.
EventCompactor   â€” trims pre-snapshot events from the in-memory ledger.
ArchiveTier      â€” writes discarded events to a gzip-compressed JSONL file.

Usage::

    from dadbot.core.compaction import CompactionPolicy, EventCompactor, ArchiveTier

    archive = ArchiveTier("runtime/archives")
    compactor = EventCompactor(
        policy=CompactionPolicy(max_events=5_000),
        archive=archive,
    )
    report = compactor.compact(ledger=ledger, snapshot=engine.latest())
    if report["compacted"]:
        print(f"Removed {report['events_removed']} events; archive: {report['archive_path']}")
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Compaction policy
# ---------------------------------------------------------------------------

class CompactionPolicy:
    """Threshold-based policy that decides whether compaction should run."""

    def __init__(
        self,
        *,
        max_events: int = 10_000,
        max_age_seconds: float = 86_400.0,
        min_snapshot_distance: int = 100,
    ) -> None:
        """
        Args:
            max_events: Compact when the in-memory event count exceeds this.
            max_age_seconds: Compact when the oldest event is older than this.
            min_snapshot_distance: Keep at least this many events *before* the
                snapshot head even after compaction (safety margin).
        """
        self.max_events             = max(1, int(max_events))
        self.max_age_seconds        = max(1.0, float(max_age_seconds))
        self.min_snapshot_distance  = max(0, int(min_snapshot_distance))

    def should_compact(
        self,
        *,
        event_count: int,
        oldest_event_timestamp: float | None = None,
        events_since_snapshot: int = 0,
    ) -> bool:
        if event_count >= self.max_events:
            return True
        if (
            oldest_event_timestamp is not None
            and (time.time() - oldest_event_timestamp) >= self.max_age_seconds
        ):
            return True
        return False


# ---------------------------------------------------------------------------
# Archive tier
# ---------------------------------------------------------------------------

class ArchiveTier:
    """Writes discarded events to gzip-compressed JSONL archive files.

    Archive filenames are timestamped to avoid collisions across multiple
    compaction cycles:  ``ledger[-<label>]-YYYYMMDD-HHMMSS.archive.gz``
    """

    def __init__(self, archive_dir: str | Path) -> None:
        self._dir = Path(archive_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def archive(
        self,
        events: list[dict[str, Any]],
        *,
        label: str = "",
    ) -> Path:
        """Write events to a new archive file.  Returns the archive path."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        suffix = f"-{label}" if label else ""
        filename = f"ledger{suffix}-{ts}.archive.gz"
        path = self._dir / filename
        with self._lock:
            with gzip.open(str(path), "wt", encoding="utf-8") as gz:
                for event in events:
                    gz.write(json.dumps(event, default=str) + "\n")
        return path

    def list_archives(self) -> list[Path]:
        return sorted(self._dir.glob("*.archive.gz"))

    def load_archive(self, path: Path) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        with gzip.open(str(path), "rt", encoding="utf-8") as gz:
            for line in gz:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError:
                    pass
        return events

    def total_archived_events(self) -> int:
        total = 0
        for path in self.list_archives():
            total += len(self.load_archive(path))
        return total


# ---------------------------------------------------------------------------
# Event compactor
# ---------------------------------------------------------------------------

class EventCompactor:
    """Trims in-memory ledger events before the snapshot head.

    Compaction only runs when:
      - a snapshot exists (no snapshot â†’ cannot compact safely), AND
      - the CompactionPolicy says the threshold is exceeded (or force=True).

    Events trimmed from the ledger are written to the ArchiveTier (if provided)
    before removal so they can be audited or replayed later.
    """

    def __init__(
        self,
        *,
        policy: CompactionPolicy | None = None,
        archive: ArchiveTier | None = None,
    ) -> None:
        self._policy           = policy or CompactionPolicy()
        self._archive          = archive
        self._compaction_count = 0

    def compact(
        self,
        *,
        ledger,
        snapshot: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run compaction if policy permits.

        Args:
            ledger: ExecutionLedger instance.
            snapshot: Latest snapshot dict (from SnapshotEngine.latest()).
            force: Bypass policy threshold check.

        Returns:
            Report dict: compacted (bool), events_removed (int), archive_path (str|None).
        """
        events = ledger.read()
        n = len(events)

        if not events:
            return {"compacted": False, "events_removed": 0, "archive_path": None, "reason": "empty"}

        if snapshot is None:
            return {
                "compacted": False,
                "events_removed": 0,
                "archive_path": None,
                "reason": "no_snapshot",
            }

        head_seq = int(snapshot.get("head_sequence") or 0)
        oldest_ts = float((events[0].get("timestamp")) or 0.0)
        events_since_snapshot = max(0, n - head_seq)

        if not force and not self._policy.should_compact(
            event_count=n,
            oldest_event_timestamp=oldest_ts,
            events_since_snapshot=events_since_snapshot,
        ):
            return {
                "compacted": False,
                "events_removed": 0,
                "archive_path": None,
                "reason": "below_threshold",
            }

        # Keep min_snapshot_distance events before the snapshot head.
        cutoff = max(0, head_seq - self._policy.min_snapshot_distance)
        events_to_archive = events[:cutoff]
        events_to_keep    = events[cutoff:]

        if not events_to_archive:
            return {
                "compacted": False,
                "events_removed": 0,
                "archive_path": None,
                "reason": "nothing_to_compact",
            }

        archive_path: str | None = None
        if self._archive is not None:
            path = self._archive.archive(events_to_archive, label="compact")
            archive_path = str(path)

        # Replace the in-memory event list directly.
        # EventCompactor is an authorised system operation â€” direct access
        # to _events is intentional here (same as snapshot restore).
        with ledger._lock:
            ledger._events = list(events_to_keep)

        self._compaction_count += 1
        return {
            "compacted": True,
            "events_removed": len(events_to_archive),
            "archive_path": archive_path,
            "compaction_count": self._compaction_count,
        }

    @property
    def compaction_count(self) -> int:
        return self._compaction_count
