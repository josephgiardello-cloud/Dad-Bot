"""Durable per-turn resume point storage.

Writes a small JSON record after each stage completes so a crashed turn can be
recovered on the next execution attempt.  Records are file-backed, one file per
trace_id, written atomically via write-then-rename.

Architecture role
-----------------
This is a thin persistence layer; it has no imports from graph, kernel, or any
domain logic.  TurnGraph writes here after each successful stage; ExecutionRecovery
reads here on execute() startup to reconstruct completed_stages.

Record schema (v1)
------------------
{
  "schema_version": "1",
  "turn_id":              "<trace_id>",
  "last_completed_stage": "<stage_name>",
  "next_stage":           "<stage_name or empty string>",
  "checkpoint_hash":      "<32-char hex from TurnContext.last_checkpoint_hash>",
  "completed_stages":     ["temporal", "health", ...],
  "created_at":           <float epoch>,
  "updated_at":           <float epoch>,
}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ResumePoint:
    """Immutable snapshot of a durable turn resume record."""

    turn_id: str
    last_completed_stage: str
    next_stage: str
    checkpoint_hash: str
    completed_stages: tuple[str, ...]
    created_at: float
    updated_at: float
    # Stage that was executing when the process last checkpointed.  Non-empty means
    # a crash may have happened mid-execution; the stage call_id is deterministic
    # so external tools can deduplicate via hash(turn_id + ':' + in_flight_stage).
    in_flight_stage: str = ""

    def is_expired(self, *, max_age_seconds: float) -> bool:
        """Return True if this record is older than max_age_seconds."""
        return (time.time() - self.updated_at) > max_age_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "turn_id": self.turn_id,
            "last_completed_stage": self.last_completed_stage,
            "next_stage": self.next_stage,
            "checkpoint_hash": self.checkpoint_hash,
            "completed_stages": list(self.completed_stages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "in_flight_stage": self.in_flight_stage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResumePoint:
        return cls(
            turn_id=str(d.get("turn_id") or ""),
            last_completed_stage=str(d.get("last_completed_stage") or ""),
            next_stage=str(d.get("next_stage") or ""),
            checkpoint_hash=str(d.get("checkpoint_hash") or ""),
            completed_stages=tuple(d.get("completed_stages") or []),
            created_at=float(d.get("created_at") or time.time()),
            updated_at=float(d.get("updated_at") or time.time()),
            in_flight_stage=str(d.get("in_flight_stage") or ""),
        )


class TurnResumeStore:
    """Durable per-turn resume point store.

    Persists resume records to *store_dir* as ``<turn_id>.resume.json``.
    Writes are atomic: the payload is written to a temp file then renamed
    into place so a partial write never leaves a corrupt record.

    Thread-safe via an internal RLock.
    """

    def __init__(self, store_dir: Path) -> None:
        self._dir = Path(store_dir)
        self._lock = RLock()

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _record_path(self, turn_id: str) -> Path:
        # Sanitize: only hex-alphanumeric characters are valid in trace_ids.
        safe = "".join(c for c in str(turn_id) if c.isalnum() or c in "-_")[:64]
        return self._dir / f"{safe}.resume.json"

    def save(
        self,
        *,
        turn_id: str,
        last_completed_stage: str,
        next_stage: str,
        checkpoint_hash: str,
        completed_stages: list[str],
        created_at: float | None = None,
    ) -> ResumePoint:
        """Persist a resume record for *turn_id*.

        Subsequent calls for the same turn_id update the record in-place.
        The ``created_at`` timestamp is set only on the first write.
        """
        with self._lock:
            self._ensure_dir()
            path = self._record_path(turn_id)
            now = time.time()

            # Preserve original created_at if record already exists.
            existing_created_at: float = created_at or now
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    existing_created_at = float(existing.get("created_at") or now)
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

            record = ResumePoint(
                turn_id=turn_id,
                last_completed_stage=last_completed_stage,
                next_stage=next_stage,
                checkpoint_hash=checkpoint_hash,
                completed_stages=tuple(completed_stages),
                created_at=existing_created_at,
                updated_at=now,
            )

            payload = json.dumps(record.to_dict(), sort_keys=True, indent=2)
            # Atomic write: temp file + rename.
            tmp_path = path.with_suffix(".tmp")
            try:
                tmp_path.write_text(payload, encoding="utf-8")
                # os.replace is atomic on POSIX; on Windows it overwrites the destination.
                os.replace(str(tmp_path), str(path))
            except OSError:
                # Clean up temp file on failure; let the caller decide whether to
                # treat this as fatal.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

        return record

    def load(self, turn_id: str) -> ResumePoint | None:
        """Return the stored resume point for *turn_id*, or None if not found.

        Returns None (rather than raising) for corrupt or missing records.
        """
        with self._lock:
            path = self._record_path(turn_id)
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if str(data.get("schema_version")) != _SCHEMA_VERSION:
                    return None
                return ResumePoint.from_dict(data)
            except (OSError, ValueError, json.JSONDecodeError):
                return None

    def clear(self, turn_id: str) -> None:
        """Remove the resume record for *turn_id* (called on successful completion)."""
        with self._lock:
            path = self._record_path(turn_id)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def mark_started(self, turn_id: str, stage_name: str) -> None:
        """Record that *stage_name* is about to execute for *turn_id*.

        Called BEFORE the node runs.  If a crash occurs between this call and
        the subsequent ``save()`` marking the stage as completed, the resume
        record will show ``in_flight_stage=stage_name`` on the next startup.

        ExecutionRecovery uses this to inject a deterministic ``_stage_call_id``
        into TurnContext.state so that external tools can deduplicate effects.

        This is a best-effort write: failures are silently swallowed so that
        a storage hiccup never aborts execution.
        """
        with self._lock:
            self._ensure_dir()
            path = self._record_path(turn_id)
            now = time.time()
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data["in_flight_stage"] = stage_name
                    data["updated_at"] = now
                    tmp_path = path.with_suffix(".tmp")
                    try:
                        tmp_path.write_text(
                            json.dumps(data, sort_keys=True, indent=2),
                            encoding="utf-8",
                        )
                        os.replace(str(tmp_path), str(path))
                    except OSError:
                        tmp_path.unlink(missing_ok=True)
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            else:
                # No record yet — create a minimal one so the in_flight marker
                # is durable even if this is the very first stage.
                minimal = ResumePoint(
                    turn_id=turn_id,
                    last_completed_stage="",
                    next_stage=stage_name,
                    checkpoint_hash="",
                    completed_stages=(),
                    created_at=now,
                    updated_at=now,
                    in_flight_stage=stage_name,
                )
                try:
                    payload = json.dumps(minimal.to_dict(), sort_keys=True, indent=2)
                    tmp_path = path.with_suffix(".tmp")
                    tmp_path.write_text(payload, encoding="utf-8")
                    os.replace(str(tmp_path), str(path))
                except OSError:
                    pass

    def list_pending(self) -> list[ResumePoint]:
        """Return all stored (potentially incomplete) resume records.

        Used at startup to discover turns that need recovery.
        """
        results: list[ResumePoint] = []
        with self._lock:
            if not self._dir.exists():
                return results
            for path in sorted(self._dir.glob("*.resume.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if str(data.get("schema_version")) == _SCHEMA_VERSION:
                        results.append(ResumePoint.from_dict(data))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        return results

    def purge_expired(self, *, max_age_seconds: float) -> int:
        """Delete resume records older than *max_age_seconds*.

        Returns number of records removed.
        """
        removed = 0
        with self._lock:
            if not self._dir.exists():
                return 0
            for path in list(self._dir.glob("*.resume.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    updated_at = float(data.get("updated_at") or 0)
                    if (time.time() - updated_at) > max_age_seconds:
                        path.unlink(missing_ok=True)
                        removed += 1
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        return removed
