"""Loop session snapshots: persist and recover AgentDriverLoop mid-run state.

Each snapshot is written as a JSON file under session_logs/<session_id>.loop.json.
This provides the persistence layer for cold-start recovery:

    snap = LoopSessionSnapshot.load("my-session")
    if snap and snap.is_incomplete:
        loop = AgentDriverLoop(kernel, policy=snap.original_policy)
        result = loop.run(snap.initial_observation, session_id="my-session",
                          resume_from=snap)

The snapshot tracks:
  - session_id + original initial_observation
  - last committed turn_index
  - stop_reason (None = still running)
  - policy snapshot (so resume honours the same limits)
  - wall-clock timestamps for start/end
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default snapshot directory  (sibling of where Dad.py lives)
# ---------------------------------------------------------------------------

_DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent.parent.parent / "session_logs"


def _snapshot_path(session_id: str, base_dir: Path | None = None) -> Path:
    d = Path(base_dir) if base_dir else _DEFAULT_SNAPSHOT_DIR
    d.mkdir(parents=True, exist_ok=True)
    # Sanitize session_id for filesystem safety
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
    return d / f"{safe}.loop.json"


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class LoopSessionSnapshot:
    """Durable record of a loop run — written at start, updated at each committed turn."""

    session_id: str
    initial_observation: str

    # Mutable during run
    last_committed_turn: int = 0          # 0 = not yet started
    completed_turns: int = 0
    failures: int = 0
    stop_reason: str | None = None        # None = still running / incomplete
    started_at: str = ""
    finished_at: str = ""

    # Policy snapshot (plain dict for serialisation)
    policy: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------
    @property
    def is_incomplete(self) -> bool:
        """True when the loop was interrupted before reaching a terminal stop reason."""
        return self.stop_reason is None

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, base_dir: Path | None = None) -> Path:
        """Write the snapshot to disk.  Returns the path written."""
        path = _snapshot_path(self.session_id, base_dir)
        payload = {
            "session_id": self.session_id,
            "initial_observation": self.initial_observation,
            "last_committed_turn": self.last_committed_turn,
            "completed_turns": self.completed_turns,
            "failures": self.failures,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "policy": self.policy,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_id: str, base_dir: Path | None = None) -> LoopSessionSnapshot | None:
        """Load an existing snapshot, or return None if none exists."""
        path = _snapshot_path(session_id, base_dir)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    @classmethod
    def delete(cls, session_id: str, base_dir: Path | None = None) -> bool:
        """Delete a snapshot file.  Returns True if deleted."""
        path = _snapshot_path(session_id, base_dir)
        if path.exists():
            path.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# Snapshot manager — wraps a loop run with auto-persist at each committed turn
# ---------------------------------------------------------------------------


class LoopSnapshotManager:
    """Writes snapshots around an AgentDriverLoop.run() call.

    Pass the hooks returned by this manager to AgentDriverLoop.run() to get
    automatic cold-start recovery support.

    Usage:
        from dadbot.runtime.loop_snapshot import LoopSnapshotManager

        mgr = LoopSnapshotManager(session_id, initial_observation, policy)
        result = loop.run(
            initial_observation,
            session_id=session_id,
            reflection_hook=mgr.wrap_reflection(base_hook),
        )
        mgr.finalize(result)
    """

    def __init__(
        self,
        session_id: str,
        initial_observation: str,
        policy: Any = None,
        *,
        base_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.base_dir = base_dir
        self._policy_dict: dict[str, Any] = {}
        if policy is not None:
            try:
                self._policy_dict = {
                    "max_turns": int(getattr(policy, "max_turns", 8)),
                    "max_failures": int(getattr(policy, "max_failures", 2)),
                    "max_consecutive_noop": int(getattr(policy, "max_consecutive_noop", 2)),
                }
            except Exception:
                pass

        from datetime import datetime, timezone

        self._snap = LoopSessionSnapshot(
            session_id=session_id,
            initial_observation=initial_observation,
            policy=self._policy_dict,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._snap.save(base_dir)

    def wrap_reflection(self, base_hook: Any = None) -> Any:
        """Return a reflection hook that persists a snapshot after each committed turn."""
        snap = self._snap
        base_dir = self.base_dir

        def hook(ctx: dict[str, Any]) -> dict[str, Any]:
            result = dict(base_hook(ctx) if callable(base_hook) else {})
            # Update snapshot with latest committed turn
            snap.last_committed_turn = int(ctx.get("turn_index", snap.last_committed_turn))
            snap.completed_turns = snap.last_committed_turn
            snap.save(base_dir)
            return result

        return hook

    def finalize(self, result: Any) -> None:
        """Mark the snapshot as complete and persist the final state."""
        from datetime import datetime, timezone

        self._snap.stop_reason = str(getattr(result, "stop_reason", "unknown"))
        self._snap.completed_turns = int(getattr(result, "completed_turns", 0))
        self._snap.failures = int(getattr(result, "failures", 0))
        self._snap.finished_at = datetime.now(timezone.utc).isoformat()
        self._snap.save(self.base_dir)


__all__ = [
    "LoopSessionSnapshot",
    "LoopSnapshotManager",
]
