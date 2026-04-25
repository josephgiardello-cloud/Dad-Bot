from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from threading import RLock
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger


class CheckpointIntegrityError(RuntimeError):
    pass


class DurableCheckpoint:
    """Durable ledger-head checkpoint with hash-chained integrity.

    Contract:
    - A checkpoint records the exact ledger sequence + replay hash at a point in time.
    - On startup, assert_resume_at_head() verifies the current ledger head matches the
      last saved checkpoint.  If it doesn't, it raises CheckpointIntegrityError and
      the runtime must NOT proceed.
    - Checkpoints are hash-chained: each new checkpoint includes the hash of the
      previous one, forming a tamper-evident sequence.
    """

    def __init__(self, *, ledger: ExecutionLedger) -> None:
        self._ledger = ledger
        self._lock = RLock()
        self._checkpoints: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, *, label: str = "runtime") -> dict[str, Any]:
        """Snapshot current ledger head into a new checkpoint and return it."""
        events = self._ledger.read()
        replay_hash = self._ledger.replay_hash()
        sequence = int(events[-1].get("sequence") or 0) if events else 0
        event_count = len(events)

        with self._lock:
            prev_hash = self._checkpoints[-1]["checkpoint_hash"] if self._checkpoints else ""

        checkpoint = {
            "label": str(label or "runtime"),
            "created_at": time.time(),
            "ledger_sequence": sequence,
            "ledger_event_count": event_count,
            "replay_hash": str(replay_hash or ""),
            "prev_checkpoint_hash": str(prev_hash or ""),
        }
        checkpoint["checkpoint_hash"] = self._hash_checkpoint(checkpoint)

        with self._lock:
            self._checkpoints.append(deepcopy(checkpoint))

        return deepcopy(checkpoint)

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def assert_resume_at_head(self) -> dict[str, Any]:
        """Assert the current ledger head matches the last saved checkpoint.

        Raises CheckpointIntegrityError if:
        - there is a saved checkpoint AND the current replay hash does not match it, or
        - there is a saved checkpoint AND the current event count is less than recorded.

        Returns the verification report dict.
        """
        with self._lock:
            if not self._checkpoints:
                return {
                    "ok": True,
                    "reason": "no checkpoints saved yet â€” clean start",
                    "checkpoint_count": 0,
                }
            last = deepcopy(self._checkpoints[-1])

        current_events = self._ledger.read()
        current_replay_hash = self._ledger.replay_hash()
        current_event_count = len(current_events)

        expected_replay_hash = str(last.get("replay_hash") or "")
        expected_event_count = int(last.get("ledger_event_count") or 0)

        report = {
            "ok": False,
            "checkpoint_label": str(last.get("label") or ""),
            "checkpoint_hash": str(last.get("checkpoint_hash") or ""),
            "checkpoint_created_at": float(last.get("created_at") or 0.0),
            "expected_replay_hash": expected_replay_hash,
            "actual_replay_hash": current_replay_hash,
            "expected_event_count": expected_event_count,
            "actual_event_count": current_event_count,
            "checkpoint_count": 0,
        }
        with self._lock:
            report["checkpoint_count"] = len(self._checkpoints)

        if current_event_count < expected_event_count:
            msg = (
                f"Ledger truncation detected: expected at least {expected_event_count} events, "
                f"found {current_event_count}. Refusing to resume."
            )
            raise CheckpointIntegrityError(msg)

        if current_replay_hash != expected_replay_hash:
            msg = (
                f"Ledger replay hash mismatch: expected {expected_replay_hash!r}, "
                f"got {current_replay_hash!r}. Ledger may be corrupted or partially written."
            )
            raise CheckpointIntegrityError(msg)

        report["ok"] = True
        report["reason"] = "ledger head matches last checkpoint"
        return report

    def verify_chain_integrity(self) -> dict[str, Any]:
        """Verify every checkpoint's hash chain is unbroken."""
        with self._lock:
            chain = list(self._checkpoints)

        violations: list[str] = []
        for index, checkpoint in enumerate(chain):
            expected = self._hash_checkpoint(
                {key: value for key, value in checkpoint.items() if key != "checkpoint_hash"}
            )
            if checkpoint.get("checkpoint_hash") != expected:
                violations.append(
                    f"Checkpoint {index} hash mismatch: stored={checkpoint.get('checkpoint_hash')!r}"
                )
            if index > 0:
                prev = chain[index - 1]
                if checkpoint.get("prev_checkpoint_hash") != prev.get("checkpoint_hash"):
                    violations.append(
                        f"Checkpoint {index} prev_hash broken: "
                        f"expected={prev.get('checkpoint_hash')!r}, "
                        f"got={checkpoint.get('prev_checkpoint_hash')!r}"
                    )

        return {
            "ok": len(violations) == 0,
            "chain_length": len(chain),
            "violations": violations,
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._checkpoints[-1]) if self._checkpoints else None

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._checkpoints)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_checkpoint(data: dict[str, Any]) -> str:
        serialized = json.dumps(
            {key: value for key, value in sorted(data.items()) if key != "checkpoint_hash"},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
