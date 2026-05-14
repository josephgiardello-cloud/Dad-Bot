"""Tool IO Recording and Replay Infrastructure.

This module provides immutable records of all tool inputs and outputs during
a turn execution. These records are persisted in checkpoints and used during
replay to bypass re-execution and return deterministic recorded outputs.

Key classes:
  - ToolIORecord: Immutable record of one tool call (input, output, latency)
  - ToolIOLedger: Sequential log of all tool IO in a turn

See archive/docs/legacy-root-notes/PRIORITY_1_IO_DETERMINISM_SEALING.md for architecture details.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _stable_payload_hash(payload: Any) -> str:
    """Deterministic hash of a payload for content addressing."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


@dataclass(frozen=True)
class ToolIORecord:
    """Immutable record of one complete tool execution IO.

    Captures input parameters, actual output, execution status, and metadata.
    Used for deterministic replay: on replay, return this recorded output
    instead of re-executing the tool.

    Attributes:
        sequence: Order in the turn's tool execution sequence (1-based)
        tool_name: Canonical tool name (e.g., "set_reminder", "web_search")
        input_hash: Stable hash of {tool_name, parameters} for fast lookup
        input_payload: Raw parameters passed to tool
        output_payload: Actual tool output (return value)
        output_hash: Stable hash of output_payload
        status: "ok" | "error" | "partial" | "replayed" | "degraded"
        latency_ms: Wall-clock latency of the execution
        error: Error message if status == "error", else ""
        metadata: Optional additional context (permissions, retry count, etc.)
    """

    sequence: int
    tool_name: str
    input_hash: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]
    output_hash: str
    status: str
    latency_ms: float
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_replay_hit(self) -> bool:
        """True if this record was returned during replay (not live execution)."""
        return self.status == "replayed"

    def is_success(self) -> bool:
        """True if execution succeeded (status ok/partial/cached/replayed)."""
        return self.status in {"ok", "partial", "cached", "replayed"}

    def matches_request(self, tool_name: str, input_hash: str) -> bool:
        """True if this record matches a given tool request."""
        return self.tool_name == tool_name and self.input_hash == input_hash

    def lookup_key(
        self,
        *,
        tool_version: str = "",
        environment_fingerprint: str = "",
    ) -> str:
        """Unique key for fast lookup.

        The replay key is intentionally versioned so that tool code changes or
        environment drift fail closed instead of replaying a stale result.
        """
        return "|".join(
            [
                self.tool_name,
                self.input_hash,
                str(tool_version or ""),
                str(environment_fingerprint or ""),
            ],
        )


@dataclass
class ToolIOLedger:
    """Sequential log of all tool IO in a turn.

    Maintains insertion order (for replay sequence) and an index for
    O(1) lookup by (tool_name, input_hash) during replay injection.

    On checkpoint save, serialize via to_dict().
    On checkpoint load, restore via from_dict().
    """

    records: list[ToolIORecord] = field(default_factory=list)
    _by_request_hash: dict[str, ToolIORecord] = field(default_factory=dict, init=False, repr=False)

    @staticmethod
    def _build_lookup_key(
        tool_name: str,
        input_hash: str,
        *,
        tool_version: str = "",
        environment_fingerprint: str = "",
    ) -> str:
        return "|".join(
            [
                str(tool_name or ""),
                str(input_hash or ""),
                str(tool_version or ""),
                str(environment_fingerprint or ""),
            ],
        )

    def append(self, record: ToolIORecord) -> None:
        """Add a new tool IO record to the ledger.

        Maintains both sequential order (for full replay) and hash index
        (for O(1) replay injection lookup).
        """
        if not isinstance(record, ToolIORecord):
            raise TypeError(f"append requires ToolIORecord, got {type(record)}")
        self.records.append(record)
        metadata = dict(record.metadata or {})
        self._by_request_hash[
            record.lookup_key(
                tool_version=str(metadata.get("tool_version") or ""),
                environment_fingerprint=str(metadata.get("environment_fingerprint") or ""),
            )
        ] = record

    def lookup(
        self,
        tool_name: str,
        input_hash: str,
        *,
        tool_version: str = "",
        environment_fingerprint: str = "",
    ) -> ToolIORecord | None:
        """Fast O(1) lookup for replay injection.

        Returns the recorded output for a given tool + input, or None if
        not found (indicating this is the first execution, not a replay).
        """
        key = self._build_lookup_key(
            tool_name,
            input_hash,
            tool_version=tool_version,
            environment_fingerprint=environment_fingerprint,
        )
        record = self._by_request_hash.get(key)
        if record is not None:
            return record

        # Compatibility fallback: older callers/tests may not provide versioned
        # lookup coordinates. In that case, return the latest matching record
        # by tool_name + input_hash.
        if not str(tool_version or "") and not str(environment_fingerprint or ""):
            for candidate in reversed(self.records):
                if candidate.matches_request(str(tool_name or ""), str(input_hash or "")):
                    return candidate
        return None

    def all_records(self) -> list[ToolIORecord]:
        """Return all records in execution order."""
        return list(self.records)

    def successful_calls(self) -> int:
        """Count of successful tool calls (is_success() == True)."""
        return sum(1 for r in self.records if r.is_success())

    def failed_calls(self) -> int:
        """Count of failed tool calls."""
        return sum(1 for r in self.records if not r.is_success())

    def to_dict(self) -> dict[str, Any]:
        """Serialize ledger to a dict for checkpoint persistence.

        Includes both raw records and metadata for validation.
        """
        return {
            "records": [asdict(r) for r in self.records],
            "sequence_count": len(self.records),
            "successful_count": self.successful_calls(),
            "failed_count": self.failed_calls(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolIOLedger:
        """Deserialize ledger from checkpoint dict.

        Reconstructs both records and the fast-lookup index.
        """
        if not isinstance(data, dict):
            return cls()  # Empty ledger on bad input

        log = cls()
        for rec_dict in data.get("records", []):
            if not isinstance(rec_dict, dict):
                continue
            try:
                record = ToolIORecord(**rec_dict)
                log.append(record)
            except TypeError:
                # Skip records with missing/bad fields
                continue
        return log

    def is_empty(self) -> bool:
        """True if no tool calls have been recorded."""
        return len(self.records) == 0

    def last_record(self) -> ToolIORecord | None:
        """Return the most recent tool IO record, or None if empty."""
        return self.records[-1] if self.records else None

    def clear(self) -> None:
        """Clear all records (for resetting between turns if needed)."""
        self.records.clear()
        self._by_request_hash.clear()


__all__ = [
    "ToolIORecord",
    "ToolIOLedger",
]
