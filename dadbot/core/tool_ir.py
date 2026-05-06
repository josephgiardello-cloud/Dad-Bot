from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    args: dict[str, Any]
    intent: str
    expected_output: str
    priority: int = 100


@dataclass(frozen=True)
class ToolExecution:
    tool_name: str
    input_hash: str
    output: Any
    latency: float
    status: str
    deterministic_id: str


@dataclass(frozen=True)
class ToolExecutionPlan:
    requests: list[ToolRequest] = field(default_factory=list)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: str
    output: Any
    deterministic_id: str


# ---------------------------------------------------------------------------
# Phase 3: ToolStatus taxonomy + ToolContractResult
# ---------------------------------------------------------------------------


class ToolStatus(enum.Enum):
    """Structured failure taxonomy for contractual tool use.

    SUCCESS           — Tool executed and returned valid output.
    RETRY             — Transient failure; safe to retry with same args.
    CONTRACT_VIOLATION — Required arguments missing or schema mismatch.
                         Surfaces repair_hint to the Planner repair loop.
    FATAL             — Non-recoverable failure; abort the tool call branch.
    """

    SUCCESS = "success"
    RETRY = "retry"
    CONTRACT_VIOLATION = "contract_violation"
    FATAL = "fatal"


@dataclass(frozen=True)
class ToolContractResult:
    """Structured tool result with failure taxonomy for the ValidationGate.

    On CONTRACT_VIOLATION, ``repair_hint`` is forwarded to the Planner so the
    repair loop can emit a corrected tool request.
    """

    tool_name: str
    status: ToolStatus
    data: Any
    error_context: dict[str, Any]
    repair_hint: str = ""


def stable_tool_input_hash(tool_name: str, args: dict[str, Any]) -> str:
    payload = {
        "tool_name": str(tool_name or "").strip().lower(),
        "args": dict(args or {}),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def deterministic_tool_id(tool_name: str, args: dict[str, Any]) -> str:
    return stable_tool_input_hash(tool_name, args)[:24]


def normalize_tool_results(
    values: list[ToolResult | dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for value in list(values or []):
        if isinstance(value, ToolResult):
            normalized.append(
                {
                    "tool_name": value.tool_name,
                    "status": value.status,
                    "output": value.output,
                    "deterministic_id": value.deterministic_id,
                },
            )
            continue
        item = dict(value or {})
        normalized.append(
            {
                "tool_name": str(item.get("tool_name") or ""),
                "status": str(item.get("status") or "ok"),
                "output": item.get("output"),
                "deterministic_id": str(item.get("deterministic_id") or ""),
            },
        )
    return normalized


def build_execution_event(
    tool_name: str,
    args: dict[str, Any],
    output: Any,
    status: str,
    started_at: float,
) -> ToolExecution:
    input_hash = stable_tool_input_hash(tool_name, args)
    return ToolExecution(
        tool_name=str(tool_name or ""),
        input_hash=input_hash,
        output=output,
        latency=round(max(time.perf_counter() - float(started_at), 0.0), 6),
        status=str(status or "ok"),
        deterministic_id=input_hash[:24],
    )


# ---------------------------------------------------------------------------
# Phase 3: Event-sourced execution stream
# ---------------------------------------------------------------------------


class ToolEventType(enum.Enum):
    """Lifecycle event types in the tool execution event stream."""

    REQUESTED = "requested"
    EXECUTED = "executed"
    FAILED = "failed"
    MERGED = "merged"


@dataclass(frozen=True)
class ToolEvent:
    """Immutable event record in the tool execution event stream.

    ``input_hash`` and ``output_hash`` are content-addressed identifiers so
    event streams can be replayed and compared without reference to wall-clock
    time or latency noise.
    """

    event_type: ToolEventType
    tool_id: str
    sequence: int
    input_hash: str
    output_hash: str
    payload: dict[str, Any]

    @classmethod
    def requested(
        cls,
        tool_id: str,
        sequence: int,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolEvent:
        input_hash = stable_tool_input_hash(tool_name, args)
        return cls(
            event_type=ToolEventType.REQUESTED,
            tool_id=tool_id,
            sequence=sequence,
            input_hash=input_hash,
            output_hash="",
            payload={"tool_name": tool_name, "args": dict(args)},
        )

    @classmethod
    def executed(
        cls,
        tool_id: str,
        sequence: int,
        tool_name: str,
        args: dict[str, Any],
        output: Any,
        status: str = "ok",
    ) -> ToolEvent:
        input_hash = stable_tool_input_hash(tool_name, args)
        output_hash = hashlib.sha256(
            json.dumps(output, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:24]
        return cls(
            event_type=ToolEventType.EXECUTED,
            tool_id=tool_id,
            sequence=sequence,
            input_hash=input_hash,
            output_hash=output_hash,
            payload={
                "tool_name": tool_name,
                "args": dict(args),
                "output": output,
                "status": status,
            },
        )

    @classmethod
    def failed(
        cls,
        tool_id: str,
        sequence: int,
        tool_name: str,
        args: dict[str, Any],
        error: str,
    ) -> ToolEvent:
        input_hash = stable_tool_input_hash(tool_name, args)
        return cls(
            event_type=ToolEventType.FAILED,
            tool_id=tool_id,
            sequence=sequence,
            input_hash=input_hash,
            output_hash="",
            payload={"tool_name": tool_name, "args": dict(args), "error": error},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "tool_id": self.tool_id,
            "sequence": self.sequence,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "payload": dict(self.payload),
        }


@dataclass
class ToolEventLog:
    """Ordered, append-only log of ToolEvents.

    The ToolExecutor is a *reducer* over this log: given an event log and an
    initial state, it deterministically derives the final execution state.
    Replaying the same event log always produces the same final state.
    """

    events: list[ToolEvent] = field(default_factory=list)

    def append(self, event: ToolEvent) -> None:
        self.events.append(event)

    def event_count(self) -> int:
        return len(self.events)

    def events_for_tool(self, tool_id: str) -> list[ToolEvent]:
        return [e for e in self.events if e.tool_id == tool_id]

    def replay_hash(self) -> str:
        """Deterministic hash of the full event log (replay identity)."""
        payload = [e.to_dict() for e in self.events]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]


def reduce_events_to_results(log: ToolEventLog) -> list[dict[str, Any]]:
    """Reducer: given a ToolEventLog, derive the canonical list of tool results.

    ToolExecutor is a reducer over events — this is the pure reduction function.
    Only EXECUTED and FAILED events contribute to the output state.
    Events are processed in sequence order for determinism.
    """
    ordered = sorted(log.events, key=lambda e: e.sequence)
    results: list[dict[str, Any]] = []
    for event in ordered:
        if event.event_type == ToolEventType.EXECUTED:
            results.append(
                {
                    "tool_id": event.tool_id,
                    "tool_name": str(event.payload.get("tool_name") or ""),
                    "status": str(event.payload.get("status") or "ok"),
                    "output": event.payload.get("output"),
                    "input_hash": event.input_hash,
                    "output_hash": event.output_hash,
                    "sequence": event.sequence,
                },
            )
        elif event.event_type == ToolEventType.FAILED:
            results.append(
                {
                    "tool_id": event.tool_id,
                    "tool_name": str(event.payload.get("tool_name") or ""),
                    "status": "error",
                    "output": str(event.payload.get("error") or ""),
                    "input_hash": event.input_hash,
                    "output_hash": event.output_hash,
                    "sequence": event.sequence,
                },
            )
    return results
