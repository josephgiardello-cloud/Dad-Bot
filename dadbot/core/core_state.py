from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from dadbot.core.canonical_event import canonicalize_event_payload

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-"
    r"[0-9a-fA-F]{12}$",
)


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def deterministic_id(*, parent_state_hash: str, input_event: "InputEvent") -> str:
    payload = {
        "parent_state_hash": str(parent_state_hash or ""),
        "event_type": str(input_event.event_type or "").strip().lower(),
        "payload": canonicalize_event_payload(dict(input_event.payload or {})),
        "metadata": dict(input_event.metadata or {}),
    }
    return _stable_hash(payload)


@dataclass(frozen=True, slots=True)
class InputEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    summary: str
    category: str
    mood: str
    updated_at: str = ""
    created_at: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "MemoryEntry":
        source = dict(payload or {})
        return MemoryEntry(
            summary=str(source.get("summary") or "").strip(),
            category=str(source.get("category") or "").strip(),
            mood=str(source.get("mood") or "").strip(),
            updated_at=str(source.get("updated_at") or "").strip(),
            created_at=str(source.get("created_at") or "").strip(),
            payload=canonicalize_event_payload(source),
        )

    def sort_key(self) -> tuple[str, str, str, str, str, str]:
        payload_hash = _stable_hash(self.payload)
        return (
            self.updated_at,
            self.created_at,
            self.summary,
            self.category,
            self.mood,
            payload_hash,
        )


@dataclass(frozen=True, slots=True)
class SortedMemoryVector:
    entries: tuple[MemoryEntry, ...] = ()

    @staticmethod
    def from_entries(entries: list[MemoryEntry] | tuple[MemoryEntry, ...]) -> "SortedMemoryVector":
        sorted_entries = tuple(sorted(list(entries or ()), key=lambda item: item.sort_key()))
        return SortedMemoryVector(entries=sorted_entries)

    def insert(self, entry: MemoryEntry) -> "SortedMemoryVector":
        return SortedMemoryVector.from_entries([*self.entries, entry])

    def to_dict_list(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in self.entries:
            output.append(
                {
                    "summary": item.summary,
                    "category": item.category,
                    "mood": item.mood,
                    "updated_at": item.updated_at,
                    "created_at": item.created_at,
                    **dict(item.payload or {}),
                },
            )
        return output


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    event_id: str
    event_type: str
    parent_event_id: str
    parent_state_hash: str
    payload: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_input(
        *,
        parent_state_hash: str,
        input_event: InputEvent,
        parent_event_id: str,
    ) -> "CanonicalEvent":
        event_id = deterministic_id(parent_state_hash=parent_state_hash, input_event=input_event)
        if _UUID_RE.match(event_id):
            raise ValueError("CanonicalEvent must use deterministic hash identifiers only")
        return CanonicalEvent(
            event_id=event_id,
            event_type=str(input_event.event_type or "").strip().lower(),
            parent_event_id=str(parent_event_id or "").strip(),
            parent_state_hash=str(parent_state_hash or "").strip(),
            payload=canonicalize_event_payload(dict(input_event.payload or {})),
        )


@dataclass(frozen=True, slots=True)
class ExecutionState:
    trace_id: str = ""
    last_response: str = ""
    should_end: bool = False


@dataclass(frozen=True, slots=True)
class CoreState:
    version: int = 0
    memory: SortedMemoryVector = field(default_factory=SortedMemoryVector)
    memory_kv: dict[str, Any] = field(default_factory=dict)
    events: tuple[CanonicalEvent, ...] = ()
    execution: ExecutionState = field(default_factory=ExecutionState)

    def state_hash(self) -> str:
        payload = {
            "version": int(self.version),
            "memory": self.memory.to_dict_list(),
            "memory_kv": canonicalize_event_payload(dict(self.memory_kv or {})),
            "events": [
                {
                    "event_id": item.event_id,
                    "event_type": item.event_type,
                    "parent_event_id": item.parent_event_id,
                    "parent_state_hash": item.parent_state_hash,
                    "payload": dict(item.payload or {}),
                }
                for item in self.events
            ],
            "execution": {
                "trace_id": self.execution.trace_id,
                "last_response": self.execution.last_response,
                "should_end": bool(self.execution.should_end),
            },
        }
        return _stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "memory": self.memory.to_dict_list(),
            "memory_kv": canonicalize_event_payload(dict(self.memory_kv or {})),
            "events": [
                {
                    "event_id": item.event_id,
                    "event_type": item.event_type,
                    "parent_event_id": item.parent_event_id,
                    "parent_state_hash": item.parent_state_hash,
                    "payload": dict(item.payload or {}),
                }
                for item in self.events
            ],
            "execution": {
                "trace_id": self.execution.trace_id,
                "last_response": self.execution.last_response,
                "should_end": bool(self.execution.should_end),
            },
        }

    @staticmethod
    def _parse_memory_entries(source: dict[str, Any]) -> "list[MemoryEntry]":
        entries: list[MemoryEntry] = []
        for item in list(source.get("memory") or []):
            if isinstance(item, dict):
                entries.append(MemoryEntry.from_payload(item))
        return entries

    @staticmethod
    def _parse_canonical_events(source: dict[str, Any]) -> "list[CanonicalEvent]":
        events: list[CanonicalEvent] = []
        for item in list(source.get("events") or []):
            if not isinstance(item, dict):
                continue
            events.append(
                CanonicalEvent(
                    event_id=str(item.get("event_id") or "").strip(),
                    event_type=str(item.get("event_type") or "").strip().lower(),
                    parent_event_id=str(item.get("parent_event_id") or "").strip(),
                    parent_state_hash=str(item.get("parent_state_hash") or "").strip(),
                    payload=canonicalize_event_payload(dict(item.get("payload") or {})),
                ),
            )
        return events

    @staticmethod
    def _parse_execution_state(source: dict[str, Any]) -> "ExecutionState":
        ep = dict(source.get("execution") or {})
        return ExecutionState(
            trace_id=str(ep.get("trace_id") or "").strip(),
            last_response=str(ep.get("last_response") or ""),
            should_end=bool(ep.get("should_end", False)),
        )

    @staticmethod
    def from_dict(payload: dict[str, Any] | None) -> "CoreState":
        source = dict(payload or {})
        memory_entries = CoreState._parse_memory_entries(source)
        events = CoreState._parse_canonical_events(source)
        execution = CoreState._parse_execution_state(source)
        return CoreState(
            version=int(source.get("version") or 0),
            memory=SortedMemoryVector.from_entries(memory_entries),
            memory_kv=canonicalize_event_payload(dict(source.get("memory_kv") or {})),
            events=tuple(events),
            execution=execution,
        )


@dataclass(frozen=True, slots=True)
class MemoryView:
    core_state: CoreState

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return self.core_state.memory.entries


@dataclass(frozen=True, slots=True)
class GraphView:
    core_state: CoreState

    @property
    def adjacency(self) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for event in self.core_state.events:
            parent = str(event.parent_event_id or "root")
            graph.setdefault(parent, []).append(event.event_id)
        return {key: sorted(value) for key, value in graph.items()}


@dataclass(frozen=True, slots=True)
class ExecutionView:
    core_state: CoreState

    @property
    def state(self) -> ExecutionState:
        return self.core_state.execution


@dataclass(frozen=True, slots=True)
class CanonicalEventView:
    core_state: CoreState

    @property
    def events(self) -> tuple[CanonicalEvent, ...]:
        return self.core_state.events


@dataclass(frozen=True, slots=True)
class FacadeView:
    core_state: CoreState

    def as_payload(self) -> dict[str, Any]:
        return {
            "state_hash": self.core_state.state_hash(),
            "version": int(self.core_state.version),
            "memory_count": len(self.core_state.memory.entries),
            "event_count": len(self.core_state.events),
            "last_event_id": str(self.core_state.events[-1].event_id if self.core_state.events else ""),
            "trace_id": self.core_state.execution.trace_id,
            "should_end": bool(self.core_state.execution.should_end),
        }


@dataclass(frozen=True, slots=True)
class CoreStateViews:
    memory: MemoryView
    graph: GraphView
    execution: ExecutionView
    canonical: CanonicalEventView
    facade: FacadeView


def project_views(core_state: CoreState) -> CoreStateViews:
    return CoreStateViews(
        memory=MemoryView(core_state=core_state),
        graph=GraphView(core_state=core_state),
        execution=ExecutionView(core_state=core_state),
        canonical=CanonicalEventView(core_state=core_state),
        facade=FacadeView(core_state=core_state),
    )


def memory_projection(
    core_state: CoreState,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a mutable-style memory dict from canonical CoreState.

    This projection is reconstructible from CoreState and must never be used as
    an authority surface.
    """
    projected = canonicalize_event_payload(dict(defaults or {}))
    projected.update(canonicalize_event_payload(dict(core_state.memory_kv or {})))
    projected["memories"] = core_state.memory.to_dict_list()
    return projected


def _memory_entries_from_list(memories_list: Any) -> SortedMemoryVector:
    if not isinstance(memories_list, list):
        return SortedMemoryVector.from_entries(())
    return SortedMemoryVector.from_entries(
        [MemoryEntry.from_payload(item) for item in memories_list if isinstance(item, dict)],
    )


def _apply_legacy_memory_initialized(
    *,
    payload: dict[str, Any],
) -> tuple[SortedMemoryVector, dict[str, Any]]:
    store = dict(payload.get("store") or {})
    memory = _memory_entries_from_list(list(store.get("memories") or []))
    memory_kv = {str(key): value for key, value in store.items() if str(key) != "memories"}
    return memory, memory_kv


def _apply_legacy_memory_appended(
    *,
    payload: dict[str, Any],
    memory: SortedMemoryVector,
    memory_kv: dict[str, Any],
) -> tuple[SortedMemoryVector, dict[str, Any]]:
    entry = payload.get("entry")
    if isinstance(entry, dict):
        memory = memory.insert(MemoryEntry.from_payload(entry))
    return memory, memory_kv


def _apply_legacy_memory_updated(
    *,
    payload: dict[str, Any],
    memory: SortedMemoryVector,
    memory_kv: dict[str, Any],
) -> tuple[SortedMemoryVector, dict[str, Any]]:
    key = str(payload.get("key") or "").strip()
    if key == "memories":
        memory = _memory_entries_from_list(payload.get("value"))
    elif key:
        memory_kv[key] = payload.get("value")
    return memory, memory_kv


def _apply_legacy_memory_deleted(
    *,
    payload: dict[str, Any],
    memory: SortedMemoryVector,
    memory_kv: dict[str, Any],
) -> tuple[SortedMemoryVector, dict[str, Any]]:
    key = str(payload.get("key") or "").strip()
    if key == "memories":
        memory = SortedMemoryVector.from_entries(())
    elif key:
        memory_kv.pop(key, None)
    return memory, memory_kv


def _apply_legacy_memory_projection_event(
    *,
    event_type: str,
    payload: dict[str, Any],
    memory: SortedMemoryVector,
    memory_kv: dict[str, Any],
) -> tuple[SortedMemoryVector, dict[str, Any]]:
    if event_type == "memoryinitialized":
        return _apply_legacy_memory_initialized(payload=payload)

    if event_type == "memoryappended":
        return _apply_legacy_memory_appended(payload=payload, memory=memory, memory_kv=memory_kv)

    if event_type == "memoryupdated":
        return _apply_legacy_memory_updated(payload=payload, memory=memory, memory_kv=memory_kv)

    if event_type == "memorydeleted":
        return _apply_legacy_memory_deleted(payload=payload, memory=memory, memory_kv=memory_kv)

    if event_type == "memorycleared":
        memory = SortedMemoryVector.from_entries(())
        memory_kv = {}
    return memory, memory_kv


def _apply_replacement_and_upsert_events(
    *,
    event_type: str,
    payload: dict[str, Any],
    memory: SortedMemoryVector,
) -> SortedMemoryVector:
    if event_type == "memories_replaced":
        # Full replacement: CoreState.memory becomes the sole authority for the memories list.
        # Legacy memory_store["memories"] is derived from this after transition.
        memories_list = payload.get("memories")
        if isinstance(memories_list, list):
            memory = _memory_entries_from_list(memories_list)
    elif event_type in {"memory_upsert", "turn_committed"}:
        memories = payload.get("memory_retrieval_set")
        if isinstance(memories, list):
            for item in memories:
                if isinstance(item, dict):
                    memory = memory.insert(MemoryEntry.from_payload(item))
    return memory


def _apply_execution_transition(
    *,
    execution: ExecutionState,
    event_type: str,
    payload: dict[str, Any],
) -> ExecutionState:
    if event_type != "turn_committed":
        return execution
    return replace(
        execution,
        trace_id=str(payload.get("trace_id") or execution.trace_id or "").strip(),
        last_response=str(payload.get("response") or ""),
        should_end=bool(payload.get("should_end", False)),
    )


def transition(current_state: CoreState, input_event: InputEvent) -> CoreState:
    state = current_state
    parent_state_hash = state.state_hash()
    parent_event_id = state.events[-1].event_id if state.events else ""
    canonical_event = CanonicalEvent.from_input(
        parent_state_hash=parent_state_hash,
        input_event=input_event,
        parent_event_id=parent_event_id,
    )

    memory = state.memory
    memory_kv = dict(state.memory_kv or {})
    payload = dict(input_event.payload or {})
    event_type = str(input_event.event_type or "").strip().lower()

    memory, memory_kv = _apply_legacy_memory_projection_event(
        event_type=event_type,
        payload=payload,
        memory=memory,
        memory_kv=memory_kv,
    )
    memory = _apply_replacement_and_upsert_events(
        event_type=event_type,
        payload=payload,
        memory=memory,
    )

    # "store_key_mutated": tracks non-memories key changes (version/event recorded, no memory change).
    # "tool_called": routes tool-call observability through the event log.
    # "job_submitted" / "job_completed": background-job lifecycle routing.
    # These event types intentionally produce no memory projection change; they advance
    # the version and append to the canonical event log only.

    execution = _apply_execution_transition(
        execution=state.execution,
        event_type=event_type,
        payload=payload,
    )

    return CoreState(
        version=int(state.version) + 1,
        memory=memory,
        memory_kv=canonicalize_event_payload(memory_kv),
        events=(*state.events, canonical_event),
        execution=execution,
    )


__all__ = [
    "CanonicalEvent",
    "CanonicalEventView",
    "CoreState",
    "CoreStateViews",
    "ExecutionState",
    "ExecutionView",
    "FacadeView",
    "GraphView",
    "InputEvent",
    "MemoryEntry",
    "MemoryView",
    "SortedMemoryVector",
    "deterministic_id",
    "project_views",
    "memory_projection",
    "transition",
]
