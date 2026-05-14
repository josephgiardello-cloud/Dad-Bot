from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field  # pyright: ignore[reportMissingImports]

type ChunkCallback = Callable[[str], Any]
type TurnResult = tuple[str | None, bool]


class ExecutionMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"
    RECOVERY = "recovery"


class TurnDelivery(StrEnum):
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"
    STREAM_ASYNC = "stream_async"


@dataclass(slots=True)
class SovereignContext:
    session_id: str = "default"
    tenant_id: str = "default"
    trace_id: str = ""
    request_id: str = ""
    execution_mode: ExecutionMode = ExecutionMode.LIVE
    policy_scope: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": str(self.session_id or "default"),
            "tenant_id": str(self.tenant_id or "default"),
            "trace_id": str(self.trace_id or ""),
            "request_id": str(self.request_id or ""),
            "execution_mode": self.execution_mode.value,
            "policy_scope": str(self.policy_scope or "default"),
        }


class UserInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class MessageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class NodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    input_hash: str = ""
    output_hash: str = ""
    tool_result_hash: str = ""
    memory_read_hash: str = ""
    memory_write_hash: str = ""


class ToolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    input_hash: str = ""
    output_hash: str = ""


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: UserInput
    mode: ExecutionMode = ExecutionMode.LIVE
    delivery: TurnDelivery = TurnDelivery.SYNC
    session_id: str = "default"
    timeout_seconds: float | None = None
    context: SovereignContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str | None = None
    should_end: bool = False
    mode: ExecutionMode = ExecutionMode.LIVE
    delivery: TurnDelivery = TurnDelivery.SYNC

    def as_result(self) -> TurnResult:
        return self.reply, self.should_end


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    step_id: int = 0
    current_node: str = ""
    node_history: list[NodeRecord] = Field(default_factory=list)
    short_term_context: list[MessageRecord] = Field(default_factory=list)
    memory_ref: str | None = None
    tool_trace: list[ToolRecord] = Field(default_factory=list)
    invariance_hash: str = ""

    def recompute_invariance_hash(self) -> str:
        payload = {
            "run_id": str(self.run_id),
            "step_id": int(self.step_id),
            "current_node": str(self.current_node),
            "node_history": [item.model_dump() for item in self.node_history],
            "short_term_context": [item.model_dump() for item in self.short_term_context],
            "memory_ref": str(self.memory_ref or ""),
            "tool_trace": [item.model_dump() for item in self.tool_trace],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()
        self.invariance_hash = digest
        return digest


class TurnExecutor(Protocol):
    def run_turn(
        self,
        input: UserInput,
        state: AgentState,
        *,
        chunk_callback: ChunkCallback | None = None,
        mode: ExecutionMode = ExecutionMode.LIVE,
    ) -> TurnResult: ...


class TurnRuntimeContract(Protocol):
    def execute_turn(
        self,
        request: TurnRequest,
        *,
        state: AgentState | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> TurnResponse | Awaitable[TurnResponse]: ...


class ExecutionEntry:
    """Single authoritative runtime execution entry surface.

    Runtime adapters must bind this at construction and call ``execute_turn`` only.
    """

    def __init__(
        self,
        execute_turn: Callable[[TurnRequest], TurnResponse | Awaitable[TurnResponse]] | None,
    ) -> None:
        if not callable(execute_turn):
            raise RuntimeError("ExecutionEntry requires a callable execute_turn at initialization")
        self._execute_turn = execute_turn

    def execute_turn(self, request: TurnRequest) -> TurnResponse | Awaitable[TurnResponse]:
        return self._execute_turn(request)


def live_turn_request(
    text: str,
    attachments: list[dict[str, Any]] | None = None,
    *,
    delivery: TurnDelivery = TurnDelivery.SYNC,
    timeout_seconds: float | None = None,
    context: SovereignContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> TurnRequest:
    resolved_context = context or SovereignContext()
    return TurnRequest(
        input=UserInput(text=str(text or ""), attachments=list(attachments or [])),
        mode=ExecutionMode.LIVE,
        delivery=delivery,
        session_id=str(resolved_context.session_id or "default"),
        timeout_seconds=timeout_seconds,
        context=resolved_context,
        metadata=dict(metadata or {}),
    )
