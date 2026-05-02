from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


ChunkCallback: TypeAlias = Callable[[str], Any]
TurnResult: TypeAlias = tuple[str | None, bool]


class ExecutionMode(str, Enum):
    LIVE = "live"
    REPLAY = "replay"
    RECOVERY = "recovery"


class TurnDelivery(str, Enum):
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"
    STREAM_ASYNC = "stream_async"


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


def live_turn_request(
    text: str,
    attachments: list[dict[str, Any]] | None = None,
    *,
    delivery: TurnDelivery = TurnDelivery.SYNC,
    session_id: str = "default",
    timeout_seconds: float | None = None,
) -> TurnRequest:
    return TurnRequest(
        input=UserInput(text=str(text or ""), attachments=list(attachments or [])),
        mode=ExecutionMode.LIVE,
        delivery=delivery,
        session_id=str(session_id or "default"),
        timeout_seconds=timeout_seconds,
    )
