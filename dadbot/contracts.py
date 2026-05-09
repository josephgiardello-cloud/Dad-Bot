from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeAlias
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator  # pyright: ignore[reportMissingImports]

Attachment: TypeAlias = dict[str, Any]
AttachmentList: TypeAlias = list[Attachment]
ChunkCallback: TypeAlias = Callable[[str], Any]
PreparedTurnResult: TypeAlias = tuple[str | None, str | None, bool, str, AttachmentList]
FinalizedTurnResult: TypeAlias = tuple[str | None, bool]


class SovereignEventType(StrEnum):
    PLANNER_DECISION = "PLANNER_DECISION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    POLICY_VETO = "POLICY_VETO"
    LOGIC_BRANCH = "LOGIC_BRANCH"
    GENERIC = "GENERIC"


class PlannerDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["PLANNER_DECISION"] = "PLANNER_DECISION"
    planner_node: str = ""
    selected_branch: str = ""
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = ""
    output_hash: str = ""
    error: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VetoReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = ""
    message: str = ""
    severity: str = "high"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["TOOL_EXECUTION"] = "TOOL_EXECUTION"
    tool_name: str = ""
    status: str = "pending"
    input_hash: str = ""
    output_hash: str = ""
    tool_result: ToolResultPayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyVetoPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["POLICY_VETO"] = "POLICY_VETO"
    policy_rule: str = ""
    reason: str = ""
    severity: str = "high"
    veto_reason: VetoReason | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogicBranchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["LOGIC_BRANCH"] = "LOGIC_BRANCH"
    branch_name: str = ""
    condition: str = ""
    outcome: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenericSovereignPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["GENERIC"] = "GENERIC"
    data: dict[str, Any] = Field(default_factory=dict)


SovereignEventPayload: TypeAlias = Annotated[
    PlannerDecisionPayload
    | ToolExecutionPayload
    | PolicyVetoPayload
    | LogicBranchPayload
    | GenericSovereignPayload,
    Field(discriminator="kind"),
]


def _sovereign_event_checksum(
    *,
    event_id: UUID,
    timestamp: datetime,
    turn_id: str,
    event_type: str,
    payload: dict[str, Any],
    previous_checksum: str,
) -> str:
    canonical = {
        "event_id": str(event_id),
        "timestamp": timestamp.astimezone(UTC).isoformat(),
        "turn_id": str(turn_id or ""),
        "event_type": str(event_type or ""),
        "payload": payload,
        "previous_checksum": str(previous_checksum or ""),
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()
    return f"evtchk-{digest}"


class SovereignEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    turn_id: str
    event_type: str
    payload: SovereignEventPayload
    previous_checksum: str = ""
    checksum: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_type(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        payload = normalized.get("payload")
        payload_kind = ""
        if isinstance(payload, dict):
            payload_kind = str(payload.get("kind") or "")
        else:
            payload_kind = str(getattr(payload, "kind", "") or "")
        event_type = str(normalized.get("event_type") or "").strip()
        normalized["event_type"] = event_type or payload_kind or SovereignEventType.GENERIC.value
        return normalized

    @model_validator(mode="after")
    def _normalize_and_validate_checksum(self) -> SovereignEvent:
        normalized_type = str(self.event_type or "").strip() or str(self.payload.kind)
        if self.payload.kind != SovereignEventType.GENERIC.value and normalized_type != self.payload.kind:
            raise ValueError(
                f"event_type {normalized_type!r} must match payload kind {self.payload.kind!r}",
            )
        expected_checksum = _sovereign_event_checksum(
            event_id=self.event_id,
            timestamp=self.timestamp,
            turn_id=self.turn_id,
            event_type=normalized_type,
            payload=self.payload.model_dump(mode="json"),
            previous_checksum=self.previous_checksum,
        )
        if str(self.checksum or "").strip() and self.checksum != expected_checksum:
            raise ValueError("checksum mismatch for sovereign event")
        self.checksum = expected_checksum
        return self

    def verify_checksum(self, previous_checksum: str = "") -> bool:
        expected = _sovereign_event_checksum(
            event_id=self.event_id,
            timestamp=self.timestamp,
            turn_id=self.turn_id,
            event_type=self.event_type,
            payload=self.payload.model_dump(mode="json"),
            previous_checksum=previous_checksum,
        )
        return self.previous_checksum == str(previous_checksum or "") and self.checksum == expected

    def to_ledger_event(self) -> dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "turn_id": str(self.turn_id or ""),
            "event_type": str(self.event_type or ""),
            "payload": self.payload.model_dump(mode="json"),
            "previous_checksum": str(self.previous_checksum or ""),
            "checksum": str(self.checksum or ""),
        }


class SupportsDadBotAccess(Protocol):
    bot: Any


class SupportsProfileContext(Protocol):
    def build_profile_block(self, fact_ids: list[str] | None = None) -> str: ...

    def relevant_fact_ids_for_input(self, user_input: str) -> list[str]: ...

    def matching_topics(self, user_input: str) -> list[dict[str, Any]]: ...


class SupportsLongTermSignals(Protocol):
    def build_wisdom_context(self, user_input: str) -> str | None: ...

    def build_deep_pattern_context(
        self,
        user_input: str,
        limit: int | None = None,
    ) -> str | None: ...

    def trait_impact(self, entry: dict[str, Any]) -> float: ...

    def decayed_trait_strength(self, entry: dict[str, Any]) -> float: ...


class SupportsRelationshipSnapshot(Protocol):
    def build_prompt_context(self) -> str: ...


class SupportsRelationshipMemory(Protocol):
    def relationship_state(self) -> dict[str, Any]: ...


class SupportsProfileRuntimePersona(Protocol):
    def effective_behavior_rules(self) -> list[str]: ...


class SupportsContextRuntime(Protocol):
    STYLE: dict[str, Any]
    session_summary: str | None
    profile_context: SupportsProfileContext
    profile_runtime: SupportsProfileRuntimePersona
    relationship_manager: SupportsRelationshipSnapshot
    long_term_signals: SupportsLongTermSignals

    def last_saved_mood(self) -> str: ...

    def memory_context_limit_for_input(self, user_input: str) -> int: ...

    def graph_retrieval_for_input(
        self,
        user_input: str,
        limit: int = 3,
    ) -> dict[str, Any] | None: ...

    def relevant_archive_entries_for_input(
        self,
        user_input: str,
        limit: int = 2,
    ) -> list[dict[str, Any]]: ...

    def relevant_memories_for_input(
        self,
        user_input: str,
        limit: int = 3,
        graph_result: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def build_active_consolidated_context(self, user_input: str) -> str | None: ...

    def persona_evolution_history(self) -> list[dict[str, Any]]: ...

    def active_persona_trait_entries(self, limit: int = 4) -> list[dict[str, Any]]: ...

    def life_patterns(self) -> list[dict[str, Any]]: ...

    def relationship_timeline(self) -> str | None: ...

    def session_archive(self) -> list[dict[str, Any]]: ...

    def build_consolidated_memory_context(self) -> str | None: ...

    def build_graph_summary_context(self, limit: int = 3) -> str | None: ...

    def natural_list(self, values: list[str]) -> str: ...


class SupportsToneRuntime(Protocol):
    SUPPORT_ESCALATION: dict[str, Any]
    _pending_daily_checkin_context: bool

    def normalize_mood(self, mood: str | None) -> str: ...


class SupportsRelationshipRuntime(Protocol):
    RELATIONSHIP_REFLECTION_INTERVAL: int
    session_summary: str | None
    last_relationship_reflection_turn: int
    memory_manager: SupportsRelationshipMemory
    profile_context: SupportsProfileContext

    def normalize_mood(self, mood: str | None) -> str: ...

    def tokenize(self, text: str | None) -> set[str]: ...

    def relationship_hypothesis_profiles(self) -> dict[str, dict[str, Any]]: ...

    def default_relationship_hypotheses(self) -> list[dict[str, Any]]: ...

    def natural_list(self, values: list[str]) -> str: ...

    def transcript_from_messages(self, messages: list[dict[str, Any]]) -> str: ...

    def session_turn_count(self) -> int: ...

    def prompt_history(self) -> list[dict[str, Any]]: ...

    def call_ollama_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    def extract_ollama_message_content(self, response: dict[str, Any]) -> str: ...

    def record_runtime_issue(
        self,
        stage: str,
        fallback: str,
        exc: Exception | None = None,
    ) -> None: ...

    def parse_model_json_content(self, content: str) -> dict[str, Any]: ...

    def clamp_score(self, score: int) -> int: ...

    def mutate_memory_store(self, **changes: Any) -> dict[str, Any]: ...

    def update_trait_impact_from_relationship_feedback(
        self,
        trust_delta: int,
        openness_delta: int,
    ) -> None: ...

    def infer_memory_category(self, user_input: str) -> str: ...


class SupportsTurnProcessingRuntime(Protocol):
    LIGHT_MODE: bool
    runtime_config: Any
    _session_lock: Any
    _pending_daily_checkin_context: bool
    _active_tool_observation_context: str | None
    history: list[dict[str, Any]]
    session_moods: list[str]

    def agentic_tool_settings(self) -> dict[str, Any]: ...

    def update_planner_debug(self, **fields: Any) -> dict[str, Any]: ...

    def get_available_tools(self) -> list[dict[str, Any]]: ...

    def call_ollama_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    async def call_ollama_chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    def parse_model_json_content(self, content: str) -> dict[str, Any]: ...

    def add_reminder(self, title: str, due_text: str = "") -> dict[str, Any] | None: ...

    @property
    def reply_finalization(self) -> Any: ...

    @property
    def mood_manager(self) -> Any: ...

    @property
    def memory(self) -> Any: ...

    @property
    def relationship(self) -> Any: ...

    def lookup_web(self, query: str) -> dict[str, Any] | None: ...

    def normalize_lookup_query(self, user_input: str) -> str: ...

    def normalize_chat_attachments(
        self,
        attachments: AttachmentList | None = None,
    ) -> AttachmentList: ...

    def enrich_multimodal_attachments(
        self,
        attachments: AttachmentList | None = None,
        user_input: str = "",
    ) -> AttachmentList: ...

    def compose_user_turn_text(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> str: ...

    def is_session_exit_command(self, stripped_input: str) -> bool: ...

    def persist_conversation(self) -> None: ...

    def mark_chat_thread_closed(self, closed: bool = True) -> dict[str, Any] | None: ...

    def prompt_history(self) -> list[dict[str, Any]]: ...

    def session_turn_count(self) -> int: ...

    def handle_memory_command(self, user_input: str) -> str | None: ...

    def handle_tool_command(self, user_input: str) -> str | None: ...

    def get_memory_reply(self, user_input: str) -> str | None: ...

    def get_fact_reply(self, user_input: str) -> str | None: ...

    def autonomous_tool_result_for_input(
        self,
        user_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
    ) -> tuple[str | None, str | None]: ...

    def set_active_tool_observation(self, observation: str | None) -> None: ...

    def history_attachment_metadata(
        self,
        attachment: dict[str, Any],
    ) -> dict[str, Any]: ...

    def sync_active_thread_snapshot(self) -> None: ...

    def schedule_post_turn_maintenance(
        self,
        user_input: str,
        current_mood: str,
    ) -> dict[str, Any]: ...

    def call_ollama_chat_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str: ...

    async def call_ollama_chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str: ...

    def build_chat_request_messages(
        self,
        user_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
    ) -> list[dict[str, object]]: ...

    def critique_reply(
        self,
        user_input: str,
        draft_reply: str,
        current_mood: str,
    ) -> str: ...

    async def critique_reply_async(
        self,
        user_input: str,
        draft_reply: str,
        current_mood: str,
    ) -> str: ...

    def validate_reply(self, user_input: str, candidate_reply: str) -> str: ...

    def begin_planner_debug(
        self,
        user_input: str,
        current_mood: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DadBotContext:
    """Live compatibility wrapper for extracted managers during the DadBot split."""

    bot: Any

    @classmethod
    def from_runtime(
        cls,
        runtime: Any | SupportsDadBotAccess | DadBotContext,
    ) -> DadBotContext:
        if isinstance(runtime, cls):
            return runtime
        if hasattr(runtime, "bot"):
            return cls(runtime.bot)
        return cls(runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.bot, name)

    @property
    def active_model(self) -> str:
        return str(self.bot.ACTIVE_MODEL)

    @property
    def active_embedding_model(self) -> str | None:
        value = getattr(self.bot, "ACTIVE_EMBEDDING_MODEL", None)
        if value is None:
            return None
        return str(value)

    @property
    def tenant_id(self) -> str:
        return str(self.bot.TENANT_ID)

    @property
    def profile_path(self) -> Path:
        return Path(self.bot.PROFILE_PATH)

    @property
    def memory_path(self) -> Path:
        return Path(self.bot.MEMORY_PATH)

    @property
    def semantic_memory_db_path(self) -> Path:
        return Path(self.bot.SEMANTIC_MEMORY_DB_PATH)

    @property
    def graph_store_db_path(self) -> Path:
        return Path(self.bot.GRAPH_STORE_DB_PATH)

    @property
    def session_log_dir(self) -> Path:
        return Path(self.bot.SESSION_LOG_DIR)


__all__ = [
    "Attachment",
    "AttachmentList",
    "ChunkCallback",
    "DadBotContext",
    "FinalizedTurnResult",
    "GenericSovereignPayload",
    "LogicBranchPayload",
    "PlannerDecisionPayload",
    "PreparedTurnResult",
    "PolicyVetoPayload",
    "SovereignEvent",
    "SovereignEventPayload",
    "SovereignEventType",
    "SupportsContextRuntime",
    "SupportsDadBotAccess",
    "SupportsLongTermSignals",
    "SupportsProfileContext",
    "SupportsProfileRuntimePersona",
    "SupportsRelationshipMemory",
    "SupportsRelationshipRuntime",
    "SupportsRelationshipSnapshot",
    "SupportsToneRuntime",
    "ToolExecutionPayload",
    "SupportsTurnProcessingRuntime",
]
