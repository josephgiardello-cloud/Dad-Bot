from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeAlias

Attachment: TypeAlias = dict[str, Any]
AttachmentList: TypeAlias = list[Attachment]
ChunkCallback: TypeAlias = Callable[[str], Any]
PreparedTurnResult: TypeAlias = tuple[str | None, str | None, bool, str, AttachmentList]
FinalizedTurnResult: TypeAlias = tuple[str | None, bool]


class SupportsDadBotAccess(Protocol):
    bot: Any


class SupportsProfileContext(Protocol):
    def build_profile_block(self, fact_ids: list[str] | None = None) -> str: ...

    def relevant_fact_ids_for_input(self, user_input: str) -> list[str]: ...

    def matching_topics(self, user_input: str) -> list[dict[str, Any]]: ...


class SupportsLongTermSignals(Protocol):
    def build_wisdom_context(self, user_input: str) -> str | None: ...

    def build_deep_pattern_context(self, user_input: str, limit: int | None = None) -> str | None: ...

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

    def graph_retrieval_for_input(self, user_input: str, limit: int = 3) -> dict[str, Any] | None: ...

    def relevant_archive_entries_for_input(self, user_input: str, limit: int = 2) -> list[dict[str, Any]]: ...

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

    def call_ollama_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...

    def extract_ollama_message_content(self, response: dict[str, Any]) -> str: ...

    def record_runtime_issue(self, stage: str, fallback: str, exc: Exception | None = None) -> None: ...

    def parse_model_json_content(self, content: str) -> dict[str, Any]: ...

    def clamp_score(self, score: int) -> int: ...

    def mutate_memory_store(self, **changes: Any) -> dict[str, Any]: ...

    def update_trait_impact_from_relationship_feedback(self, trust_delta: int, openness_delta: int) -> None: ...

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

    def call_ollama_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...

    async def call_ollama_chat_async(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...

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

    def normalize_chat_attachments(self, attachments: AttachmentList | None = None) -> AttachmentList: ...

    def enrich_multimodal_attachments(self, attachments: AttachmentList | None = None, user_input: str = "") -> AttachmentList: ...

    def compose_user_turn_text(self, user_input: str, attachments: AttachmentList | None = None) -> str: ...

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

    def history_attachment_metadata(self, attachment: dict[str, Any]) -> dict[str, Any]: ...

    def sync_active_thread_snapshot(self) -> None: ...

    def schedule_post_turn_maintenance(self, user_input: str, current_mood: str) -> dict[str, Any]: ...

    def call_ollama_chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...

    async def call_ollama_chat_stream_async(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...

    def build_chat_request_messages(
        self,
        user_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
    ) -> list[dict[str, object]]: ...

    def critique_reply(self, user_input: str, draft_reply: str, current_mood: str) -> str: ...

    async def critique_reply_async(self, user_input: str, draft_reply: str, current_mood: str) -> str: ...

    def validate_reply(self, user_input: str, candidate_reply: str) -> str: ...

    def begin_planner_debug(self, user_input: str, current_mood: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# L3/L4/L5 substrate protocols
# ---------------------------------------------------------------------------


class SupportsEventAuthority(Protocol):
    """Protocol for EventAuthority: canonical source-of-truth event layer."""
    def append(self, event: dict[str, Any]) -> int: ...
    def derive_state(self) -> dict[str, Any]: ...
    def is_defined(self) -> bool: ...
    def assert_defined(self) -> None: ...
    def rebuild_state_from_events(self, events: list[dict[str, Any]]) -> dict[str, Any]: ...
    def authority_hash(self) -> str: ...


class SupportsStatelessExecution(Protocol):
    """Protocol for StatelessExecutor: pure-function execution mode."""
    def execute(
        self,
        user_input: str,
        event_log: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> Any: ...
    def is_bootstrapped(self, event_log: list[dict[str, Any]]) -> bool: ...


class SupportsMemoryStateSpace(Protocol):
    """Protocol for MemoryStateVector: ordered state vector space."""
    entries: tuple[dict[str, Any], ...]
    space_hash: str

    @classmethod
    def from_memories(cls, memories: list[dict[str, Any]]) -> Any: ...
    def project(self, indices: list[int]) -> Any: ...
    def to_list(self) -> list[dict[str, Any]]: ...


class SupportsEvolutionHooks(Protocol):
    """Protocol for GraphIntrospectionAPI: read-only introspection + hooks."""
    def introspect(self, dag: Any) -> dict[str, Any]: ...
    def hook_pre_mutation(self, callback: Any) -> None: ...
    def hook_post_mutation(self, callback: Any) -> None: ...


@dataclass(frozen=True)
class DadBotContext:
    """Live compatibility wrapper for extracted managers during the DadBot split."""

    bot: Any

    @classmethod
    def from_runtime(cls, runtime: Any | SupportsDadBotAccess | "DadBotContext") -> "DadBotContext":
        if isinstance(runtime, cls):
            return runtime
        if hasattr(runtime, "bot"):
            return cls(getattr(runtime, "bot"))
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
    "PreparedTurnResult",
    "SupportsContextRuntime",
    "SupportsDadBotAccess",
    "SupportsLongTermSignals",
    "SupportsProfileContext",
    "SupportsProfileRuntimePersona",
    "SupportsRelationshipMemory",
    "SupportsRelationshipRuntime",
    "SupportsRelationshipSnapshot",
    "SupportsToneRuntime",
    "SupportsTurnProcessingRuntime",
]
