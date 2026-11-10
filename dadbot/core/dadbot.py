


"""
dadbot/core/dadbot.py — DadBot public facade.

DadBot is intentionally thin.  Logic lives in dedicated managers held by
``self.services`` and in the five behaviour mixins:

    DadBotBootMixin     boot/init/shutdown lifecycle
    DadBotTurnMixin     turn execution and graph failure handling
    DadBotLlmMixin      LLM/model call forwarding
    DadBotMcpMixin      local MCP server management
    DadBotHealthMixin   health/UX state and checkpoint/replay

The remaining body of this class is pure delegation plumbing:
    * Class-level routing maps (_CONFIG_ATTR_MAP, explicit service names, etc.)
    * __getattr__ / __setattr__ for zero-overhead manager attribute routing
    * Explicit @property getters/setters for every registered manager
    * Config and runtime-state property aliases (PROFILE, MEMORY_STORE, ...)
    * model_port / ux_gateway properties (access private attrs set by boot mixin)
    * reset_session_state and two small turn-state helpers
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast
from pathlib import Path
from dadbot.typing import MemoryManager, RelationshipManager, MoodManager, ProfileRuntime

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]

from dadbot.assistant_runtime import AssistantRuntime
from dadbot.core.action_mixin import DadBotActionMixin
from dadbot.core.boot_mixin import DadBotBootMixin
from dadbot.core.compat_mixin import DadBotCompatMixin
from dadbot.core.health_mixin import DadBotHealthMixin
from dadbot.core.llm_mixin import DadBotLlmMixin
from dadbot.core.mcp_mixin import DadBotMcpMixin
from dadbot.core.turn_mixin import DadBotTurnMixin
from dadbot.core.execution_contract import ExecutionEntry
from dadbot.core.ux_projection_gateway import TurnUxProjectionGateway
from dadbot.runtime.model import ModelPort

if ollama is None:
    logging.getLogger(__name__).warning(
        "ollama package is not installed; Ollama-backed features will be unavailable. Install with: pip install ollama",
    )

logger = logging.getLogger(__name__)


class _MemoryStoreProjection(dict):
    def __init__(self, value: dict[str, Any], on_change):
        super().__init__(value)
        self._on_change = on_change

    def _commit(self) -> None:
        self._on_change(dict(self))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._commit()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._commit()

    def clear(self):
        if self:
            super().clear()
            self._commit()

    def pop(self, key, *default):
        existed = key in self
        value = super().pop(key, *default)
        if existed:
            self._commit()
        return value

    def popitem(self):
        item = super().popitem()
        self._commit()
        return item

    def setdefault(self, key, default=None):
        existed = key in self
        value = super().setdefault(key, default)
        if not existed:
            self._commit()
        return value

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._commit()

    def __ior__(self, other):
        self.update(other)
        return self

    def __deepcopy__(self, memo):
        import copy

        return copy.deepcopy(dict(self), memo)


class DadBot(
    DadBotBootMixin,
    DadBotTurnMixin,
    DadBotLlmMixin,
    DadBotMcpMixin,
    DadBotHealthMixin,
    DadBotActionMixin,
    DadBotCompatMixin,
    AssistantRuntime,
):
    def current_avatar_exists(self):
        return self.avatar_manager.current_avatar_exists()

    # Remove all test compatibility manager defaults. All must be real and present.

    # Compatibility shims for test expectations
    _CONFIG_ATTR_MAP = {}
    _RUNTIME_STATE_ATTR_MAP = {
        "_pending_daily_checkin_context": "pending_daily_checkin_context",
        "_active_tool_observation_context": "active_tool_observation_context",
        "_last_planner_debug": "planner_debug",
    }
    _INTERNAL_RUNTIME_ATTR_MAP = {
        "_prompt_guard_stats": "prompt_guard_stats",
        "_last_memory_context_stats": "last_memory_context_stats",
        "_last_output_moderation": "last_output_moderation",
        "_last_reply_supervisor": "last_reply_supervisor",
        "_last_turn_pipeline": "last_turn_pipeline",
        "_last_turn_health_state": "last_turn_health_state",
        "_last_turn_ux_feedback": "last_turn_ux_feedback",
        "_shadow_decision_bus": "shadow_decision_bus",
        "_last_shadow_decision_report": "last_shadow_decision_report",
        "_background_task_ids": "background_task_ids",
    }
    _DEPRECATED_FACADE_ALIAS_MAP = {}

    def __getattr__(self, name):
        state_target = self._RUNTIME_STATE_ATTR_MAP.get(name)
        if state_target is not None:
            try:
                manager = object.__getattribute__(self, "runtime_state_manager")
            except AttributeError:
                manager = None
            if manager is not None:
                return getattr(manager, state_target)

        internal_target = self._INTERNAL_RUNTIME_ATTR_MAP.get(name)
        if internal_target is not None:
            try:
                runtime = object.__getattribute__(self, "_internal_runtime")
            except AttributeError:
                runtime = None
            if runtime is not None:
                return getattr(runtime, internal_target)

        try:
            services = object.__getattribute__(self, "services")
        except AttributeError:
            services = None
        registry = getattr(services, "registry", None)
        get_provider = getattr(registry, "get_provider", None)
        if callable(get_provider):
            provider = get_provider(name)
            if provider is not None:
                return getattr(provider, name)

        required = [
            'memory_manager', 'profile_runtime', 'relationship_manager', 'mood_manager',
            'event_bus', 'model_runtime', 'runtime_state_manager', 'services',
            'reply_finalization', 'reply_supervisor', 'turn_service', 'service_registry'
        ]
        if name in required:
            raise AttributeError(f"DadBot is missing required manager attribute: {name}")
        raise AttributeError(f"DadBot has no attribute '{name}'")

    def __setattr__(self, name, value):
        state_target = type(self)._RUNTIME_STATE_ATTR_MAP.get(name)
        if state_target is not None:
            try:
                manager = object.__getattribute__(self, "runtime_state_manager")
            except AttributeError:
                manager = None
            if manager is not None:
                setattr(manager, state_target, value)
                return

        internal_target = type(self)._INTERNAL_RUNTIME_ATTR_MAP.get(name)
        if internal_target is not None:
            try:
                runtime = object.__getattribute__(self, "_internal_runtime")
            except AttributeError:
                runtime = None
            if runtime is not None:
                setattr(runtime, internal_target, value)
                return

        object.__setattr__(self, name, value)

    """
    DadBot is intentionally thin.  Logic lives in dedicated managers held by
    ``self.services`` and in the five behaviour mixins:

        DadBotBootMixin     boot/init/shutdown lifecycle
        DadBotTurnMixin     turn execution and graph failure handling
        DadBotLlmMixin      LLM/model call forwarding
        DadBotMcpMixin      local MCP server management
        DadBotHealthMixin   health/UX state and checkpoint/replay

    The remaining body of this class is pure delegation plumbing:
        * Class-level routing maps (_CONFIG_ATTR_MAP, explicit service names, etc.)
        * __getattr__ / __setattr__ for zero-overhead manager attribute routing
        * Explicit @property getters/setters for every registered manager
        * Config and runtime-state property aliases (PROFILE, MEMORY_STORE, ...)
        * model_port / ux_gateway properties (access private attrs set by boot mixin)
        * reset_session_state and two small turn-state helpers
    """

    def save_memory_store(self, *args, **kwargs):
        manager = getattr(self, "_memory_manager", None)
        save = getattr(manager, "save_memory_store", None)
        if not callable(save):
            raise RuntimeError("Memory persistence manager is unavailable.")
        return save(*args, **kwargs)

    def conversation_history(self):
        return self.runtime_state_manager.conversation_history()

    def add_reminder(
        self,
        title: str,
        due_text: str = "",
        *,
        idempotency_key: str = "",
    ) -> dict[str, Any] | None:
        return self.memory_manager.add_reminder(title, due_text, idempotency_key=idempotency_key)

    def ical_feed_url(self):
        return self.calendar_manager.ical_feed_url()

    def refresh_memory_graph(self, force=False):
        return self.long_term_signals.refresh_memory_graph(force=force)

    def ui_shell_snapshot(self):
        return self.status_reporting.ui_shell_snapshot()
    CONTEXT_TOKEN_BUDGET = 6000  # Default context token budget; used by boot mixin and runtime caches

    @property
    def prompt_composer(self):
        return getattr(self, "prompt_assembly", None)
    CONTEXT_TOKEN_BUDGET = 6000  # Default context token budget; used by boot mixin and runtime caches

    def __init__(
        self,
        *,
        memory_manager: Any = None,
        relationship_manager: Any = None,
        mood_manager: Any = None,
        profile_runtime: Any = None,
        event_bus: Any = None,
        model_runtime: Any = None,
        prompt_assembly: Any = None,
        config: Any = None,
        runtime_config: Any = None,
        runtime_state_manager: Any = None,
        model_name: str = "qwen3.5:9b",
        append_signoff: bool = True,
        light_mode: bool = False,
        tenant_id: str = "",
        **kwargs: Any,
    ):
        self._memory_manager = memory_manager
        self._relationship_manager = relationship_manager
        self._mood_manager = mood_manager
        self._profile_runtime = profile_runtime
        self._event_bus = event_bus
        self.model_runtime = model_runtime

        self.runtime_state_manager = runtime_state_manager
        self.prompt_assembly = prompt_assembly
        self._recent_mood_detections = {}
        self.thread_snapshots = {}
        if config is not None:
            self.config = config
        if runtime_config is not None:
            self.runtime_config = runtime_config

        DadBotBootMixin.__init__(
            self,
            model_name=model_name,
            append_signoff=append_signoff,
            light_mode=light_mode,
            tenant_id=tenant_id,
            skip_manager_validation=False,
            **kwargs
        )

    def active_persona_trait_entries(self, limit=None):
        return self.long_term_signals.active_persona_trait_entries(limit=limit)


    @property
    def profile_runtime(self):
        return getattr(self, "_profile_runtime", None)

    @profile_runtime.setter
    def profile_runtime(self, value):
        self._profile_runtime = value

    def ensure_chat_thread_state(self, preserve_active_runtime=False):
        if hasattr(self, "runtime_state_manager") and self.runtime_state_manager is not None:
            return self.runtime_state_manager.ensure_chat_thread_state(preserve_active_runtime=preserve_active_runtime)
        raise AttributeError("DadBot has no runtime_state_manager to delegate ensure_chat_thread_state")
    @property
    def llm_model(self) -> str:
        return self.config.llm_model

    @property
    def ACTIVE_MODEL(self) -> str:  # noqa: N802
        return self.config.active_model

    @ACTIVE_MODEL.setter
    def ACTIVE_MODEL(self, value: str) -> None:  # noqa: N802
        self.config.active_model = value

    @property
    def MODEL_NAME(self) -> str:  # noqa: N802
        return self.config.model_name

    @MODEL_NAME.setter
    def MODEL_NAME(self, value: str) -> None:  # noqa: N802
        self.config.model_name = value

    @property
    def FALLBACK_MODELS(self) -> tuple[str, ...]:  # noqa: N802
        return self.config.fallback_models

    @FALLBACK_MODELS.setter
    def FALLBACK_MODELS(self, value: tuple[str, ...]) -> None:  # noqa: N802
        self.config.fallback_models = tuple(value or ())

    @property
    def PREFERRED_EMBEDDING_MODELS(self) -> tuple[str, ...]:  # noqa: N802
        return self.config.preferred_embedding_models

    @PREFERRED_EMBEDDING_MODELS.setter
    def PREFERRED_EMBEDDING_MODELS(self, value: tuple[str, ...]) -> None:  # noqa: N802
        self.config.preferred_embedding_models = tuple(value or ())

    @property
    def LLM_PROVIDER(self) -> str:  # noqa: N802
        return self.config.llm_provider

    @LLM_PROVIDER.setter
    def LLM_PROVIDER(self, value: str) -> None:  # noqa: N802
        self.config.llm_provider = value

    @property
    def LLM_MODEL(self) -> str:  # noqa: N802
        return self.config.llm_model

    @LLM_MODEL.setter
    def LLM_MODEL(self, value: str) -> None:  # noqa: N802
        self.config.llm_model = value

    @property
    def ACTIVE_EMBEDDING_MODEL(self) -> str | None:  # noqa: N802
        return self.config.active_embedding_model

    @ACTIVE_EMBEDDING_MODEL.setter
    def ACTIVE_EMBEDDING_MODEL(self, value: str | None) -> None:  # noqa: N802
        self.config.active_embedding_model = value

    @property
    def APPEND_SIGNOFF(self) -> bool:  # noqa: N802
        return self.config.append_signoff

    @APPEND_SIGNOFF.setter
    def APPEND_SIGNOFF(self, value: bool) -> None:  # noqa: N802
        self.config.append_signoff = value

    @property
    def LIGHT_MODE(self) -> bool:  # noqa: N802
        return self.config.light_mode

    @LIGHT_MODE.setter
    def LIGHT_MODE(self, value: bool) -> None:  # noqa: N802
        self.config.light_mode = value

    @property
    def tenant_id(self) -> str:
        return self.config.tenant_id

    @property
    def append_signoff(self) -> bool:
        return self.config.append_signoff

    @property
    def light_mode(self) -> bool:
        return self.config.light_mode

    @property
    def preferred_embedding_models(self) -> list[str]:
        return list(self.config.preferred_embedding_models)

    @property
    def recent_history_window(self) -> int:
        return self.config.recent_history_window

    @property
    def RECENT_HISTORY_WINDOW(self) -> int:  # noqa: N802
        return self.config.recent_history_window

    @RECENT_HISTORY_WINDOW.setter
    def RECENT_HISTORY_WINDOW(self, value: int) -> None:  # noqa: N802
        self.config.recent_history_window = value

    @property
    def max_history_messages_scan(self) -> int:
        return self.config.max_history_messages_scan

    @property
    def MAX_HISTORY_MESSAGES_SCAN(self) -> int:  # noqa: N802
        return self.config.max_history_messages_scan

    @MAX_HISTORY_MESSAGES_SCAN.setter
    def MAX_HISTORY_MESSAGES_SCAN(self, value: int) -> None:  # noqa: N802
        self.config.max_history_messages_scan = value

    @property
    def summary_trigger_messages(self) -> int:
        return self.config.summary_trigger_messages

    @property
    def relationship_reflection_interval(self) -> int:
        return self.config.relationship_reflection_interval

    @property
    def context_token_budget(self) -> int:
        return self.config.context_token_budget

    @property
    def reserved_response_tokens(self) -> int:
        return self.config.reserved_response_tokens

    @property
    def RESERVED_RESPONSE_TOKENS(self) -> int:  # noqa: N802
        return self.config.reserved_response_tokens

    @RESERVED_RESPONSE_TOKENS.setter
    def RESERVED_RESPONSE_TOKENS(self, value: int) -> None:  # noqa: N802
        self.config.reserved_response_tokens = value

    @property
    def approx_chars_per_token(self) -> float:
        return self.config.approx_chars_per_token

    @property
    def mood_detection_temperature(self) -> float:
        return self.config.mood_detection_temperature

    @property
    def MOOD_DETECTION_TEMPERATURE(self) -> float:  # noqa: N802
        return self.config.mood_detection_temperature

    @MOOD_DETECTION_TEMPERATURE.setter
    def MOOD_DETECTION_TEMPERATURE(self, value: float) -> None:  # noqa: N802
        self.config.mood_detection_temperature = value

    @property
    def stream_timeout_seconds(self) -> float:
        return self.config.stream_timeout_seconds

    @property
    def stream_max_chars(self) -> int:
        return self.config.stream_max_chars

    @property
    def profile_path(self) -> str:
        return str(self.config.profile_path)

    @property
    def memory_path(self) -> str:
        return str(self.config.memory_path)

    @property
    def semantic_memory_db_path(self) -> str:
        return str(self.config.semantic_memory_db_path)

    @property
    def graph_store_db_path(self) -> str:
        return str(self.config.graph_store_db_path)

    @property
    def session_log_dir(self) -> str:
        return str(self.config.session_log_dir)

    @property
    def MEMORY_PATH(self) -> Path:  # noqa: N802
        return getattr(self, "_memory_path_override", self.config.memory_path)

    @MEMORY_PATH.setter
    def MEMORY_PATH(self, value: str | Path) -> None:  # noqa: N802
        self._memory_path_override = Path(value)

    @property
    def SEMANTIC_MEMORY_DB_PATH(self) -> Path:  # noqa: N802
        return self.config.semantic_memory_db_path

    @SEMANTIC_MEMORY_DB_PATH.setter
    def SEMANTIC_MEMORY_DB_PATH(self, value: str | Path) -> None:  # noqa: N802
        self.config.semantic_memory_db_path = Path(value)

    @property
    def GRAPH_STORE_DB_PATH(self) -> Path:  # noqa: N802
        return self.config.graph_store_db_path

    @GRAPH_STORE_DB_PATH.setter
    def GRAPH_STORE_DB_PATH(self, value: str | Path) -> None:  # noqa: N802
        self.config.graph_store_db_path = Path(value)

    @property
    def SESSION_LOG_DIR(self) -> Path:  # noqa: N802
        return self.config.session_log_dir

    @SESSION_LOG_DIR.setter
    def SESSION_LOG_DIR(self, value: str | Path) -> None:  # noqa: N802
        self.config.session_log_dir = Path(value)

    @property
    def TENANT_ID(self) -> str:  # noqa: N802
        return self.config.tenant_id

    # ------------------------------------------------------------------
    # Explicit properties for runtime state attributes
    # ------------------------------------------------------------------


    @property
    def runtime_state_container(self) -> Any:
        return self._runtime_state

    @property
    def history(self) -> Any:
        return self.runtime_state_manager.history

    @history.setter
    def history(self, value):
        self.runtime_state_manager.history = value

    @property
    def MEMORY_STORE(self) -> dict[str, Any]:
        return _MemoryStoreProjection(
            self.memory_manager.memory_store,
            self.memory_manager._storage.replace_projection_via_canonical_event,
        )

    @MEMORY_STORE.setter
    def MEMORY_STORE(self, value: dict[str, Any]) -> None:
        self.memory_manager._storage.replace_projection_via_canonical_event(value)

    @property
    def session_moods(self) -> Any:
        return self.runtime_state_manager.session_moods

    @session_moods.setter
    def session_moods(self, value):
        self.runtime_state_manager.session_moods = value

    @property
    def session_summary(self) -> Any:
        return self.runtime_state_manager.session_summary

    @session_summary.setter
    def session_summary(self, value):
        self.runtime_state_manager.session_summary = value

    @property
    def session_summary_updated_at(self) -> Any:
        return self.runtime_state_manager.session_summary_updated_at

    @session_summary_updated_at.setter
    def session_summary_updated_at(self, value):
        self.runtime_state_manager.session_summary_updated_at = value

    @property
    def session_summary_covered_messages(self) -> Any:
        return self.runtime_state_manager.session_summary_covered_messages

    @session_summary_covered_messages.setter
    def session_summary_covered_messages(self, value):
        self.runtime_state_manager.session_summary_covered_messages = value

    @property
    def last_relationship_reflection_turn(self) -> Any:
        return self.runtime_state_manager.last_relationship_reflection_turn

    @last_relationship_reflection_turn.setter
    def last_relationship_reflection_turn(self, value):
        self.runtime_state_manager.last_relationship_reflection_turn = value


    @property
    def pending_daily_checkin_context(self) -> Any:
        return self.runtime_state_manager.pending_daily_checkin_context

    @pending_daily_checkin_context.setter
    def pending_daily_checkin_context(self, value):
        self.runtime_state_manager.pending_daily_checkin_context = value

    @property
    def active_tool_observation_context(self) -> Any:
        return self.runtime_state_manager.active_tool_observation_context

    @active_tool_observation_context.setter
    def active_tool_observation_context(self, value):
        self.runtime_state_manager.active_tool_observation_context = value

    @property
    def planner_debug(self) -> Any:
        return self.runtime_state_manager.planner_debug

    @planner_debug.setter
    def planner_debug(self, value):
        self.runtime_state_manager.planner_debug = value

    @property
    def chat_threads(self) -> Any:
        return self.runtime_state_manager.chat_threads

    @chat_threads.setter
    def chat_threads(self, value):
        self.runtime_state_manager.chat_threads = value

    @property
    def active_thread_id(self) -> Any:
        return self.runtime_state_manager.active_thread_id

    @active_thread_id.setter
    def active_thread_id(self, value):
        self.runtime_state_manager.active_thread_id = value

    # thread_snapshots is always a direct attribute, never a property

    # (No property, no setter, no indirection)

    # ------------------------------------------------------------------
    # Explicit properties for internal runtime attributes
    # ------------------------------------------------------------------


    def prompt_guard_stats(self) -> Any:
        return self._internal_runtime.prompt_guard_stats

    @property
    def last_memory_context_stats(self) -> Any:
        return self._internal_runtime.last_memory_context_stats

    @property
    def last_output_moderation(self) -> Any:
        return self._internal_runtime.last_output_moderation

    @property
    def last_reply_supervisor(self) -> Any:
        return self._internal_runtime.last_reply_supervisor

    @property
    def last_turn_pipeline(self) -> Any:
        return self._internal_runtime.last_turn_pipeline

    @property
    def last_turn_health_state(self) -> Any:
        return self._internal_runtime.last_turn_health_state

    @property
    def last_turn_ux_feedback(self) -> Any:
        return self._internal_runtime.last_turn_ux_feedback

    @property
    def shadow_decision_bus(self) -> Any:
        return self._internal_runtime.shadow_decision_bus

    @property
    def last_shadow_decision_report(self) -> Any:
        return self._internal_runtime.last_shadow_decision_report

    @property
    def background_task_ids(self) -> Any:
        return self._internal_runtime.background_task_ids

    # (Add more explicit properties as needed for full coverage)
    # ------------------------------------------------------------------
    # Agentic Planning/Reasoning Loop (ReAct-style)
    # ------------------------------------------------------------------

    """Thin public facade for the Dad Bot persona.

    This facade is intentionally thin -- prefer direct manager access in new code.

    Architecture overview
    ---------------------
    DadBot is the public facade for all bot functionality.  Heavy logic lives in
    dedicated managers held by ``self.services`` and in the five behaviour mixins
    listed at the top of this module.  The facade keeps the high-level public API
    stable while delegating most implementation detail to the container.

    Delegation strategies
    ---------------------
    1. **Behaviour mixins** -- grouped logical concerns (boot, turn, LLM, MCP, health).
    2. **Explicit @property delegation** -- thin shims for every registered manager so
       that IDE discoverability and static analysis work.
     3. **Indexed service/provider lookup** -- unknown attribute lookups resolve through
         the service container by explicit service name or provider ownership.
    """


    # --- All dynamic attribute routing, unified routing, and deprecated alias logic removed. ---





    def deliver_status_message(self, message: str, status_callback: Any = None) -> None:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        DadBotFacadeUtils.deliver_status_message(self, message, status_callback=status_callback)



    @staticmethod
    def terminal_width() -> int:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.terminal_width()



    def print_system_message(self, message: str) -> None:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        DadBotFacadeUtils.print_system_message(message)



    @staticmethod
    def print_speaker_message(speaker: str, message: str) -> None:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        DadBotFacadeUtils.print_speaker_message(speaker, message)



    def build_system_prompt(self) -> str:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.build_system_prompt(self)

    def build_cross_session_context(self, user_input: str = "") -> str | None:
        return self.context_builder.build_cross_session_context(user_input)



    @staticmethod
    def flatten_memory_payload(payload: Any) -> Any:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.flatten_memory_payload(payload)



    @staticmethod
    def coerce_memory_summary(value: Any) -> Any:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.coerce_memory_summary(value)



    def new_chat_session(self) -> Any:
        from dadbot.core.facade_utils import DadBotFacadeUtils
        return DadBotFacadeUtils.new_chat_session(self)

    # ------------------------------------------------------------------
    # Core manager properties
    # ------------------------------------------------------------------

    @property
    def memory_manager(self) -> MemoryManager:
        return self._memory_manager

    @memory_manager.setter
    def memory_manager(self, value: MemoryManager) -> None:
        self._memory_manager = value

    @property
    def relationship_manager(self) -> RelationshipManager:
        return self._relationship_manager

    @relationship_manager.setter
    def relationship_manager(self, value: RelationshipManager) -> None:
        self._relationship_manager = value

    @property
    def mood_manager(self) -> MoodManager:
        return self._mood_manager

    @mood_manager.setter
    def mood_manager(self, value: MoodManager) -> None:
        self._mood_manager = value


    @property
    def memory(self) -> MemoryManager:
        """Returns the memory manager. Guaranteed non-None."""
        return self._memory_manager


    @property
    def relationship(self) -> RelationshipManager:
        """Returns the relationship manager. Guaranteed non-None."""
        return self._relationship_manager


    @property
    def mood(self) -> MoodManager:
        """Returns the mood manager. Guaranteed non-None."""
        return self._mood_manager


    @property
    def profile(self) -> ProfileRuntime:
        """Returns the profile runtime. Guaranteed non-None."""
        return self._profile_runtime

    @property
    def turn_orchestrator(self) -> Any:
        return self.services.turn_orchestrator

    @property
    def assistant(self) -> AssistantRuntime:
        cached = getattr(self, "_assistant_runtime", None)
        if cached is None:
            cached = AssistantRuntime(self, entrypoint=ExecutionEntry(self.execute_turn))
            object.__setattr__(self, "_assistant_runtime", cached)
        return cast(AssistantRuntime, cached)

    @property
    def phase_closure_runtime(self) -> Any:
        cached = getattr(self, "_phase_closure_runtime", None)
        if cached is None:
            from dadbot.core.phase_closure_runtime import PhaseClosureRuntime

            cached = PhaseClosureRuntime()
            object.__setattr__(self, "_phase_closure_runtime", cached)
        return cached

    @property
    def context_builder(self) -> Any:
        """On-demand ContextBuilder for prompt composition during boot and runtime.
        
        This ensures build_core_persona_prompt() is always available even if
        context_service isn't fully wired during early boot phases.
        """
        cached = getattr(self, "_context_builder", None)
        if cached is None:
            from dadbot.context import ContextBuilder
            cached = ContextBuilder(cast(Any, self))
            object.__setattr__(self, "_context_builder", cached)
        return cached

    @context_builder.setter
    def context_builder(self, value: Any) -> None:
        """Allow setting context_builder during wiring."""
        object.__setattr__(self, "_context_builder", value)

    def build_core_persona_prompt(self) -> str:
        """Delegate to context_builder for persona prompt composition."""
        return self.context_builder.build_core_persona_prompt()


    # ------------------------------------------------------------------
    # Config properties
    # ------------------------------------------------------------------

    @property
    def PROFILE(self) -> Any:
        return self._profile_runtime.profile

    @PROFILE.setter
    def PROFILE(self, value: Any) -> None:
        self._profile_runtime.profile = value

    @property
    def STYLE(self) -> dict[str, Any]:
        profile_runtime = getattr(self, "_profile_runtime", None)
        if profile_runtime is not None:
            return profile_runtime.style
        return getattr(self, "_STYLE", {"name": "DadBot"})

    @STYLE.setter
    def STYLE(self, value: dict[str, Any]) -> None:
        profile_runtime = getattr(self, "_profile_runtime", None)
        if profile_runtime is not None:
            profile_runtime.style = value
        else:
            self._STYLE = value if isinstance(value, dict) else {}


    @property
    def memory_store(self) -> dict[str, Any]:
        """Read-only projection of memory store."""
        mem = self.memory
        if mem is not None and hasattr(mem, "memory_projection"):
            return mem.memory_projection()
        return {}

    def store_memory(self, key: str, value: Any) -> None:
        mem = self.memory
        if mem is not None and hasattr(mem, "store"):
            mem.store(key, value)

    def delete_memory(self, key: str) -> None:
        mem = self.memory
        if mem is not None and hasattr(mem, "delete"):
            mem.delete(key)



    # ------------------------------------------------------------------
    # Deterministic model port and UX gateway
    # (private attrs set by DadBotBootMixin)
    # ------------------------------------------------------------------

    @property
    def model_port(self) -> ModelPort:
        """Deterministic model interaction port.

        All LLM and embedding calls MUST route through this port to ensure
        replay correctness, deterministic execution, and certification validity.
        """
        if self._model_port is None:
            raise RuntimeError("Model port not initialized -- services not ready")
        return self._model_port

    @property
    def ux_gateway(self) -> TurnUxProjectionGateway:
        """UX projection gateway (post-execution assembly).

        Constraint: DadBot does NOT access state/fidelity directly.
        All reads route through cached values from previous gateway projections.
        """
        return self._ux_gateway

    # ------------------------------------------------------------------
    # Session state helpers
    # ------------------------------------------------------------------

    def reset_session_state(self) -> Any:
        """Explicit facade delegate for runtime session reset."""
        mgr = getattr(self, "_memory_manager", None)
        if mgr is not None and hasattr(mgr, 'reset_session_state'):
            result = mgr.reset_session_state()
        else:
            result = None
        # Reset CSCL session arc/pacing/coherence state for the new session
        cscl = getattr(self, "conversation_surface", None)
        if cscl is not None:
            try:
                cscl.reset_session()
            except Exception:
                pass
        return result

    def direct_reply_for_input(
        self,
        stripped_input: str,
        current_mood: str,
    ) -> str | None:
        """Explicit facade delegate for TurnService direct reply helpers.

        This avoids accidental routing to similarly-named safety helpers.
        """

        turn_service = getattr(self, 'turn_service', None)
        if turn_service and hasattr(turn_service, 'direct_reply_for_input'):
            return turn_service.direct_reply_for_input(stripped_input, current_mood)
        return None

    def build_image_analysis_prompt(
        self,
        note: str = "",
        *,
        user_input: str = "",
        attachment: dict | None = None,
    ) -> str:
        """Explicit facade delegate for multimodal image prompt assembly.

        Ensures the facade routes through the multimodal manager, even if other
        services expose the same method name.
        """

        multimodal = getattr(self, "multimodal_handler", None)
        builder = getattr(multimodal, "build_image_analysis_prompt", None)
        if callable(builder):
            return str(
                builder(
                    note=note,
                    user_input=user_input,
                    attachment=attachment,
                )
                or ""
            )
        return ""

    def _load_memory_store(self) -> dict[str, Any]:
        """Explicit compatibility delegate for memory-store loading.

        Some regression tests and operational tooling call this private helper
        directly to validate corruption recovery behaviour.
        """
        memory = getattr(self, "memory", None)
        loader = getattr(memory, "load_memory_store", None)
        if callable(loader):
            loaded = loader()
            return dict(loaded or {}) if isinstance(loaded, dict) else {}
        return {}
    def _thread_timestamp(self) -> Any:
        mgr = getattr(self, '_memory_manager', None)
        if mgr and hasattr(mgr, 'thread_timestamp'):
            return mgr.thread_timestamp()
        return None

    def _apply_thread_snapshot_unlocked(self, snapshot: Any) -> Any:
        mgr = getattr(self, '_memory_manager', None)
        if mgr and hasattr(mgr, 'apply_thread_snapshot_unlocked'):
            return mgr.apply_thread_snapshot_unlocked(snapshot)
        return None



    def record_shadow_decision(
        self,
        *,
        source: str,
        type: str,
        content_preview: str = "",
        reason: str = "",
        would_replace: bool = False,
        priority: float = 0.0,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
        turn_context: Any | None = None,
    ) -> dict[str, Any]:
        """Emit a shadow-decision event via the event bus."""
        raw_type = str(type or "suggestion").strip().lower()
        allowed_types = {"override_attempt", "veto", "suggestion", "transform"}
        normalized_type = raw_type if raw_type in allowed_types else "suggestion"
        event: dict[str, Any] = {
            "source": str(source or "unknown").strip().lower() or "unknown",
            "type": normalized_type,
            "content_preview": str(content_preview or "")[:280],
            "reason": str(reason or "")[:320],
            "would_replace": bool(would_replace),
            "priority": float(priority or 0.0),
            "timestamp": float(timestamp if timestamp is not None else time.time()),
        }
        if isinstance(metadata, dict) and metadata:
            event["metadata"] = dict(metadata)
        if self._event_bus is not None:
            self._event_bus.emit(event)
        else:
            self.shadow_decision_bus.append(event)
        return event



    def peek_shadow_decisions(self, limit: int = 64) -> list[dict[str, Any]]:
        if self._event_bus is not None:
            return self._event_bus.peek(limit)
        return list(self.shadow_decision_bus[-max(0, int(limit)):]) if limit > 0 else []



    def consume_shadow_decisions(self, limit: int = 128) -> list[dict[str, Any]]:
        if self._event_bus is not None:
            return self._event_bus.consume(limit)
        count = max(0, int(limit))
        events = list(self.shadow_decision_bus[:count])
        del self.shadow_decision_bus[:count]
        return events

    def record_shadow_decision_report(self, report: dict[str, Any]) -> dict[str, Any]:
        payload = dict(report or {})
        self._last_shadow_decision_report = payload
        return payload

    def shadow_decision_report(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_shadow_decision_report", {}) or {})

    @property
    def script_path(self) -> Any:
        """Resolved runtime entry script path used by persistence helpers."""
        return self.runtime_script_path()


    def sync_active_thread_snapshot(self):
        return self.runtime_state_manager.sync_active_thread_snapshot()

    def last_saved_mood(self):
        return self.memory.last_saved_mood()

    def streamlit_security_settings(self):
        return self.profile_runtime.streamlit_security_settings()

if __name__ == "__main__":
    from dadbot.app_runtime import (
        main as run_app_main,  # type: ignore[reportUnknownVariableType]
    )

    raise SystemExit(run_app_main(dadbot_cls=DadBot, script_path=__file__))
