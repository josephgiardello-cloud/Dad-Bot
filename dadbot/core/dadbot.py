import atexit
import asyncio
import importlib
from importlib import util as import_util  # Fixes the find_spec error
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]
from dadbot.app_runtime import build_customer_document_store, ensure_streamlit_app_file, main as run_app_main
from dadbot.core.action_mixin import DadBotActionMixin
from dadbot.core.facade_compat import DadBotFacadeCompat
from dadbot.core.compat_mixin import DadBotCompatMixin
from dadbot.core.convenience_mixin import DadBotConvenienceMixin
from dadbot.core.internal_runtime import DadBotInternalRuntime
from dadbot.core.observability import CorrelationContext, TracingContext
from dadbot.core.orchestrator import DadBotOrchestrator
from dadbot.registry import DadBotServiceContainer
from dadbot.background import BackgroundTaskManager
from dadbot.config import DadBotConfig, DadRuntimeConfig as PackagedDadRuntimeConfig
from dadbot.contracts import AttachmentList, ChunkCallback, DadBotContext, FinalizedTurnResult
from dadbot.constants import MOOD_CATEGORIES, MOOD_ALIASES
from dadbot.defaults import (
    default_planner_debug_state as _default_planner_debug_state,
)
from dadbot.utils import (
    env_truthy,
    json_dump,
    json_load,
)
from dadbot_system import (
    DadServiceClient,
    InMemoryEventBus,
    InMemoryStateStore,
    NamespacedStateStore,
    ServiceConfig,
    normalize_tenant_id,
)
from dadbot_system.state import AppStateContainer

tiktoken = importlib.import_module("tiktoken") if import_util.find_spec("tiktoken") else None
litellm = importlib.import_module("litellm") if import_util.find_spec("litellm") else None

if ollama is None:
    logging.getLogger(__name__).warning(
        "ollama package is not installed; Ollama-backed features will be unavailable. "
        "Install with: pip install ollama"
    )
if tiktoken is None:
    logging.getLogger(__name__).warning(
        "tiktoken is not installed; token counting will use the character-based estimate, "
        "which may cause context-budget drift. Install with: pip install tiktoken"
    )
if litellm is None:
    logging.getLogger(__name__).warning(
        "litellm is not installed; multi-provider LLM routing is disabled and Dad Bot will use Ollama. "
        "Install with: pip install litellm"
    )


logger = logging.getLogger(__name__)


class DadBot(DadBotCompatMixin, DadBotConvenienceMixin, DadBotActionMixin):
    """Thin public facade for the Dad Bot persona.

    This facade is intentionally thin. Prefer direct manager access where possible.

    Architecture overview
    ---------------------
    DadBot is the public facade for all bot functionality. Heavy logic lives in
    dedicated managers held by ``self.services`` and runtime configuration lives
    in ``self.config``. The facade keeps the high-level public API stable while
    delegating most implementation detail to the container.

    1. **Explicit delegation** - thin shim methods that call a specific manager.
        Used when the method signature differs from the manager, when IDE
        discoverability matters, or when extra logic must run around the call.

    2. **Auto-delegation via ``__getattr__``** - unknown attribute lookups are
        forwarded in priority order through ``_MANAGER_DELEGATE_CHAIN``.  The five
        fully-extracted managers (RuntimeHealthManager, TTSManager, AvatarManager,
        CalendarManager, EmailManager) are handled this way; explicit pass-through
        shims for those managers have been removed.

    Ongoing extraction
    ------------------
    MemoryCoordinator, RelationshipManager, and LongTermSignalsManager are partially extracted.
    Turn orchestration is now handled by TurnService in dadbot.services. Their heavy logic lives in
    their manager files but DadBot retains explicit delegation shims for backward
    compatibility while the migration is in progress.

    ``DadBotOrchestrator`` at the bottom of this module wires a graph-based turn
    pipeline (``TurnGraph``) that now drives production turn handling; streaming
    flows still reuse TurnService for chunked reply generation.
    """
    @staticmethod
    def runtime_root_path():
        return Path(__file__).resolve().parents[2]

    @classmethod
    def runtime_script_path(cls):
        return cls.runtime_root_path() / "Dad.py"

    @staticmethod
    def profile_template_path():
        return DadBot.runtime_root_path() / "dad_profile.template.json"

    @staticmethod
    def env_path(name, fallback_path):
        configured = str(os.environ.get(name) or "").strip()
        if configured:
            return Path(configured)
        return Path(fallback_path)

    @staticmethod
    def default_profile():
        template_path = DadBot.profile_template_path()
        with template_path.open("r", encoding="utf-8") as profile_template_file:
            return json_load(profile_template_file)

    @classmethod
    def initialize_profile_file(cls, profile_path=None, force=False):
        destination = Path(profile_path) if profile_path is not None else cls.runtime_root_path() / "dad_profile.json"
        if destination.exists() and not force:
            return False

        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("w", encoding="utf-8") as profile_file:
            json_dump(cls.default_profile(), profile_file, indent=2)
        return True

    # -- Auto-delegation registry ------------------------------------------------
    # Python calls __getattr__ only when normal attribute lookup fails, so any
    # explicit method on DadBot always takes precedence over this chain.
    # Managers are searched in priority order; first match wins.
    _LEGACY_MANAGER_DELEGATE_CHAIN: tuple[str, ...] = (
        "health_manager",    # RuntimeHealthManager  -- health metrics, HW optimisation
        "tts_manager",       # TTSManager            -- Piper / pyttsx3 audio synthesis
        "avatar_manager",    # AvatarManager         -- avatar image generation
        "calendar_manager",  # CalendarManager       -- iCal feed sync
        "email_manager",     # EmailManager          -- draft + SMTP delivery
        "runtime_storage",
        "profile_runtime",   # opening_message, persona management
        "memory_manager",    # embed_texts, embed_text, semantic index + embeddings cache
        "memory_query",
        "memory_coordinator",  # build_consolidated_memory_context, build_active_consolidated_context
        "long_term_signals",
        "relationship_manager",
        "runtime_state_manager",
        "status_reporting",
    )

    _SERVICE_MANAGER_DELEGATE_CHAIN: tuple[str, ...] = (
        "maintenance_scheduler",   # run_post_turn_maintenance, run_scheduled_proactive_jobs, etc.
        "conversation_persistence",# persist_conversation, save_session_log, etc.
        "session_summary_manager", # build_session_summary_prompt, refresh_session_summary
        "turn_service",            # should_offer_daily_checkin_for_turn, prepare_user_turn, etc.
        "memory_commands",         # handle_memory_command, build_memory_transcript
        "safety_support",          # detect_crisis_signal, moderate_output_reply, etc.
        "profile_context",         # age_on_date, template_context, validate_reply, etc.
        "reply_supervisor",        # run_reply_supervisor, judge_reply_alignment, etc.
        "reply_finalization",      # should_calibrate_pushback, apply_calibrated_pushback
        "multimodal_handler",      # normalize_chat_attachments, build_user_request_message, etc.
        "model_runtime",           # model_context_length, embedding_model_candidates, etc.
        "agentic_handler",         # add_reminder, lookup_web, add_calendar_event, delete_calendar_event
        "tool_registry",           # parse_tool_command, get_available_tools, etc.
        "context_builder",         # build_core_persona_prompt, build_memory_context, etc.
        "tone_context",            # build_mood_context, build_escalation_context, etc.
        "prompt_assembly",         # guard_chat_request_messages, build_chat_request_messages, etc.
        "runtime_client",          # ollama_async_client, available_model_names, ensure_ollama_ready
        "runtime_orchestration",   # background_task_snapshot, submit_background_task
        "runtime_interface",       # chat_loop, chat_loop_via_service
        "mood_manager",            # get_cached_mood_detection, remember_mood_detection
    )

    _MANAGER_DELEGATE_CHAIN: tuple[str, ...] = (
        _LEGACY_MANAGER_DELEGATE_CHAIN + _SERVICE_MANAGER_DELEGATE_CHAIN
    )

    # Maps backward-compatible facade attribute names to their canonical DadBotConfig
    # field names.  __getattr__ reads and __setattr__ writes are routed through this
    # map so callers can continue using bot.MODEL_NAME, bot.LLM_PROVIDER, etc. while
    # the single source of truth remains bot.config.<field>.
    # Canonical new code should use bot.config.<field> directly.
    _CONFIG_ATTR_MAP: dict[str, str] = {
        "MODEL_NAME": "model_name",
        "FALLBACK_MODELS": "fallback_models",
        "ACTIVE_MODEL": "active_model",
        "active_model": "active_model",
        "ACTIVE_EMBEDDING_MODEL": "active_embedding_model",
        "LLM_PROVIDER": "llm_provider",
        "LLM_MODEL": "llm_model",
        "TENANT_ID": "tenant_id",
        "tenant_id": "tenant_id",
        "APPEND_SIGNOFF": "append_signoff",
        "LIGHT_MODE": "light_mode",
        "PREFERRED_EMBEDDING_MODELS": "preferred_embedding_models",
        "RECENT_HISTORY_WINDOW": "recent_history_window",
        "MAX_HISTORY_MESSAGES_SCAN": "max_history_messages_scan",
        "SUMMARY_TRIGGER_MESSAGES": "summary_trigger_messages",
        "RELATIONSHIP_REFLECTION_INTERVAL": "relationship_reflection_interval",
        "CONTEXT_TOKEN_BUDGET": "context_token_budget",
        "RESERVED_RESPONSE_TOKENS": "reserved_response_tokens",
        "APPROX_CHARS_PER_TOKEN": "approx_chars_per_token",
        "MOOD_DETECTION_TEMPERATURE": "mood_detection_temperature",
        "STREAM_TIMEOUT_SECONDS": "stream_timeout_seconds",
        "STREAM_MAX_CHARS": "stream_max_chars",
        "PROFILE_PATH": "profile_path",
        "MEMORY_PATH": "memory_path",
        "SEMANTIC_MEMORY_DB_PATH": "semantic_memory_db_path",
        "GRAPH_STORE_DB_PATH": "graph_store_db_path",
        "SESSION_LOG_DIR": "session_log_dir",
    }

    _DEPRECATED_FACADE_ALIASES: dict[str, str] = {
        "detect_mood": "mood_manager.detect",
        "detect_mood_async": "mood_manager.detect_async",
        "fastpath_detect_mood": "mood_manager.fastpath_detect",
        "finalize_reply": "reply_finalization.append_signoff",
        "prepare_final_reply": "reply_finalization.finalize",
        "prepare_final_reply_async": "reply_finalization.finalize_async",
        "relationship_emotional_momentum": "relationship.emotional_momentum",
        "top_relationship_topics": "relationship.top_topics",
        "relationship_hypotheses": "relationship.hypotheses",
        "relationship_snapshot": "relationship.snapshot",
        "update_relationship_state": "relationship.current_state",
        "apply_relationship_feedback": "relationship.current_state",
        "build_relationship_reflection_prompt": "relationship.build_reflection_prompt",
        "reflect_relationship_state": "relationship.current_state",
        "internal_state_snapshot": "internal_state_manager.snapshot",
        "reflect_internal_state": "internal_state_manager.reflect_after_turn",
        "record_user_turn_state": "turn_service.record_user_turn_state",
        "evolved_persona_traits": "profile_runtime.evolved_persona_traits",
        "active_persona_traits": "profile_runtime.active_persona_traits",
        "effective_behavior_rules": "profile_runtime.effective_behavior_rules",
        "living_dad_snapshot": "profile_runtime.living_dad_snapshot",
    }

    _RUNTIME_STATE_ATTR_MAP: dict[str, str] = {
        "runtime_state_container": "container",
        "history": "history",
        "session_moods": "session_moods",
        "session_summary": "session_summary",
        "session_summary_updated_at": "session_summary_updated_at",
        "session_summary_covered_messages": "session_summary_covered_messages",
        "last_relationship_reflection_turn": "last_relationship_reflection_turn",
        "_pending_daily_checkin_context": "pending_daily_checkin_context",
        "_active_tool_observation_context": "active_tool_observation_context",
        "_last_planner_debug": "planner_debug",
        "chat_threads": "chat_threads",
        "active_thread_id": "active_thread_id",
        "thread_snapshots": "thread_snapshots",
    }

    _INTERNAL_RUNTIME_ATTR_MAP: dict[str, str] = {
        "_prompt_guard_stats": "prompt_guard_stats",
        "_last_memory_context_stats": "last_memory_context_stats",
        "_last_output_moderation": "last_output_moderation",
        "_last_reply_supervisor": "last_reply_supervisor",
        "_last_turn_pipeline": "last_turn_pipeline",
        "_last_turn_health_state": "last_turn_health_state",
        "_last_turn_ux_feedback": "last_turn_ux_feedback",
        "_background_task_ids": "background_task_ids",
    }

    def __getattr__(self, name: str):
        """Route unknown attribute lookups to registered manager objects.

        After ``__init__`` completes, lookups hit the O(1) provider map exposed
        by ``self.services``. During the early-init window (before the services
        container exists) the chain is walked
        directly so that manager construction can reference each other safely.

        Dunder names (``__foo__``) are never delegated to avoid interfering with
        pickling, copying, and other Python internals.
        """
        if name.startswith("__"):
            raise AttributeError(name)

        # Config proxy: names in _CONFIG_ATTR_MAP forward to self.config.
        # Canonical callers should prefer bot.config.<field> directly.
        config_attr = self.__class__._CONFIG_ATTR_MAP.get(name)
        if config_attr is not None:
            try:
                config = object.__getattribute__(self, "config")
                return getattr(config, config_attr)
            except AttributeError:
                pass  # config not yet assigned during early __init__

        runtime_state_attr = self.__class__._RUNTIME_STATE_ATTR_MAP.get(name)
        if runtime_state_attr is not None:
            try:
                runtime_state_manager = object.__getattribute__(self, "runtime_state_manager")
                return getattr(runtime_state_manager, runtime_state_attr)
            except AttributeError:
                pass

        internal_runtime_attr = self.__class__._INTERNAL_RUNTIME_ATTR_MAP.get(name)
        if internal_runtime_attr is not None:
            try:
                internal_runtime = object.__getattribute__(self, "_internal_runtime")
                return getattr(internal_runtime, internal_runtime_attr)
            except AttributeError:
                pass

        # Fast path: O(1) provider lookup once boot is complete.
        try:
            services = object.__getattribute__(self, "services")
            provider = services.get_provider(name)
            if provider is not None:
                compat = self.__dict__.get("_facade_compat")
                if compat is not None:
                    compat.warn_if_deprecated(name)
                return getattr(provider, name)
        except AttributeError:
            pass  # services not yet assigned â€” fall through to chain walk

        # Slow path: linear chain walk used only during __init__ construction.
        _sentinel = object()
        for _mgr_attr in self.__class__._MANAGER_DELEGATE_CHAIN:
            try:
                _mgr = object.__getattribute__(self, _mgr_attr)
            except AttributeError:
                continue
            _val = getattr(_mgr, name, _sentinel)
            if _val is not _sentinel:
                return _val
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name!r}'")

    def __setattr__(self, name: str, value) -> None:
        """Route mutations of config-mirrored names to self.config.

        Preserves backward compatibility for callers that set
        ``bot.MODEL_NAME``, ``bot.LLM_PROVIDER``, etc. while keeping the
        canonical state inside ``DadBotConfig``.  All other attribute writes
        fall through to normal ``object.__setattr__``.
        """
        config_attr = self.__class__._CONFIG_ATTR_MAP.get(name)
        if config_attr is not None:
            try:
                config = object.__getattribute__(self, "config")
                setattr(config, config_attr, value)
                return
            except AttributeError:
                pass  # config not yet assigned â€” fall through for early __init__ writes

        runtime_state_attr = self.__class__._RUNTIME_STATE_ATTR_MAP.get(name)
        if runtime_state_attr is not None:
            try:
                runtime_state_manager = object.__getattribute__(self, "runtime_state_manager")
                setattr(runtime_state_manager, runtime_state_attr, value)
                return
            except AttributeError:
                pass

        internal_runtime_attr = self.__class__._INTERNAL_RUNTIME_ATTR_MAP.get(name)
        if internal_runtime_attr is not None:
            try:
                internal_runtime = object.__getattribute__(self, "_internal_runtime")
                setattr(internal_runtime, internal_runtime_attr, value)
                return
            except AttributeError:
                pass
        object.__setattr__(self, name, value)

    def _get_explicit_manager(self, name: str):
        sentinel = object()
        value = self.__dict__.get(f"_{name}", sentinel)
        if value is not sentinel:
            return value
        services = self.__dict__.get("services")
        if services is not None:
            value = getattr(services, name, sentinel)
            if value is not sentinel:
                return value
        raise AttributeError(name)

    def _set_explicit_manager(self, name: str, value) -> None:
        object.__setattr__(self, f"_{name}", value)

    def _resolve_dependency(self, name: str, factory):
        """Resolve a runtime dependency from the optional injection registry.

        `dependency_registry` can be either:
        - a mapping-like object with `.get(name)`
        - a plain dict

        Registry values may be concrete instances or zero-arg callables that
        lazily build the instance.
        """
        registry = getattr(self, "_dependency_registry", None)
        candidate = None
        if registry is not None and hasattr(registry, "get"):
            candidate = registry.get(name)
        if callable(candidate):
            candidate = candidate()
        if candidate is not None:
            return candidate
        return factory()

    def _build_runtime_state_bundle(self):
        store = InMemoryStateStore()
        event_bus = InMemoryEventBus()
        container = AppStateContainer(
            f"local-{uuid.uuid4().hex}",
            self.default_planner_debug_state,
            tenant_id=self.config.tenant_id,
            store=store,
            event_bus=event_bus,
        )
        return {
            "store": store,
            "event_bus": event_bus,
            "container": container,
        }

    def _resolve_runtime_state_bundle(self):
        bundle = self._resolve_dependency("runtime_state_bundle", self._build_runtime_state_bundle)
        if isinstance(bundle, AppStateContainer):
            return {
                "store": None,
                "event_bus": None,
                "container": bundle,
            }
        if not isinstance(bundle, dict):
            raise RuntimeError("runtime_state_bundle dependency must be a dict or AppStateContainer")
        container = bundle.get("container")
        if not isinstance(container, AppStateContainer):
            raise RuntimeError("runtime_state_bundle must provide an AppStateContainer via 'container'")
        return {
            "store": bundle.get("store"),
            "event_bus": bundle.get("event_bus"),
            "container": container,
        }

    def _initialize_config(
        self,
        *,
        model_name: str,
        fallback_models: tuple[str, ...],
        append_signoff: bool,
        light_mode: bool,
        tenant_id: str,
    ) -> None:
        self.config = DadBotConfig(
            model_name=model_name,
            fallback_models=fallback_models,
            append_signoff=append_signoff,
            light_mode=light_mode,
            tenant_id=tenant_id,
        )
        self._service_config = self.config.service_config
        self.runtime_config = self.config.runtime_config

    def _ensure_runtime_paths(self) -> None:
        self.config.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.semantic_memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.graph_store_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.session_log_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_document_store(self, document_store: Any) -> None:
        self._customer_state_store = (
            document_store
            if document_store is not None
            else build_customer_document_store(self._service_config)
        )
        self._tenant_document_store = None
        if self._customer_state_store is not None:
            self._tenant_document_store = NamespacedStateStore(
                self._customer_state_store,
                f"tenant-doc:{self.config.tenant_id}",
            )

    def _initialize_services(self) -> None:
        self.bot_context = DadBotContext.from_runtime(self)
        self.context = self.bot_context
        self.services = DadBotServiceContainer(self)

        # Bootstrap managers required before profile/memory hydration.
        self.services.wire_bootstrap()

    def _hydrate_profile_and_memory(self) -> None:
        self.profile_runtime.load_profile()
        profile_llm = self.profile_runtime.profile.get("llm", {}) if isinstance(self.profile_runtime.profile, dict) else {}
        if isinstance(profile_llm, dict):
            # apply_profile_llm_settings normalises provider/model and updates
            # active_model when provider is ollama â€” no manual fan-out needed.
            self.config.apply_profile_llm_settings(
                str(profile_llm.get("provider", "")).strip().lower(),
                str(profile_llm.get("model", "")).strip(),
            )
        self.memory.load_memory_store()
        self.profile_runtime.refresh_profile_runtime()
        self.services.wire_runtime()
        self.service_registry = self.services.registry

    def _initialize_runtime_caches(self) -> None:
        self._model_metadata_cache = {}
        self._tokenizer_cache = {}
        self._tokenizer = self.initialize_tokenizer(self.config.active_model)
        self._ollama_async_client = None
        self._io_lock = RLock()
        self._session_lock = RLock()
        self._graph_refresh_lock = RLock()
        # Explicit actor isolation for sync graph turns: one in-flight turn at a time.
        self._turn_execution_lock = RLock()
        self.background_tasks = BackgroundTaskManager(max_workers=12, thread_name_prefix="dadbot-bg")
        self._semantic_index_lock = RLock()
        self._semantic_index_future = None
        self._pending_semantic_index_memories = None
        self._memory_graph_dirty = True
        self._last_memory_graph_refresh_monotonic = 0.0
        self._recent_mood_detections = {}
        self._recent_runtime_issues = deque(maxlen=12)
        self._last_runtime_health_log_monotonic = 0.0
        self._base_background_worker_limit = 12
        self._health_snapshot_interval_seconds = self.config.health_snapshot_interval_seconds
        self._proactive_heartbeat_interval_seconds = self.config.proactive_heartbeat_interval_seconds
        self._cached_runtime_health_snapshot = None
        self._last_runtime_health_snapshot_monotonic = 0.0
        self._internal_runtime = DadBotInternalRuntime(context_token_budget=self.CONTEXT_TOKEN_BUDGET)

    def _initialize_turn_orchestration(self) -> None:
        self._turn_graph_enabled = self.config.turn_graph_enabled
        self._strict_graph_mode = self.config.strict_graph_mode
        self._turn_graph_config_path = self.config.turn_graph_config_path
        self._turn_orchestrator = None

    def _start_boot_background_tasks(self) -> None:
        # Defer initial graph sync in normal runtime, but avoid starting this
        # task under pytest where it can race deterministic tests and keep
        # temporary SQLite files open during Windows cleanup.
        disable_graph_init_task = env_truthy("DADBOT_DISABLE_GRAPH_INIT_TASK", default=False)
        if self.should_start_background_tasks():
            if not disable_graph_init_task:
                self.background_tasks.submit(
                    self.refresh_memory_graph,
                    force=True,
                    task_kind="graph-init",
                )
            # Auto-sync iCal calendar feed on startup (background, non-blocking)
            if self.ical_feed_url():
                self.schedule_calendar_sync()
            if not env_truthy("DADBOT_DISABLE_PROACTIVE_HEARTBEAT", default=False):
                self.background_tasks.submit(
                    self._run_proactive_heartbeat_loop,
                    task_kind="proactive-heartbeat",
                )

    def __init__(
        self,
        model_name: str = "llama3.2",
        fallback_models: tuple[str, ...] = ("phi4:mini", "qwen3.5:4b", "gemma3:4b", "gemma2:2b", "llama3.2:1b"),
        *,
        append_signoff: bool = True,
        light_mode: bool = False,
        tenant_id: str = "",
        document_store: Any = None,
        dependency_registry: Any = None,
    ) -> None:
        self._dependency_registry = dependency_registry
        self._facade_compat = DadBotFacadeCompat(self.__class__._DEPRECATED_FACADE_ALIASES)
        self._initialize_config(
            model_name=model_name,
            fallback_models=fallback_models,
            append_signoff=append_signoff,
            light_mode=light_mode,
            tenant_id=tenant_id,
        )
        self._ensure_runtime_paths()
        self._initialize_document_store(document_store)
        self._initialize_services()
        self._hydrate_profile_and_memory()
        self._initialize_runtime_caches()
        self._initialize_turn_orchestration()
        self._validate_managers(smoke=env_truthy("DADBOT_VALIDATE_MANAGERS_SMOKE", default=False))
        atexit.register(self.shutdown)
        self.reset_session_state()
        self._start_boot_background_tasks()

    def should_start_background_tasks(self) -> bool:
        """Return True when background tasks should be started on boot.

        Returns False under pytest (where background threads can race
        deterministic tests and keep temporary files open on Windows).
        Override via the ``DADBOT_DISABLE_GRAPH_INIT_TASK`` /
        ``DADBOT_DISABLE_PROACTIVE_HEARTBEAT`` env flags for finer control.
        """
        return "PYTEST_CURRENT_TEST" not in os.environ

    def _get_turn_orchestrator(self):
        services = getattr(self, "services", None)
        if services is not None:
            return services.turn_orchestrator
        orchestrator = getattr(self, "_turn_orchestrator", None)
        if orchestrator is not None:
            return orchestrator
        orchestrator = DadBotOrchestrator(
            config_path=self._turn_graph_config_path,
            bot=self,
            strict=bool(getattr(self, "_strict_graph_mode", True)),
        )
        self._turn_orchestrator = orchestrator
        return orchestrator

    async def _run_graph_turn_async(self, user_input: str, attachments: AttachmentList | None = None) -> FinalizedTurnResult:
        return await self.turn_orchestrator.handle_turn(user_input, attachments=attachments)

    @staticmethod
    def _run_coro_in_thread(coro):
        """Run a coroutine in a worker thread with its own event loop.

        Used when the coroutine must be awaited from within an already-running
        loop (e.g. Streamlit, Jupyter) where :func:`asyncio.run` is forbidden.
        """
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def _run_graph_turn_sync(self, user_input: str, attachments: AttachmentList | None = None) -> FinalizedTurnResult:
        """Run the graph turn synchronously regardless of the calling context.

        When called from within a running event loop (e.g., Streamlit), the
        coroutine is dispatched to a dedicated thread with its own event loop
        rather than falling back to the legacy TurnProcessingManager.  This
        ensures the graph pipeline is *always* the active engine in production.

        Strict actor-isolation contract: sync callers share one execution actor,
        so turn-critical mutable state is never concurrently mutated.
        """
        with self._turn_execution_lock:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            coro = self._run_graph_turn_async(user_input, attachments=attachments)
            if loop is not None and loop.is_running():
                return self._run_coro_in_thread(coro)
            else:
                return asyncio.run(coro)

    def _validate_managers(self, *, smoke=False):
        """Compatibility wrapper for extracted facade validation."""
        self.services.validate_facade(smoke=smoke)

    def shutdown(self):
        background_tasks = getattr(self, "background_tasks", None)
        try:
            if background_tasks is not None:
                background_tasks.shutdown(wait=True)
        finally:
            save_memory_store = getattr(self, "save_memory_store", None)
            if callable(save_memory_store):
                try:
                    save_memory_store()
                except Exception as exc:
                    logger.info("Final memory store flush during shutdown failed: %s", exc)

    def _run_proactive_heartbeat_loop(self):
        interval_seconds = max(300, int(getattr(self, "_proactive_heartbeat_interval_seconds", 3600) or 3600))
        try:
            self.maintenance_scheduler.run_proactive_heartbeat()
        except Exception as exc:
            logger.info("Proactive heartbeat startup tick failed: %s", exc)

        while not self.background_tasks.wait_for_shutdown(interval_seconds):
            try:
                self.maintenance_scheduler.run_proactive_heartbeat()
            except Exception as exc:
                logger.info("Proactive heartbeat tick failed: %s", exc)

    @staticmethod
    def ollama_retryable_errors():
        base = (ConnectionError, TimeoutError, OSError)
        if ollama is None:
            return base
        return (*base, ollama.RequestError, ollama.ResponseError)

    @staticmethod
    def ollama_error_summary(exc):
        if exc is None:
            return "Unknown Ollama error"

        details = []
        # Check for Ollama-specific response errors
        if hasattr(exc, "status_code"):
            details.append(f"status={exc.status_code}")
        elif hasattr(exc, "status"):
            details.append(f"status={exc.status}")

        # Safely grab the message content
        if hasattr(exc, "error"):
            error_text = exc.error
        elif hasattr(exc, "message"):
            error_text = exc.message
        else:
            error_text = str(exc)
        if error_text:
            details.append(str(error_text)[:120])
        summary = "; ".join(details)
        return summary if summary else type(exc).__name__

    @staticmethod
    def default_planner_debug_state():
        return _default_planner_debug_state()

    @property
    def memory(self):
        return self.services.memory_manager

    @property
    def relationship(self):
        return self.services.relationship_manager

    @property
    def mood(self) -> Any:
        """Typed alias for explicit mood-manager access."""
        return self.services.mood_manager

    @property
    def profile(self) -> Any:
        """Typed alias for explicit profile-runtime access."""
        return self.services.profile_runtime

    @property
    def turn_orchestrator(self):
        return self.services.turn_orchestrator

    @property
    def runtime_storage(self):
        return self._get_explicit_manager("runtime_storage")

    @runtime_storage.setter
    def runtime_storage(self, value):
        self._set_explicit_manager("runtime_storage", value)

    @property
    def profile_runtime(self):
        return self._get_explicit_manager("profile_runtime")

    @profile_runtime.setter
    def profile_runtime(self, value):
        self._set_explicit_manager("profile_runtime", value)

    @property
    def mood_manager(self):
        return self._get_explicit_manager("mood_manager")

    @mood_manager.setter
    def mood_manager(self, value):
        self._set_explicit_manager("mood_manager", value)

    @property
    def turn_service(self):
        return self._get_explicit_manager("turn_service")

    @turn_service.setter
    def turn_service(self, value):
        self._set_explicit_manager("turn_service", value)

    @property
    def reply_finalization(self):
        return self._get_explicit_manager("reply_finalization")

    @reply_finalization.setter
    def reply_finalization(self, value):
        self._set_explicit_manager("reply_finalization", value)

    @property
    def runtime_interface(self):
        return self._get_explicit_manager("runtime_interface")

    @runtime_interface.setter
    def runtime_interface(self, value):
        self._set_explicit_manager("runtime_interface", value)

    @property
    def status_reporting(self):
        return self._get_explicit_manager("status_reporting")

    @status_reporting.setter
    def status_reporting(self, value):
        self._set_explicit_manager("status_reporting", value)

    @property
    def model_runtime(self):
        return self._get_explicit_manager("model_runtime")

    @model_runtime.setter
    def model_runtime(self, value):
        self._set_explicit_manager("model_runtime", value)

    @property
    def runtime_client(self):
        return self._get_explicit_manager("runtime_client")

    @runtime_client.setter
    def runtime_client(self, value):
        self._set_explicit_manager("runtime_client", value)

    @property
    def maintenance_scheduler(self):
        return self._get_explicit_manager("maintenance_scheduler")

    @maintenance_scheduler.setter
    def maintenance_scheduler(self, value):
        self._set_explicit_manager("maintenance_scheduler", value)

    @property
    def health_manager(self):
        return self._get_explicit_manager("health_manager")

    @health_manager.setter
    def health_manager(self, value):
        self._set_explicit_manager("health_manager", value)

    @property
    def internal_state_manager(self):
        return self._get_explicit_manager("internal_state_manager")

    @internal_state_manager.setter
    def internal_state_manager(self, value):
        self._set_explicit_manager("internal_state_manager", value)

    @property
    def runtime_state_manager(self):
        return self._get_explicit_manager("runtime_state_manager")

    @runtime_state_manager.setter
    def runtime_state_manager(self, value):
        self._set_explicit_manager("runtime_state_manager", value)

    @property
    def prompt_assembly(self):
        return self._get_explicit_manager("prompt_assembly")

    @prompt_assembly.setter
    def prompt_assembly(self, value):
        self._set_explicit_manager("prompt_assembly", value)

    @property
    def context_builder(self):
        return self._get_explicit_manager("context_builder")

    @context_builder.setter
    def context_builder(self, value):
        self._set_explicit_manager("context_builder", value)

    @property
    def tone_context(self):
        return self._get_explicit_manager("tone_context")

    @tone_context.setter
    def tone_context(self, value):
        self._set_explicit_manager("tone_context", value)

    @property
    def memory_query(self):
        return self._get_explicit_manager("memory_query")

    @memory_query.setter
    def memory_query(self, value):
        self._set_explicit_manager("memory_query", value)

    @property
    def memory_commands(self):
        return self._get_explicit_manager("memory_commands")

    @memory_commands.setter
    def memory_commands(self, value):
        self._set_explicit_manager("memory_commands", value)

    @property
    def memory_coordinator(self):
        return self._get_explicit_manager("memory_coordinator")

    @memory_coordinator.setter
    def memory_coordinator(self, value):
        self._set_explicit_manager("memory_coordinator", value)

    @property
    def long_term_signals(self):
        return self._get_explicit_manager("long_term_signals")

    @long_term_signals.setter
    def long_term_signals(self, value):
        self._set_explicit_manager("long_term_signals", value)

    @property
    def safety_support(self):
        return self._get_explicit_manager("safety_support")

    @safety_support.setter
    def safety_support(self, value):
        self._set_explicit_manager("safety_support", value)

    @property
    def reply_supervisor(self):
        return self._get_explicit_manager("reply_supervisor")

    @reply_supervisor.setter
    def reply_supervisor(self, value):
        self._set_explicit_manager("reply_supervisor", value)

    @property
    def multimodal_handler(self):
        return self._get_explicit_manager("multimodal_handler")

    @multimodal_handler.setter
    def multimodal_handler(self, value):
        self._set_explicit_manager("multimodal_handler", value)

    @property
    def runtime_orchestration(self):
        return self._get_explicit_manager("runtime_orchestration")

    @runtime_orchestration.setter
    def runtime_orchestration(self, value):
        self._set_explicit_manager("runtime_orchestration", value)

    @property
    def session_summary_manager(self):
        return self._get_explicit_manager("session_summary_manager")

    @session_summary_manager.setter
    def session_summary_manager(self, value):
        self._set_explicit_manager("session_summary_manager", value)

    @property
    def tool_registry(self):
        return self._get_explicit_manager("tool_registry")

    @tool_registry.setter
    def tool_registry(self, value):
        self._set_explicit_manager("tool_registry", value)

    @property
    def agentic_handler(self):
        return self._get_explicit_manager("agentic_handler")

    @agentic_handler.setter
    def agentic_handler(self, value):
        self._set_explicit_manager("agentic_handler", value)

    @property
    def conversation_persistence(self):
        return self._get_explicit_manager("conversation_persistence")

    @conversation_persistence.setter
    def conversation_persistence(self, value):
        self._set_explicit_manager("conversation_persistence", value)

    @property
    def PROFILE(self):
        return self.profile_runtime.profile

    @PROFILE.setter
    def PROFILE(self, value):
        self.profile_runtime.profile = value

    @property
    def MEMORY_STORE(self):
        return self.memory.memory_store

    @MEMORY_STORE.setter
    def MEMORY_STORE(self, value):
        self.memory.memory_store = value

    @property
    def STYLE(self):
        return self.profile_runtime.style

    @STYLE.setter
    def STYLE(self, value):
        self.profile_runtime.style = value

    def reset_session_state(self) -> Any:
        """Explicit facade delegate for runtime session reset."""
        return self.runtime_state_manager.reset_session_state()

    def _thread_timestamp(self):
        return self.runtime_state_manager.thread_timestamp()

    def _apply_thread_snapshot_unlocked(self, snapshot):
        return self.runtime_state_manager.apply_thread_snapshot_unlocked(snapshot)

    def runtime_health_snapshot(self, *, log_warnings=True, persist=True):
        health_manager = getattr(self, "health_manager", None)
        runtime_health_snapshot = getattr(health_manager, "runtime_health_snapshot", None)
        if callable(runtime_health_snapshot):
            return runtime_health_snapshot(log_warnings=log_warnings, persist=persist)
        return {"status": "unknown"}

    def estimate_token_count(self, text):
        if not text:
            return 0

        normalized = str(text)
        tokenizer = self.current_tokenizer(model_name=self.ACTIVE_MODEL)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(normalized))
            except Exception:
                pass

        chars_per_token = self.model_chars_per_token_estimate(self.ACTIVE_MODEL)
        return max(1, int((len(normalized) + chars_per_token - 1) // chars_per_token))

    def estimate_tokens(self, text):
        return self.estimate_token_count(text)

    def initialize_tokenizer(self, model_name=None):
        return self.model_runtime.initialize_tokenizer(model_name)

    def current_tokenizer(self, model_name=None):
        return self.model_runtime.current_tokenizer(model_name)

    def model_chars_per_token_estimate(self, model_name=None):
        return self.model_runtime.model_chars_per_token_estimate(model_name)

    # -- Intentionally kept thin shims for API stability / discoverability ------
    # Prefer direct manager access in new code:
    # - bot.mood_manager.detect(...)
    # - bot.reply_finalization.append_signoff(...)
    # - bot.relationship.reflect(...), bot.relationship.snapshot(...)
    # - bot.memory.export_memory_store(...), bot.memory.memory_graph_snapshot()
    # - bot.internal_state_manager.snapshot(...)
    #
    # These remain on DadBot for current callers, tests, and interactive usage
    # while the facade is being thinned down incrementally.

    def call_llm(
        self,
        messages,
        *,
        model=None,
        temperature=None,
        stream=False,
        purpose="chat",
        options=None,
        response_format=None,
        chunk_callback=None,
        **kwargs,
    ):
        """Main unified LLM entrypoint delegated to the runtime client manager."""
        return self.runtime_client.call_llm(
            messages,
            model=model,
            temperature=temperature,
            stream=stream,
            purpose=purpose,
            options=options,
            response_format=response_format,
            chunk_callback=chunk_callback,
            **kwargs,
        )

    async def call_llm_async(
        self,
        messages,
        *,
        model=None,
        temperature=None,
        stream=False,
        purpose="chat",
        options=None,
        response_format=None,
        chunk_callback=None,
        **kwargs,
    ):
        """Async unified LLM entrypoint delegated to the runtime client manager."""
        return await self.runtime_client.call_llm_async(
            messages,
            model=model,
            temperature=temperature,
            stream=stream,
            purpose=purpose,
            options=options,
            response_format=response_format,
            chunk_callback=chunk_callback,
            **kwargs,
        )

    def call_ollama_chat(self, messages, options=None, response_format=None, purpose="chat"):
        return self.call_llm(messages, purpose=purpose, options=options, response_format=response_format)

    async def call_ollama_chat_async(self, messages, options=None, response_format=None, purpose="chat"):
        return await self.call_llm_async(messages, purpose=purpose, options=options, response_format=response_format)

    def call_ollama_chat_with_model(self, model_name, messages, options=None, response_format=None, purpose="chat"):
        return self.call_llm(messages, model=model_name, purpose=purpose, options=options, response_format=response_format)

    def call_ollama_chat_stream(self, messages, options=None, purpose="chat", chunk_callback=None):
        return self.call_llm(messages, stream=True, purpose=purpose, options=options, chunk_callback=chunk_callback)

    async def call_ollama_chat_stream_async(self, messages, options=None, purpose="chat", chunk_callback=None):
        return await self.call_llm_async(messages, stream=True, purpose=purpose, options=options, chunk_callback=chunk_callback)

    def _record_background_task(self, task_id, *, task_kind, status, metadata=None, error=""):
        return self.runtime_orchestration.record_background_task(
            task_id,
            task_kind=task_kind,
            status=status,
            metadata=metadata,
            error=error,
        )

    def embed_texts(self, texts, purpose="semantic retrieval"):
        return self.memory.embed_texts(texts, purpose=purpose)

    def chat_loop(self):
        return self.runtime_interface.chat_loop()

    def chat_loop_via_service(self, service_client, session_id=None):
        return self.runtime_interface.chat_loop_via_service(service_client, session_id=session_id)

    async def handle_turn_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        """Canonical async turn entrypoint backed by the existing runtime path."""
        return await self.process_user_message_async(user_input, attachments=attachments)

    def handle_turn_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        """Canonical sync turn entrypoint backed by the existing runtime path."""
        return self.process_user_message(user_input, attachments=attachments)

    def _graph_failure_session_id(self) -> str:
        candidate = getattr(self, "active_thread_id", "") or getattr(self, "tenant_id", "")
        normalized = str(candidate or "").strip()
        return normalized or "default"

    @staticmethod
    def _safe_graph_failure_payload(value: Any, *, limit: int = 240) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:limit]
        if isinstance(value, dict):
            return {
                str(key)[:80]: DadBot._safe_graph_failure_payload(item, limit=limit)
                for key, item in list(value.items())[:12]
            }
        if isinstance(value, (list, tuple, set)):
            return [DadBot._safe_graph_failure_payload(item, limit=limit) for item in list(value)[:12]]
        return str(value)[:limit]

    def _emit_graph_failure_event(
        self,
        *,
        mode: str,
        correlation_id: str,
        trace_id: str,
        user_input: str,
        attachments: AttachmentList | None,
        exc: Exception,
    ) -> None:
        orchestrator = getattr(self, "_turn_orchestrator", None)
        if orchestrator is None:
            try:
                orchestrator = self._get_turn_orchestrator()
            except Exception:
                orchestrator = None
        control_plane = getattr(orchestrator, "control_plane", None)
        ledger_writer = getattr(control_plane, "ledger_writer", None)
        write_event = getattr(ledger_writer, "write_event", None)
        if not callable(write_event):
            return

        payload = {
            "mode": mode,
            "graph_enabled": True,
            "strict_graph_mode": bool(getattr(self, "_strict_graph_mode", True)),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "user_input": self._safe_graph_failure_payload(user_input),
            "attachment_count": len(attachments or []),
            "attachments": self._safe_graph_failure_payload(list(attachments or []), limit=120),
            "recorded_at": str(getattr(getattr(self, "_current_turn_time_base", None), "wall_time", "")),
        }
        write_event(
            event_type="GRAPH_EXECUTION_FAILED",
            session_id=self._graph_failure_session_id(),
            trace_id=trace_id or correlation_id,
            kernel_step_id=f"graph.failure.{mode}",
            payload={
                **payload,
                "correlation_id": correlation_id,
                "trace_id": trace_id or correlation_id,
            },
            committed=True,
        )

    def _graph_failure_reply(self, correlation_id: str) -> str:
        return self._append_signoff_compat(
            "I hit an internal graph error and stopped before touching memory or state. "
            f"Please try again. Reference ID: {correlation_id}"
        )

    def _raise_graph_execution_failure(
        self,
        exc: Exception,
        *,
        mode: str,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> None:
        correlation_id = CorrelationContext.current() or CorrelationContext.ensure()
        trace_id = TracingContext.current_trace_id() or correlation_id
        logger.exception(
            "Graph execution failed in %s mode; strict mode forbids alternate execution paths",
            mode,
            extra={
                "correlation_id": correlation_id,
                "trace_id": trace_id,
                "graph_enabled": True,
                "strict_graph_mode": bool(getattr(self, "_strict_graph_mode", True)),
                "attachment_count": len(attachments or []),
            },
        )
        self._emit_graph_failure_event(
            mode=mode,
            correlation_id=correlation_id,
            trace_id=trace_id,
            user_input=user_input,
            attachments=attachments,
            exc=exc,
        )
        raise RuntimeError("Graph execution failed in strict mode; legacy path is disabled") from exc

    def _deliver_buffered_stream_chunks(self, reply: str, chunk_callback: ChunkCallback | None) -> None:
        if callable(chunk_callback) and reply:
            chunk_callback(reply)

    def process_user_message(self, user_input: str, attachments: AttachmentList | None = None) -> FinalizedTurnResult:
        try:
            return self._run_graph_turn_sync(user_input, attachments=attachments)
        except Exception as exc:
            self._raise_graph_execution_failure(
                exc,
                mode="sync",
                user_input=user_input,
                attachments=attachments,
            )

    def _append_signoff_compat(self, text: str) -> str:
        """Apply reply signoff using manager-first compatibility resolution."""
        finalization = getattr(self, "reply_finalization", None)
        append_signoff = getattr(finalization, "append_signoff", None)
        if callable(append_signoff):
            return append_signoff(text)
        compat_finalize = getattr(self, "finalize_reply", None)
        if callable(compat_finalize):
            return compat_finalize(text)
        return str(text or "")

    async def process_user_message_async(self, user_input: str, attachments: AttachmentList | None = None) -> FinalizedTurnResult:
        try:
            return await self._run_graph_turn_async(user_input, attachments=attachments)
        except Exception as exc:
            self._raise_graph_execution_failure(
                exc,
                mode="async",
                user_input=user_input,
                attachments=attachments,
            )

    def process_user_message_stream(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        reply, should_end = self.process_user_message(user_input, attachments=attachments)
        self._deliver_buffered_stream_chunks(reply, chunk_callback)
        return reply, should_end

    async def process_user_message_stream_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        reply, should_end = await self.process_user_message_async(user_input, attachments=attachments)
        self._deliver_buffered_stream_chunks(reply, chunk_callback)
        return reply, should_end

    def current_runtime_health_snapshot(self, *, force=False, log_warnings=False, persist=False, max_age_seconds=None):
        """Cached health snapshot; explicit on DadBot so tests can patch bot.runtime_health_snapshot."""
        now = time.monotonic()
        max_age_seconds = max(0, int(max_age_seconds or 0)) if max_age_seconds is not None else max(30, int(self._health_snapshot_interval_seconds or 300))
        cached = self._cached_runtime_health_snapshot
        last = self._last_runtime_health_snapshot_monotonic
        if not force and isinstance(cached, dict) and (now - last) <= max_age_seconds:
            return dict(cached)
        snapshot = getattr(
            self,
            "runtime_health_snapshot",
            lambda **_kw: {"status": "unknown"},
        )(log_warnings=log_warnings, persist=persist)
        self._cached_runtime_health_snapshot = dict(snapshot)
        self._last_runtime_health_snapshot_monotonic = now
        return dict(snapshot)

    def turn_health_state(self) -> dict[str, Any]:
        payload = dict(getattr(self, "_last_turn_health_state", {}) or {})
        if payload:
            return payload
        return {
            "status": "OK",
            "latency_ms": 0.0,
            "memory_ops_time": 0.0,
            "graph_sync_time": 0.0,
            "inference_time": 0.0,
            "fallback_used": False,
        }

    def turn_fidelity_state(self) -> dict[str, Any]:
        payload = dict(getattr(self, "_last_turn_health_state", {}) or {})
        fidelity = dict(payload.get("fidelity") or {}) if isinstance(payload, dict) else {}
        if fidelity:
            return {
                "temporal": bool(fidelity.get("temporal", False)),
                "inference": bool(fidelity.get("inference", False)),
                "reflection": bool(fidelity.get("reflection", False)),
                "save": bool(fidelity.get("save", False)),
                "full_pipeline": bool(fidelity.get("full_pipeline", False)),
            }
        return {
            "temporal": False,
            "inference": False,
            "reflection": False,
            "save": False,
            "full_pipeline": False,
        }

    def turn_ux_feedback(self) -> dict[str, Any]:
        payload = dict(getattr(self, "_last_turn_ux_feedback", {}) or {})
        if payload:
            return payload
        return {
            "dad_is_thinking": False,
            "message": "",
            "checking_memory": False,
            "memory_message": "",
            "mood_hint": str(self.last_saved_mood() or "neutral"),
            "status": "OK",
        }

    def local_mcp_status(self) -> dict[str, Any]:
        try:
            from dadbot_system.local_mcp_server import local_mcp_status as describe_local_mcp
        except Exception as exc:
            return {
                "available": False,
                "configured": False,
                "server_name": "dadbot-local-services",
                "tool_count": 0,
                "local_state_entries": len(dict(self.MEMORY_STORE.get("mcp_local_store") or {})),
                "error": str(exc).strip() or exc.__class__.__name__,
                "running": False,
            }
        payload = dict(describe_local_mcp(self) or {})
        runtime_paths = self.local_mcp_runtime_paths()
        pid = self._read_local_mcp_pid()
        running = self._local_mcp_process_running(pid)
        if pid is not None and not running:
            try:
                runtime_paths["pid"].unlink(missing_ok=True)
            except OSError:
                pass
        payload.setdefault("local_state_entries", len(dict(self.MEMORY_STORE.get("mcp_local_store") or {})))
        payload.update(
            {
                "running": running,
                "pid": pid if running else None,
                "stdout_log_path": str(runtime_paths["stdout"]),
                "stderr_log_path": str(runtime_paths["stderr"]),
                "task_label": "Run Dad Bot MCP Server",
            }
        )
        return payload

    def local_mcp_runtime_paths(self) -> dict[str, Path]:
        # Always anchor to the project root so this matches _resolve_pid_path() in the server module.
        base_dir = self.runtime_root_path() / "session_logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        return {
            "pid": base_dir / "local_mcp_server.pid",
            "stdout": base_dir / "local_mcp_server.stdout.log",
            "stderr": base_dir / "local_mcp_server.stderr.log",
        }

    @staticmethod
    def _local_mcp_process_running(pid: int | None) -> bool:
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _read_local_mcp_pid(self) -> int | None:
        pid_path = self.local_mcp_runtime_paths()["pid"]
        if not pid_path.exists():
            return None
        try:
            return int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def local_mcp_log_tail(self, *, lines: int = 20) -> dict[str, str]:
        def _tail(path: Path) -> str:
            try:
                content = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return ""
            return "\n".join(content[-max(1, int(lines or 1)):])

        paths = self.local_mcp_runtime_paths()
        return {
            "stdout": _tail(paths["stdout"]),
            "stderr": _tail(paths["stderr"]),
        }

    def start_local_mcp_server_process(self, *, restart: bool = False) -> dict[str, Any]:
        if restart:
            self.stop_local_mcp_server_process()

        status = self.local_mcp_status()
        if status.get("running"):
            return status

        runtime_paths = self.local_mcp_runtime_paths()
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

        stdout_handle = runtime_paths["stdout"].open("ab")
        stderr_handle = runtime_paths["stderr"].open("ab")
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "dadbot_system.local_mcp_server"],
                cwd=str(self.runtime_root_path()),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()

        runtime_paths["pid"].write_text(str(process.pid), encoding="utf-8")
        return self.local_mcp_status()

    def stop_local_mcp_server_process(self) -> dict[str, Any]:
        runtime_paths = self.local_mcp_runtime_paths()
        pid = self._read_local_mcp_pid()
        if pid is not None and self._local_mcp_process_running(pid):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
        try:
            runtime_paths["pid"].unlink(missing_ok=True)
        except OSError:
            pass
        return self.local_mcp_status()

    def load_latest_graph_checkpoint(self, trace_id: str = "") -> dict[str, Any] | None:
        return self.conversation_persistence.load_latest_graph_checkpoint(trace_id=trace_id)

    def resume_turn_from_checkpoint(self, trace_id: str = "") -> dict[str, Any] | None:
        return self.conversation_persistence.resume_graph_checkpoint(trace_id=trace_id)

    def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
        return self.conversation_persistence.list_turn_events(trace_id=trace_id, limit=limit)

    def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
        return self.conversation_persistence.replay_turn_events(trace_id=trace_id)

    def validate_replay_determinism(self, trace_id: str, expected_lock_hash: str = "") -> dict[str, Any]:
        return self.conversation_persistence.validate_replay_determinism(
            trace_id=trace_id,
            expected_lock_hash=expected_lock_hash,
        )

    def build_local_mcp_server(self):
        from dadbot_system.local_mcp_server import build_server

        return build_server(bot=self)

    def run_local_mcp_server(self) -> None:
        from dadbot_system.local_mcp_server import main as run_local_mcp_main

        run_local_mcp_main(bot=self)


if __name__ == "__main__":
    raise SystemExit(run_app_main(dadbot_cls=DadBot, script_path=__file__))

