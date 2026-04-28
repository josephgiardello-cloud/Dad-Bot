"""DadBotBootMixin — boot/init/shutdown lifecycle for the DadBot facade.

Extracted from DadBot to reduce the god-class surface. Contains:
- Static runtime-path utilities (runtime_root_path, etc.)
- All __init__ helper stages (_initialize_config, _initialize_services, …)
- __init__ itself
- shutdown / proactive-heartbeat loop
- Ollama error helpers and default_planner_debug_state static utility
"""
from __future__ import annotations

import atexit
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from threading import RLock
from typing import Any

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]

from dadbot.defaults import default_planner_debug_state as _default_planner_debug_state
from dadbot.utils import env_truthy, json_dump, json_load
from dadbot.core.facade_compat import DadBotFacadeCompat


class DadBotLifecycleState:
    """Observable lifecycle states for a DadBot instance.

    States advance monotonically through the boot sequence; SHUTDOWN is
    terminal.  Introspect via ``bot.lifecycle_state``.
    """
    UNINITIALIZED: str = "uninitialized"
    CONFIG: str = "config"
    PATHS: str = "paths"
    DOCUMENT_STORE: str = "document_store"
    SERVICES: str = "services"
    HYDRATED: str = "hydrated"
    CACHES: str = "caches"
    ORCHESTRATION: str = "orchestration"
    READY: str = "ready"
    SHUTDOWN: str = "shutdown"


class DadBotBootMixin:
    """Boot, init, and shutdown lifecycle for the DadBot facade.

    This mixin owns the full __init__ sequence (split into explicit
    _initialize_* stages for testability) plus the shutdown hook and
    background-task bootstrap. All methods operate on ``self`` which is
    always the concrete DadBot instance at runtime.
    """

    # ------------------------------------------------------------------
    # Static / class path utilities
    # ------------------------------------------------------------------

    _EMBEDDED_DEFAULT_PROFILE: dict[str, Any] = {
        "name": "Dad",
        "relationship": "father",
        "conversation_style": {
            "tone": "supportive",
            "humor": "dad-jokes",
            "directness": "balanced",
        },
        "llm": {
            "provider": "ollama",
            "model": "llama3.2",
        },
        "preferences": {
            "append_signoff": True,
        },
    }

    @staticmethod
    def runtime_root_path() -> Path:
        # dadbot/core/boot_mixin.py → parents[0]=dadbot/core, [1]=dadbot, [2]=root
        return Path(__file__).resolve().parents[2]

    @classmethod
    def runtime_script_path(cls) -> Path:
        return cls.runtime_root_path() / "Dad.py"

    @staticmethod
    def profile_template_path() -> Path:
        return Path(__file__).resolve().parents[2] / "dad_profile.template.json"

    @staticmethod
    def env_path(name: str, fallback_path) -> Path:
        configured = str(os.environ.get(name) or "").strip()
        if configured:
            return Path(configured)
        return Path(fallback_path)

    @staticmethod
    def default_profile() -> dict:
        template_path = Path(__file__).resolve().parents[2] / "dad_profile.template.json"
        try:
            with template_path.open("r", encoding="utf-8") as f:
                payload = json_load(f)
            if isinstance(payload, dict) and payload:
                return payload
        except Exception:
            pass
        # Deterministic bootstrap fallback for clean/test environments.
        return json.loads(json.dumps(DadBotBootMixin._EMBEDDED_DEFAULT_PROFILE))

    @classmethod
    def initialize_profile_file(cls, profile_path=None, force: bool = False) -> bool:
        destination = (
            Path(profile_path)
            if profile_path is not None
            else cls.runtime_root_path() / "dad_profile.json"
        )
        if destination.exists() and not force:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as f:
            json_dump(cls.default_profile(), f, indent=2)
        return True

    # ------------------------------------------------------------------
    # Runtime-state bundle helpers
    # ------------------------------------------------------------------

    def _build_runtime_state_bundle(self) -> dict:
        from dadbot_system import InMemoryEventBus, InMemoryStateStore
        from dadbot_system.state import AppStateContainer

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

    def _resolve_runtime_state_bundle(self) -> dict:
        from dadbot_system.state import AppStateContainer

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

    # ------------------------------------------------------------------
    # Init stages
    # ------------------------------------------------------------------

    def _initialize_config(
        self,
        *,
        model_name: str,
        fallback_models: tuple[str, ...],
        append_signoff: bool,
        light_mode: bool,
        tenant_id: str,
    ) -> None:
        from dadbot.config import DadBotConfig

        self.config = DadBotConfig(
            model_name=model_name,
            fallback_models=fallback_models,
            append_signoff=append_signoff,
            light_mode=light_mode,
            tenant_id=tenant_id,
        )
        self._service_config = self.config.service_config
        self.runtime_config = self.config.runtime_config
        self._lifecycle_state = DadBotLifecycleState.CONFIG

    def _ensure_runtime_paths(self) -> None:
        self.config.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.semantic_memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.graph_store_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.session_log_dir.mkdir(parents=True, exist_ok=True)
        self._lifecycle_state = DadBotLifecycleState.PATHS

    def _initialize_document_store(self, document_store: Any) -> None:
        from dadbot_system import NamespacedStateStore

        from dadbot.app_runtime import build_customer_document_store

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
        self._lifecycle_state = DadBotLifecycleState.DOCUMENT_STORE

    def _initialize_services(self) -> None:
        # Deferred import keeps cold module import topology lean.
        import importlib

        from dadbot.contracts import DadBotContext
        from dadbot.core.ux_projection_gateway import TurnUxProjectionGateway
        from dadbot.registry import DadBotServiceContainer
        from dadbot.runtime.mcp import LocalMcpServerController

        invariance = importlib.import_module("dadbot.core.invariance_contract")
        build_boundary_compliance = getattr(invariance, "build_boundary_compliance")
        get_evaluation_contract = getattr(invariance, "get_evaluation_contract")
        serialize_boundary_declarations = getattr(invariance, "serialize_boundary_declarations")

        self.bot_context = DadBotContext.from_runtime(self)
        self.context = self.bot_context
        self.evaluation_contract = get_evaluation_contract()
        self._boundary_contracts = {
            "boot": build_boundary_compliance("boot"),
        }
        self.contract_compliance = {
            "evaluation_contract": self.evaluation_contract.model_dump(mode="json"),
            "boundary_contracts": serialize_boundary_declarations(self._boundary_contracts),
        }
        self.services = DadBotServiceContainer(self)

        # Bootstrap managers required before profile/memory hydration.
        self.services.wire_bootstrap()
        self.services.registry.declare_boundary_compliance(self._boundary_contracts["boot"])

        # Placeholder until _hydrate_profile_and_memory wires the real adapter.
        self._model_port: Any | None = None

        # MCP server controller and UX projection gateway (wired early).
        self._mcp_controller = LocalMcpServerController(self.runtime_root_path())
        self._ux_gateway = TurnUxProjectionGateway()
        self._lifecycle_state = DadBotLifecycleState.SERVICES

    def _hydrate_profile_and_memory(self) -> None:
        from dadbot.runtime.model import ModelConfig, OllamaModelAdapter

        self.profile_runtime.load_profile()
        profile_llm = (
            self.profile_runtime.profile.get("llm", {})
            if isinstance(self.profile_runtime.profile, dict)
            else {}
        )
        if isinstance(profile_llm, dict):
            self.config.apply_profile_llm_settings(
                str(profile_llm.get("provider", "")).strip().lower(),
                str(profile_llm.get("model", "")).strip(),
            )
        self.memory.load_memory_store()
        self.profile_runtime.refresh_profile_runtime()
        self.services.wire_runtime()
        self.services.registry.declare_boundary_compliance(self._boundary_contracts["boot"])
        self.service_registry = self.services.registry

        # Wire the deterministic model-interaction port now that runtime is ready.
        config = ModelConfig(
            active_model=str(self.config.active_model or "llama3.2"),
            temperature=None,
            request_timeout_seconds=30.0,
        )
        self._model_port = OllamaModelAdapter(
            runtime_client=self.runtime_client,
            model_runtime=self.model_runtime,
            config=config,
        )
        self._lifecycle_state = DadBotLifecycleState.HYDRATED

    def _initialize_runtime_caches(self) -> None:
        from dadbot.background import BackgroundTaskManager
        from dadbot.core.internal_runtime import DadBotInternalRuntime

        self._model_metadata_cache: dict = {}
        self._tokenizer_cache: dict = {}
        self._tokenizer = self.initialize_tokenizer(self.config.active_model)
        self._ollama_async_client = None
        self._io_lock = RLock()
        self._session_lock = RLock()
        self._graph_refresh_lock = RLock()
        self._turn_execution_lock = RLock()
        self.background_tasks = BackgroundTaskManager(max_workers=12, thread_name_prefix="dadbot-bg")
        self._semantic_index_lock = RLock()
        self._semantic_index_future = None
        self._pending_semantic_index_memories = None
        self._memory_graph_dirty = True
        self._last_memory_graph_refresh_monotonic = 0.0
        self._recent_mood_detections: dict = {}
        self._recent_runtime_issues: deque = deque(maxlen=12)
        self._last_runtime_health_log_monotonic = 0.0
        self._base_background_worker_limit = 12
        self._health_snapshot_interval_seconds = self.config.health_snapshot_interval_seconds
        self._proactive_heartbeat_interval_seconds = self.config.proactive_heartbeat_interval_seconds
        self._cached_runtime_health_snapshot = None
        self._last_runtime_health_snapshot_monotonic = 0.0
        self._internal_runtime = DadBotInternalRuntime(context_token_budget=self.CONTEXT_TOKEN_BUDGET)
        self._lifecycle_state = DadBotLifecycleState.CACHES

    def _initialize_turn_orchestration(self) -> None:
        self._turn_graph_enabled = self.config.turn_graph_enabled
        self._strict_graph_mode = self.config.strict_graph_mode
        self._turn_graph_config_path = self.config.turn_graph_config_path
        self._turn_orchestrator = None
        self._lifecycle_state = DadBotLifecycleState.ORCHESTRATION

    def _start_boot_background_tasks(self) -> None:
        disable_graph_init_task = env_truthy("DADBOT_DISABLE_GRAPH_INIT_TASK", default=False)
        if self.should_start_background_tasks():
            if not disable_graph_init_task:
                self.background_tasks.submit(
                    self.refresh_memory_graph,
                    force=True,
                    task_kind="graph-init",
                )
            if self.ical_feed_url():
                self.schedule_calendar_sync()
            if not env_truthy("DADBOT_DISABLE_PROACTIVE_HEARTBEAT", default=False):
                self.background_tasks.submit(
                    self._run_proactive_heartbeat_loop,
                    task_kind="proactive-heartbeat",
                )
        self._lifecycle_state = DadBotLifecycleState.READY

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

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
        self._lifecycle_state = DadBotLifecycleState.UNINITIALIZED
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

    # ------------------------------------------------------------------
    # Runtime lifecycle
    # ------------------------------------------------------------------

    @property
    def lifecycle_state(self) -> DadBotLifecycleState:
        """Queryable lifecycle state — useful for health checks and diagnostics."""
        return getattr(self, "_lifecycle_state", DadBotLifecycleState.UNINITIALIZED)

    def _guard_ready(self) -> None:
        """Raise RuntimeError if the bot has not yet reached READY state.

        Call from turn-entry methods that must not run during boot or after
        shutdown to produce a clear, actionable error rather than an obscure
        AttributeError from an un-wired manager.
        """
        state = self.lifecycle_state
        if state == DadBotLifecycleState.SHUTDOWN:
            raise RuntimeError(
                "DadBot has been shut down and cannot process new requests."
            )
        if state != DadBotLifecycleState.READY:
            raise RuntimeError(
                f"DadBot is not ready yet (current lifecycle state: {state!r}). "
                "Wait until boot completes before sending messages."
            )

    def should_start_background_tasks(self) -> bool:
        """Return True when background tasks should start on boot.

        Returns False under pytest to avoid races with deterministic tests
        and keep temporary SQLite files closed on Windows cleanup.
        """
        return "PYTEST_CURRENT_TEST" not in os.environ

    def shutdown(self) -> None:
        self._lifecycle_state = DadBotLifecycleState.SHUTDOWN
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
                    import logging
                    logging.getLogger(__name__).info(
                        "Final memory store flush during shutdown failed: %s", exc
                    )

    def _run_proactive_heartbeat_loop(self) -> None:
        interval_seconds = max(
            300,
            int(getattr(self, "_proactive_heartbeat_interval_seconds", 3600) or 3600),
        )
        try:
            self.maintenance_scheduler.run_proactive_heartbeat()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).info("Proactive heartbeat startup tick failed: %s", exc)

        while not self.background_tasks.wait_for_shutdown(interval_seconds):
            try:
                self.maintenance_scheduler.run_proactive_heartbeat()
            except Exception as exc:
                import logging
                logging.getLogger(__name__).info("Proactive heartbeat tick failed: %s", exc)

    # ------------------------------------------------------------------
    # Ollama and model utilities (static)
    # ------------------------------------------------------------------

    @staticmethod
    def ollama_retryable_errors() -> tuple:
        base = (ConnectionError, TimeoutError, OSError)
        if ollama is None:
            return base
        return (*base, ollama.RequestError, ollama.ResponseError)

    @staticmethod
    def ollama_error_summary(exc) -> str:
        if exc is None:
            return "Unknown Ollama error"
        details = []
        if hasattr(exc, "status_code"):
            details.append(f"status={exc.status_code}")
        elif hasattr(exc, "status"):
            details.append(f"status={exc.status}")
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
