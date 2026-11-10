from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dadbot.core.execution_contract import SovereignContext, TurnDelivery, live_turn_request

from .contracts import (
    DEFAULT_TENANT_ID,
    ChatRequest,
    ChatResponse,
    EventEnvelope,
    EventType,
    ServiceConfig,
    TelemetrySettings,
    WorkerResult,
    WorkerTask,
)
from .execution_identity import ExecutionIdentity, require_tenant_id
from .runtime_signals import configure_logging, configure_tracing, shutdown_tracing, start_span
from .state import StateStore

logger = logging.getLogger(__name__)

AUTHORITY_UNAVAILABLE = "Delegated execution authority is unavailable; authenticated execution denied"
WORKER_AUTHORITY_STARTING = 0
WORKER_AUTHORITY_AVAILABLE = 1
WORKER_AUTHORITY_QUARANTINED = 2
WORKER_AUTHORITY_UNAVAILABLE = 3
AUTHORITY_UNAVAILABLE_AFTER_FAILURES = 3
AUTHORITY_RETRY_INITIAL_SECONDS = 0.1
AUTHORITY_RETRY_MAX_SECONDS = 1.6


@dataclass(slots=True)
class _AuthorityPollState:
    consecutive_failures: int = 0

    @property
    def status(self) -> int:
        if self.consecutive_failures == 0:
            return WORKER_AUTHORITY_AVAILABLE
        if self.consecutive_failures >= AUTHORITY_UNAVAILABLE_AFTER_FAILURES:
            return WORKER_AUTHORITY_UNAVAILABLE
        return WORKER_AUTHORITY_QUARANTINED

    def record(self, available: bool) -> float:
        if available:
            self.consecutive_failures = 0
            return 0.0
        self.consecutive_failures += 1
        exponent = min(self.consecutive_failures - 1, 30)
        return min(AUTHORITY_RETRY_INITIAL_SECONDS * (2**exponent), AUTHORITY_RETRY_MAX_SECONDS)


def _coerce_turn_result(response_obj: object) -> tuple[str, bool]:
    if hasattr(response_obj, "as_result") and callable(getattr(response_obj, "as_result", None)):
        result = response_obj.as_result()
        if isinstance(result, tuple) and len(result) >= 2:
            return str(result[0] or ""), bool(result[1])
    if isinstance(response_obj, tuple) and len(response_obj) >= 2:
        return str(response_obj[0] or ""), bool(response_obj[1])
    reply = str(getattr(response_obj, "reply", "") or "")
    should_end = bool(getattr(response_obj, "should_end", False))
    return reply, should_end


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int
    backoff_seconds: float
    timeout_seconds: float


class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_timeout_seconds: float):
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.failure_count = 0
        self.opened_at = 0.0

    def allow_request(self) -> bool:
        if self.failure_count < self.failure_threshold:
            return True
        if time.monotonic() - self.opened_at >= self.reset_timeout_seconds:
            self.failure_count = 0
            self.opened_at = 0.0
            return True
        return False

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = 0.0

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.monotonic()


class LocalMultiprocessBroker:
    def __init__(self, max_queue_size: int = 256):
        context = mp.get_context("spawn")
        self.request_queue = context.Queue(maxsize=max_queue_size)
        self.result_queue = context.Queue(maxsize=max_queue_size)

    def enqueue(self, task: WorkerTask) -> None:
        self.request_queue.put(task)

    def get_result_nowait(self) -> WorkerResult | None:
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def get_result(self, timeout: float | None = None) -> WorkerResult | None:
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class DadBotTaskProcessor:
    def __init__(self, config: ServiceConfig, bot_factory=None):
        self.config = config
        self._bot_factory = bot_factory
        self._bots: dict[tuple[str, str, str], object] = {}
        self._runtime_state_store: StateStore | None = None
        self._session_revocation_cursor: int | None = None
        self._session_revocation_lock = threading.Lock()
        self._session_revocation_unavailable = False
        self._circuit = CircuitBreaker(
            config.workers.circuit_breaker_failures,
            config.workers.circuit_breaker_reset_seconds,
        )

    def poll_session_revocations(self) -> bool:
        if not self.config.security.auth_required:
            return True
        if not self.config.security.authority_sqlite_path:
            self._bots.clear()
            if not self._session_revocation_unavailable:
                logger.warning("Worker cache invalidation authority path is not configured")
            self._session_revocation_unavailable = True
            return False
        from .delegation import AuthorityUnavailableError, SQLiteDelegationStore

        try:
            store = SQLiteDelegationStore(
                self.config.security.authority_sqlite_path,
                self.config.security.token_secret,
                issuer=self.config.security.token_issuer,
            )
            with self._session_revocation_lock:
                if self._session_revocation_cursor is None:
                    self._session_revocation_cursor = store.session_revocation_cursor()
                else:
                    revocations = store.session_revocations_after(self._session_revocation_cursor)
                    for revision, tenant_id, session_id in revocations:
                        self._evict_session_bots(tenant_id, session_id)
                        self._session_revocation_cursor = revision
        except (AuthorityUnavailableError, ValueError) as exc:
            self._bots.clear()
            if not self._session_revocation_unavailable:
                logger.warning("Worker cache invalidation authority unavailable: %s", exc)
            self._session_revocation_unavailable = True
            return False
        else:
            if self._session_revocation_unavailable:
                logger.info("Worker cache invalidation authority recovered")
            self._session_revocation_unavailable = False
            return True

    def _evict_session_bots(self, tenant_id: str, session_id: str) -> None:
        for key in tuple(self._bots):
            if key[0] == tenant_id and key[1] == session_id:
                del self._bots[key]

    def _get_bot(self, session_id: str, requested_model: str, *, tenant_id: str = DEFAULT_TENANT_ID):
        model_name = requested_model or self.config.default_model
        normalized_tenant_id = require_tenant_id(tenant_id)
        key = (normalized_tenant_id, session_id, model_name)
        bot = self._bots.get(key)
        if bot is None:
            if self._bot_factory is not None:
                bot = self._bot_factory(model_name=model_name, tenant_id=normalized_tenant_id)
            else:
                from Dad import DadBot

                bot_kwargs: dict[str, Any] = {
                    "model_name": model_name,
                    "tenant_id": normalized_tenant_id,
                }
                dependencies: dict[str, Any] = {
                    "storage_security_mode": (
                        "AUTHENTICATED" if self.config.security.auth_required else "TRUSTED_LOCAL"
                    ),
                }
                from dadbot.runtime.env_loader import is_production_environment

                if is_production_environment():
                    from dadbot.app_runtime import build_runtime_state_bundle, build_service_state_store

                    if self._runtime_state_store is None:
                        self._runtime_state_store = build_service_state_store(self.config)
                    dependencies["runtime_state_bundle"] = build_runtime_state_bundle(
                            self._runtime_state_store,
                            tenant_id=normalized_tenant_id,
                            planner_debug_factory=DadBot.default_planner_debug_state,
                        )
                bot_kwargs["dependency_registry"] = dependencies
                bot = DadBot(**bot_kwargs)
            if getattr(getattr(bot, "config", None), "tenant_id", None) != normalized_tenant_id:
                raise ValueError("worker bot tenant identity mismatch")
            if hasattr(bot, "ensure_ollama_ready"):
                try:
                    ready = bool(bot.ensure_ollama_ready())
                except Exception as exc:
                    logger.warning("Worker model readiness check failed for %s: %s", model_name, exc)
                else:
                    if not ready:
                        logger.warning(
                            "Worker preflight could not confirm model %s; continuing so DadBot can use its runtime fallback model selection.",
                            model_name,
                        )
            self._bots[key] = bot
        if getattr(getattr(bot, "config", None), "tenant_id", None) != normalized_tenant_id:
            raise ValueError("worker bot tenant identity mismatch")
        return bot

    def _build_request_attachments(self, request: ChatRequest) -> list[dict]:
        attachments = []
        for attachment in request.attachments:
            payload = attachment.to_dict()
            if attachment.type == "image" and attachment.data_b64:
                payload["image_b64"] = attachment.data_b64
            attachments.append(payload)
        return attachments

    @staticmethod
    def _request_policy_metadata(request: ChatRequest) -> dict[str, object] | None:
        metadata = dict(request.metadata or {})
        service_policy = dict(metadata.get("service_policy") or {})
        if not service_policy:
            return None
        principal = dict(metadata.get("auth") or {})
        if principal:
            service_policy.setdefault("principal", principal)
        return service_policy

    async def _execute_request_async(self, bot, request: ChatRequest) -> tuple[str, bool]:
        if self.config.security.auth_required:
            from .delegation import require_live_authority

            require_live_authority()
        attachments = self._build_request_attachments(request)
        request_policy = self._request_policy_metadata(request)
        previous_policy = getattr(bot, "_service_request_policy", None)
        if request_policy is not None:
            bot._service_request_policy = request_policy
        try:
            try:
                execute_turn = bot.execute_turn
            except AttributeError as exc:
                raise RuntimeError("service shell requires bot.execute_turn") from exc
            if not callable(execute_turn):
                raise RuntimeError("service shell requires bot.execute_turn")

            turn_request = live_turn_request(
                request.user_input,
                attachments=attachments,
                delivery=TurnDelivery.ASYNC,
                metadata={"source_mode": True} if request.metadata.get("source_mode") is True else {},
                context=SovereignContext(
                    session_id=str(request.session_id or "default"),
                    tenant_id=str(request.tenant_id or DEFAULT_TENANT_ID),
                    request_id=str(request.request_id or ""),
                    trace_id=str(request.request_id or ""),
                    policy_scope="dadbot_system.worker",
                ),
            )
            response_obj = execute_turn(turn_request)
            if inspect.isawaitable(response_obj):
                response_obj = await response_obj
            return _coerce_turn_result(response_obj)
        finally:
            if request_policy is not None:
                bot._service_request_policy = previous_policy

    def process(self, task: WorkerTask) -> WorkerResult:
        from .delegation import AuthorityUnavailableError, SQLiteDelegationStore, _execution_authority
        from .security import AuthorizationError

        if not self.poll_session_revocations():
            return WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
                status="failed",
                error=AUTHORITY_UNAVAILABLE,
                retryable=True,
            )
        store = None
        if self.config.security.auth_required:
            if self.config.security.authority_sqlite_path:
                try:
                    store = SQLiteDelegationStore(
                        self.config.security.authority_sqlite_path,
                        self.config.security.token_secret,
                        issuer=self.config.security.token_issuer,
                    )
                    store.verify(task.delegation, task, operation="execute")
                except AuthorityUnavailableError as exc:
                    logger.warning(
                        "Task %s could not verify delegation while authority is unavailable: %s",
                        task.task_id,
                        exc,
                    )
                    return WorkerResult(
                        task_id=task.task_id,
                        session_id=task.session_id,
                        request_id=task.request.request_id,
                        tenant_id=task.tenant_id,
                        status="failed",
                        error=AUTHORITY_UNAVAILABLE,
                        retryable=True,
                    )
                except (AuthorizationError, ValueError) as exc:
                    self._evict_session_bots(task.tenant_id, task.session_id)
                    logger.warning("Task %s failed delegation verification: %s", task.task_id, exc)
                    store = None
            if store is None:
                self._evict_session_bots(task.tenant_id, task.session_id)
                logger.error("%s (task_id=%s)", AUTHORITY_UNAVAILABLE, task.task_id)
                return WorkerResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    request_id=task.request.request_id,
                    tenant_id=task.tenant_id,
                    status="failed",
                    error=AUTHORITY_UNAVAILABLE,
                )
        try:
            identity = ExecutionIdentity(
                tenant_id=task.tenant_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                task_id=task.task_id,
            )
            identity.require_match(
                tenant_id=task.request.tenant_id,
                session_id=task.request.session_id,
                request_id=task.request.request_id,
            )
        except ValueError as exc:
            logger.warning("Rejected worker task with invalid identity: %s", exc)
            return WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
                status="failed",
                error=str(exc),
            )
        def finish_failure(error: str) -> WorkerResult:
            result = WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
                status="failed",
                error=error,
            )
            if store is not None:
                try:
                    result.authority_proof = store.sign_result(task, result)
                except (AuthorizationError, AuthorityUnavailableError) as exc:
                    logger.warning("Task %s lost authority before failure publication: %s", task.task_id, exc)
                    result.error = AUTHORITY_UNAVAILABLE
            return result

        if not self._circuit.allow_request():
            error = "Circuit breaker is open for worker requests"
            return finish_failure(error)

        policy = RetryPolicy(
            max_attempts=max(1, self.config.workers.max_retries + 1),
            backoff_seconds=self.config.workers.retry_backoff_seconds,
            timeout_seconds=self.config.workers.task_timeout_seconds,
        )

        last_error = ""
        for attempt in range(1, policy.max_attempts + 1):
            with start_span(
                "worker.process_task", task_id=task.task_id, request_id=task.request.request_id, attempt=attempt
            ):
                authority_token = None
                try:
                    if store is not None:
                        store.verify(task.delegation, task, operation="execute")
                        authority_token = _execution_authority.set((store, task))
                    bot = self._get_bot(task.session_id, task.request.requested_model, tenant_id=task.tenant_id)
                    if store is not None:
                        ledger = getattr(bot, "_execution_ledger", None)
                        mark_ledger = getattr(ledger, "require_authenticated_authority", None)
                        if callable(mark_ledger):
                            mark_ledger()
                        memory = getattr(bot, "memory_manager", None)
                        storage = getattr(memory, "_storage", None)
                        mark_storage = getattr(storage, "require_authenticated_authority", None)
                        if callable(mark_storage):
                            mark_storage()
                    if task.session_state:
                        session_snapshot = dict(task.session_state)
                        # Memory is tenant-scoped and durable; the session copy may predate the task's side effects.
                        session_snapshot.pop("memory_store", None)
                        bot.load_session_state_snapshot(session_snapshot)

                    bot._last_tool_action_audit = []
                    reply, should_end = asyncio.run(
                        asyncio.wait_for(
                            self._execute_request_async(bot, task.request),
                            timeout=policy.timeout_seconds,
                        )
                    )

                    self._circuit.record_success()
                    response = ChatResponse(
                        session_id=task.session_id,
                        request_id=task.request.request_id,
                        reply=reply,
                        tenant_id=task.tenant_id,
                        should_end=should_end,
                        active_model=str(getattr(bot, "ACTIVE_MODEL", "") or ""),
                        status="completed",
                    )
                    session_state = bot.snapshot_session_state()
                    action_audit = getattr(bot, "_last_tool_action_audit", [])
                    if isinstance(action_audit, list):
                        session_state["last_tool_actions"] = [
                            dict(action) for action in action_audit[-50:] if isinstance(action, dict)
                        ]
                    result = WorkerResult(
                        task_id=task.task_id,
                        session_id=task.session_id,
                        request_id=task.request.request_id,
                        status="completed",
                        tenant_id=task.tenant_id,
                        session_state=session_state,
                        response=response,
                    )
                    if store is not None:
                        result.authority_proof = store.sign_result(task, result)
                    return result
                except (AuthorizationError, AuthorityUnavailableError) as exc:
                    self._evict_session_bots(task.tenant_id, task.session_id)
                    logger.warning("Task %s lost execution authority: %s", task.task_id, exc)
                    return WorkerResult(
                        task_id=task.task_id,
                        session_id=task.session_id,
                        request_id=task.request.request_id,
                        tenant_id=task.tenant_id,
                        status="failed",
                        error=AUTHORITY_UNAVAILABLE,
                        retryable=isinstance(exc, AuthorityUnavailableError),
                    )
                except TimeoutError:
                    last_error = f"Worker timed out after {policy.timeout_seconds:.1f}s"
                except Exception as exc:
                    last_error = str(exc)
                finally:
                    if authority_token is not None:
                        _execution_authority.reset(authority_token)

                self._circuit.record_failure()
                if attempt < policy.max_attempts:
                    time.sleep(policy.backoff_seconds * attempt)

        return finish_failure(last_error)


def worker_process_main(  # noqa: PLR0913
    config_payload: dict,
    request_queue,
    result_queue,
    event_queue=None,
    telemetry_initializer: Callable[[TelemetrySettings], Any] | None = None,
    stop_event=None,
    authority_status=None,
    worker_index: int = 0,
    *,
    bot_factory: Callable[..., Any] | None = None,
):
    config = ServiceConfig.from_dict(config_payload)
    configure_logging(config.telemetry)
    if telemetry_initializer is None:
        configure_tracing(config.telemetry)
    else:
        telemetry_initializer(config.telemetry)
    processor = DadBotTaskProcessor(config, bot_factory=bot_factory)
    worker_name = mp.current_process().name

    if event_queue is not None:
        event_queue.put(
            EventEnvelope(
                session_id="system",
                event_type=EventType.WORKER_STARTED,
                tenant_id=DEFAULT_TENANT_ID,
                payload={"worker": worker_name},
            )
        )

    try:
        poll_state = _AuthorityPollState()
        pending_task = None
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if config.security.auth_required:
                authority_available = processor.poll_session_revocations()
                retry_delay = poll_state.record(authority_available)
                if authority_status is not None:
                    authority_status[worker_index] = poll_state.status
                if not authority_available:
                    if stop_event is not None:
                        stop_event.wait(retry_delay)
                    else:
                        time.sleep(retry_delay)
                    continue
                if pending_task is None:
                    try:
                        task = request_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                else:
                    task = pending_task
            else:
                try:
                    task = request_queue.get(timeout=0.1 if stop_event is not None else None)
                except queue.Empty:
                    continue
            if task is None:
                break
            result = processor.process(task)
            if config.security.auth_required and result.retryable:
                pending_task = task
                retry_delay = poll_state.record(False)
                if authority_status is not None:
                    authority_status[worker_index] = poll_state.status
                if stop_event is not None:
                    stop_event.wait(retry_delay)
                else:
                    time.sleep(retry_delay)
                continue
            pending_task = None
            result_queue.put(result)
    finally:
        if config.telemetry.otel_enabled:
            shutdown_tracing()

    if event_queue is not None:
        event_queue.put(
            EventEnvelope(
                session_id="system",
                event_type=EventType.WORKER_STOPPED,
                tenant_id=DEFAULT_TENANT_ID,
                payload={"worker": worker_name},
            )
        )


class WorkerProcessManager:
    def __init__(
        self,
        broker: LocalMultiprocessBroker,
        config: ServiceConfig,
        event_queue=None,
        *,
        telemetry_initializer: Callable[[TelemetrySettings], Any] | None = None,
    ):
        self.broker = broker
        self.config = config
        self.event_queue = event_queue
        self._telemetry_initializer = telemetry_initializer
        self._processes: list[mp.Process] = []
        self._stop_event = None
        self._authority_status = None

    def start(self) -> None:
        if self._processes:
            return
        context = mp.get_context("spawn")
        worker_count = max(1, self.config.workers.worker_count)
        self._stop_event = context.Event()
        if self.config.security.auth_required:
            self._authority_status = context.Array("b", worker_count, lock=True)
        for index in range(worker_count):
            process = context.Process(
                target=worker_process_main,
                args=(
                    self.config.to_dict(),
                    self.broker.request_queue,
                    self.broker.result_queue,
                    self.event_queue,
                    self._telemetry_initializer,
                    self._stop_event,
                    self._authority_status,
                    index,
                ),
                name=f"dadbot-worker-{index + 1}",
            )
            process.start()
            self._processes.append(process)

    def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if not self.config.security.auth_required:
            for _ in self._processes:
                self.broker.request_queue.put(None)
        for process in self._processes:
            process.join(timeout=5)
        self._processes.clear()
        self._stop_event = None
        self._authority_status = None

    @property
    def worker_count(self) -> int:
        return len(self._processes)

    @property
    def is_ready(self) -> bool:
        return self.readiness_status == "available"

    @property
    def readiness_status(self) -> str:
        expected = max(1, self.config.workers.worker_count)
        processes_ready = len(self._processes) == expected and all(
            process.is_alive() and process.exitcode is None for process in self._processes
        )
        if not processes_ready:
            return "unavailable"
        if self.config.security.auth_required:
            if self._authority_status is None:
                return "starting"
            statuses = [int(self._authority_status[index]) for index in range(expected)]
            if any(status == WORKER_AUTHORITY_UNAVAILABLE for status in statuses):
                return "unavailable"
            if any(status == WORKER_AUTHORITY_QUARANTINED for status in statuses):
                return "quarantined"
            if all(status == WORKER_AUTHORITY_AVAILABLE for status in statuses):
                return "available"
            return "starting"
        return "available"
