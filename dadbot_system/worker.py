from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

from .contracts import (
    DEFAULT_TENANT_ID,
    ChatRequest,
    ChatResponse,
    EventEnvelope,
    EventType,
    ServiceConfig,
    WorkerResult,
    WorkerTask,
    normalize_tenant_id,
)
from .runtime_signals import configure_logging, start_span

logger = logging.getLogger(__name__)


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
        self._bots: dict[tuple[str, str], object] = {}
        self._circuit = CircuitBreaker(
            config.workers.circuit_breaker_failures,
            config.workers.circuit_breaker_reset_seconds,
        )

    def _get_bot(self, session_id: str, requested_model: str, *, tenant_id: str = DEFAULT_TENANT_ID):
        model_name = requested_model or self.config.default_model
        normalized_tenant_id = normalize_tenant_id(tenant_id)
        key = (normalized_tenant_id, session_id, model_name)
        bot = self._bots.get(key)
        if bot is None:
            if self._bot_factory is not None:
                try:
                    bot = self._bot_factory(model_name=model_name, tenant_id=normalized_tenant_id)
                except TypeError:
                    bot = self._bot_factory(model_name=model_name)
            else:
                from Dad import DadBot

                bot = DadBot(model_name=model_name, tenant_id=normalized_tenant_id)
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
        attachments = self._build_request_attachments(request)
        request_policy = self._request_policy_metadata(request)
        previous_policy = getattr(bot, "_service_request_policy", None)
        if request_policy is not None:
            bot._service_request_policy = request_policy
        try:
            if hasattr(bot, "process_user_message_async"):
                return await bot.process_user_message_async(request.user_input, attachments=attachments)
            return bot.process_user_message(request.user_input, attachments=attachments)
        finally:
            if request_policy is not None:
                bot._service_request_policy = previous_policy

    def process(self, task: WorkerTask) -> WorkerResult:
        if not self._circuit.allow_request():
            error = "Circuit breaker is open for worker requests"
            return WorkerResult(
                task_id=task.task_id,
                session_id=task.session_id,
                request_id=task.request.request_id,
                tenant_id=task.tenant_id,
                status="failed",
                error=error,
            )

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
                try:
                    bot = self._get_bot(task.session_id, task.request.requested_model, tenant_id=task.tenant_id)
                    if task.session_state:
                        bot.load_session_state_snapshot(task.session_state)

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
                    return WorkerResult(
                        task_id=task.task_id,
                        session_id=task.session_id,
                        request_id=task.request.request_id,
                        status="completed",
                        tenant_id=task.tenant_id,
                        session_state=bot.snapshot_session_state(),
                        response=response,
                    )
                except TimeoutError:
                    last_error = f"Worker timed out after {policy.timeout_seconds:.1f}s"
                except Exception as exc:
                    last_error = str(exc)

                self._circuit.record_failure()
                if attempt < policy.max_attempts:
                    time.sleep(policy.backoff_seconds * attempt)

        return WorkerResult(
            task_id=task.task_id,
            session_id=task.session_id,
            request_id=task.request.request_id,
            status="failed",
            tenant_id=task.tenant_id,
            error=last_error,
        )


def worker_process_main(config_payload: dict, request_queue, result_queue, event_queue=None):
    config = ServiceConfig.from_dict(config_payload)
    configure_logging(config.telemetry)
    processor = DadBotTaskProcessor(config)
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

    while True:
        task = request_queue.get()
        if task is None:
            break
        result = processor.process(task)
        result_queue.put(result)

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
    def __init__(self, broker: LocalMultiprocessBroker, config: ServiceConfig, event_queue=None):
        self.broker = broker
        self.config = config
        self.event_queue = event_queue
        self._processes: list[mp.Process] = []

    def start(self) -> None:
        if self._processes:
            return
        context = mp.get_context("spawn")
        for index in range(max(1, self.config.workers.worker_count)):
            process = context.Process(
                target=worker_process_main,
                args=(self.config.to_dict(), self.broker.request_queue, self.broker.result_queue, self.event_queue),
                name=f"dadbot-worker-{index + 1}",
            )
            process.start()
            self._processes.append(process)

    def shutdown(self) -> None:
        for _ in self._processes:
            self.broker.request_queue.put(None)
        for process in self._processes:
            process.join(timeout=5)
        self._processes.clear()

    @property
    def worker_count(self) -> int:
        return len(self._processes)
