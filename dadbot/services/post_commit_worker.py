from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.failure_policy_engine import FailurePolicyRules
from dadbot.core.ledger_backend import FileWALLedgerBackend
from dadbot.core.memory_consolidation_outbox import (
    pending_memory_consolidation_work,
    record_memory_consolidation_completion,
)
from dadbot.core.post_commit_events import POST_COMMIT_READY, PostCommitEvent

logger = logging.getLogger(__name__)


class _PostCommitCapability:
    """Minimal capability surface granted to the post-commit background worker.

    Holds the post-commit callables and ledger reference needed for memory
    maintenance. It does not store a direct ``bot`` attribute.
    """

    __slots__ = (
        "_consolidate_source",
        "_forget",
        "_ledger",
        "_queue_semantic_index",
        "_tenant_id",
    )

    def __init__(self, bot: Any) -> None:
        memory_coordinator = getattr(bot, "memory_coordinator", None)
        self._consolidate_source = getattr(memory_coordinator, "consolidate_memories_from_source", None)
        self._forget = getattr(memory_coordinator, "apply_controlled_forgetting", None)
        memory_manager = getattr(bot, "memory_manager", None)
        self._queue_semantic_index = getattr(memory_manager, "queue_semantic_memory_index", None)
        candidate_ledger = getattr(bot, "_execution_ledger", None)
        self._ledger = candidate_ledger if isinstance(candidate_ledger, ExecutionLedger) else None
        self._tenant_id = str(getattr(getattr(bot, "config", None), "tenant_id", "") or "")
        # bot reference is intentionally NOT stored

    def supports_durable_consolidation(self) -> bool:
        ledger = self._ledger
        backend = None if ledger is None else ledger._backend
        return bool(
            isinstance(backend, FileWALLedgerBackend)
            and backend._security_mode != "AUTHENTICATED"
            and self._tenant_id
        )

    def pending_consolidation_work(self) -> list[dict[str, Any]]:
        ledger = self._ledger
        if ledger is None or not self.supports_durable_consolidation():
            return []
        return pending_memory_consolidation_work(
            ledger.read(),
            tenant_id=self._tenant_id,
        )

    def consolidate_source(self, work: dict[str, Any]) -> str:
        if not callable(self._consolidate_source):
            raise RuntimeError("Durable memory consolidation capability is unavailable")
        ledger = self._ledger
        if ledger is None:
            raise RuntimeError("Durable memory consolidation ledger is unavailable")
        current = ledger.replay_tenant_memory(work["tenant_id"])
        if work["work_id"] in current.operations:
            return "succeeded"
        source = ledger.replay_tenant_memory_at_revision(
            work["tenant_id"],
            work["source_revision"],
        )
        operation = source.operations.get(work["source_operation_id"])
        if operation is None:
            raise RuntimeError("Durable consolidation source operation is missing")
        result = self._consolidate_source(work, operation["state"])
        if not isinstance(result, str) or result not in {"succeeded", "skipped"}:
            raise RuntimeError("Durable consolidation returned an invalid status")
        return result

    def complete_consolidation(self, work: dict[str, Any], *, status: str) -> None:
        if self._ledger is None:
            raise RuntimeError("Durable memory consolidation ledger is unavailable")
        record_memory_consolidation_completion(self._ledger, work, status=status)

    def forget(self, *, turn_context: Any) -> None:
        if callable(self._forget):
            self._forget(turn_context=turn_context)

    def index_tool_result(self, *, turn_context: Any) -> None:
        if not callable(self._queue_semantic_index):
            return
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return
        entry = state.get("last_tool_result")
        if not isinstance(entry, dict):
            return
        result_text = str(entry.get("text") or "").strip()
        tool_name = str(entry.get("tool") or "tool_result").strip()
        if not result_text or len(result_text) < 10:
            return
        self._queue_semantic_index([{
            "summary": result_text,
            "category": f"tool_result:{tool_name}",
            "mood": "neutral",
        }])


class PostCommitWorker:
    """Background worker for post-commit reasoning and maintenance work.

    Holds a ``_PostCommitCapability`` rather than a direct ``bot`` reference.
    The event bus reference is stored separately.
    """

    _RETRY_BASE_DELAY_MS = 1_000.0
    _RETRY_MAX_DELAY_SECONDS = 30.0
    _MAX_RETRY_ATTEMPT = 32
    _QUEUE_POLL_SECONDS = 0.2
    _IDLE_POLL_SECONDS = 5.0

    def __init__(self, bot: Any, *, max_queue_size: int = 200) -> None:
        self._capability = _PostCommitCapability(bot)
        self._max_queue_size = max_queue_size
        self._stop_event = Event()
        self._subscriber: Queue[Any] | None = None
        self._thread: Thread | None = None
        self._initialize_retry_tracking()
        # Store only the event bus — not bot itself
        self._event_bus = getattr(bot, "_runtime_event_bus", None)
        self.start()

    def _initialize_retry_tracking(self) -> None:
        self._retry_attempts: dict[str, int] = {}
        self._retry_deadlines: dict[str, float] = {}
        self._outbox_retry_attempt = 0
        self._outbox_retry_deadline: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        subscribe = getattr(self._event_bus, "subscribe", None)
        if callable(subscribe):
            subscriber = subscribe(max_queue_size=self._max_queue_size)
            if isinstance(subscriber, Queue):
                self._subscriber = subscriber
            else:
                logger.warning("PostCommitWorker event bus subscribe() did not return a queue")
        else:
            logger.warning("PostCommitWorker runtime event bus missing subscribe()")
        if self._subscriber is None and not self._capability.supports_durable_consolidation():
            return
        self._thread = Thread(
            target=self._run,
            name="dadbot-post-commit-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        self._drain_pending_consolidation_work()
        while not self._stop_event.is_set():
            timeout = self._next_wait_seconds()
            if self._subscriber is None:
                if self._stop_event.wait(timeout):
                    return
                self._drain_pending_consolidation_work()
                continue
            try:
                event = self._subscriber.get(timeout=timeout)
            except Empty:
                if self._has_due_retry():
                    self._drain_pending_consolidation_work()
                continue
            if getattr(event, "event_type", "") != POST_COMMIT_READY:
                continue
            self._handle_post_commit_event(event)

    def _next_wait_seconds(self) -> float:
        idle_timeout = self._IDLE_POLL_SECONDS if self._subscriber is None else self._QUEUE_POLL_SECONDS
        deadlines = list(self._retry_deadlines.values())
        if self._outbox_retry_deadline is not None:
            deadlines.append(self._outbox_retry_deadline)
        if not deadlines:
            return idle_timeout
        return max(0.0, min(idle_timeout, min(deadlines) - time.monotonic()))

    def _has_due_retry(self) -> bool:
        now = time.monotonic()
        return (
            self._outbox_retry_deadline is not None
            and self._outbox_retry_deadline <= now
        ) or any(deadline <= now for deadline in self._retry_deadlines.values())

    @classmethod
    def _retry_delay_seconds(cls, attempt: int) -> float:
        return FailurePolicyRules.compute_retry_delay(
            attempt=min(max(1, attempt), cls._MAX_RETRY_ATTEMPT),
            base_delay_ms=cls._RETRY_BASE_DELAY_MS,
            max_delay_seconds=cls._RETRY_MAX_DELAY_SECONDS,
            jitter_ratio=0.2,
        )

    def _schedule_work_retry(self, work_id: str) -> None:
        attempt = min(
            self._retry_attempts.get(work_id, 0) + 1,
            self._MAX_RETRY_ATTEMPT,
        )
        self._retry_attempts[work_id] = attempt
        self._retry_deadlines[work_id] = time.monotonic() + self._retry_delay_seconds(attempt)

    def _schedule_outbox_retry(self) -> None:
        self._outbox_retry_attempt = min(
            self._outbox_retry_attempt + 1,
            self._MAX_RETRY_ATTEMPT,
        )
        self._outbox_retry_deadline = (
            time.monotonic() + self._retry_delay_seconds(self._outbox_retry_attempt)
        )

    def _handle_turn_outcome(self, work: dict[str, Any]) -> str:
        work_id = str(work["work_id"])
        outcome_status = work.get("turn_outcome_status")
        if outcome_status not in {"succeeded", "failed"}:
            self._retry_attempts.pop(work_id, None)
            self._retry_deadlines[work_id] = (
                time.monotonic() + self._IDLE_POLL_SECONDS
            )
            return "deferred"
        if outcome_status == "succeeded":
            return "ready"
        try:
            self._capability.complete_consolidation(work, status="skipped")
        except Exception as exc:  # noqa: BLE001
            self._schedule_work_retry(work_id)
            logger.warning(
                "Unable to acknowledge failed-turn consolidation work_id=%s: %s",
                work_id,
                exc,
            )
            return "failed"
        self._retry_attempts.pop(work_id, None)
        self._retry_deadlines.pop(work_id, None)
        return "skipped"

    def _process_eligible_work(self, work: dict[str, Any]) -> str:
        work_id = str(work["work_id"])
        retry_deadline = self._retry_deadlines.get(work_id)
        if retry_deadline is not None and retry_deadline > time.monotonic():
            return "deferred"
        try:
            status = self._capability.consolidate_source(work)
            self._capability.complete_consolidation(work, status=status)
        except Exception as exc:  # noqa: BLE001
            self._schedule_work_retry(work_id)
            logger.warning(
                "Durable memory consolidation remains pending work_id=%s: %s",
                work_id,
                exc,
            )
            return "failed"
        self._retry_attempts.pop(work_id, None)
        self._retry_deadlines.pop(work_id, None)
        return "succeeded"

    def _process_pending_work_items(self, work_items: list[dict[str, Any]]) -> str:
        pending_ids = {str(work["work_id"]) for work in work_items}
        for stale_id in set(self._retry_attempts).difference(pending_ids):
            self._retry_attempts.pop(stale_id, None)
            self._retry_deadlines.pop(stale_id, None)
        had_failure = False
        deferred = False
        for work in work_items:
            outcome = self._handle_turn_outcome(work)
            if outcome == "deferred":
                deferred = True
                continue
            if outcome == "failed":
                had_failure = True
                continue
            if outcome == "skipped":
                continue
            result = self._process_eligible_work(work)
            if result == "deferred":
                deferred = True
            elif result == "failed":
                had_failure = True
        if had_failure:
            return "failed"
        if deferred:
            return "deferred"
        return "succeeded"

    def _drain_pending_consolidation_work(self) -> str:
        now = time.monotonic()
        if self._outbox_retry_deadline is not None and self._outbox_retry_deadline > now:
            return "deferred"
        try:
            work_items = self._capability.pending_consolidation_work()
        except Exception as exc:  # noqa: BLE001
            self._schedule_outbox_retry()
            logger.error("Unable to read durable memory consolidation outbox: %s", exc)
            return "failed"
        self._outbox_retry_attempt = 0
        self._outbox_retry_deadline = None
        if not work_items:
            self._retry_attempts.clear()
            self._retry_deadlines.clear()
            return "skipped"
        return self._process_pending_work_items(work_items)

    @staticmethod
    def _append_memory_write_intent(
        turn_context: Any,
        *,
        op: str,
        status: str,
        error: str = "",
    ) -> None:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        if not isinstance(state, dict):
            return

        intent = {
            "op": str(op or ""),
            "status": str(status or ""),
            "error": str(error or ""),
            "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        intents = list(state.get("memory_write_intents") or [])
        intents.append(intent)
        state["memory_write_intents"] = intents

        summary = {
            "version": "1.0",
            "intent_count": len(intents),
            "intents": list(intents),
            "memory_retrieval_set_after_commit": list(state.get("memory_retrieval_set") or []),
        }
        state["memory_delta_summary"] = summary

        if isinstance(metadata, dict):
            metadata["memory_write_intents"] = list(intents)
            metadata["memory_delta_summary"] = dict(summary)

    def _handle_post_commit_event(self, event: PostCommitEvent) -> None:
        turn_context = getattr(event, "payload", {}).get("turn_context")
        self._append_memory_write_intent(
            turn_context,
            op="consolidate_memories",
            status="started",
        )
        consolidation_status = self._drain_pending_consolidation_work()
        self._append_memory_write_intent(
            turn_context,
            op="consolidate_memories",
            status=consolidation_status,
            error="Durable consolidation work remains pending"
            if consolidation_status == "failed"
            else "",
        )
        self._append_memory_write_intent(
            turn_context,
            op="apply_controlled_forgetting",
            status="started",
        )
        try:
            self._capability.forget(turn_context=turn_context)
            self._append_memory_write_intent(
                turn_context,
                op="apply_controlled_forgetting",
                status="succeeded",
            )
        except Exception as exc:  # noqa: BLE001
            self._append_memory_write_intent(
                turn_context,
                op="apply_controlled_forgetting",
                status="failed",
                error=str(exc),
            )
            logger.warning("Post-commit apply_controlled_forgetting failed (non-fatal): %s", exc)
        try:
            self._capability.index_tool_result(turn_context=turn_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Post-commit index_tool_result failed (non-fatal): %s", exc)
