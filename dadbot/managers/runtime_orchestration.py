from __future__ import annotations

import uuid
from collections import Counter
from concurrent.futures import Future

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.models import BackgroundTaskRecord
from dadbot_system import EventType


class RuntimeOrchestrationManager:
	"""Owns background task bookkeeping and async wrappers around durable runtime work."""

	NONCRITICAL_TASK_KINDS = {
		"post-turn-maintenance",
		"graph-init",
		"memory-graph-refresh",
		"durable-synthesis",
	}

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	@staticmethod
	def _normalize_task_status(status):
		normalized = str(status or "unknown").strip().lower() or "unknown"
		return normalized if normalized in {"queued", "running", "completed", "failed", "unknown"} else "unknown"

	def _normalize_background_task_record(self, payload):
		normalized_payload = {
			"task_id": str(payload.get("task_id") or "").strip(),
			"session_id": str(payload.get("session_id") or "").strip(),
			"task_kind": str(payload.get("task_kind") or "background").strip() or "background",
			"status": self._normalize_task_status(payload.get("status")),
			"metadata": dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
			"created_at": str(payload.get("created_at") or "").strip() or None,
			"updated_at": str(payload.get("updated_at") or "").strip() or None,
			"queued_at": str(payload.get("queued_at") or "").strip() or None,
			"started_at": str(payload.get("started_at") or "").strip() or None,
			"completed_at": str(payload.get("completed_at") or "").strip() or None,
			"failed_at": str(payload.get("failed_at") or "").strip() or None,
			"error": str(payload.get("error") or "").strip(),
		}
		return BackgroundTaskRecord.model_validate(normalized_payload).model_dump(mode="python")

	def record_background_task(self, task_id, *, task_kind, status, metadata=None, error=""):
		task_store = getattr(self.bot.runtime_state_container, "store", None)
		payload = {}
		if task_store is not None:
			payload = dict(task_store.load_task(task_id) or {})

		metadata_payload = dict(payload.get("metadata") or {})
		metadata_payload.update(dict(metadata or {}))
		timestamp = self.bot.runtime_timestamp()
		payload.update(
			{
				"task_id": task_id,
				"session_id": self.bot.runtime_state_container.session_id,
				"task_kind": str(task_kind or "background"),
				"status": str(status or "unknown"),
				"metadata": metadata_payload,
				"updated_at": timestamp,
			}
		)
		payload.setdefault("created_at", timestamp)
		if status == "queued":
			payload.setdefault("queued_at", timestamp)
		elif status == "running":
			payload.setdefault("started_at", timestamp)
		elif status == "completed":
			payload["completed_at"] = timestamp
			payload["error"] = ""
		elif status == "failed":
			payload["failed_at"] = timestamp
			payload["error"] = str(error or "").strip()

		payload = self._normalize_background_task_record(payload)

		if task_store is not None:
			task_store.save_task(task_id, payload)
		if task_id not in self.bot._background_task_ids:
			self.bot._background_task_ids.append(task_id)
		self.bot.runtime_state_container.record_event(
			EventType.STATE_UPDATED,
			{
				"reason": f"background_task.{payload['status']}",
				"task_id": task_id,
				"task_kind": payload["task_kind"],
				"status": payload["status"],
				"metadata": metadata_payload,
				"error": payload.get("error", ""),
			},
		)
		return payload

	def background_task_snapshot(self, limit=8):
		task_store = getattr(self.bot.runtime_state_container, "store", None)
		task_payloads = []
		if task_store is not None:
			for task_id in self.bot._background_task_ids:
				payload = task_store.load_task(task_id)
				if isinstance(payload, dict):
					task_payloads.append(self._normalize_background_task_record(payload))

		counts = Counter(str(payload.get("status") or "unknown") for payload in task_payloads)
		recent = []
		for payload in reversed(task_payloads[-max(0, int(limit or 0)) :]):
			recent.append(
				{
					"task_id": payload.get("task_id", ""),
					"task_kind": payload.get("task_kind", "background"),
					"status": payload.get("status", "unknown"),
					"updated_at": payload.get("updated_at"),
					"error": payload.get("error", ""),
					"metadata": dict(payload.get("metadata") or {}),
				}
			)

		return {
			"tracked": len(task_payloads),
			"queued": counts.get("queued", 0),
			"running": counts.get("running", 0),
			"completed": counts.get("completed", 0),
			"failed": counts.get("failed", 0),
			"recent": recent,
		}

	def submit_background_task(self, func, *args, task_kind="background", metadata=None, **kwargs):
		health = self.bot.current_runtime_health_snapshot(
			log_warnings=False,
			persist=True,
			max_age_seconds=120,
		)
		effective_limit = self.bot.adaptive_background_worker_limit(health)
		task_kind_name = str(task_kind or "background")
		task_metadata = dict(metadata or {})
		task_metadata.setdefault("health_level", health.get("level", "green"))
		task_metadata.setdefault("adaptive_worker_limit", effective_limit)
		running_tasks = int(self.background_task_snapshot(limit=8).get("running", 0) or 0)

		def _completed_future(payload):
			future = Future()
			future.set_result(payload)
			return future

		if task_kind_name in self.NONCRITICAL_TASK_KINDS and self.bot.should_delay_noncritical_maintenance(health):
			task_id = uuid.uuid4().hex
			task_metadata.update(
				{
					"deferred_by_health": True,
					"deferred_reason": "high-pressure runtime",
				}
			)
			self.record_background_task(task_id, task_kind=task_kind_name, status="queued", metadata=task_metadata)
			self.record_background_task(task_id, task_kind=task_kind_name, status="completed", metadata=task_metadata)
			return _completed_future({"deferred": True, "task_id": task_id})

		if self.bot.LIGHT_MODE and task_kind_name in self.NONCRITICAL_TASK_KINDS and running_tasks >= effective_limit:
			task_id = uuid.uuid4().hex
			task_metadata.update(
				{
					"deferred_by_health": True,
					"deferred_reason": "adaptive worker throttle",
				}
			)
			self.record_background_task(task_id, task_kind=task_kind_name, status="queued", metadata=task_metadata)
			self.record_background_task(task_id, task_kind=task_kind_name, status="completed", metadata=task_metadata)
			return _completed_future({"deferred": True, "task_id": task_id})

		task_id = uuid.uuid4().hex
		self.record_background_task(task_id, task_kind=task_kind, status="queued", metadata=task_metadata)

		def run_tracked_task():
			self.record_background_task(task_id, task_kind=task_kind, status="running", metadata=task_metadata)
			try:
				result = func(*args, **kwargs)
			except Exception as exc:
				self.record_background_task(
					task_id,
					task_kind=task_kind,
					status="failed",
					metadata=task_metadata,
					error=self.bot.ollama_error_summary(exc),
				)
				raise
			self.record_background_task(task_id, task_kind=task_kind, status="completed", metadata=task_metadata)
			return result

		future = self.bot.background_tasks.submit(run_tracked_task)
		future.dadbot_task_id = task_id
		future.dadbot_task_kind = str(task_kind or "background")
		return future

	def persist_conversation_async(self):
		snapshot = self.bot.snapshot_session_state()
		return self.submit_background_task(
			self.bot.persist_conversation_snapshot,
			snapshot,
			task_kind="conversation-persist",
			metadata={
				"history_messages": len(snapshot.get("history", [])),
				"turn_count": self.bot.session_turn_count(),
			},
		)


__all__ = ["RuntimeOrchestrationManager"]
