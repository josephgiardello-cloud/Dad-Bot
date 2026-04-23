from __future__ import annotations

import base64
import copy
import gzip
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dadbot.contracts import DadBotContext, SupportsDadBotAccess


class ConversationPersistenceManager:
	"""Owns conversation persistence, snapshot rehydration, and session-log writes."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	def persist_conversation(self) -> None:
		if not self.bot.LIGHT_MODE:
			self.bot.refresh_session_summary(force=True)
			self.bot.relationship.reflect(force=True)
		chat_history = self.bot.conversation_history()
		self.bot.update_memory_store(chat_history)
		if not self.bot.LIGHT_MODE:
			self.bot.consolidate_memories()
		self.bot.archive_session_context(chat_history)
		if not self.bot.LIGHT_MODE:
			self.bot.refresh_relationship_timeline(force=True)
			self.bot.detect_life_patterns()
			self.bot.evolve_persona()
			self.bot.refresh_memory_graph()
		self.save_session_log(chat_history)

	def persist_conversation_snapshot(self, snapshot: dict) -> None:
		snapshot_payload = copy.deepcopy(dict(snapshot or {}))
		chat_history = list(snapshot_payload.get("history", []))
		if chat_history and chat_history[0].get("role") == "system":
			chat_history = chat_history[1:]
		if not chat_history:
			return

		previous_snapshot = self.bot.snapshot_session_state()
		self.bot.load_session_state_snapshot(snapshot_payload)

		try:
			self.bot.update_memory_store(chat_history)
			if not self.bot.LIGHT_MODE:
				self.bot.consolidate_memories()
			self.bot.archive_session_context(chat_history)
			if not self.bot.LIGHT_MODE:
				self.bot.refresh_relationship_timeline(force=True)
				self.bot.detect_life_patterns()
				self.bot.evolve_persona()
				self.bot.refresh_memory_graph()
			self.save_session_log(chat_history)
		finally:
			self.bot.load_session_state_snapshot(previous_snapshot)

	def save_session_log(self, history: list[dict]) -> None:
		payload = {
			"created_at": datetime.now().isoformat(timespec="seconds"),
			"tenant_id": self.bot.config.tenant_id,
			"model": self.bot.config.active_model,
			"embedding_model": self.bot.config.active_embedding_model,
			"session_summary": self.bot.session_summary,
			"relationship_state": self.bot.relationship_state(),
			"history": history,
		}
		timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		io_lock = getattr(self.bot, "_io_lock", None)
		if io_lock is None:
			if self.bot._tenant_document_store is not None:
				self.bot._tenant_document_store.save_session_state(f"session-log:{timestamp}", payload)
				return
			self.bot.config.session_log_dir.mkdir(parents=True, exist_ok=True)
			session_path = self.bot.config.session_log_dir / f"session-{timestamp}.json"
			self.bot.write_json_atomically(session_path, payload, backup=False)
			return

		with io_lock:
			if self.bot._tenant_document_store is not None:
				self.bot._tenant_document_store.save_session_state(f"session-log:{timestamp}", payload)
				return
			self.bot.SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
			session_path = self.bot.SESSION_LOG_DIR / f"session-{timestamp}.json"
			self.bot.write_json_atomically(session_path, payload, backup=False)

	def persist_graph_checkpoint(self, checkpoint: dict) -> None:
		payload = copy.deepcopy(dict(checkpoint or {}))
		if not payload:
			return
		payload["session_state"] = self.bot.snapshot_session_state()

		trace_id = str(payload.get("trace_id") or "unknown").strip() or "unknown"
		stage = str(payload.get("stage") or "stage").strip().replace(" ", "-") or "stage"
		status = str(payload.get("status") or "unknown").strip().replace(" ", "-") or "unknown"
		timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		binary_payload = gzip.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
		wrapped_payload = {
			"format": "gzip+pickle",
			"trace_id": trace_id,
			"stage": stage,
			"status": status,
			"created_at": datetime.now().isoformat(timespec="seconds"),
			"payload_b64": base64.b64encode(binary_payload).decode("ascii"),
		}

		if self.bot._tenant_document_store is not None:
			checkpoint_key = f"graph-checkpoint:{trace_id}:{timestamp}:{stage}:{status}"
			self.bot._tenant_document_store.save_session_state(checkpoint_key, wrapped_payload)
			self.bot._tenant_document_store.save_session_state(f"graph-checkpoint-latest:{trace_id}", wrapped_payload)
			self.bot._tenant_document_store.save_session_state("graph-checkpoint-latest", wrapped_payload)
			return

		checkpoint_dir = self.bot.SESSION_LOG_DIR / "graph_checkpoints"
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		checkpoint_path = checkpoint_dir / f"{timestamp}-{trace_id[:12]}-{stage}-{status}.bin"
		latest_path = checkpoint_dir / f"latest-{trace_id[:12]}.bin"
		latest_global_path = checkpoint_dir / "latest.bin"
		checkpoint_path.write_bytes(binary_payload)
		latest_path.write_bytes(binary_payload)
		latest_global_path.write_bytes(binary_payload)

		# Mirror graph checkpoints into an append-only turn-event stream so the
		# runtime can be replayed deterministically without depending on mutable state.
		self.persist_turn_event(
			{
				"event_type": "graph_checkpoint",
				"trace_id": trace_id,
				"stage": stage,
				"status": status,
				"checkpoint": copy.deepcopy(payload),
				"occurred_at": datetime.now().isoformat(timespec="seconds"),
			}
		)

	def _turn_event_dir(self) -> Path:
		event_dir = self.bot.SESSION_LOG_DIR / "turn_events"
		event_dir.mkdir(parents=True, exist_ok=True)
		return event_dir

	def _turn_event_path(self, trace_id: str) -> Path:
		normalized = str(trace_id or "unknown").strip() or "unknown"
		return self._turn_event_dir() / f"{normalized[:40]}.jsonl"

	def _snapshot_path(self, trace_id: str) -> Path:
		normalized = str(trace_id or "unknown").strip() or "unknown"
		return self._turn_event_dir() / f"snapshot-{normalized[:40]}.json"

	def _compute_next_sequence(self, trace_id: str) -> int:
		events = self.list_turn_events(trace_id=trace_id)
		if not events:
			return 1
		return int(events[-1].get("sequence") or len(events)) + 1

	def persist_turn_event(self, event: dict[str, Any]) -> None:
		payload = copy.deepcopy(dict(event or {}))
		if not payload:
			return
		trace_id = str(payload.get("trace_id") or "unknown").strip() or "unknown"
		payload["trace_id"] = trace_id
		payload.setdefault("event_id", uuid.uuid4().hex)
		payload.setdefault("occurred_at", datetime.now().isoformat(timespec="seconds"))
		payload.setdefault("sequence", self._compute_next_sequence(trace_id))

		if self.bot._tenant_document_store is not None:
			key = f"turn-events:{trace_id}"
			existing = self.bot._tenant_document_store.load_session_state(key)
			events = list(existing) if isinstance(existing, list) else []
			events.append(payload)
			self.bot._tenant_document_store.save_session_state(key, events)
			self.bot._tenant_document_store.save_session_state(f"turn-events-latest:{trace_id}", payload)
		else:
			event_path = self._turn_event_path(trace_id)
			line = json.dumps(payload, ensure_ascii=True)
			with event_path.open("a", encoding="utf-8") as handle:
				handle.write(line + "\n")

		# Snapshot compaction every 25 events: replay can start from latest snapshot.
		sequence = int(payload.get("sequence") or 0)
		if sequence > 0 and sequence % 25 == 0:
			replay = self.replay_turn_events(trace_id=trace_id)
			snapshot_payload = {
				"trace_id": trace_id,
				"created_at": datetime.now().isoformat(timespec="seconds"),
				"last_sequence": sequence,
				"phase": replay.get("phase"),
				"state": replay.get("replayed_state", {}),
				"metadata": replay.get("replayed_metadata", {}),
			}
			if self.bot._tenant_document_store is not None:
				self.bot._tenant_document_store.save_session_state(f"turn-snapshot-latest:{trace_id}", snapshot_payload)
			else:
				snapshot_path = self._snapshot_path(trace_id)
				snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=True), encoding="utf-8")

	def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
		normalized = str(trace_id or "").strip()
		if not normalized:
			return []

		events: list[dict[str, Any]] = []
		if self.bot._tenant_document_store is not None:
			stored = self.bot._tenant_document_store.load_session_state(f"turn-events:{normalized}")
			if isinstance(stored, list):
				events = [item for item in stored if isinstance(item, dict)]
		else:
			event_path = self._turn_event_path(normalized)
			if event_path.exists():
				for raw_line in event_path.read_text(encoding="utf-8").splitlines():
					line = str(raw_line or "").strip()
					if not line:
						continue
					try:
						item = json.loads(line)
					except Exception:
						continue
					if isinstance(item, dict):
						events.append(item)

		events.sort(key=lambda item: int(item.get("sequence") or 0))
		if limit and limit > 0:
			return events[-limit:]
		return events

	@staticmethod
	def _fold_events(events: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
		replayed_state: dict[str, Any] = {}
		replayed_metadata: dict[str, Any] = {}
		phase = "PLAN"
		seen_lock_hashes: set[str] = set()
		for event in events:
			if not isinstance(event, dict):
				continue
			determinism_lock = event.get("determinism_lock")
			if isinstance(determinism_lock, dict):
				lock_hash = str(determinism_lock.get("lock_hash") or "").strip()
				if lock_hash:
					seen_lock_hashes.add(lock_hash)
			checkpoint = event.get("checkpoint")
			if isinstance(checkpoint, dict):
				state = checkpoint.get("state")
				metadata = checkpoint.get("metadata")
				if isinstance(state, dict):
					replayed_state.update(copy.deepcopy(state))
				if isinstance(metadata, dict):
					replayed_metadata.update(copy.deepcopy(metadata))
				replayed_determinism = metadata.get("determinism") if isinstance(metadata, dict) else None
				if isinstance(replayed_determinism, dict):
					lock_hash = str(replayed_determinism.get("lock_hash") or "").strip()
					if lock_hash:
						seen_lock_hashes.add(lock_hash)
				checkpoint_phase = str(checkpoint.get("phase") or "").strip()
				if checkpoint_phase:
					phase = checkpoint_phase
			event_phase = str(event.get("phase") or "").strip()
			if event_phase:
				phase = event_phase

		determinism_summary = {
			"lock_hash": next(iter(seen_lock_hashes)) if len(seen_lock_hashes) == 1 else "",
			"lock_hashes": sorted(seen_lock_hashes),
			"consistent": len(seen_lock_hashes) <= 1,
		}
		return replayed_state, replayed_metadata, phase, determinism_summary

	def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
		normalized = str(trace_id or "").strip()
		events = self.list_turn_events(trace_id=normalized)
		replayed_state, replayed_metadata, phase, determinism = self._fold_events(events)
		return {
			"trace_id": normalized,
			"events": events,
			"phase": phase,
			"replayed_state": replayed_state,
			"replayed_metadata": replayed_metadata,
			"determinism": determinism,
			"event_count": len(events),
		}

	def validate_replay_determinism(self, trace_id: str, expected_lock_hash: str = "") -> dict[str, Any]:
		replay = self.replay_turn_events(trace_id=trace_id)
		determinism = dict(replay.get("determinism") or {})
		observed_hash = str(determinism.get("lock_hash") or "").strip()
		expected_hash = str(expected_lock_hash or "").strip()
		matches_expected = True
		if expected_hash:
			matches_expected = observed_hash == expected_hash
		return {
			"trace_id": str(trace_id or "").strip(),
			"consistent": bool(determinism.get("consistent", True)),
			"observed_lock_hash": observed_hash,
			"expected_lock_hash": expected_hash,
			"matches_expected": matches_expected,
			"lock_hashes": list(determinism.get("lock_hashes") or []),
		}

	def load_latest_graph_checkpoint(self, trace_id: str = "") -> dict | None:
		encoded_payload = None
		normalized_trace_id = str(trace_id or "").strip()
		if self.bot._tenant_document_store is not None:
			lookup_key = f"graph-checkpoint-latest:{normalized_trace_id}" if normalized_trace_id else "graph-checkpoint-latest"
			encoded_payload = self.bot._tenant_document_store.load_session_state(lookup_key)
		else:
			checkpoint_dir = self.bot.SESSION_LOG_DIR / "graph_checkpoints"
			checkpoint_path = checkpoint_dir / (f"latest-{normalized_trace_id[:12]}.bin" if normalized_trace_id else "latest.bin")
			if checkpoint_path.exists():
				try:
					return pickle.loads(gzip.decompress(checkpoint_path.read_bytes()))
				except Exception:
					return None
		if not isinstance(encoded_payload, dict):
			return None
		payload_b64 = str(encoded_payload.get("payload_b64") or "").strip()
		if not payload_b64:
			return None
		try:
			return pickle.loads(gzip.decompress(base64.b64decode(payload_b64)))
		except Exception:
			return None

	def resume_graph_checkpoint(self, trace_id: str = "") -> dict | None:
		checkpoint = self.load_latest_graph_checkpoint(trace_id=trace_id)
		if not isinstance(checkpoint, dict):
			return None
		session_state = checkpoint.get("session_state")
		if isinstance(session_state, dict):
			self.bot.load_session_state_snapshot(session_state)
		return checkpoint


__all__ = ["ConversationPersistenceManager"]