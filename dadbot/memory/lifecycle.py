"""Memory catalog and lifecycle accessors sub-component.

Extracted from MemoryManager so that all catalog access (reminders, session
archive, consolidated memories, etc.) lives in one focused class. MemoryManager
keeps delegation shims so all existing call-sites continue to work unchanged.
"""

from __future__ import annotations

from datetime import date, datetime


class MemoryLifecycleManager:
	"""Owns all memory catalog accessors and lifecycle management.

	Depends on:
	- ``bot``     â€” for MEMORY_STORE access, runtime config, normalization
	- ``manager`` â€” back-reference to MemoryManager for normalize/mutate methods
	"""

	def __init__(self, bot, manager) -> None:
		self.bot = bot
		self._manager = manager

	# ------------------------------------------------------------------ Catalog accessors (read + normalize + mutate if needed)

	def reminder_catalog(self, include_done=False):
		reminders = [
			reminder
			for reminder in self.bot.MEMORY_STORE.get("reminders", [])
			if include_done or reminder.get("status") != "done"
		]
		reminders.sort(
			key=lambda item: (
				item.get("status") == "done",
				item.get("due_at") or "9999-12-31T23:59:59",
				item.get("due_text", ""),
				item.get("updated_at", ""),
			)
		)
		return reminders

	def session_archive(self):
		archive = self.bot.MEMORY_STORE.get("session_archive", [])
		cleaned = self.bot.runtime_config.tail([
			entry
			for entry in (self._manager.normalize_session_archive_entry(item) for item in archive)
			if entry is not None
		], "session_archive")
		if cleaned != archive:
			self._manager.mutate_memory_store(session_archive=cleaned)
		return cleaned

	def narrative_memories(self):
		entries = self.bot.MEMORY_STORE.get("narrative_memories", [])
		cleaned = [dict(item) for item in list(entries or []) if isinstance(item, dict)]
		if cleaned != entries:
			self._manager.mutate_memory_store(narrative_memories=cleaned)
		return cleaned

	def relationship_timeline(self):
		timeline = str(self.bot.MEMORY_STORE.get("relationship_timeline") or "").strip()
		if timeline != self.bot.MEMORY_STORE.get("relationship_timeline", ""):
			self._manager.mutate_memory_store(relationship_timeline=timeline)
		return timeline

	def relationship_history(self, limit=60):
		try:
			max_items = max(1, int(limit or 1))
		except (TypeError, ValueError):
			max_items = 60
		history = self.bot.MEMORY_STORE.get("relationship_history", [])
		if not isinstance(history, list):
			return []
		return [dict(item) for item in history[-max_items:] if isinstance(item, dict)]

	def persona_evolution_history(self):
		history = self.bot.runtime_config.tail([
			entry
			for entry in (self._manager.normalize_persona_evolution_entry(item) for item in self.bot.MEMORY_STORE.get("persona_evolution", []))
			if entry is not None
		], "persona_evolution")
		if history != self.bot.MEMORY_STORE.get("persona_evolution"):
			self._manager.mutate_memory_store(persona_evolution=history)
		return history

	def wisdom_catalog(self):
		insights = self.bot.runtime_config.tail([
			entry
			for entry in (self._manager.normalize_wisdom_entry(item) for item in self.bot.MEMORY_STORE.get("wisdom_insights", []))
			if entry is not None
		], "wisdom_insights")
		if insights != self.bot.MEMORY_STORE.get("wisdom_insights"):
			self._manager.mutate_memory_store(wisdom_insights=insights)
		return insights

	def life_patterns(self):
		patterns = self.bot.runtime_config.tail([
			entry
			for entry in (self._manager.normalize_life_pattern_entry(item) for item in self.bot.MEMORY_STORE.get("life_patterns", []))
			if entry is not None
		], "life_patterns")
		if patterns != self.bot.MEMORY_STORE.get("life_patterns"):
			self._manager.mutate_memory_store(life_patterns=patterns)
		return patterns

	def pending_proactive_messages(self):
		messages = self.bot.runtime_config.tail([
			entry
			for entry in (self._manager.normalize_proactive_message_entry(item) for item in self.bot.MEMORY_STORE.get("pending_proactive_messages", []))
			if entry is not None
		], "pending_proactive_messages")
		if messages != self.bot.MEMORY_STORE.get("pending_proactive_messages"):
			self._manager.mutate_memory_store(pending_proactive_messages=messages)
		return messages

	def consolidated_memories(self):
		consolidated = self.bot.runtime_config.tail([
			entry
			for entry in (self._manager.normalize_consolidated_memory_entry(item) for item in self.bot.MEMORY_STORE.get("consolidated_memories", []))
			if entry is not None
		], "consolidated_memories")
		if consolidated != self.bot.MEMORY_STORE.get("consolidated_memories"):
			self._manager.mutate_memory_store(consolidated_memories=consolidated)
		return consolidated

	def memory_graph_snapshot(self):
		graph = self._manager.normalize_memory_graph(self.bot.MEMORY_STORE.get("memory_graph"))
		if graph != self.bot.MEMORY_STORE.get("memory_graph"):
			self._manager.mutate_memory_store(memory_graph=graph)
		return graph

	def memory_catalog(self):
		raw_memories = self.bot.MEMORY_STORE.get("memories", [])
		normalized_memories = self._manager.clean_memory_entries(raw_memories)
		if normalized_memories != raw_memories:
			self._manager.mutate_memory_store(memories=normalized_memories)
		self._manager.queue_semantic_memory_index(normalized_memories)
		return normalized_memories

	def last_saved_mood(self):
		return self.bot.normalize_mood(self.bot.MEMORY_STORE.get("last_mood"))

	def recent_mood_history(self):
		raw_history = self.bot.MEMORY_STORE.get("recent_moods", [])
		cleaned = []
		for item in raw_history[-self.bot.runtime_config.window("recent_mood_history", 12):]:
			if isinstance(item, dict):
				cleaned.append({
					"mood": self.bot.normalize_mood(item.get("mood")),
					"date": item.get("date") or date.today().isoformat(),
				})
			elif isinstance(item, str):
				cleaned.append({
					"mood": self.bot.normalize_mood(item),
					"date": date.today().isoformat(),
				})
		if cleaned != raw_history:
			self._manager.mutate_memory_store(recent_moods=cleaned)
		return cleaned

	def relationship_state(self):
		current = self.bot.MEMORY_STORE.get("relationship_state")
		normalized = self._manager.normalize_relationship_state(current)
		if normalized != current:
			self._manager.mutate_memory_store(relationship_state=normalized)
		return normalized

	# ------------------------------------------------------------------ Lifecycle mutations

	def queue_proactive_message(self, message, source="general"):
		entry = self._manager.normalize_proactive_message_entry({
			"message": message,
			"source": source,
			"created_at": datetime.now().isoformat(timespec="seconds"),
		})
		if entry is None:
			return None

		queued = self.pending_proactive_messages()
		dedup_window = self.bot.runtime_config.window("pending_proactive_dedup", 3)
		if any(self.bot.normalize_memory_text(item.get("message", "")) == self.bot.normalize_memory_text(entry["message"]) for item in queued[-dedup_window:]):
			return None

		queued.append(entry)
		self._manager.mutate_memory_store(pending_proactive_messages=self.bot.runtime_config.tail(queued, "pending_proactive_messages"))
		return entry

	def consume_proactive_message(self):
		queued = self.pending_proactive_messages()
		if not queued:
			return None
		entry = queued[0]
		self._manager.mutate_memory_store(pending_proactive_messages=queued[1:])
		return entry


__all__ = ["MemoryLifecycleManager"]
