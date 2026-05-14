from __future__ import annotations


class MemoryLifecycleMixin:
    """Storage/lifecycle delegation mixin for the physical memory layer."""

    # Storage I/O delegation shims
    # All persistence is owned by MemoryStorageBackend.

    def _write_json_memory_store_unlocked(self):
        return self._storage._write_json_memory_store_unlocked()

    def _write_memory_store_unlocked(self):
        return self._storage._write_memory_store_unlocked()

    def _load_memory_store(self):
        return self._storage.load_memory_store()

    def prepare_memory_store_for_save(self):
        return self._storage.prepare_memory_store_for_save()

    def save_memory_store(self):
        projection = self.normalize_memory_store(self.memory_projection())
        return self._storage._write_memory_store_unlocked(projection)

    def export_memory_store(self, export_path):
        return self._storage.export_memory_store(export_path)

    def clear_memory_store(self):
        return self._storage.clear_memory_projection()

    # Lifecycle/catalog delegation shims

    def reminder_catalog(self, include_done=False):
        return self._lifecycle.reminder_catalog(include_done=include_done)

    def session_archive(self):
        # Deterministic hard-fact surface: no transformation, pure delegation.
        return self._lifecycle.session_archive()

    def narrative_memories(self):
        return self._lifecycle.narrative_memories()

    def relationship_timeline(self):
        return self._lifecycle.relationship_timeline()

    def relationship_history(self, limit=60):
        return self._lifecycle.relationship_history(limit=limit)

    def persona_evolution_history(self):
        return self._lifecycle.persona_evolution_history()

    def wisdom_catalog(self):
        return self._lifecycle.wisdom_catalog()

    def life_patterns(self):
        return self._lifecycle.life_patterns()

    def pending_proactive_messages(self):
        return self._lifecycle.pending_proactive_messages()

    def queue_proactive_message(self, message, source="general"):
        return self._lifecycle.queue_proactive_message(message, source=source)

    def consume_proactive_message(self):
        return self._lifecycle.consume_proactive_message()

    def consolidated_memories(self):
        return self._lifecycle.consolidated_memories()

    def memory_graph_snapshot(self):
        return self._lifecycle.memory_graph_snapshot()

    def memory_catalog(self):
        # Deterministic hard-fact surface: no transformation, pure delegation.
        return self._lifecycle.memory_catalog()

    def last_saved_mood(self):
        return self._lifecycle.last_saved_mood()

    def recent_mood_history(self):
        return self._lifecycle.recent_mood_history()

    def relationship_state(self):
        return self._lifecycle.relationship_state()
