from __future__ import annotations


class DadBotActionMixin:
    """Convenience actions layered on top of the manager-owned runtime surface."""

    def record_relationship_history_point(self, *, trust_level, openness_level, source="turn"):
        history = list(self.memory.relationship_history(limit=180))
        point = {
            "recorded_at": self.runtime_timestamp(),
            "trust_level": self.clamp_score(trust_level),
            "openness_level": self.clamp_score(openness_level),
            "source": str(source or "turn").strip().lower() or "turn",
        }
        history.append(point)
        self.memory.mutate_memory_store(relationship_history=history[-180:], save=True)
        return point

    def soft_reset_session_context(self, preserve_recent_summary=True):
        preserved_summary = str(self.session_summary or "").strip()
        if not preserved_summary:
            transcript = self.transcript_from_messages(self.history[-8:])
            if transcript:
                preserved_summary = transcript[:380]
        self.reset_session_state()
        if preserve_recent_summary and preserved_summary:
            self.session_summary = preserved_summary
            self.session_summary_updated_at = self.runtime_timestamp()
            self.session_summary_covered_messages = 0
        if hasattr(self.runtime_state_manager, "sync_active_thread_snapshot"):
            self.runtime_state_manager.sync_active_thread_snapshot()
        else:
            self._apply_thread_snapshot_unlocked(
                getattr(self.runtime_state_manager, "thread_snapshot", lambda: {})()
            )
        return {
            "mode": "soft",
            "preserved_summary": preserved_summary if preserve_recent_summary else "",
        }
