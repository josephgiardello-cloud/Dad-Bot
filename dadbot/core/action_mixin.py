from __future__ import annotations


class DadBotActionMixin:
    """Convenience actions layered on top of the manager-owned runtime surface."""

    def _queue_or_apply_memory_patch(self, **patch):
        normalized_patch = dict(patch or {})
        # Apply immediately for test visibility and correctness.
        # During graph execution commits, this ensures mutations are visible to observers.
        self.memory.mutate_memory_store(**normalized_patch, save=True)
        return "applied"

    def record_relationship_history_point(
        self,
        *,
        trust_level,
        openness_level,
        source="turn",
    ):
        history = list(self.memory.relationship_history(limit=180))
        point = {
            "recorded_at": self.runtime_timestamp(),
            "trust_level": self.clamp_score(trust_level),
            "openness_level": self.clamp_score(openness_level),
            "source": str(source or "turn").strip().lower() or "turn",
        }
        history.append(point)
        self._queue_or_apply_memory_patch(relationship_history=history[-180:])
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
                getattr(self.runtime_state_manager, "thread_snapshot", dict)(),
            )
        return {
            "mode": "soft",
            "preserved_summary": preserved_summary if preserve_recent_summary else "",
        }
