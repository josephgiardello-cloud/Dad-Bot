from __future__ import annotations

import logging
import sqlite3
from typing import Any

from dadbot.services.source_grounding import GroundingViolation, SourceGroundingService

logger = logging.getLogger(__name__)


def render_source_mode(st: Any, bot: Any, thread_id: str) -> bool:
    modes = dict(st.session_state.get("thread_source_modes") or {})
    enabled = st.sidebar.checkbox(
        "Jor-El source mode",
        value=bool(modes.get(thread_id, False)),
        key=f"source-mode-{thread_id}",
        help="Quote profile facts and explicitly approved memories only. No model or tools.",
    )
    modes[thread_id] = enabled
    st.session_state["thread_source_modes"] = modes
    if not enabled:
        return False
    st.sidebar.caption(
        "Extractive quotes only. Match percentages are retrieval overlap, not truth confidence. "
        "Attachments and conversation history are not sources. Auto-speech is disabled. "
        "Crisis safety responses take precedence."
    )
    with st.sidebar.expander("Review memory citation approvals"):
        service: SourceGroundingService = bot.service_registry.get("source_grounding_service")
        try:
            sources = service.memory_sources()
            approved = service.approvals.approved_digests()
            if not sources:
                st.caption("No memory records to review.")
            for source in sources:
                st.code(source.text, language=None)
                is_approved = approved.get(source.source_id) == source.digest
                st.caption(f"{source.source_id}: {'Approved' if is_approved else 'Not approved'}")
                if st.button(
                    "Revoke citation approval" if is_approved else "Approve for citations",
                    key=f"citation-approval-{source.source_id}",
                ):
                    service.set_memory_approval(source, approved=not is_approved)
                    st.rerun()
        except (GroundingViolation, sqlite3.Error, OSError):
            logger.exception("Memory citation approval operation failed")
            st.error("Could not read or save citation approval. Reload and retry; approval was not confirmed.")
    return bool(enabled)
