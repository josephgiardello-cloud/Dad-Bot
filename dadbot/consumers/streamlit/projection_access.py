from __future__ import annotations

import logging
from typing import Any

from dadbot.runtime_core.event_api import RuntimeEventAPI
from dadbot.runtime_core.store import ConversationStore


logger = logging.getLogger(__name__)


def load_thread_projection(
    *,
    api: RuntimeEventAPI | Any,
    thread_id: str,
    version: str = ConversationStore.THREAD_VIEW_DEFAULT_VERSION,
    seed_messages: list[dict] | None = None,
) -> Any:
    """Read a detached thread projection through the sanctioned consumer boundary."""
    normalized_thread_id = str(thread_id or "default")
    if version != ConversationStore.THREAD_VIEW_DEFAULT_VERSION:
        logger.warning(
            "Non-default projection version requested through streamlit consumer boundary: %s",
            version,
        )
    if seed_messages is not None:
        api.seed_thread(normalized_thread_id, list(seed_messages or []))
    return api.get_view(normalized_thread_id, version=version)