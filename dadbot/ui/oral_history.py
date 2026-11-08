from __future__ import annotations

import json
from typing import Any

from dadbot.memory.oral_history_evidence import create_reviewed_evidence
from dadbot_system.state import InMemoryStateStore


def save_reviewed_transcript(bot: Any, transcript: str) -> dict[str, Any] | None:
    text = str(transcript or "").strip()
    if not text:
        return None
    evidence = create_reviewed_evidence(text)
    memory = bot.add_memory(
        text,
        category="oral_history",
        oral_history_evidence=evidence,
    )
    if memory is None:
        return None

    document_store = getattr(bot, "_tenant_document_store", None)
    try:
        if document_store is not None and not isinstance(bot._customer_state_store, InMemoryStateStore):
            load = getattr(document_store, "load_memory_state", document_store.load_session_state)
            persisted = load("memory")
        else:
            with bot.MEMORY_PATH.open("r", encoding="utf-8") as handle:
                persisted = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Could not verify the oral-history memory in durable storage.") from exc
    records = persisted.get("memories", []) if isinstance(persisted, dict) else []
    if not isinstance(records, list) or not any(
        isinstance(record, dict)
        and record.get("summary") == text
        and record.get("oral_history_evidence") == evidence
        for record in records
    ):
        raise RuntimeError("The oral-history memory was not confirmed in durable storage.")
    return memory
