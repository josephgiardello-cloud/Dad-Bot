from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from dadbot.context import ContextBuilder
from dadbot.core.execution_trace_context import (
    ensure_execution_trace_root,
    record_execution_step,
)
from dadbot.utils import significant_tokens as _significant_tokens

logger = logging.getLogger(__name__)


class ContextService:
    """Service wrapper for memory/profile/relationship context composition.

    When a ``semantic_index`` (``SQLiteSemanticIndex`` or ``PGVectorSemanticIndex``)
    is wired at construction time, ``build_context`` augments the standard context
    sections with targeted semantic search results â€” RAG over long-term memory.
    This lets the model surface relevant history beyond the most-recent window
    without bloating the prompt with the entire memory store.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        memory_manager: Any,
        semantic_index: Any = None,
    ):
        self.context_builder = context_builder
        self.memory_manager = memory_manager
        self.semantic_index = semantic_index  # Optional; wired by ServiceRegistry.boot()

    def _runtime(self) -> Any:
        return getattr(self.context_builder, "bot", None)

    def _estimate_tokens(self, text: str | None) -> int:
        runtime = self._runtime()
        if runtime is None:
            return 0
        return int(runtime.estimate_token_count(text or "") or 0)

    def get_snapshot(self, session_id: str) -> dict[str, Any]:
        """Return an immutable per-session memory snapshot for the current turn."""
        runtime = self._runtime()
        if runtime is None:
            return {
                "recent_buffer": [],
                "rolling_summary": "",
                "structured_memory": {},
                "full_history_id": "",
            }

        history = list(runtime.conversation_history() or [])
        recent_messages = []
        recent_builder = getattr(self.context_builder, "_recent_turn_messages", None)
        if callable(recent_builder):
            recent_messages = list(recent_builder(max_turns=12) or [])
        else:
            recent_messages = [dict(m) for m in history[-24:] if isinstance(m, dict)]

        summary_text = str(runtime.session_summary or "")
        claims = list(
            getattr(runtime, "_last_hierarchical_memory_stats", {}).get(
                "structured_claims",
            )
            or [],
        )
        full_history_id = hashlib.sha256(
            json.dumps(
                {
                    "session_id": str(session_id or "default"),
                    "history_len": len(history),
                    "recent_tail": [
                        {
                            "role": str(m.get("role") or ""),
                            "content": str(m.get("content") or "")[:120],
                        }
                        for m in recent_messages[-6:]
                    ],
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8"),
        ).hexdigest()[:16]

        return {
            "recent_buffer": [dict(m) for m in recent_messages],
            "rolling_summary": summary_text,
            "structured_memory": {"claims": [dict(c) for c in claims]},
            "full_history_id": f"hist-{full_history_id}",
        }

    @staticmethod
    def _deterministic_memory_fingerprint(snapshot: dict[str, Any]) -> str:
        claims = list((snapshot.get("structured_memory") or {}).get("claims") or [])
        compact_claims = [
            {
                "type": str(item.get("type") or ""),
                "summary": str(item.get("summary") or ""),
                "category": str(item.get("category") or ""),
            }
            for item in claims
            if isinstance(item, dict)
        ]
        normalized = {
            "recent_count": len(list(snapshot.get("recent_buffer") or [])),
            "rolling_summary": str(snapshot.get("rolling_summary") or ""),
            "claims": compact_claims,
            "full_history_id": str(snapshot.get("full_history_id") or ""),
        }
        return hashlib.sha256(
            json.dumps(
                normalized,
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            ).encode("utf-8"),
        ).hexdigest()[:16]

    def _maybe_schedule_background_compression(
        self,
        turn_context: Any,
        *,
        context_total_tokens: int,
        recent_tokens: int,
    ) -> bool:
        runtime = self._runtime()
        if runtime is None:
            return False

        metadata = getattr(turn_context, "metadata", None)
        determinism = dict(metadata.get("determinism") or {}) if isinstance(metadata, dict) else {}
        if bool(determinism.get("enforced", False)):
            # Strict mode must remain deterministic; skip background summarization side effects.
            if isinstance(metadata, dict):
                metadata["compression_scheduled"] = False
                metadata["compression_blocking"] = False
                metadata["compression_disabled_in_strict"] = True
            return False

        should_start_background_tasks = getattr(
            runtime,
            "should_start_background_tasks",
            None,
        )
        if callable(should_start_background_tasks) and not bool(
            should_start_background_tasks(),
        ):
            return False

        if isinstance(metadata, dict) and metadata.get("compression_scheduled"):
            return False

        budget = max(1, int(getattr(runtime, "CONTEXT_TOKEN_BUDGET", 6000) or 6000))
        threshold = max(128, int(budget * 0.75))
        recent_pressure = recent_tokens >= max(80, int(threshold * 0.25))
        over_threshold = context_total_tokens >= threshold
        if not (over_threshold or recent_pressure):
            if isinstance(metadata, dict):
                metadata["compression_scheduled"] = False
                metadata["compression_blocking"] = False
            return False

        summary_manager = getattr(runtime, "session_summary_manager", None)
        refresh = getattr(summary_manager, "refresh_session_summary", None)
        background_tasks = getattr(runtime, "background_tasks", None)
        submit = getattr(background_tasks, "submit", None)
        if not callable(refresh) or not callable(submit):
            return False

        try:
            # Fire-and-forget scheduling only; never await in the foreground turn.
            submit(refresh, force=True, task_kind="context-compression")
            if isinstance(metadata, dict):
                metadata["compression_scheduled"] = True
                metadata["compression_blocking"] = False
                metadata["compression_trigger_tokens"] = context_total_tokens
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # build_context helpers
    # ------------------------------------------------------------------

    def _build_base_context_sections(
        self,
        user_input: str,
        temporal_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the core context dict from all context_builder methods."""
        return {
            "core_persona": self.context_builder.build_core_persona_prompt(),
            "dynamic_profile": self.context_builder.build_dynamic_profile_context(),
            "relationship": self.context_builder.build_relationship_context(),
            "session_summary": self.context_builder.build_session_summary_context(),
            "memory": self.context_builder.build_memory_context(user_input),
            "relevant": self.context_builder.build_relevant_context(user_input),
            "cross_session": self.context_builder.build_cross_session_context(user_input),
            "temporal": temporal_snapshot,
        }

    def _extract_layer_stats(self) -> dict[str, Any]:
        """Return a copy of the bot's last hierarchical memory stats (or empty dict)."""
        bot = getattr(self.context_builder, "bot", None)
        return dict(getattr(bot, "_last_hierarchical_memory_stats", {}) or {}) if bot else {}

    def _run_semantic_rag(
        self,
        user_input: str,
        ctx: dict[str, Any],
        state: Any,
    ) -> None:
        """Query the semantic index and write hits into *ctx* and *state*."""
        if self.semantic_index is None or not user_input.strip():
            return
        try:
            tokens = _significant_tokens(user_input)
            query_embedding = None
            embed_text = getattr(self.memory_manager, "embed_text", None)
            if callable(embed_text):
                query_embedding = embed_text(user_input)
            hits = self.semantic_index.fetch_candidates(
                query_embedding=query_embedding,
                query_tokens=tokens,
                query_category="general",
                query_mood="neutral",
                limit=5,
            )
            record_execution_step(
                "memory_retrieval",
                payload={
                    "query_token_count": len(list(tokens or [])),
                    "has_embedding": bool(query_embedding is not None),
                    "hit_count": len(list(hits or [])),
                },
                required=True,
            )
            if hits:
                base_summaries = {str(m.get("summary", "")) for m in (ctx.get("memory") or []) if isinstance(m, dict)}
                unique_hits = [h for h in hits if str(h.get("summary", "")) not in base_summaries]
                if unique_hits:
                    ctx["semantic"] = unique_hits
                if isinstance(state, dict):
                    state["memory_retrieval_set"] = list(unique_hits or hits)
        except Exception as exc:
            logger.debug("ContextService: semantic index query failed (non-fatal): %s", exc)

    def _stamp_build_results(
        self,
        turn_context: Any,
        memory_snapshot: dict[str, Any],
        layer_stats: dict[str, Any],
        context_total_tokens: int,
        context_build_ms: float,
    ) -> None:
        """Stamp determinism hashes, timing metadata, and schedule compression."""
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        recent_tokens = int(layer_stats.get("recent_tokens", 0) or 0)
        summary_tokens = int(layer_stats.get("summary_tokens", 0) or 0)
        structured_tokens = int(layer_stats.get("structured_tokens", 0) or 0)

        if isinstance(metadata, dict):
            determinism = dict(metadata.get("determinism") or {})
            if bool(determinism.get("enforced", False)):
                memory_fingerprint = self._deterministic_memory_fingerprint(memory_snapshot)
                base_lock_hash = str(determinism.get("lock_hash") or "")
                composite_lock_hash = hashlib.sha256(
                    json.dumps(
                        {"base_lock_hash": base_lock_hash, "memory_fingerprint": memory_fingerprint},
                        sort_keys=True,
                        ensure_ascii=True,
                    ).encode("utf-8"),
                ).hexdigest()
                determinism["memory_fingerprint"] = memory_fingerprint
                determinism["lock_hash"] = composite_lock_hash
                determinism["lock_id"] = f"det-{composite_lock_hash[:16]}"
                determinism["lock_version"] = max(2, int(determinism.get("lock_version") or 2))
                metadata["determinism"] = determinism
                self.context_builder.bot._last_memory_fingerprint = memory_fingerprint
            metadata["context_build_ms"] = context_build_ms
            metadata["prefill_ms"] = float(metadata.get("prefill_ms") or 0.0)
            metadata["recent_tokens"] = recent_tokens
            metadata["summary_tokens"] = summary_tokens
            metadata["structured_tokens"] = structured_tokens
            metadata["context_total_tokens"] = context_total_tokens

        if isinstance(state, dict):
            state["_timing_context_build_ms"] = context_build_ms

        compression_scheduled = self._maybe_schedule_background_compression(
            turn_context,
            context_total_tokens=context_total_tokens,
            recent_tokens=recent_tokens,
        )
        if isinstance(metadata, dict):
            metadata["compression_scheduled"] = bool(
                metadata.get("compression_scheduled") or compression_scheduled,
            )

    def build_context(self, turn_context: Any) -> dict[str, Any]:
        started = time.perf_counter()
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError("TemporalNode required — execution invalid")
        user_input = str(getattr(turn_context, "user_input", "") or "")
        session_id = str(
            (getattr(turn_context, "metadata", {}) or {}).get("control_plane", {}).get("session_id") or "default",
        )
        with ensure_execution_trace_root(
            operation="memory_read",
            prompt="[context-service-build]",
            metadata={"source": "ContextService.build_context", "session_id": session_id},
            required=True,
        ):
            record_execution_step(
                "memory_read",
                payload={
                    "source": "ContextService.build_context",
                    "session_id": session_id,
                    "query_length": len(user_input),
                },
                required=True,
            )

        temporal_snapshot: dict[str, Any] = {}
        temporal_builder = getattr(turn_context, "temporal_snapshot", None)
        if callable(temporal_builder):
            temporal_snapshot = temporal_builder()
        wall_time = str(temporal_snapshot.get("wall_time") or "").strip()
        wall_date = str(temporal_snapshot.get("wall_date") or "").strip()
        if not wall_time or not wall_date:
            raise RuntimeError("TemporalNode required — execution invalid")

        ctx = self._build_base_context_sections(user_input, temporal_snapshot)
        memory_snapshot = self.get_snapshot(session_id)
        layer_stats = self._extract_layer_stats()

        ctx["memory_layers"] = {
            "selected_layers": list(layer_stats.get("selected_layers") or []),
            "structured_claims": list(layer_stats.get("structured_claims") or []),
            "recent_tokens": int(layer_stats.get("recent_tokens", 0) or 0),
            "summary_tokens": int(layer_stats.get("summary_tokens", 0) or 0),
            "structured_tokens": int(layer_stats.get("structured_tokens", 0) or 0),
        }

        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["memory_recent_buffer"] = list(memory_snapshot.get("recent_buffer") or [])
            state["memory_rolling_summary"] = str(memory_snapshot.get("rolling_summary") or "")
            state["memory_structured"] = dict(memory_snapshot.get("structured_memory") or {})
            state["memory_full_history_id"] = str(memory_snapshot.get("full_history_id") or "")
            state["memory_retrieval_set"] = []

        self._run_semantic_rag(user_input, ctx, state)

        context_total_tokens = sum(
            self._estimate_tokens(str(ctx.get(key) or ""))
            for key in (
                "core_persona",
                "dynamic_profile",
                "relationship",
                "session_summary",
                "memory",
                "relevant",
                "cross_session",
            )
        )
        context_total_tokens += (
            int(layer_stats.get("recent_tokens", 0) or 0)
            + int(layer_stats.get("summary_tokens", 0) or 0)
            + int(layer_stats.get("structured_tokens", 0) or 0)
        )

        context_build_ms = round((time.perf_counter() - started) * 1000, 3)
        self._stamp_build_results(turn_context, memory_snapshot, layer_stats, context_total_tokens, context_build_ms)
        return ctx
