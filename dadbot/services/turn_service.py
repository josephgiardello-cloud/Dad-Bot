from __future__ import annotations

import json
import inspect
import logging
import warnings
from typing import Any, cast

from dadbot.contracts import (
    AttachmentList,
    ChunkCallback,
    DadBotContext,
    FinalizedTurnResult,
    PreparedTurnResult,
    SupportsTurnProcessingRuntime,
)
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.core.execution_contract import (
    SovereignContext,
    TurnDelivery,
    TurnResponse,
    live_turn_request,
)
from dadbot.core.execution_result_unified import get_unified_execution_result
from dadbot.core.memory_set_invariants import (
    assert_memory_set_invariants,
    record_causal_step_locked,
    MemorySetInvariantViolation,
)
from dadbot.core.system_state_algebra import (
    evaluate_system_state_algebra,
    persist_system_state_algebra,
)
from dadbot.core.graph import (
    LedgerMutationOp,
    MutationIntent,
    MutationKind,
    TurnContext,
)
from dadbot.core.tool_executor import execute_tool
from dadbot.core.turn_coherence import mark_turn_coherence, reset_turn_coherence
from dadbot.managers.reply_generation import ReplyGenerationManager
from dadbot.models import AgenticToolPlan
from dadbot.services.memory_service import MemoryService
from dadbot.services.llm_call_adapter import LLMCallAdapter
from dadbot.services.turn_state_mutator import TurnStateMutator
from dadbot.services._tool_pipeline_mixin import (
    ToolPipelineMixin,
    _DEFER_TOOL_BIASES,
    _PlannerDecision,
    _REMINDER_TOOL_NAMES,
    _ReflectionDecision,
    _SET_REMINDER_TOOL,
    _TOOL_NAME_ALIASES,
    _TOOL_VISIBILITY_SETTINGS,
    _WEB_SEARCH_TOOL,
    _is_event_loop_closed_error,
)

# Keep the historic logger name so existing tests and log pipelines remain stable
# while the implementation moves from managers/ to services/.
logger = logging.getLogger("dadbot.managers.turn_processing")


class TurnService(ToolPipelineMixin):
    """Own per-turn preparation, reply generation, and finalization flows.

    This replaces the old TurnProcessingManager. The concrete implementation
    now lives under dadbot.services and is exposed through ``turn_service``.
    """

    def __init__(self, bot: DadBotContext | SupportsTurnProcessingRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self.reply_generation = ReplyGenerationManager(self.context)
        self._llm_adapter = LLMCallAdapter(self.bot)
        self._memory_service = MemoryService(self.bot)
        self._state_mutator = TurnStateMutator(self.bot)

    def _pipeline_timestamp(self) -> str:
        return self._state_mutator.pipeline_timestamp()

    def _store_turn_pipeline(self, payload: dict[str, object]) -> dict[str, object]:
        return self._state_mutator.store_turn_pipeline(payload)

    def _legacy_pipeline_overridden(self) -> bool:
        prepare = getattr(self, "prepare_user_turn", None)
        finalize = getattr(self, "finalize_user_turn", None)
        if not callable(prepare) or not callable(finalize):
            return False
        prepare_func = getattr(prepare, "__func__", None)
        finalize_func = getattr(finalize, "__func__", None)
        if prepare_func is None or finalize_func is None:
            return True
        return (
            prepare_func is not TurnService.prepare_user_turn
            or finalize_func is not TurnService.finalize_user_turn
        )

    def _start_turn_pipeline(self, mode: str, user_input: str) -> dict[str, object]:
        return self._state_mutator.start_turn_pipeline(mode, user_input)

    def _update_turn_pipeline(self, **fields) -> dict[str, object] | None:
        return self._state_mutator.update_turn_pipeline(**fields)

    def _append_turn_pipeline_step(
        self,
        name: str,
        status: str = "completed",
        detail: str = "",
        **metadata,
    ) -> dict[str, object] | None:
        return self._state_mutator.append_turn_pipeline_step(
            name,
            status,
            detail,
            **metadata,
        )

    def _ensure_compat_pipeline_steps(self, names: tuple[str, ...]) -> None:
        snapshot = self.turn_pipeline_snapshot()
        if not isinstance(snapshot, dict):
            return
        existing = {
            str(step.get("name") or "")
            for step in list(snapshot.get("steps") or [])
            if isinstance(step, dict)
        }
        for name in names:
            if name in existing:
                continue
            self._append_turn_pipeline_step(
                name,
                detail="compat alias on spine-delegated turn",
            )

    def _complete_turn_pipeline(
        self,
        *,
        final_path: str = "",
        reply_source: str = "",
        should_end: bool = False,
        error: str = "",
    ) -> dict[str, object] | None:
        return self._state_mutator.complete_turn_pipeline(
            final_path=final_path,
            reply_source=reply_source,
            should_end=should_end,
            error=error,
        )

    def turn_pipeline_snapshot(self):
        return self._state_mutator.turn_pipeline_snapshot()

    def should_offer_daily_checkin_for_turn(self) -> bool:
        return self.bot.memory.should_do_daily_checkin() and self.bot.session_turn_count() == 0

    def record_user_turn_state(
        self,
        stripped_input: str,
        current_mood: str,
        turn_context: Any | None = None,
    ) -> None:
        should_offer_daily_checkin = self.should_offer_daily_checkin_for_turn()

        # If turn_context is provided, use the strict-mode path through mutation queue
        if turn_context is not None:
            mutation_queue = getattr(turn_context, "mutation_queue", None)
            if mutation_queue is None:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Strict mode turn_context lacks mutation_queue; using fallback",
                )
                # Fall through to direct write below
            else:
                try:
                    mutation_queue.queue(
                        MutationIntent(
                            type=MutationKind.LEDGER,
                            payload={
                                "op": LedgerMutationOp.RECORD_TURN_STATE.value,
                                "mood": str(current_mood or "neutral"),
                                "should_offer_daily_checkin": bool(
                                    should_offer_daily_checkin,
                                ),
                            },
                            requires_temporal=False,
                            source="turn_service.record_user_turn_state",
                        ),
                    )

                    # Queue mood mutation for SaveNode drain at the turn commit boundary.
                    # All persistent mood writes flow through state["_pending_mood_updates"] →
                    # PersistenceService._apply_pending_save_boundary_mutations → SaveNode.
                    queued_mood = str(current_mood or "neutral")
                    state = getattr(turn_context, "state", None)
                    if isinstance(state, dict):
                        pending_moods = list(state.get("_pending_mood_updates") or [])
                        pending_moods.append({"mood": queued_mood})
                        state["_pending_mood_updates"] = pending_moods
                    # Preserve same-turn UX behavior: tone blending depends on this
                    # transient flag before SaveNode drains queued mutations.
                    self.bot._last_should_offer_daily_checkin = bool(
                        should_offer_daily_checkin,
                    )
                    self.bot._pending_daily_checkin_context = bool(
                        should_offer_daily_checkin,
                    )
                    return
                except RuntimeError as guard_exc:
                    # MutationGuard violation: mutations locked outside SaveNode.
                    # Gracefully fall through to compatibility path for legacy callers.
                    if "MutationGuard" in str(guard_exc):
                        logger = logging.getLogger(__name__)
                        logger.debug(
                            f"MutationGuard prevented recording outside SaveNode; using fallback: {guard_exc}",
                        )
                    else:
                        raise

        # Fallback: direct write when turn_context is not available (e.g., in tests)
        # This maintains compatibility with legacy code paths.
        # All direct attribute writes are owned by TurnStateMutator.
        self._state_mutator.write_mood_fallback(
            current_mood,
            should_offer_daily_checkin,
        )

    def direct_reply_for_input(
        self,
        stripped_input: str,
        current_mood: str,
    ) -> str | None:
        parse_tool_command = getattr(self.bot, "parse_tool_command", None)
        parsed_tool_command = parse_tool_command(stripped_input) if callable(parse_tool_command) else None
        parsed_action = str((parsed_tool_command or {}).get("action") or "").strip()
        # Executable tool commands must flow through planner->executor coupling.
        if parsed_action in {_SET_REMINDER_TOOL, "web_lookup"}:
            return None

        crisis_reply = self.bot.safety_support.direct_reply_for_input(stripped_input)
        if crisis_reply is not None:
            return crisis_reply

        for direct_reply in (
            self.bot.handle_memory_command(stripped_input),
            self.bot.handle_tool_command(stripped_input),
            self.bot.get_memory_reply(stripped_input),
            self.bot.get_fact_reply(stripped_input),
        ):
            if direct_reply is not None:
                return self.bot.reply_finalization.finalize(
                    direct_reply,
                    current_mood,
                    stripped_input,
                )
        return None

    def prepare_user_turn(
        self,
        stripped_input: str,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> PreparedTurnResult:
        reset_turn_coherence(self.bot)
        self._start_turn_pipeline("sync", stripped_input)
        normalized_attachments = self.bot.normalize_chat_attachments(attachments)
        normalized_attachments = self.bot.enrich_multimodal_attachments(
            normalized_attachments,
            user_input=stripped_input,
        )
        self._append_turn_pipeline_step(
            "normalize_attachments",
            detail=f"attachments={len(normalized_attachments)}",
        )
        turn_text = self.bot.compose_user_turn_text(
            stripped_input,
            normalized_attachments,
        )
        self._append_turn_pipeline_step(
            "compose_turn_text",
            detail="composed user turn text" if turn_text else "empty composed turn",
        )
        if not turn_text:
            self._complete_turn_pipeline(final_path="empty_input", reply_source="none")
            return None, None, False, "", normalized_attachments

        if self.bot.is_session_exit_command(stripped_input):
            self.bot.persist_conversation()
            self.bot.mark_chat_thread_closed(closed=True)
            self._append_turn_pipeline_step(
                "session_exit",
                detail="handled exit command",
            )
            self._complete_turn_pipeline(
                final_path="session_exit",
                reply_source="exit_command",
                should_end=True,
            )
            return (
                None,
                self.bot.reply_finalization.append_signoff(
                    "Catch ya later, Tony! Always here if you need me.",
                ),
                True,
                turn_text,
                normalized_attachments,
            )

        mood_history_window = self.bot.runtime_config.window(
            "mood_detection_context",
            6,
        )
        if self.bot.LIGHT_MODE:
            current_mood = "neutral"
        else:
            current_mood = self.bot.mood_manager.detect(
                turn_text,
                self.bot.prompt_history()[-mood_history_window:],
            )
        self._update_turn_pipeline(current_mood=current_mood)
        self._append_turn_pipeline_step(
            "detect_mood",
            detail=f"current_mood={current_mood}",
        )
        self.record_user_turn_state(turn_text, current_mood, turn_context=turn_context)
        self._append_turn_pipeline_step(
            "record_turn_state",
            detail="saved mood and relationship state",
        )
        self.bot.begin_planner_debug(stripped_input, current_mood)
        self._append_turn_pipeline_step(
            "begin_planner_debug",
            detail="initialized planner debug state",
        )
        self.bot.prompt_assembly.begin_turn_memory_context(
            stripped_input,
            user_id=str(getattr(self.bot, "TENANT_ID", "default") or "default"),
            session_id=str(getattr(self.bot, "active_thread_id", "default") or "default"),
        )

        if direct_reply := self.direct_reply_for_input(stripped_input, current_mood):
            self.bot.update_planner_debug(
                planner_status="skipped",
                planner_reason="A direct reply path handled this turn before tool planning.",
                final_path="direct_reply",
            )
            self._append_turn_pipeline_step(
                "direct_reply",
                detail="reply produced before tool planning",
            )
            self._update_turn_pipeline(
                final_path="direct_reply",
                reply_source="direct_reply",
            )
            return current_mood, direct_reply, False, turn_text, normalized_attachments

        mark_turn_coherence(self.bot, "tool_decision_origin")
        auto_reply, tool_observation = self.plan_agentic_tools(
            stripped_input,
            current_mood,
            normalized_attachments,
            turn_context=turn_context,
        )
        planner_snapshot = self.bot.planner_debug_snapshot()
        self._append_turn_pipeline_step(
            "plan_tools",
            detail=f"planner_status={planner_snapshot.get('planner_status', 'idle')}",
        )
        if auto_reply is not None:
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "planner_tool",
                ).strip()
                or "planner_tool",
                reply_source="planner_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self.bot.set_active_tool_observation(tool_observation)
        if tool_observation:
            self._append_turn_pipeline_step(
                "tool_observation",
                detail="captured tool observation for reply generation",
            )
        if auto_reply is not None:
            planner_snapshot = self.bot.planner_debug_snapshot()
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "heuristic_tool",
                ).strip()
                or "heuristic_tool",
                reply_source="heuristic_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self._update_turn_pipeline(
            final_path="model_reply",
            reply_source="model_generation",
        )
        return current_mood, None, False, turn_text, normalized_attachments

    async def prepare_user_turn_async(
        self,
        stripped_input: str,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> PreparedTurnResult:
        reset_turn_coherence(self.bot)
        self._start_turn_pipeline("async", stripped_input)
        normalized_attachments = self.bot.normalize_chat_attachments(attachments)
        normalized_attachments = self.bot.enrich_multimodal_attachments(
            normalized_attachments,
            user_input=stripped_input,
        )
        self._append_turn_pipeline_step(
            "normalize_attachments",
            detail=f"attachments={len(normalized_attachments)}",
        )
        turn_text = self.bot.compose_user_turn_text(
            stripped_input,
            normalized_attachments,
        )
        self._append_turn_pipeline_step(
            "compose_turn_text",
            detail="composed user turn text" if turn_text else "empty composed turn",
        )
        if not turn_text:
            self._complete_turn_pipeline(final_path="empty_input", reply_source="none")
            return None, None, False, "", normalized_attachments

        if self.bot.is_session_exit_command(stripped_input):
            self.bot.persist_conversation()
            self.bot.mark_chat_thread_closed(closed=True)
            self._append_turn_pipeline_step(
                "session_exit",
                detail="handled exit command",
            )
            self._complete_turn_pipeline(
                final_path="session_exit",
                reply_source="exit_command",
                should_end=True,
            )
            return (
                None,
                self.bot.reply_finalization.append_signoff(
                    "Catch ya later, Tony! Always here if you need me.",
                ),
                True,
                turn_text,
                normalized_attachments,
            )

        mood_history_window = self.bot.runtime_config.window(
            "mood_detection_context",
            6,
        )
        if self.bot.LIGHT_MODE:
            current_mood = "neutral"
        else:
            current_mood = await self.bot.mood_manager.detect_async(
                turn_text,
                self.bot.prompt_history()[-mood_history_window:],
            )
        self._update_turn_pipeline(current_mood=current_mood)
        self._append_turn_pipeline_step(
            "detect_mood",
            detail=f"current_mood={current_mood}",
        )
        self.record_user_turn_state(turn_text, current_mood, turn_context=turn_context)
        self._append_turn_pipeline_step(
            "record_turn_state",
            detail="saved mood and relationship state",
        )
        self.bot.begin_planner_debug(stripped_input, current_mood)
        self._append_turn_pipeline_step(
            "begin_planner_debug",
            detail="initialized planner debug state",
        )
        self.bot.prompt_assembly.begin_turn_memory_context(
            stripped_input,
            user_id=str(getattr(self.bot, "TENANT_ID", "default") or "default"),
            session_id=str(getattr(self.bot, "active_thread_id", "default") or "default"),
        )

        if direct_reply := self.direct_reply_for_input(stripped_input, current_mood):
            self.bot.update_planner_debug(
                planner_status="skipped",
                planner_reason="A direct reply path handled this turn before tool planning.",
                final_path="direct_reply",
            )
            self._append_turn_pipeline_step(
                "direct_reply",
                detail="reply produced before tool planning",
            )
            self._update_turn_pipeline(
                final_path="direct_reply",
                reply_source="direct_reply",
            )
            return current_mood, direct_reply, False, turn_text, normalized_attachments

        mark_turn_coherence(self.bot, "tool_decision_origin")
        auto_reply, tool_observation = await self.plan_agentic_tools_async(
            stripped_input,
            current_mood,
            normalized_attachments,
            turn_context=turn_context,
        )
        planner_snapshot = self.bot.planner_debug_snapshot()
        self._append_turn_pipeline_step(
            "plan_tools",
            detail=f"planner_status={planner_snapshot.get('planner_status', 'idle')}",
        )
        if auto_reply is not None:
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "planner_tool",
                ).strip()
                or "planner_tool",
                reply_source="planner_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self.bot.set_active_tool_observation(tool_observation)
        if tool_observation:
            self._append_turn_pipeline_step(
                "tool_observation",
                detail="captured tool observation for reply generation",
            )
        if auto_reply is not None:
            planner_snapshot = self.bot.planner_debug_snapshot()
            self._update_turn_pipeline(
                final_path=str(
                    planner_snapshot.get("final_path") or "heuristic_tool",
                ).strip()
                or "heuristic_tool",
                reply_source="heuristic_tool",
            )
            return current_mood, auto_reply, False, turn_text, normalized_attachments

        self._update_turn_pipeline(
            final_path="model_reply",
            reply_source="model_generation",
        )
        return current_mood, None, False, turn_text, normalized_attachments

    def finalize_user_turn(
        self,
        stripped_input: str,
        current_mood: str | None,
        dad_reply: str | None,
        attachments: AttachmentList | None = None,
        turn_context: Any | None = None,
    ) -> FinalizedTurnResult:
        if turn_context is None:
            raise RuntimeError(
                "Strict mode requires turn_context for finalize_user_turn",
            )
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        if mutation_queue is None:
            raise RuntimeError("SaveNode context missing mutation_queue in strict mode")

        self.validate_execution_truth_contract(turn_context, enforce=True)

        self._append_turn_pipeline_step(
            "finalize_turn",
            detail="persisted conversation turn",
        )

        user_turn = {"role": "user", "content": stripped_input, "mood": current_mood}
        if attachments:
            user_turn["attachments"] = [self.bot.history_attachment_metadata(attachment) for attachment in attachments]

        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={
                    "op": LedgerMutationOp.APPEND_HISTORY.value,
                    "entry": user_turn,
                },
                requires_temporal=False,
                source="turn_service.finalize_user_turn.user_entry",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={
                    "op": LedgerMutationOp.APPEND_HISTORY.value,
                    "entry": {"role": "assistant", "content": dad_reply},
                },
                requires_temporal=False,
                source="turn_service.finalize_user_turn.assistant_entry",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={"op": LedgerMutationOp.CLEAR_TURN_CONTEXT.value},
                requires_temporal=False,
                source="turn_service.finalize_user_turn.clear_turn_context",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={"op": LedgerMutationOp.SYNC_THREAD_SNAPSHOT.value},
                requires_temporal=False,
                source="turn_service.finalize_user_turn.sync_thread_snapshot",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={
                    "op": LedgerMutationOp.SCHEDULE_MAINTENANCE.value,
                    "turn_text": stripped_input,
                    "mood": current_mood,
                },
                requires_temporal=False,
                source="turn_service.finalize_user_turn.schedule_maintenance",
            ),
        )
        mutation_queue.queue(
            MutationIntent(
                type=MutationKind.LEDGER,
                payload={"op": LedgerMutationOp.HEALTH_SNAPSHOT.value},
                requires_temporal=False,
                source="turn_service.finalize_user_turn.health_snapshot",
            ),
        )
        policy_trace_events = list(getattr(turn_context, "state", {}).get("policy_trace_events") or [])
        if policy_trace_events:
            mutation_queue.queue(
                MutationIntent(
                    type=MutationKind.LEDGER,
                    payload={
                        "op": LedgerMutationOp.POLICY_TRACE_EVENT.value,
                        "events": policy_trace_events,
                    },
                    requires_temporal=False,
                    source="turn_service.finalize_user_turn.policy_trace_event",
                ),
            )
        if bool(getattr(turn_context, "metadata", {}).get("audit_mode", False)):
            mutation_queue.queue(
                MutationIntent(
                    type=MutationKind.LEDGER,
                    payload={
                        "op": LedgerMutationOp.CAPABILITY_AUDIT_EVENT.value,
                        "scenario": "runtime_turn",
                    },
                    requires_temporal=False,
                    source="turn_service.finalize_user_turn.capability_audit_event",
                ),
            )
        return dad_reply, False

    # ---------------------------------------------------------------------------
    # Public API: thin shells — accept input, invoke kernel, return output.
    # TurnService is an input adapter only.  No state mutations happen here.
    # ---------------------------------------------------------------------------

    def _resolve_persistence_service(self) -> Any:
        services = getattr(self.bot, "services", None)
        persistence_service = getattr(services, "persistence_service", None)
        if persistence_service is not None:
            return persistence_service
        registry = getattr(self.bot, "service_registry", None)
        get = getattr(registry, "get", None)
        if callable(get):
            try:
                persistence_service = get("persistence_service")
            except Exception:
                persistence_service = None
        if persistence_service is None:
            cached = getattr(self.bot, "_compat_persistence_service", None)
            if cached is not None:
                return cached
            from dadbot.managers.conversation_persistence import (
                ConversationPersistenceManager,
            )
            from dadbot.services.persistence import PersistenceService
            from dadbot.services.post_commit_worker import PostCommitWorker

            persistence_service = PersistenceService(
                ConversationPersistenceManager(self.bot),
                turn_service=self,
            )
            if getattr(self.bot, "_post_commit_worker", None) is None:
                self.bot._post_commit_worker = PostCommitWorker(self.bot)
            self.bot._compat_persistence_service = persistence_service
        return persistence_service

    @staticmethod
    def _compat_turn_context(
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> TurnContext:
        return TurnContext(user_input=user_input, attachments=attachments)

    # DEPRECATED — NO NEW CALLERS. This compat shim bridges legacy direct-call
    # surfaces that bypass the TurnGraph execution context. Target removal once
    # all callers are migrated to the graph pipeline. Expiry: 2026-Q3.
    # All callers must be migrated to process_user_message() or graph.execute().
    def _finalize_direct_compat_turn(
        self,
        stripped_input: str,
        current_mood: str | None,
        dad_reply: str | None,
        attachments: AttachmentList | None = None,
        *,
        turn_context: TurnContext | None = None,
    ) -> FinalizedTurnResult:
        return self._finalize_turn_compat_context(
            stripped_input,
            current_mood,
            dad_reply,
            attachments,
            turn_context=turn_context,
            mark_legacy_direct_compat=True,
            emit_deprecation_warning=True,
        )

    def _finalize_turn_compat_context(
        self,
        stripped_input: str,
        current_mood: str | None,
        dad_reply: str | None,
        attachments: AttachmentList | None = None,
        *,
        turn_context: TurnContext | None = None,
        mark_legacy_direct_compat: bool = False,
        emit_deprecation_warning: bool = False,
    ) -> FinalizedTurnResult:
        if emit_deprecation_warning:
            warnings.warn(
                "_finalize_direct_compat_turn is a legacy compat shim scheduled for removal in 2026-Q3. "
                "Migrate callers to process_user_message() or the TurnGraph pipeline.",
                DeprecationWarning,
                stacklevel=3,
            )
        context = turn_context or self._compat_turn_context(stripped_input, attachments)
        metadata = getattr(context, "metadata", None)
        if mark_legacy_direct_compat and isinstance(metadata, dict):
            metadata.setdefault("legacy_direct_compat", True)
        context.state["turn_text"] = stripped_input
        context.state["mood"] = current_mood or "neutral"
        context.state["norm_attachments"] = attachments or []
        persistence_service = self._resolve_persistence_service()
        with ensure_execution_trace_root(
            operation="turn_service_finalize_direct_compat_turn",
            prompt="[turn-service-direct-compat-finalize]",
            metadata={"source": "TurnService._finalize_direct_compat_turn"},
            required=True,
        ):
            return persistence_service.finalize_turn(context, (dad_reply, False))

    def _run_orchestrator_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None,
        *,
        chunk_callback: ChunkCallback | None = None,
        stream: bool = False,
    ) -> FinalizedTurnResult:
        response_obj = self.bot.execute_turn(
            live_turn_request(
                user_input,
                attachments=list(attachments or []),
                delivery=TurnDelivery.STREAM if stream else TurnDelivery.SYNC,
                context=SovereignContext(
                    session_id=str(getattr(self.bot, "active_thread_id", "") or "default"),
                    tenant_id=str(getattr(self.bot, "TENANT_ID", "default") or "default"),
                ),
            ),
            chunk_callback=chunk_callback,
        )
        if inspect.isawaitable(response_obj):
            raise RuntimeError("execute_turn returned an awaitable in sync mode")
        return self._coerce_turn_response(response_obj)

    @staticmethod
    def _coerce_turn_response(response_obj: Any) -> FinalizedTurnResult:
        if hasattr(response_obj, "as_result"):
            result_payload = response_obj.as_result()
            if isinstance(result_payload, tuple) and len(result_payload) >= 2:
                return str(result_payload[0] or ""), bool(result_payload[1])
            return str(getattr(response_obj, "reply", "") or ""), bool(
                getattr(response_obj, "should_end", False),
            )

        if isinstance(response_obj, tuple):
            return (
                str(response_obj[0] if len(response_obj) >= 1 else ""),
                bool(response_obj[1] if len(response_obj) >= 2 else False),
            )

        return str(getattr(response_obj, "reply", "") or ""), bool(
            getattr(response_obj, "should_end", False),
        )

    async def _run_orchestrator_async(
        self,
        user_input: str,
        attachments: AttachmentList | None,
        *,
        chunk_callback: ChunkCallback | None = None,
        stream: bool = False,
    ) -> FinalizedTurnResult:
        response_obj = self.bot.execute_turn(
            live_turn_request(
                user_input,
                attachments=list(attachments or []),
                delivery=TurnDelivery.STREAM_ASYNC if stream else TurnDelivery.ASYNC,
                context=SovereignContext(
                    session_id=str(getattr(self.bot, "active_thread_id", "") or "default"),
                    tenant_id=str(getattr(self.bot, "TENANT_ID", "default") or "default"),
                ),
            ),
            chunk_callback=chunk_callback,
        )
        if inspect.isawaitable(response_obj):
            response_obj = await cast("Any", response_obj)
        return self._coerce_turn_response(response_obj)

    def process_user_message(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False
        if self._legacy_pipeline_overridden():
            current_mood, direct_reply, should_end, turn_text, normalized_attachments = self.prepare_user_turn(
                stripped_input,
                attachments,
            )
            if should_end:
                return str(direct_reply or ""), True
            if direct_reply is None:
                dad_reply = self.reply_generation.generate_validated_reply(
                    stripped_input,
                    turn_text,
                    str(current_mood or "neutral"),
                    normalized_attachments,
                )
            else:
                dad_reply = str(direct_reply)
            return self.finalize_user_turn(
                stripped_input,
                current_mood,
                dad_reply,
                attachments=normalized_attachments,
            )
        self._append_turn_pipeline_step(
            "spine_delegate",
            status="running",
            detail="delegating turn execution to execute_turn spine",
        )
        result = self._run_orchestrator_sync(stripped_input, attachments)
        self._ensure_compat_pipeline_steps(
            (
                "detect_mood",
                "generate_reply",
                "finalize_turn",
            ),
        )
        self._append_turn_pipeline_step(
            "spine_delegate",
            detail="execute_turn spine completed",
        )
        return result

    async def process_user_message_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False
        self._append_turn_pipeline_step(
            "spine_delegate",
            status="running",
            detail="delegating async turn execution to execute_turn spine",
        )
        result = await self._run_orchestrator_async(stripped_input, attachments)
        self._append_turn_pipeline_step(
            "spine_delegate",
            detail="async execute_turn spine completed",
        )
        return result

    def process_user_message_stream(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False
        self._update_turn_pipeline(mode="stream_sync")
        self._append_turn_pipeline_step(
            "spine_delegate",
            status="running",
            detail="delegating streamed sync turn to execute_turn spine",
        )
        result = self._run_orchestrator_sync(
            stripped_input,
            attachments,
            chunk_callback=chunk_callback,
            stream=True,
        )
        self._append_turn_pipeline_step(
            "spine_delegate",
            detail="streamed sync execute_turn spine completed",
        )
        return result

    async def process_user_message_stream_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        chunk_callback: ChunkCallback | None = None,
    ) -> FinalizedTurnResult:
        stripped_input = user_input.strip()
        if not stripped_input and not attachments:
            return None, False

        # Compatibility branch for test harnesses that monkeypatch the stream
        # implementation directly on the bot instance. In production runtime,
        # stream execution stays on the execute_turn/control-plane spine.
        stream_impl = getattr(self.bot, "call_ollama_chat_stream_async", None)
        if callable(stream_impl) and not inspect.ismethod(stream_impl):
            current_mood, direct_reply, should_end, turn_text, normalized_attachments = await self.prepare_user_turn_async(
                stripped_input,
                attachments,
            )
            if should_end:
                return str(direct_reply or ""), True
            if direct_reply is None:
                dad_reply = await self.reply_generation.generate_validated_reply_async(
                    stripped_input,
                    turn_text,
                    str(current_mood or "neutral"),
                    normalized_attachments,
                    stream=True,
                    chunk_callback=chunk_callback,
                )
            else:
                dad_reply = str(direct_reply)

            self.bot.history.append({"role": "user", "content": stripped_input})
            self.bot.history.append({"role": "assistant", "content": str(dad_reply or "")})
            return str(dad_reply or ""), False

        streamed = False

        async def _emit_chunk(chunk: str) -> None:
            if not callable(chunk_callback):
                return
            emitted = chunk_callback(chunk)
            if inspect.isawaitable(emitted):
                await cast("Any", emitted)

        async def _wrapped_chunk_callback(chunk: str) -> None:
            nonlocal streamed
            if not chunk:
                return
            streamed = True
            await _emit_chunk(str(chunk))

        self._update_turn_pipeline(mode="stream_async")
        self._append_turn_pipeline_step(
            "spine_delegate",
            status="running",
            detail="delegating streamed async turn to execute_turn spine",
        )
        result = await self._run_orchestrator_async(
            stripped_input,
            attachments,
            chunk_callback=_wrapped_chunk_callback,
            stream=True,
        )
        if not streamed and callable(chunk_callback):
            buffered_reply = str(result[0] or "")
            if buffered_reply:
                head, sep, tail = buffered_reply.partition(" ")
                if sep:
                    await _emit_chunk(head)
                    await _emit_chunk(f"{sep}{tail}")
                else:
                    await _emit_chunk(buffered_reply)
        self._append_turn_pipeline_step(
            "spine_delegate",
            detail="streamed async execute_turn spine completed",
        )
        return result


__all__ = ["TurnService"]
