from __future__ import annotations

import logging
import math
from typing import Any

from dadbot.core.determinism import DeterminismBoundary, DeterminismMode, DeterminismViolation

logger = logging.getLogger(__name__)


class AgentService:
    """Service wrapper for primary LLM/tool turn execution.

    Drives the InferenceNode of the TurnGraph: runs mood detection, direct reply
    checks, agentic tool planning, and LLM generation.  The turn is NOT finalized
    here â€” history append, maintenance scheduling, and persistence are the
    SaveNode / PersistenceService responsibility.

    Calling ``bot.process_user_message*`` here would re-enter the graph and cause
    infinite recursion.  Instead, we call ``bot.turn_service.prepare_user_turn_async``
    directly (which bypasses the graph check) and then ``reply_generation.generate_validated_reply``.
    """

    def __init__(self, bot: Any):
        self.bot = bot

    @staticmethod
    def _policy_for_hypothesis(hypothesis_name: str) -> dict[str, str]:
        mapping = {
            "acute_stress": {
                "policy": "stabilize_then_solve",
                "tool_bias": "minimal_tools",
            },
            "supportive_baseline": {
                "policy": "supportive_problem_solving",
                "tool_bias": "planner_default",
            },
            "guarded_distance": {
                "policy": "low_pressure_support",
                "tool_bias": "defer_tools_unless_explicit",
            },
            "positive_rebound": {
                "policy": "reinforce_momentum",
                "tool_bias": "optional_tools",
            },
        }
        return dict(mapping.get(str(hypothesis_name or ""), {
            "policy": "supportive_problem_solving",
            "tool_bias": "planner_default",
        }))

    def _bayesian_execution_state(self, *, turn_text: str, current_mood: str) -> dict[str, Any]:
        relationship = getattr(self.bot, "relationship_manager", None)
        if relationship is None:
            return {
                "required": True,
                "applied": False,
                "reason": "relationship_manager_unavailable",
                "policy": "supportive_problem_solving",
            }

        state = relationship.current_state()
        posterior = relationship.build_hypothesis_posteriors(state, turn_text, current_mood)
        if not posterior:
            return {
                "required": True,
                "applied": False,
                "reason": "empty_posterior",
                "policy": "supportive_problem_solving",
            }

        top = posterior[0]
        entropy = 0.0
        for entry in posterior:
            prob = float(entry.get("probability", 0.0) or 0.0)
            if prob > 0:
                entropy -= prob * math.log(prob, 2)

        policy = self._policy_for_hypothesis(str(top.get("name") or ""))
        return {
            "required": True,
            "applied": True,
            "active_hypothesis": str(top.get("name") or "supportive_baseline"),
            "active_probability": float(top.get("probability", 0.0) or 0.0),
            "entropy_bits": round(entropy, 4),
            "policy": policy["policy"],
            "tool_bias": policy["tool_bias"],
            "posterior": [
                {
                    "name": str(entry.get("name") or ""),
                    "probability": float(entry.get("probability", 0.0) or 0.0),
                }
                for entry in posterior
            ],
        }

    @staticmethod
    def _tool_execution_envelope(bot: Any, turn_context: Any, *, reply_source: str) -> dict[str, Any]:
        planner_snapshot = {}
        planner_debug_snapshot = getattr(bot, "planner_debug_snapshot", None)
        if callable(planner_debug_snapshot):
            try:
                planner_snapshot = dict(planner_debug_snapshot() or {})
            except Exception:
                planner_snapshot = {}

        selected_tool = str(
            planner_snapshot.get("planner_tool")
            or planner_snapshot.get("fallback_tool")
            or ""
        ).strip()
        parameters = dict(planner_snapshot.get("planner_parameters") or {})
        observation = str(
            planner_snapshot.get("planner_observation")
            or planner_snapshot.get("fallback_observation")
            or turn_context.state.get("active_tool_observation")
            or ""
        ).strip()

        return {
            "executor": "turn_service_tool_executor_v1",
            "status": str(planner_snapshot.get("planner_status") or "idle"),
            "final_path": str(planner_snapshot.get("final_path") or "model_reply"),
            "selected_tool": selected_tool or None,
            "parameters": parameters,
            "observation": observation,
            "reply_source": str(reply_source or "model_generation"),
            "deterministic": True,
        }

    @staticmethod
    def _normalize_turn_input(turn_context: Any) -> tuple[str, Any, str]:
        user_input = str(getattr(turn_context, "user_input", "") or "")
        attachments = getattr(turn_context, "attachments", None)
        return user_input, attachments, user_input.strip()

    async def _prepare_user_turn(
        self,
        bot: Any,
        stripped: str,
        attachments: Any,
        turn_context: Any,
    ) -> tuple[Any, Any, Any, str, Any] | None:
        service = bot.turn_service
        if service is None:
            logger.error("AgentService: bot.turn_service is not wired; cannot run inference")
            return None
        try:
            return await service.prepare_user_turn_async(stripped, attachments, turn_context=turn_context)
        except Exception as exc:
            logger.error("AgentService: prepare_user_turn_async failed: %s", exc)
            return None

    @staticmethod
    def _record_turn_preinference_state(turn_context: Any, *, mood: Any, turn_text: str, stripped: str, norm_attachments: Any) -> None:
        turn_context.state["mood"] = mood or "neutral"
        turn_context.state["turn_text"] = turn_text or stripped
        turn_context.state["norm_attachments"] = norm_attachments

    def _record_bayesian_state(self, bot: Any, turn_context: Any, *, turn_text: str, stripped: str, mood: Any) -> dict[str, Any]:
        bayesian_state = self._bayesian_execution_state(
            turn_text=turn_text or stripped,
            current_mood=mood or "neutral",
        )
        turn_context.state["bayesian_state"] = bayesian_state
        turn_context.metadata["bayesian_state"] = bayesian_state
        tool_bias = str(bayesian_state.get("tool_bias") or "planner_default")
        try:
            bot.update_planner_debug(
                bayesian_tool_bias=tool_bias,
                bayesian_policy=str(bayesian_state.get("policy") or ""),
            )
        except Exception:
            pass
        return bayesian_state

    @staticmethod
    def _record_relationship_snapshot(bot: Any, turn_context: Any) -> None:
        try:
            rel_state = bot.relationship_manager.current_state()
            turn_context.state["tony_score"] = rel_state.get("score", 50)
            turn_context.state["tony_level"] = rel_state.get("level", "steady")
            turn_context.metadata["relationship"] = rel_state
        except Exception:
            turn_context.state.setdefault("tony_score", 50)
            turn_context.state.setdefault("tony_level", "steady")

    @staticmethod
    def _llm_reply_with_boundary(
        *,
        bot: Any,
        turn_context: Any,
        stripped: str,
        turn_text: str,
        mood: Any,
        norm_attachments: Any,
    ) -> str:
        reply_gen = getattr(bot, "reply_generation", None)
        if reply_gen is None:
            logger.error("AgentService: bot.reply_generation is not wired")
            return "I can't generate a reply right now. Try again."

        boundary: DeterminismBoundary = getattr(turn_context, "determinism_boundary", None) or DeterminismBoundary()

        def _llm_call():
            return reply_gen.generate_validated_reply(
                stripped,
                turn_text or stripped,
                mood or "neutral",
                norm_attachments,
                stream=False,
            )

        try:
            reply = boundary.capture("inference.llm_reply", _llm_call)
        except DeterminismViolation as exc:
            logger.error("AgentService: determinism boundary violated: %s", exc)
            reply = "I'm having trouble saving my thoughts right now - let me try again in a moment."
        except Exception as exc:
            logger.error("AgentService: LLM inference failed: %s", exc)
            reply = "I'm having trouble saving my thoughts right now - let me try again in a moment."

        boundary_snapshot = boundary.snapshot()
        turn_context.state["determinism_boundary"] = boundary_snapshot
        turn_context.metadata["determinism_boundary"] = boundary_snapshot
        return reply

    async def run_agent(self, turn_context: Any, rich_context: dict[str, Any] | None = None):
        bot = self.bot
        _user_input, attachments, stripped = self._normalize_turn_input(turn_context)

        if not stripped and not attachments:
            turn_context.state["already_finalized"] = True
            return ("", False)

        prepared = await self._prepare_user_turn(bot, stripped, attachments, turn_context)
        if prepared is None:
            return ("I'm having trouble saving my thoughts right now - let me try again in a moment.", False)
        mood, early_reply, should_end, turn_text, norm_attachments = prepared

        self._record_turn_preinference_state(
            turn_context,
            mood=mood,
            turn_text=turn_text,
            stripped=stripped,
            norm_attachments=norm_attachments,
        )
        self._record_bayesian_state(bot, turn_context, turn_text=turn_text, stripped=stripped, mood=mood)
        self._record_relationship_snapshot(bot, turn_context)

        # Session exit: prepare_user_turn_async already persisted; skip SaveNode work.
        if should_end:
            turn_context.state["already_finalized"] = True
            turn_context.state["tool_execution_envelope"] = self._tool_execution_envelope(
                bot,
                turn_context,
                reply_source="session_exit",
            )
            return (early_reply or "", True)

        # Direct reply (safety intercept, memory command, tool command, fact reply)
        if early_reply is not None:
            turn_context.state["tool_execution_envelope"] = self._tool_execution_envelope(
                bot,
                turn_context,
                reply_source="direct_or_tool_reply",
            )
            return (early_reply, False)

        reply = self._llm_reply_with_boundary(
            bot=bot,
            turn_context=turn_context,
            stripped=stripped,
            turn_text=turn_text,
            mood=mood,
            norm_attachments=norm_attachments,
        )

        turn_context.state["tool_execution_envelope"] = self._tool_execution_envelope(
            bot,
            turn_context,
            reply_source="model_generation",
        )

        return (reply, False)
