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
    here — history append, maintenance scheduling, and persistence are the
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

    async def run_agent(self, turn_context: Any, rich_context: dict[str, Any] | None = None):
        bot = self.bot
        user_input = str(getattr(turn_context, "user_input", "") or "")
        attachments = getattr(turn_context, "attachments", None)
        stripped = user_input.strip()

        if not stripped and not attachments:
            turn_context.state["already_finalized"] = True
            return ("", False)

        # Delegate pre-LLM work to TurnService (mood, direct reply,
        # agentic tool planning) without going through DadBot.process_user_message*
        # which would re-enter the graph and create infinite recursion.
        turn_service = getattr(bot, "turn_service", None)
        if turn_service is None:
            logger.error("AgentService: bot.turn_service is not wired; cannot run inference")
            return ("I'm having trouble thinking right now. Try again in a moment.", False)

        try:
            mood, early_reply, should_end, turn_text, norm_attachments = (
                await turn_service.prepare_user_turn_async(stripped, attachments)
            )
        except Exception as exc:
            logger.error("AgentService: prepare_user_turn_async failed: %s", exc)
            return ("Something went sideways on my end. Give me a second.", False)

        # Stash turn context metadata for downstream nodes (SafetyNode, SaveNode)
        turn_context.state["mood"] = mood or "neutral"
        turn_context.state["turn_text"] = turn_text or stripped
        turn_context.state["norm_attachments"] = norm_attachments

        # Required Bayesian step in the execution path: belief update + policy.
        bayesian_state = self._bayesian_execution_state(
            turn_text=turn_text or stripped,
            current_mood=mood or "neutral",
        )
        turn_context.state["bayesian_state"] = bayesian_state
        turn_context.metadata["bayesian_state"] = bayesian_state

        # Push the governing tool_bias from the Bayesian state into the planner debug
        # store so that plan_agentic_tools (which is called by prepare_user_turn_async
        # earlier in this same turn) can gate tool selection correctly.
        # NOTE: prepare_user_turn_async is called above; the governing bias is persisted
        # here for subsequent turns and for the tool envelope metadata.
        tool_bias = str(bayesian_state.get("tool_bias") or "planner_default")
        try:
            bot.update_planner_debug(bayesian_tool_bias=tool_bias, bayesian_policy=str(bayesian_state.get("policy") or ""))
        except Exception:
            pass

        # Record TONY relationship score AFTER prepare_user_turn_async has updated
        # relationship state so the score reflects the current turn.
        try:
            rel_state = bot.relationship_manager.current_state()
            turn_context.state["tony_score"] = rel_state.get("score", 50)
            turn_context.state["tony_level"] = rel_state.get("level", "steady")
            turn_context.metadata["relationship"] = rel_state
        except Exception:
            turn_context.state.setdefault("tony_score", 50)
            turn_context.state.setdefault("tony_level", "steady")

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

        # LLM inference — routed through the determinism boundary so that:
        #   RECORD mode: call LLM, seal the response for this turn.
        #   REPLAY mode: return sealed response without calling LLM.
        #   OPEN mode:   call LLM directly (no enforcement).
        reply_gen = getattr(bot, "reply_generation", None)
        if reply_gen is None:
            logger.error("AgentService: bot.reply_generation is not wired")
            return ("I can't generate a reply right now. Try again.", False)

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
            reply = "Something went sideways on my end. Try again in a moment."
        except Exception as exc:
            logger.error("AgentService: LLM inference failed: %s", exc)
            reply = "Something went sideways on my end. Try again in a moment."

        # Persist boundary snapshot into turn state after execution.
        turn_context.state["determinism_boundary"] = boundary.snapshot()
        turn_context.metadata["determinism_boundary"] = boundary.snapshot()

        turn_context.state["tool_execution_envelope"] = self._tool_execution_envelope(
            bot,
            turn_context,
            reply_source="model_generation",
        )

        return (reply, False)
