from __future__ import annotations

import logging
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dadbot.core.turn_coherence import mark_turn_coherence
from dadbot.runtime.model.model_call_port import ModelPort

logger = logging.getLogger(__name__)


class _EmotionalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mood: Literal["positive", "neutral", "stressed", "sad", "frustrated", "tired"]
    reply: str = Field(min_length=1)


class AgentService:
    """Generate non-authoritative model candidates for the turn graph."""

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
        return dict(
            mapping.get(
                str(hypothesis_name or ""),
                {
                    "policy": "supportive_problem_solving",
                    "tool_bias": "planner_default",
                },
            ),
        )

    def _bayesian_execution_state(
        self,
        *,
        turn_text: str,
        current_mood: str,
    ) -> dict[str, Any]:
        relationship = getattr(self.bot, "relationship_manager", None)
        if relationship is None:
            return {
                "required": True,
                "applied": False,
                "reason": "relationship_manager_unavailable",
                "policy": "supportive_problem_solving",
            }

        state = relationship.current_state()
        posterior = relationship.build_hypothesis_posteriors(
            state,
            turn_text,
            current_mood,
        )
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
    def _tool_execution_envelope(
        bot: Any,
        turn_context: Any,
        *,
        reply_source: str,
    ) -> dict[str, Any]:
        planner_snapshot = {}
        planner_debug_snapshot = getattr(bot, "planner_debug_snapshot", None)
        if callable(planner_debug_snapshot):
            try:
                planner_snapshot = dict(planner_debug_snapshot() or {})
            except Exception:
                planner_snapshot = {}

        selected_tool = str(
            planner_snapshot.get("planner_tool") or planner_snapshot.get("fallback_tool") or "",
        ).strip()
        parameters = dict(planner_snapshot.get("planner_parameters") or {})
        observation = str(
            planner_snapshot.get("planner_observation")
            or planner_snapshot.get("fallback_observation")
            or turn_context.state.get("active_tool_observation")
            or "",
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
            logger.error(
                "AgentService: bot.turn_service is not wired; cannot run inference",
            )
            return None
        try:
            return await service.prepare_user_turn_async(
                stripped,
                attachments,
                turn_context=turn_context,
            )
        except Exception as exc:
            logger.error("AgentService: prepare_user_turn_async failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _record_turn_preinference_state(
        turn_context: Any,
        *,
        mood: Any,
        turn_text: str,
        stripped: str,
        norm_attachments: Any,
    ) -> None:
        turn_context.state["mood"] = mood or "neutral"
        turn_context.state["turn_text"] = turn_text or stripped
        turn_context.state["norm_attachments"] = norm_attachments

    def _record_bayesian_state(
        self,
        bot: Any,
        turn_context: Any,
        *,
        turn_text: str,
        stripped: str,
        mood: Any,
    ) -> dict[str, Any]:
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

    async def run_agent(
        self,
        turn_context: Any,
        rich_context: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        return await self.generate_candidate(turn_context, rich_context)

    async def generate_candidate(
        self,
        turn_context: Any,
        _rich_context: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        bot = self.bot
        _, attachments, stripped = self._normalize_turn_input(turn_context)

        if not stripped and not attachments:
            turn_context.state["already_finalized"] = True
            return ("", False)

        batch_mood = bool(
            not bot.LIGHT_MODE
            and bot.mood_manager.needs_model_detection(stripped, bot.prompt_history())
        )
        turn_context.state["_batch_mood_with_reply"] = batch_mood
        prepared = await self._prepare_user_turn(
            bot,
            stripped,
            attachments,
            turn_context,
        )
        if prepared is None:
            raise RuntimeError("AgentService could not prepare the user turn")
        mood, early_reply, should_end, turn_text, norm_attachments = prepared
        if batch_mood and early_reply is not None and not should_end:
            bot.turn_service.record_user_turn_state(turn_text, str(mood or "neutral"), turn_context=turn_context)

        self._record_turn_preinference_state(
            turn_context,
            mood=mood,
            turn_text=turn_text,
            stripped=stripped,
            norm_attachments=norm_attachments,
        )
        self._record_bayesian_state(
            bot,
            turn_context,
            turn_text=turn_text,
            stripped=stripped,
            mood=mood,
        )
        self._record_relationship_snapshot(bot, turn_context)

        # Session exit: prepare_user_turn_async already persisted; skip SaveNode work.
        if should_end:
            turn_context.state["already_finalized"] = True
            turn_context.state["tool_execution_envelope"] = self._tool_execution_envelope(
                bot,
                turn_context,
                reply_source="agent_service",
            )
            final_reply = (
                early_reply[0]
                if isinstance(early_reply, tuple) and early_reply
                else early_reply
            )
            return str(final_reply or ""), True

        turn_context.state["tool_execution_envelope"] = self._tool_execution_envelope(
            bot,
            turn_context,
            reply_source="agent_service",
        )
        if early_reply is not None:
            if isinstance(early_reply, tuple) and len(early_reply) >= 2:
                return str(early_reply[0] or ""), bool(early_reply[1])
            return str(early_reply or ""), False

        prompt_assembly = getattr(bot, "prompt_assembly", None)
        build_messages = getattr(prompt_assembly, "build_chat_request_messages", None)
        if not callable(build_messages):
            raise RuntimeError("AgentService requires prompt_assembly.build_chat_request_messages")
        messages_value = build_messages(
            turn_text or stripped,
            str(mood or "neutral"),
            norm_attachments,
        )
        if not isinstance(messages_value, list):
            raise RuntimeError("Prompt assembly returned an invalid chat message list")
        messages: list[dict[str, Any]] = []
        for message in messages_value:
            if not isinstance(message, dict):
                raise RuntimeError("Prompt assembly returned a non-dictionary chat message")
            normalized_message: dict[str, Any] = {}
            for key, value in message.items():
                if not isinstance(key, str):
                    raise RuntimeError("Prompt assembly returned a non-string chat message key")
                normalized_message[key] = value
            messages.append(normalized_message)
        model_port = getattr(bot, "model_port", None)
        if not isinstance(model_port, ModelPort):
            raise RuntimeError("AgentService requires a wired asynchronous model port")
        active_model = str(getattr(bot, "ACTIVE_MODEL", "") or "").strip()
        if not active_model:
            raise RuntimeError("AgentService requires a configured active model")
        generation_kwargs: dict[str, Any] = {}
        if batch_mood:
            instruction = (
                "Classify the user's emotional state and answer in the same response. "
                "Return JSON with exactly mood and reply. Mood must be positive, neutral, "
                "stressed, sad, frustrated, or tired. The reply is the complete natural "
                "answer to the user, not an explanation of the classification. "
                "Be supportive and concise; acknowledge the feeling before giving advice. "
                "Do not add a closing signoff; the reply finalizer adds it separately. "
                "Do not include analysis or reasoning in either field."
            )
            cadence = bot.cadence_settings()
            if bot.session_turn_count() % cadence["family_echo_turn_interval"] == 0:
                instruction += (
                    "\nIf mood is sad, stressed, or positive and the user hasn't mentioned "
                    "their mother, you may weave this optional family reference into reply "
                    "instead of generating it separately. Use the family facts below only; "
                    "ignore their standalone output instructions and keep the JSON contract:\n"
                    + bot.build_family_echo_prompt(turn_text, str(mood or "neutral"))
                )
            if messages and messages[0].get("role") == "system":
                messages[0] = {**messages[0], "content": f"{instruction}\n\n{messages[0].get('content', '')}"}
            else:
                messages.insert(0, {"role": "system", "content": instruction})
            generation_kwargs["response_format"] = _EmotionalAnswer.model_json_schema()
            if bot.model_runtime.normalized_llm_provider() == "ollama":
                generation_kwargs["think"] = False
        response = await model_port.generate_async(messages, model=active_model, purpose="chat", **generation_kwargs)
        if batch_mood:
            answer = _EmotionalAnswer.model_validate_json(str(response or ""))
            mark_turn_coherence(bot, "family_echo_batched")
            mood = answer.mood
            response = answer.reply
            self._record_turn_preinference_state(
                turn_context, mood=mood, turn_text=turn_text, stripped=stripped,
                norm_attachments=norm_attachments,
            )
            self._record_bayesian_state(bot, turn_context, turn_text=turn_text, stripped=stripped, mood=mood)
            bot.turn_service.record_user_turn_state(turn_text, mood, turn_context=turn_context)
            bot.turn_service._update_turn_pipeline(current_mood=mood)
            cache_key, _ = bot.mood_manager.get_cached_mood_detection(turn_text, bot.prompt_history())
            bot.mood_manager.remember_mood_detection(cache_key, mood)
        reply = str(response or "").strip()
        if not reply:
            raise RuntimeError("AgentService received an empty model response")
        return reply, False
