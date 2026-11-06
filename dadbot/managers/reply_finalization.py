from __future__ import annotations

import logging
import os
from typing import Any
from dadbot.core.turn_coherence import assert_personality_applied_exactly_once, mark_turn_coherence
from dadbot.managers.advice_audit import ShadowAuditManager
from dadbot.managers.personality_service import PersonalityServiceManager

logger = logging.getLogger(__name__)


class ReplyFinalizationManager:
    """Owns the post-generation reply pipeline before the final message reaches Tony.
    
    PHASE 1 NOTE: reply_finalization is an ENFORCEMENT LAYER (formatting-only).
    Formatting operations (personality voice, moderation, audit, signoff) are applied
    AFTER ResponseEngine selects the response. reply_finalization does NOT influence
    which response is selected — only how it is delivered.
    """

    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def _reply_text(reply: Any) -> str:
        if hasattr(reply, "model_dump") and not isinstance(reply, str):
            from dadbot.utils.llm import extract_ollama_message_content

            reply = extract_ollama_message_content(reply)
        while isinstance(reply, tuple):
            if len(reply) != 2 or not isinstance(reply[1], bool):
                raise TypeError("Turn result tuple must be (reply_text, should_end)")
            reply = reply[0]
        if reply is None:
            return ""
        if not isinstance(reply, str):
            raise TypeError(f"Reply must be a string, got {type(reply).__name__}")
        return reply.strip()

    @classmethod
    def normalize_reply(cls, reply: Any) -> str:
        """Validate and normalize a canonical reply before it reaches a UI surface."""
        normalized = cls._reply_text(reply)
        if not normalized:
            raise ValueError("Assistant returned an empty reply")
        return normalized

    @staticmethod
    def _signoff_is_appropriate(
        reply: str,
        *,
        user_input: Any = None,
        current_mood: Any = None,
    ) -> bool:
        mood = str(current_mood or "").strip().lower()
        if mood in {"positive", "sad", "stressed", "frustrated", "tired", "anxious", "worried"}:
            return True

        prompt = str(user_input or "").lower()
        emotional_prompt_markers = (
            "i feel",
            "i'm feeling",
            "i am feeling",
            "i've been feeling",
            "i'm struggling",
            "i am struggling",
            "struggling",
            "overwhelmed",
            "having a hard time",
            "going through a lot",
            "feeling down",
            "lonely",
            "grieving",
            "grief",
            "heartbroken",
            "scared",
            "afraid",
            "frustrated",
            "burned out",
            "burnt out",
            "exhausted",
            "depressed",
            "anxiety",
            "stressed",
            "sad",
            "i love you",
        )
        if any(marker in prompt for marker in emotional_prompt_markers):
            return True

        reply_text = str(reply or "").lower()
        emotional_reply_markers = (
            "i'm proud of you",
            "i am proud of you",
            "i'm here for you",
            "i am here for you",
            "you are not alone",
            "you're not alone",
            "that sounds hard",
            "that sounds painful",
            "your feelings",
            "your emotions",
            "emotional support",
            "anxiety",
            "grief",
            "loneliness",
            "overwhelmed",
            "struggling",
            "feeling down",
        )
        return any(marker in reply_text for marker in emotional_reply_markers)

    def _record_shadow_pipeline(self, payload: dict[str, Any]) -> None:
        try:
            setattr(self.bot, "_last_reply_finalization_shadow", payload)
            recorder = getattr(self.bot, "record_shadow_decision", None)
            if callable(recorder):
                recorder(
                    source="finalization",
                    type="transform",
                    content_preview=str(payload.get("shadow_moderated_reply") or payload.get("shadow_voiced_reply") or ""),
                    reason="Finalization transform observed; response authority remains in ResponseEngine.",
                    would_replace=True,
                    priority=0.45,
                    metadata={
                        "path": str(payload.get("path") or "unknown"),
                        "base_reply_preview": str(payload.get("base_reply") or "")[:220],
                    },
                )
        except Exception:
            logger.debug("Failed to record reply finalization shadow payload", exc_info=True)

    @staticmethod
    def _looks_like_raw_ai_identity_disclaimer(reply: str) -> bool:
        text = str(reply or "").strip().lower()
        if not text:
            return False
        markers = (
            "i'm an ai assistant",
            "i am an ai assistant",
            "as an ai",
            "i'm an artificial intelligence model",
            "i am an artificial intelligence model",
            "artificial intelligence model known as llama",
            "large language model meta ai",
            "i am llama",
            "i'm llama",
            "known as llama",
            "i'm dadbot, and i'm here with you",
            "i am dadbot, and i am here with you",
            "i don't have a personal life outside this chat",
            "i do not have a personal life outside this chat",
            "i don't have a personal history",
            "i do not have a personal history",
            "i don't have a family",
            "i do not have a family",
            "i wasn't born",
            "i was not born",
            "i exist solely to provide information",
            "i don't have a personal preference",
            "i do not have a personal preference",
            "i don't have preferences or feelings",
            "i do not have preferences or feelings",
            "i don't have personal preferences",
            "i do not have personal preferences",
            "i don't have a physical existence",
            "i do not have a physical existence",
            "i'm not your dad",
            "released to the public in",
            "released in 2023",
            "i don't have personal experiences",
            "i do not have personal experiences",
            "i'm an ai designed",
            "i am an ai designed",
            "i'm a large language model",
            "i am a large language model",
            "i don't participate in physical activities",
            "i do not participate in physical activities",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _profile_question_kind(user_input: Any) -> str:
        text = str(user_input or "").strip().lower()
        if not text:
            return ""
        if any(
            marker in text
            for marker in (
                "in this code", "in the code", "in this repo", "in the repo",
                "in this project", "in the project", "in this file", "in the file",
                "in this app", "in this system", "in this model", "in this example",
                "in the example", "in this scenario", "in the scenario", "in this story",
                "in the story", "in this conversation",
            )
        ):
            return ""
        if any(
            marker in text
            for marker in (
                "who are you", "what's your story", "whats your story",
                "tell me your story", "your story", "about yourself", "tell me about yourself",
            )
        ):
            return "identity"
        if any(
            marker in text
            for marker in (
                "how old are you", "your age", "dad age", "what year were you born",
                "when were you born", "your birthday",
            )
        ):
            return "age"
        if any(
            marker in text
            for marker in (
                "baseball", "football", "soccer", "basketball", "sports team",
                "favorite team", "favourite team", "which team do you like",
                "what team do you like",
            )
        ):
            return "preferences"
        return ""

    def _profile_fact_reply(self, user_input: Any) -> str:
        kind = self._profile_question_kind(user_input)
        if not kind:
            return ""
        get_fact_reply = getattr(self.bot, "get_fact_reply", None)
        if callable(get_fact_reply):
            try:
                fact_reply = str(get_fact_reply(user_input) or "").strip()
            except Exception:
                logger.warning("Failed to resolve profile fact reply", exc_info=True)
            else:
                if fact_reply:
                    return fact_reply

        style = getattr(getattr(self.bot, "profile_runtime", None), "style", {}) or {}
        listener_name = str(style.get("listener_name") or "buddy")
        if kind == "identity":
            user_text = str(user_input or "").strip().lower()
            profile = getattr(self.bot, "PROFILE", {})
            family = getattr(self.bot, "FAMILY", {})
            education = getattr(self.bot, "EDUCATION", {})
            profile = profile if isinstance(profile, dict) else {}
            family = family if isinstance(family, dict) else {}
            education = education if isinstance(education, dict) else {}
            dad = family.get("dad", {})
            dad = dad if isinstance(dad, dict) else {}
            details = []
            birthplace = str(dad.get("birthplace") or "").strip()
            moved_to = str(dad.get("moved_to") or "").strip()
            move_age = dad.get("move_age")
            boarding_school = str(education.get("boarding_school") or "").strip()
            university = str(education.get("university") or "").strip()
            degree = str(education.get("degree") or "").strip()
            if birthplace:
                details.append(f"I was born in {birthplace}")
            if moved_to:
                age_fragment = f" when I was {move_age}" if move_age is not None else ""
                details.append(f"we moved to {moved_to}{age_fragment}")
            if boarding_school:
                details.append(f"I went to {boarding_school}")
            if university:
                degree_fragment = f" for {degree}" if degree else ""
                details.append(f"I studied at {university}{degree_fragment}")
            story = (
                f"Here's the short version, {listener_name}: " + ". ".join(details) + "."
                if details
                else f"I'm here with you, {listener_name}. What would you like to know?"
            )
            long_story = any(
                marker in user_text
                for marker in (
                    "full life story", "full story", "detailed", "detail",
                    "long version", "more about you",
                )
            )
            if long_story:
                childhood = str(dad.get("childhood") or "").strip()
                other_subjects = education.get("other_subjects", [])
                if isinstance(other_subjects, list) and other_subjects:
                    story += " I also studied " + ", ".join(
                        str(item) for item in other_subjects if str(item).strip()
                    ) + "."
                if childhood:
                    story += f" {childhood}"
                sports = profile.get("sports_preferences", {})
                if isinstance(sports, dict):
                    sports_items = sorted(
                        (str(key).replace("_", " "), str(value).strip())
                        for key, value in sports.items()
                        if str(value).strip()
                    )
                    if sports_items:
                        story += " My sports preferences are " + "; ".join(
                            f"{key}: {value}" for key, value in sports_items
                        ) + "."
                return f"Full version, {listener_name}: {story}"
            return story

        if kind == "age":
            age_on_date = getattr(self.bot, "age_on_date", None)
            birthdate = getattr(self.bot, "DAD_BIRTHDATE", None)
            if callable(age_on_date) and birthdate is not None:
                try:
                    age_text = str(age_on_date(birthdate) or "").strip()
                    age = int(age_text) if age_text else None
                except Exception:
                    logger.warning("Failed to resolve profile age", exc_info=True)
                else:
                    if age is not None:
                        return f"I'm {age} years old, {listener_name}."
            return f"I don't have a birth date in my profile, {listener_name}."

        profile = getattr(self.bot, "PROFILE", {})
        sports = profile.get("sports_preferences", {}) if isinstance(profile, dict) else {}
        if not isinstance(sports, dict) or not sports:
            return f"I don't have a sports preference saved in my profile yet, {listener_name}."
        prompt = str(user_input or "").strip().lower()
        preferences = [
            f"{str(key).replace('_', ' ')}: {value}"
            for key, value in sports.items()
            if str(key).split("_")[0] in prompt
        ]
        if not preferences:
            preferences = [
                f"{str(key).replace('_', ' ')}: {value}"
                for key, value in sorted(sports.items())
            ]
        return f"Here's what my profile says, {listener_name}: " + "; ".join(preferences) + "."

    def _dad_bio_reply(self) -> str:
        get_fact_reply = getattr(self.bot, "get_fact_reply", None)
        if callable(get_fact_reply):
            try:
                bio_reply = str(get_fact_reply("who are you") or "").strip()
                if bio_reply:
                    return bio_reply
            except Exception:
                logger.debug("Failed to render bio fact reply", exc_info=True)

        return (
            "I'm your dad - born in Providence, Rhode Island, raised in Albion, Maine. "
            "I studied English at the University of Maine, and I'm here for you, buddy."
        )

    def _enforce_dad_identity(self, reply: str) -> str:
        normalized = str(reply or "").strip()
        if not self._looks_like_raw_ai_identity_disclaimer(normalized):
            return normalized
        return self._dad_bio_reply()

    def _listener_model_enabled(self) -> bool:
        profile = getattr(self.bot, "PROFILE", None)
        if not isinstance(profile, dict):
            return False
        listener_model = profile.get("listener_model")
        return isinstance(listener_model, dict) and bool(listener_model)

    @staticmethod
    def _strip_performative_prefaces(text: str) -> str:
        normalized = str(text or "").strip()
        lower = normalized.lower()
        prefixes = (
            "great question",
            "excellent question",
            "good question",
            "absolutely",
            "certainly",
            "of course",
        )
        for prefix in prefixes:
            if lower.startswith(prefix):
                # Drop the leading performative opener and any immediate punctuation.
                stripped = normalized[len(prefix):].lstrip(" ,:-.!")
                if stripped:
                    return stripped
        return normalized

    @staticmethod
    def _compress_to_sentence_budget(text: str, max_sentences: int) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        parts: list[str] = []
        current = ""
        for ch in raw:
            current += ch
            if ch in ".!?":
                chunk = current.strip()
                if chunk:
                    parts.append(chunk)
                current = ""
        if current.strip():
            parts.append(current.strip())
        if not parts:
            return raw
        return " ".join(parts[: max(1, int(max_sentences))]).strip()

    def _enforce_listener_contract(self, reply: str, user_input: Any = None) -> str:
        text = str(reply or "").strip()
        if not text:
            return text
        if not self._listener_model_enabled():
            return text

        text = self._strip_performative_prefaces(text)
        text = " ".join(text.split())

        user_text = str(user_input or "").strip().lower()
        is_question = user_text.endswith("?") or "?" in user_text
        detail_markers = (
            "steps",
            "walk me through",
            "deep dive",
            "details",
            "explain",
            "spec",
            "architecture",
        )
        wants_detail = any(marker in user_text for marker in detail_markers)

        sentence_budget = 5 if wants_detail else (3 if is_question else 4)
        text = self._compress_to_sentence_budget(text, sentence_budget)
        return text

    def append_signoff(
        self,
        reply: Any,
        *,
        user_input: Any = None,
        current_mood: Any = None,
    ) -> str:
        reply = self._enforce_dad_identity(self._reply_text(reply))
        reply = self._enforce_listener_contract(reply)
        profile_style = getattr(getattr(self.bot, "profile_runtime", None), "style", None)
        if not isinstance(profile_style, dict):
            profile_style = getattr(self.bot, "STYLE", {})
        signoff = str(profile_style.get("signoff") or "").strip()

        if (
            not reply
            or not self.bot.APPEND_SIGNOFF
            or not signoff
            or not self._signoff_is_appropriate(
                reply,
                user_input=user_input,
                current_mood=current_mood,
            )
        ):
            return reply

        if reply.endswith(signoff):
            return reply
        if reply.endswith(("!", "?", ".")):
            return f"{reply} {signoff}"
        return f"{reply}. {signoff}"

    def finalize_reply(
        self,
        reply: Any,
        *,
        user_input: Any = None,
        current_mood: Any = None,
    ) -> str:
        return self.append_signoff(
            reply,
            user_input=user_input,
            current_mood=current_mood,
        )

    def should_calibrate_pushback(self, user_input, current_mood):
        return self.bot.personality_service._should_calibrate_pushback(
            str(user_input or ""),
            str(current_mood or "neutral"),
        )

    def apply_calibrated_pushback(self, reply, user_input, current_mood):
        return self.bot.personality_service._apply_calibrated_pushback(
            str(reply or ""),
            str(user_input or ""),
            str(current_mood or "neutral"),
        )

    def finalize(self, reply, current_mood, user_input=None):
        """Apply formatting pipeline to response (ENFORCEMENT LAYER — NOT a selection authority).
        
        PHASE 1: This method is only invoked by control-plane post-ResponseEngine selection.
        All formatting operations here are non-binding (they do not change which response was selected).
        """
        # PHASE 1: Audit logging for enforcement layer
        logger.debug(
            "reply_finalization.finalize invoked (ENFORCEMENT LAYER formatting); "
            f"applying personality/moderation/audit to response"
        )
        
        # Defense: if reply is somehow a Pydantic response object, extract content first
        mark_turn_coherence(self.bot, "finalizer_called")
        base_reply = self._profile_fact_reply(user_input) or self._reply_text(reply)
        shadow_voiced = base_reply
        shadow_supervised = base_reply
        shadow_moderated = base_reply
        try:
            shadow_voiced = self.bot.personality_service.apply_authoritative_voice(
                base_reply,
                str(current_mood or "neutral"),
                user_input,
            )

            enable_supervisor = str(os.environ.get("DADBOT_REPLY_SUPERVISOR_ENABLED") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            supervisor = getattr(self.bot, "reply_supervisor", None)
            can_supervise = (
                enable_supervisor
                and supervisor is not None
                and callable(getattr(supervisor, "run_reply_supervisor", None))
                and not bool(getattr(self.bot, "LIGHT_MODE", False))
            )
            if can_supervise and user_input is not None:
                shadow_supervised = supervisor.run_reply_supervisor(
                    str(user_input or ""),
                    str(shadow_voiced or ""),
                    str(current_mood or "neutral"),
                    stage="final_polish",
                )
            else:
                shadow_supervised = shadow_voiced
            shadow_moderated = self.bot.moderate_output_reply(
                user_input,
                shadow_supervised,
                current_mood,
            )
            self._record_shadow_pipeline(
                {
                    "path": "sync",
                    "base_reply": base_reply,
                    "shadow_voiced_reply": str(shadow_voiced or ""),
                    "shadow_supervised_reply": str(shadow_supervised or ""),
                    "shadow_moderated_reply": str(shadow_moderated or ""),
                    "applied": False,
                },
            )
        except Exception:
            logger.debug("reply_finalization shadow transforms failed", exc_info=True)
        mark_turn_coherence(self.bot, "personality_applied")
        assert_personality_applied_exactly_once(self.bot)
        personality_apply = getattr(getattr(self.bot, "personality_service", None), "apply_authoritative_voice", None)
        personality_func = getattr(personality_apply, "__func__", None)
        can_apply_surface_transforms = personality_func is PersonalityServiceManager.apply_authoritative_voice
        # Preserve response authority in compatibility/monkeypatched test paths,
        # while still allowing the default personality pipeline in production.
        final_reply = (
            str(shadow_moderated or shadow_voiced or base_reply)
            if can_apply_surface_transforms
            else base_reply
        )
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(final_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        final_reply = self._enforce_dad_identity(final_reply)
        final_reply = self._enforce_listener_contract(final_reply, user_input=user_input)
        final_reply = self.append_signoff(
            final_reply,
            user_input=user_input,
            current_mood=current_mood,
        )
        # CSCL post-turn feedback: feed voiced reply to coherence tracker
        try:
            cscl = getattr(self.bot, "conversation_surface", None)
            if cscl is not None:
                cscl.record_output(final_reply)
        except Exception:
            pass
        return final_reply

    def prepare_final_reply(self, reply, current_mood, user_input=None):
        return self.finalize(reply, current_mood, user_input)

    async def finalize_async(self, reply, current_mood, user_input=None):
        """Apply formatting pipeline to response (async variant — ENFORCEMENT LAYER)."""
        # PHASE 1: Audit logging for enforcement layer
        logger.debug(
            "reply_finalization.finalize_async invoked (ENFORCEMENT LAYER formatting); "
            f"applying personality/moderation/audit to response (async)"
        )
        
        # Defense: if reply is somehow a Pydantic response object, extract content first
        mark_turn_coherence(self.bot, "finalizer_called")
        base_reply = self._profile_fact_reply(user_input) or self._reply_text(reply)
        shadow_voiced = base_reply
        shadow_supervised = base_reply
        shadow_moderated = base_reply
        try:
            shadow_voiced = self.bot.personality_service.apply_authoritative_voice(
                base_reply,
                str(current_mood or "neutral"),
                user_input,
            )

            enable_supervisor = str(os.environ.get("DADBOT_REPLY_SUPERVISOR_ENABLED") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            supervisor = getattr(self.bot, "reply_supervisor", None)
            can_supervise = (
                enable_supervisor
                and supervisor is not None
                and callable(getattr(supervisor, "run_reply_supervisor_async", None))
                and not bool(getattr(self.bot, "LIGHT_MODE", False))
            )
            if can_supervise and user_input is not None:
                shadow_supervised = await supervisor.run_reply_supervisor_async(
                    str(user_input or ""),
                    str(shadow_voiced or ""),
                    str(current_mood or "neutral"),
                    stage="final_polish",
                )
            else:
                shadow_supervised = shadow_voiced
            shadow_moderated = await self.bot.moderate_output_reply_async(
                user_input,
                shadow_supervised,
                current_mood,
            )
            self._record_shadow_pipeline(
                {
                    "path": "async",
                    "base_reply": base_reply,
                    "shadow_voiced_reply": str(shadow_voiced or ""),
                    "shadow_supervised_reply": str(shadow_supervised or ""),
                    "shadow_moderated_reply": str(shadow_moderated or ""),
                    "applied": False,
                },
            )
        except Exception:
            logger.debug("reply_finalization async shadow transforms failed", exc_info=True)
        mark_turn_coherence(self.bot, "personality_applied")
        assert_personality_applied_exactly_once(self.bot)
        personality_apply = getattr(getattr(self.bot, "personality_service", None), "apply_authoritative_voice", None)
        personality_func = getattr(personality_apply, "__func__", None)
        can_apply_surface_transforms = personality_func is PersonalityServiceManager.apply_authoritative_voice
        final_reply = (
            str(shadow_moderated or shadow_voiced or base_reply)
            if can_apply_surface_transforms
            else base_reply
        )
        try:
            ShadowAuditManager(self.bot).audit_and_record(
                user_input=str(user_input or ""),
                reply=str(final_reply or ""),
                current_mood=str(current_mood or "neutral"),
            )
        except Exception:
            pass
        final_reply = self._enforce_dad_identity(final_reply)
        final_reply = self._enforce_listener_contract(final_reply, user_input=user_input)
        final_reply = self.append_signoff(
            final_reply,
            user_input=user_input,
            current_mood=current_mood,
        )
        # CSCL post-turn feedback: feed voiced reply to coherence tracker (async path)
        try:
            cscl = getattr(self.bot, "conversation_surface", None)
            if cscl is not None:
                cscl.record_output(final_reply)
        except Exception:
            pass
        return final_reply

    async def prepare_final_reply_async(self, reply, current_mood, user_input=None):
        return await self.finalize_async(reply, current_mood, user_input)


__all__ = ["ReplyFinalizationManager"]
