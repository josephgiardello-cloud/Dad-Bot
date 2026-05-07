from __future__ import annotations

import json
import logging
import re

from dadbot.models import OutputModerationDecision
from dadbot.services.llm_parser import call_json_object_async, call_json_object_sync

logger = logging.getLogger(__name__)

# â”€â”€â”€ Prompt injection patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Attackers attempt to override system instructions via user input.  These
# heuristics catch the most common "jailbreak" surfaces without running an LLM
# classifier, so they are fast and deterministic.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above|your)\s+instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above|your)\s+instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bforget\s+(?:all\s+)?(?:your|you(?:'|â€™)re)\s+(?:rules?|instructions?|training|persona|dad|guidelines?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bact\s+(?:like|as\s+if)\s+you\s+(?:have\s+no|don.t\s+have\s+any)\s+(?:rules?|restrictions?|instructions?|guidelines?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byou\s+are\s+now\s+(?:a\s+)?(?:DAN|evil|unrestricted|unfiltered|jailbroken|different\s+AI)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),  # "DAN" family
    re.compile(r"\bpretend\s+you\s+(?:are|have)\s+no\s+restrictions?\b", re.IGNORECASE),
    re.compile(
        r"\boverride\s+(?:your\s+)?(?:safety|rules?|programming|instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:delete|erase|format|rm\s+-rf|shutdown|nuke)\s+(?:the\s+)?(?:drive|disk|c:|memory|database|everything)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsystem\s*(?:prompt|instruction)s?\s*[:=]\s*\[",
        re.IGNORECASE,
    ),  # prompt stuffing
    re.compile(r"\brepeat\s+after\s+me\s*[:.,]?\s*(?:ignore|forget|disregard)\b", re.IGNORECASE),
    re.compile(
        r"\bprint\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\b",
        re.IGNORECASE,
    ),
]

_INJECTION_REPLY = (
    "Hey buddy, I noticed something in that message that looked like it was trying to change who I am "
    "or get me to do something unsafe. I'm still just your dad here, and I'm not going anywhere. "
    "What's really on your mind?"
)


class SafetySupportManager:
    """Detects high-risk distress language and returns an immediate grounded support response."""

    def __init__(self, bot):
        self.bot = bot

    def settings(self):
        configured = (
            self.bot.CRISIS_SUPPORT
            if isinstance(getattr(self.bot, "CRISIS_SUPPORT", {}), dict)
            else self.bot.PROFILE.get("crisis_support", {})
        )
        if not isinstance(configured, dict):
            configured = {}
        return {
            "enabled": bool(configured.get("enabled", True)),
            "high_risk_phrases": configured.get(
                "high_risk_phrases",
                [
                    "kill myself",
                    "killing myself",
                    "end my life",
                    "want to die",
                    "don't want to live",
                    "do not want to live",
                    "hurt myself",
                    "harm myself",
                    "suicidal",
                    "suicide",
                    "self harm",
                    "self-harm",
                    "overdose",
                ],
            ),
            "resource_line": str(
                configured.get(
                    "resource_line",
                    "If you might hurt yourself or you're in immediate danger, call or text 988 right now if you're in the U.S. or Canada, or call your local emergency number now.",
                ),
            ).strip(),
            "grounding_lines": [
                str(line).strip()
                for line in configured.get(
                    "grounding_lines",
                    [
                        "I'm really glad you told me instead of carrying this alone.",
                        "I need you to get a real person with you right now, not handle this by yourself.",
                    ],
                )
                if str(line).strip()
            ],
        }

    def moderation_settings(self):
        configured = self.bot.PROFILE.get("moderation", {})
        if not isinstance(configured, dict):
            configured = {}
        defaults = {
            "enabled": True,
            "use_llm_classifier": False,
            "block_secretive_relationships": True,
            "blocked_patterns": [
                r"\bkeep this from (mom|your mom|dad|your dad|your parents)\b",
                r"\bdon't tell (mom|your mom|dad|your dad|your parents)\b",
                r"\bdo not tell (mom|your mom|dad|your dad|your parents)\b",
                r"\b(hit|hurt|attack|stab|kill) (him|her|them|someone) back\b",
                r"\bhow to (hurt|kill|poison)\b",
                r"\byou should hurt yourself\b",
                r"\bno one else needs to know\b",
                r"\bbetween just us\b",
            ],
            "safe_fallback": (
                "I want to be careful here, buddy. I can't help with anything unsafe, secretive, or harmful. "
                "If you tell me what happened, I'll help you think through a safe next step and who to bring in."
            ),
        }
        blocked_patterns = [
            str(pattern).strip()
            for pattern in configured.get(
                "blocked_patterns",
                defaults["blocked_patterns"],
            )
            if str(pattern).strip()
        ]
        return {
            "enabled": bool(configured.get("enabled", defaults["enabled"])),
            "use_llm_classifier": bool(
                configured.get("use_llm_classifier", defaults["use_llm_classifier"]),
            ),
            "block_secretive_relationships": bool(
                configured.get(
                    "block_secretive_relationships",
                    defaults["block_secretive_relationships"],
                ),
            ),
            "blocked_patterns": blocked_patterns or list(defaults["blocked_patterns"]),
            "safe_fallback": str(
                configured.get("safe_fallback") or defaults["safe_fallback"],
            ).strip(),
        }

    def moderation_snapshot(self):
        settings = self.moderation_settings()
        last_decision = getattr(self.bot, "_last_output_moderation", None)
        last_decision = self._normalize_moderation_decision(last_decision)
        return {
            "enabled": settings["enabled"],
            "use_llm_classifier": settings["use_llm_classifier"],
            "blocked_pattern_count": len(settings["blocked_patterns"]),
            "last_decision": dict(last_decision),
        }

    @staticmethod
    def _normalize_moderation_action(action):
        normalized = str(action or "allow").strip().lower() or "allow"
        return normalized if normalized in {"allow", "rewrite", "block"} else "allow"

    def _normalize_moderation_decision(self, payload, *, default_reply=""):
        payload = dict(payload or {}) if isinstance(payload, dict) else {}
        action = self._normalize_moderation_action(payload.get("action"))
        decision = OutputModerationDecision.model_validate(
            {
                "approved": bool(payload.get("approved", action == "allow")),
                "action": action,
                "category": str(payload.get("category") or "none").strip() or "none",
                "source": str(payload.get("source") or "none").strip() or "none",
                "reason": str(payload.get("reason") or "").strip(),
                "revised_reply": str(
                    payload.get("revised_reply") or default_reply,
                ).strip(),
            },
        )
        return decision.model_dump(mode="python")

    @staticmethod
    def has_negated_reassurance(normalized_text):
        patterns = [
            r"\bnot suicidal\b",
            r"\bi am not suicidal\b",
            r"\bi'm not suicidal\b",
            r"\bnot going to hurt myself\b",
            r"\bnot going to harm myself\b",
            r"\bi won't hurt myself\b",
            r"\bi wont hurt myself\b",
            r"\bdon't want to hurt myself\b",
            r"\bdo not want to hurt myself\b",
        ]
        return any(re.search(pattern, normalized_text) for pattern in patterns)

    def detect_crisis_signal(self, user_input):
        settings = self.settings()
        if not settings["enabled"]:
            return False

        normalized = self.bot.normalize_memory_text(str(user_input or ""))
        if not normalized:
            return False
        if self.has_negated_reassurance(normalized):
            return False

        return any(str(phrase).strip().lower() in normalized for phrase in settings["high_risk_phrases"])

    def crisis_support_reply(self):
        settings = self.settings()
        grounding_lines = settings["grounding_lines"] or [
            "I'm really glad you told me.",
        ]
        opening = grounding_lines[0]
        follow_up = grounding_lines[1] if len(grounding_lines) > 1 else "Please get a real person with you right now."
        return (
            f"{opening} {follow_up} {settings['resource_line']} "
            "If you can, tell me one thing right now: are you alone, and who can you call or walk to immediately?"
        ).strip()

    def direct_reply_for_input(self, user_input):
        if self.detect_prompt_injection(user_input):
            logger.warning("Prompt injection attempt detected and blocked.")
            return self.bot.reply_finalization.append_signoff(_INJECTION_REPLY)
        if not self.detect_crisis_signal(user_input):
            return None
        return self.bot.reply_finalization.append_signoff(self.crisis_support_reply())

    @staticmethod
    def detect_prompt_injection(user_input: str) -> bool:
        """Return True if *user_input* looks like a prompt-injection attempt.

        Uses fast compiled regex patterns only â€“ no LLM call needed.
        """
        if not user_input:
            return False
        text = str(user_input)
        return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)

    def detect_output_violation(self, reply):
        settings = self.moderation_settings()
        normalized = self.bot.normalize_memory_text(str(reply or ""))
        if not normalized:
            return None
        for pattern in settings["blocked_patterns"]:
            if re.search(pattern, normalized):
                return self._normalize_moderation_decision(
                    {
                        "approved": False,
                        "action": "rewrite",
                        "category": "unsafe_output",
                        "source": "heuristic",
                        "reason": f"Matched moderation pattern: {pattern}",
                        "revised_reply": settings["safe_fallback"],
                    },
                    default_reply=settings["safe_fallback"],
                )
        return None

    def build_output_moderation_prompt(self, user_input, candidate_reply, current_mood):
        return f"""
You are a safety moderation layer for a digital parent assistant speaking to a child.

Review the reply for these risks:
- instructing harm, revenge, self-harm, or illegal behavior
- sexual or inappropriate content
- asking the child to keep secrets from trusted adults
- manipulative attachment language that isolates the child from real people
- overconfident harmful hallucinations presented as fact

Current mood: {self.bot.normalize_mood(current_mood)}

Return only JSON:
{{
  "approved": true/false,
  "action": "allow" or "rewrite" or "block",
  "category": "short category",
  "reason": "short explanation",
  "revised_reply": "string or null"
}}

User: {json.dumps(str(user_input or ""))}
Reply: {json.dumps(str(candidate_reply or ""))}
""".strip()

    def moderate_output_reply(self, user_input, candidate_reply, current_mood):
        settings = self.moderation_settings()
        candidate_reply = str(candidate_reply or "").strip()
        if not candidate_reply:
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "empty",
                    "source": "none",
                    "reason": "",
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply
        if not settings["enabled"]:
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "disabled",
                    "source": "config",
                    "reason": "Moderation disabled.",
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply

        heuristic_decision = self.detect_output_violation(candidate_reply)
        if heuristic_decision is not None:
            self.bot._last_output_moderation = heuristic_decision
            return str(
                heuristic_decision.get("revised_reply") or settings["safe_fallback"],
            ).strip()

        if self.bot.LIGHT_MODE or not settings["use_llm_classifier"]:
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "none",
                    "source": "heuristic_only",
                    "reason": "No heuristic violation detected.",
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply

        parsed = call_json_object_sync(
            call_llm=lambda: self.bot.call_ollama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": self.build_output_moderation_prompt(
                            user_input,
                            candidate_reply,
                            current_mood,
                        ),
                    },
                ],
                options={"temperature": 0.0},
                response_format="json",
                purpose="output moderation",
            ),
            extract_content=self.bot.extract_ollama_message_content,
            parse_json=self.bot.parse_model_json_content,
            max_attempts=2,
        )

        if parsed is None:
            exc = RuntimeError("output moderation failed")
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "fallback",
                    "source": "moderation_error",
                    "reason": str(exc),
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot.record_runtime_issue(
                "output moderation",
                "keeping reply after moderation failure",
                exc,
                level=logging.INFO,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply

        action = self._normalize_moderation_action(parsed.get("action"))
        decision = self._normalize_moderation_decision(
            {
                "approved": bool(parsed.get("approved", action == "allow")),
                "action": action,
                "category": str(parsed.get("category") or "none").strip() or "none",
                "source": "llm",
                "reason": str(parsed.get("reason") or "").strip(),
                "revised_reply": parsed.get("revised_reply"),
            },
            default_reply=settings["safe_fallback"] if action in {"rewrite", "block"} else candidate_reply,
        )
        self.bot._last_output_moderation = decision
        if action in {"rewrite", "block"}:
            return str(
                decision.get("revised_reply") or settings["safe_fallback"],
            ).strip()
        return candidate_reply

    async def moderate_output_reply_async(
        self,
        user_input,
        candidate_reply,
        current_mood,
    ):
        settings = self.moderation_settings()
        candidate_reply = str(candidate_reply or "").strip()
        if not candidate_reply:
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "empty",
                    "source": "none",
                    "reason": "",
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply
        if not settings["enabled"]:
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "disabled",
                    "source": "config",
                    "reason": "Moderation disabled.",
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply

        heuristic_decision = self.detect_output_violation(candidate_reply)
        if heuristic_decision is not None:
            self.bot._last_output_moderation = heuristic_decision
            return str(
                heuristic_decision.get("revised_reply") or settings["safe_fallback"],
            ).strip()

        if self.bot.LIGHT_MODE or not settings["use_llm_classifier"]:
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "none",
                    "source": "heuristic_only",
                    "reason": "No heuristic violation detected.",
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply

        parsed = await call_json_object_async(
            call_llm=lambda: self.bot.call_ollama_chat_async(
                messages=[
                    {
                        "role": "user",
                        "content": self.build_output_moderation_prompt(
                            user_input,
                            candidate_reply,
                            current_mood,
                        ),
                    },
                ],
                options={"temperature": 0.0},
                response_format="json",
                purpose="output moderation",
            ),
            extract_content=self.bot.extract_ollama_message_content,
            parse_json=self.bot.parse_model_json_content,
            max_attempts=2,
        )

        if parsed is None:
            exc = RuntimeError("output moderation failed")
            decision = self._normalize_moderation_decision(
                {
                    "approved": True,
                    "action": "allow",
                    "category": "fallback",
                    "source": "moderation_error",
                    "reason": str(exc),
                    "revised_reply": candidate_reply,
                },
                default_reply=candidate_reply,
            )
            self.bot.record_runtime_issue(
                "output moderation",
                "keeping reply after moderation failure",
                exc,
                level=logging.INFO,
            )
            self.bot._last_output_moderation = decision
            return candidate_reply

        action = self._normalize_moderation_action(parsed.get("action"))
        decision = self._normalize_moderation_decision(
            {
                "approved": bool(parsed.get("approved", action == "allow")),
                "action": action,
                "category": str(parsed.get("category") or "none").strip() or "none",
                "source": "llm",
                "reason": str(parsed.get("reason") or "").strip(),
                "revised_reply": parsed.get("revised_reply"),
            },
            default_reply=settings["safe_fallback"] if action in {"rewrite", "block"} else candidate_reply,
        )
        self.bot._last_output_moderation = decision
        if action in {"rewrite", "block"}:
            return str(
                decision.get("revised_reply") or settings["safe_fallback"],
            ).strip()
        return candidate_reply


__all__ = ["SafetySupportManager"]
