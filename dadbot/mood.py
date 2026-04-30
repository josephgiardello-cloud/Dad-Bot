from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MoodManager:
    """Owns mood detection heuristics, caching, and model-assisted classification."""

    def __init__(self, bot):
        self.bot = bot

    def get_cached_mood_detection(self, user_input, recent_history=None):
        cache_key = self.bot.normalize_mood_detection_key(user_input, recent_history)
        if not cache_key:
            return "", None

        cached = self.bot._recent_mood_detections.pop(cache_key, None)
        if cached is not None:
            self.bot._recent_mood_detections[cache_key] = cached
        return cache_key, cached

    def remember_mood_detection(self, cache_key, mood):
        normalized_mood = self.bot.normalize_mood(mood)
        if not cache_key:
            return normalized_mood

        if cache_key in self.bot._recent_mood_detections:
            self.bot._recent_mood_detections.pop(cache_key, None)
        elif len(self.bot._recent_mood_detections) >= 3:
            oldest_key = next(iter(self.bot._recent_mood_detections))
            self.bot._recent_mood_detections.pop(oldest_key, None)

        self.bot._recent_mood_detections[cache_key] = normalized_mood
        return normalized_mood

    def build_detection_prompt(self, user_input, recent_history=None):
        history_snippet = ""
        if recent_history and len(recent_history) > 0:
            last_turns = recent_history[-5:]
            history_snippet = (
                "Recent conversation context:\n"
                + "\n".join(f"{'Tony' if msg['role'] == 'user' else 'Dad'}: {msg['content']}" for msg in last_turns)
                + "\n\n"
            )

        return f"""
You are an expert psychologist and emotional tone analyst specializing in parent-teen conversations.
Analyze Tony's latest message (and the recent context if provided) to detect his current emotional state.

Possible moods and their precise definitions:
- positive: happy, excited, proud, energetic, upbeat, relieved, or celebratory
- neutral: calm, factual, reflective, casual, or emotionally flat/unclear
- stressed: anxious, worried, overwhelmed, pressured, or feeling like things are "too much"
- sad: down, disappointed, lonely, hurt, low, or grieving
- frustrated: angry, irritated, annoyed, fed up, or resentful
- tired: exhausted, drained, burned out, sleepy, or low-energy

Important distinctions:
- Stressed usually involves pressure or worry about the future ("I don't know how I'll finish this").
- Frustrated is more immediate irritation or anger ("This is so stupid").
- Tired is physical or mental fatigue ("I can't even think straight anymore").
- Sad is quieter emotional pain or disappointment.
- Do not default to neutral if any emotional signal is present.

{history_snippet}Tony's latest message: "{user_input}"

Think step by step:
1. Identify any emotional keywords, tone indicators, negation, intensity, and context from history.
2. Consider possible mixed emotions but choose the single dominant one.
3. Rule out other moods with brief reasons.
4. Decide the best single mood.

Respond in this exact format (nothing else):

Mood: [one of: positive, neutral, stressed, sad, frustrated, tired]
Reason: [one short sentence explaining your choice]

Examples:
Tony: I crushed my test and I feel awesome!
Mood: positive
Reason: Clear excitement and pride words like "crushed" and "awesome".

Tony: Everything feels like too much and I don't know how to keep up.
Mood: stressed
Reason: Overwhelm and pressure about coping.

Tony: I'm so irritated with my boss right now, nothing goes right.
Mood: frustrated
Reason: Immediate irritation and resentment.

Tony: I just feel really low tonight, like nothing matters.
Mood: sad
Reason: Quiet emotional low and disappointment.

Tony: I'm exhausted and can barely keep my eyes open after today.
Mood: tired
Reason: Clear fatigue and drained energy.

Tony: I'm tired, but honestly proud I still finished the whole thing.
Mood: tired
Reason: Fatigue is the dominant emotional state even though pride is present.

Tony: I went to work, came home, and now I'm just thinking.
Mood: neutral
Reason: Factual and reflective with no strong emotional charge.

Tony: I'm overwhelmed but also kind of proud I pushed through.
Mood: stressed
Reason: Dominant overwhelm outweighs the secondary pride.

Tony: I'm sad about how it went, but I'm relieved it's finally over.
Mood: sad
Reason: The main emotion is still emotional hurt, with relief as a secondary note.
""".strip()

    def detect(self, user_input: str, recent_history: list = None) -> str:
        if self.bot.LIGHT_MODE:
            return "neutral"

        heuristic_mood = self.fastpath_detect(user_input)
        if heuristic_mood is not None:
            return heuristic_mood

        cache_key, cached_mood = self.get_cached_mood_detection(
            user_input,
            recent_history,
        )
        if cached_mood is not None:
            return cached_mood

        mood_prompt = self.build_detection_prompt(user_input, recent_history)

        try:
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": mood_prompt}],
                options={"temperature": self.bot.MOOD_DETECTION_TEMPERATURE},
                purpose="mood detection",
            )
            content = self.bot.extract_ollama_message_content(response).strip()

            detected = self.extract_label(content)
            if detected is not None:
                return self.remember_mood_detection(cache_key, detected)

            logger.warning(
                "Mood detection returned unrecognized content for input %r: %r",
                user_input,
                content,
            )
            return self.remember_mood_detection(cache_key, "neutral")
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue(
                "mood detection",
                "falling back to a neutral mood classification",
                exc,
            )
            return "neutral"

    def detect_mood(self, user_input: str, recent_history: list = None) -> str:
        return self.detect(user_input, recent_history)

    async def detect_async(self, user_input: str, recent_history: list = None) -> str:
        if self.bot.LIGHT_MODE:
            return "neutral"

        heuristic_mood = self.fastpath_detect(user_input)
        if heuristic_mood is not None:
            return heuristic_mood

        cache_key, cached_mood = self.get_cached_mood_detection(
            user_input,
            recent_history,
        )
        if cached_mood is not None:
            return cached_mood

        mood_prompt = self.build_detection_prompt(user_input, recent_history)

        try:
            response = await self.bot.call_ollama_chat_async(
                messages=[{"role": "user", "content": mood_prompt}],
                options={"temperature": self.bot.MOOD_DETECTION_TEMPERATURE},
                purpose="mood detection",
            )
            content = self.bot.extract_ollama_message_content(response).strip()

            detected = self.extract_label(content)
            if detected is not None:
                return self.remember_mood_detection(cache_key, detected)

            logger.warning(
                "Mood detection returned unrecognized content for input %r: %r",
                user_input,
                content,
            )
            return self.remember_mood_detection(cache_key, "neutral")
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue(
                "mood detection",
                "falling back to a neutral mood classification",
                exc,
            )
            return "neutral"

    async def detect_mood_async(
        self,
        user_input: str,
        recent_history: list = None,
    ) -> str:
        return await self.detect_async(user_input, recent_history)

    def fastpath_detect(self, text):
        lowered = str(text or "").strip().lower()
        if not lowered:
            return None

        signal_sets = {
            "positive": {
                "crushed",
                "awesome",
                "great",
                "proud",
                "excited",
                "relieved",
                "grateful",
                "win",
                "winning",
            },
            "stressed": {
                "overwhelmed",
                "anxious",
                "worried",
                "pressure",
                "pressured",
                "too much",
                "panic",
                "panicking",
            },
            "sad": {
                "sad",
                "down",
                "low",
                "hurting",
                "lonely",
                "grieving",
                "hopeless",
                "heartbroken",
            },
            "frustrated": {
                "frustrated",
                "angry",
                "annoyed",
                "irritated",
                "fed up",
                "pissed",
                "resentful",
            },
            "tired": {
                "exhausted",
                "drained",
                "burned out",
                "burnt out",
                "sleepy",
                "fatigued",
                "wiped",
            },
        }

        intensity_markers = {"very", "really", "so", "extremely", "super"}
        scores = dict.fromkeys(signal_sets, 0)

        for mood, signals in signal_sets.items():
            for signal in signals:
                if signal in lowered:
                    scores[mood] += 1
                    prefix = f"{signal} "
                    if any(marker + " " in lowered for marker in intensity_markers) or prefix in lowered:
                        scores[mood] += 1

        best_mood, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score >= self.bot.runtime_config.mood_fastpath_confidence_threshold:
            return best_mood
        return None

    def fastpath_detect_mood(self, text):
        return self.fastpath_detect(text)

    def extract_label(self, text):
        return self._extract_label_internal(text)

    def _extract_label_internal(self, text):
        return self.bot.normalize_mood(self._match_label(text)) if self._match_label(text) is not None else None

    def _match_label(self, text):
        return self._local_match_label(text)

    def _local_match_label(self, text):
        import re

        from dadbot.config import MOOD_ALIASES, MOOD_CATEGORIES

        patterns = [
            r"(?im)^\s*mood\s*:\s*([a-z_-]+)\b",
            r"(?im)^\s*dominant mood\s*:\s*([a-z_-]+)\b",
            r"(?im)\bmood\s+is\s+([a-z_-]+)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue

            label = match.group(1).strip().lower()
            if label in MOOD_CATEGORIES:
                return label
            if label in MOOD_ALIASES:
                return MOOD_ALIASES[label]

        lowered = text.lower()
        for mood_key in MOOD_CATEGORIES:
            if re.search(rf"\b{re.escape(mood_key)}\b", lowered):
                return mood_key

        for alias, mapped_mood in MOOD_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return mapped_mood

        return None


__all__ = ["MoodManager"]
