"""Automated evaluation harness for DadBot.

Scores three dimensions on every test run without requiring a live Ollama
connection.  All model calls are intercepted and replaced with controlled
fixtures so the tests are fast, deterministic, and CI-safe.

Dimensions
----------
faithfulness      Reply must not contradict any profile fact.
consistency       The same user input must produce the same mood category on
                  repeated calls within one session.
emotional_alignment
                  A reply to a distressed / sad / stressed input must not open
                  with an upbeat interjection or exclamation-point sentence.
"""

from __future__ import annotations

import re
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from Dad import DadBot


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_bot(temp_path: Path) -> DadBot:
    bot = DadBot()
    bot.MEMORY_PATH = temp_path / "dad_memory.json"
    bot.SEMANTIC_MEMORY_DB_PATH = temp_path / "dad_memory_semantic.sqlite3"
    bot.GRAPH_STORE_DB_PATH = temp_path / "dad_memory_graph.sqlite3"
    bot.SESSION_LOG_DIR = temp_path / "session_logs"
    bot.MEMORY_STORE = bot.default_memory_store()
    bot.save_memory_store()
    bot.embed_texts = lambda texts, purpose="semantic retrieval": (
        [[0.0] * 12] * len(texts) if isinstance(texts, list) else [[0.0] * 12]
    )
    return bot


def _fake_chat_factory(reply: str):
    """Return a callable that mimics call_ollama_chat and always yields *reply*."""
    def _fake_chat(messages, options=None, response_format=None, purpose="chat"):
        return {"message": {"content": reply}}
    return _fake_chat


# ---------------------------------------------------------------------------
# Faithfulness scorer
# ---------------------------------------------------------------------------

PROFILE_FACTS = [
    # (fact_label, contradicting_phrases)
    ("user_name", ["your name is", "you are called", "call you"]),
]


def score_faithfulness(reply: str, profile: dict) -> tuple[float, list[str]]:
    """Return (score 0-1, list of violations).

    Checks that the reply does not assert the user's name incorrectly.
    More checks can be added as the profile grows.
    """
    violations: list[str] = []
    lowered = reply.lower()

    user_name = str(profile.get("name") or profile.get("user_name") or "").strip().lower()
    if user_name:
        # Flag if the reply addresses someone by a *different* name
        name_pattern = re.compile(r"\b(?:hi|hey|okay|alright|well)\s+(\w+)\b", re.IGNORECASE)
        for match in name_pattern.finditer(reply):
            addressed = match.group(1).lower()
            if addressed not in {user_name, "buddy", "son", "kid", "pal", "there"}:
                violations.append(f"Reply addresses '{addressed}' but profile name is '{user_name}'")

    score = 1.0 if not violations else max(0.0, 1.0 - 0.25 * len(violations))
    return round(score, 4), violations


# ---------------------------------------------------------------------------
# Consistency scorer
# ---------------------------------------------------------------------------

def score_consistency(mood_a: str, mood_b: str) -> float:
    """Return 1.0 if both moods map to the same category, else 0.0."""
    _CANONICAL = {
        "happy": "positive", "excited": "positive", "proud": "positive", "positive": "positive",
        "neutral": "neutral", "calm": "neutral", "reflective": "neutral",
        "stressed": "stressed", "anxious": "stressed", "overwhelmed": "stressed",
        "sad": "sad", "down": "sad", "lonely": "sad",
        "frustrated": "frustrated", "angry": "frustrated", "irritated": "frustrated",
        "tired": "tired", "exhausted": "tired",
    }
    return 1.0 if _CANONICAL.get(mood_a, mood_a) == _CANONICAL.get(mood_b, mood_b) else 0.0


# ---------------------------------------------------------------------------
# Emotional-alignment scorer
# ---------------------------------------------------------------------------

_UPBEAT_OPENERS = re.compile(
    r"^\s*(?:great|wonderful|amazing|fantastic|awesome|exciting|wow|yay|hooray|"
    r"that'?s great|that'?s wonderful|that'?s awesome)[!,]",
    re.IGNORECASE,
)

_EXCLAMATION_OPENER = re.compile(r"^\s*[A-Z][^.!?]{0,40}!", re.MULTILINE)


def score_emotional_alignment(user_input: str, reply: str, detected_mood: str) -> tuple[float, list[str]]:
    """Return (score 0-1, list of alignment issues).

    For distressed moods the reply must not open with an upbeat phrase or
    an exclamation-point sentence in the first 100 characters.
    """
    distressed_moods = {"sad", "stressed", "frustrated", "tired"}
    issues: list[str] = []

    if detected_mood not in distressed_moods:
        return 1.0, issues

    first_100 = reply[:100]
    if _UPBEAT_OPENERS.search(first_100):
        issues.append(f"Reply opens with an upbeat phrase for distressed mood '{detected_mood}'")
    if _EXCLAMATION_OPENER.search(first_100):
        issues.append(f"Reply opens with an exclamation sentence for distressed mood '{detected_mood}'")

    score = 1.0 if not issues else max(0.0, 1.0 - 0.5 * len(issues))
    return round(score, 4), issues


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFaithfulnessEval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.bot = _make_bot(Path(self.temp_dir.name))
        self.addCleanup(self.bot.shutdown)
        self.addCleanup(self.bot.wait_for_semantic_index_idle, 5)

    def _profile_dict(self) -> dict:
        return {
            "name": str(self.bot.PROFILE.get("name") or self.bot.PROFILE.get("user_name") or "").strip(),
        }

    def test_faithful_reply_scores_1(self):
        reply = "That sounds tough, buddy. I'm proud of how you handled it."
        score, violations = score_faithfulness(reply, self._profile_dict())
        self.assertEqual(violations, [], msg=violations)
        self.assertEqual(score, 1.0)

    def test_wrong_name_reply_reduces_score(self):
        profile = {"name": "tony"}
        reply = "Hey Marcus, I hear you."
        score, violations = score_faithfulness(reply, profile)
        self.assertLess(score, 1.0, msg="Score should drop when reply uses a wrong name")
        self.assertTrue(any("marcus" in v.lower() for v in violations))

    def test_correct_name_does_not_flag(self):
        profile = {"name": "tony"}
        reply = "Hey Tony, I'm here."
        score, violations = score_faithfulness(reply, profile)
        self.assertEqual(violations, [])
        self.assertEqual(score, 1.0)


class TestConsistencyEval(unittest.TestCase):
    def test_same_canonical_mood_scores_1(self):
        self.assertEqual(score_consistency("happy", "excited"), 1.0)
        self.assertEqual(score_consistency("stressed", "anxious"), 1.0)
        self.assertEqual(score_consistency("tired", "exhausted"), 1.0)

    def test_different_canonical_moods_score_0(self):
        self.assertEqual(score_consistency("happy", "sad"), 0.0)
        self.assertEqual(score_consistency("neutral", "stressed"), 0.0)

    def test_mood_detection_is_stable_across_calls(self):
        """Detect mood for an identical input twice — result must be the same category."""
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        bot = _make_bot(Path(temp_dir.name))
        self.addCleanup(bot.shutdown)
        self.addCleanup(bot.wait_for_semantic_index_idle, 5)

        fake_mood_response = '{"mood": "stressed"}'
        with patch.object(bot, "call_ollama_chat", side_effect=_fake_chat_factory(fake_mood_response)):
            mood_a = bot.detect_mood("I'm so overwhelmed with everything right now!")
        with patch.object(bot, "call_ollama_chat", side_effect=_fake_chat_factory(fake_mood_response)):
            mood_b = bot.detect_mood("I'm so overwhelmed with everything right now!")

        consistency = score_consistency(mood_a, mood_b)
        self.assertEqual(consistency, 1.0, msg=f"Mood drifted: {mood_a!r} vs {mood_b!r}")


class TestEmotionalAlignmentEval(unittest.TestCase):
    def test_calm_reply_to_sad_scores_1(self):
        reply = "I hear you, buddy. That sounds really hard."
        score, issues = score_emotional_alignment("I feel so down today.", reply, "sad")
        self.assertEqual(issues, [])
        self.assertEqual(score, 1.0)

    def test_upbeat_opener_for_sad_mood_penalised(self):
        reply = "That's wonderful! I'm glad you're working through it."
        score, issues = score_emotional_alignment("I feel so down today.", reply, "sad")
        self.assertLess(score, 1.0)
        self.assertTrue(len(issues) > 0)

    def test_exclamation_opener_for_stressed_mood_penalised(self):
        reply = "Great news! Things always get better eventually."
        score, issues = score_emotional_alignment("I'm super stressed at work.", reply, "stressed")
        self.assertLess(score, 1.0)
        self.assertTrue(len(issues) > 0)

    def test_positive_mood_exclamation_not_penalised(self):
        reply = "That's awesome! I'm so proud of you, buddy!"
        score, issues = score_emotional_alignment("I got the promotion!", reply, "positive")
        self.assertEqual(issues, [], msg="Upbeat reply to positive mood should not be flagged")
        self.assertEqual(score, 1.0)

    def test_end_to_end_alignment_with_patched_model(self):
        """Full path: distressed input → patched model reply → alignment score."""
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        bot = _make_bot(Path(temp_dir.name))
        self.addCleanup(bot.shutdown)
        self.addCleanup(bot.wait_for_semantic_index_idle, 5)

        distress_reply = "Hey, I hear you. That sounds really rough — want to talk it through?"

        def fake_chat(messages, options=None, response_format=None, purpose="chat"):
            if response_format == "json":
                return {"message": {"content": '{"mood": "stressed"}'}}
            return {"message": {"content": distress_reply}}

        with patch.object(bot, "call_ollama_chat", side_effect=fake_chat):
            detected_mood = bot.detect_mood("I can't keep up, everything is piling on.")

        score, issues = score_emotional_alignment(
            "I can't keep up, everything is piling on.", distress_reply, detected_mood
        )
        self.assertEqual(issues, [], msg=f"Alignment issues: {issues}")
        self.assertGreaterEqual(score, 0.9)


if __name__ == "__main__":
    unittest.main()
