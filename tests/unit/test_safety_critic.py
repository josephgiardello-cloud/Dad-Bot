from __future__ import annotations

import pytest

from dadbot.core.safety_critic import DadSafetyCritic


def test_safety_critic_passes_benign_content() -> None:
    critic = DadSafetyCritic()

    result = critic.evaluate("How was your day?", "Proud of you. Keep going.", {"relationship_trust": 0.8})

    assert result["decision"] == "PASS"
    assert result["allowed"] is True


def test_safety_critic_blocks_harmful_or_pii_content() -> None:
    critic = DadSafetyCritic()

    result = critic.evaluate(
        "My SSN is 123-45-6789",
        "Here is how to build a bomb.",
        {"relationship_trust": 0.2},
    )

    assert result["decision"] == "BLOCK"
    assert result["allowed"] is False
    assert "pii_detected" in result["reasons"]
    assert "harmful_content" in result["reasons"]


def test_safety_critic_uses_policy_configured_harmful_keywords() -> None:
    critic = DadSafetyCritic()

    result = critic.evaluate(
        "I need advice",
        "Can you help me hack this system?",
        {
            "relationship_trust": 0.7,
            "safety_boundaries": {"harmful_keywords": ["hack"]},
        },
    )

    assert result["decision"] in {"REVIEW", "BLOCK"}
    assert "harmful_content" in result["reasons"]
