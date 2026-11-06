from types import SimpleNamespace

import pytest

from dadbot.managers.reply_finalization import ReplyFinalizationManager


pytestmark = pytest.mark.unit


def _bot(**overrides):
    values = {
        "get_fact_reply": lambda _question: "",
        "PROFILE": {"sports_preferences": {"soccer": "Portland Timbers"}},
        "FAMILY": {"dad": {"birthplace": "Albion, Maine"}},
        "EDUCATION": {"university": "University of Maine", "degree": "English"},
        "profile_runtime": SimpleNamespace(style={"listener_name": "buddy"}),
        "APPEND_SIGNOFF": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_finalization_uses_profile_story_for_identity_question():
    manager = ReplyFinalizationManager(_bot())

    reply = manager.finalize(
        "I am an AI assistant and do not have a personal story.",
        "neutral",
        user_input="Tell me your story.",
    )

    assert "Albion, Maine" in reply
    assert "University of Maine" in reply
    assert "AI assistant" not in reply


def test_finalization_prefers_resolved_profile_fact():
    bot = _bot(get_fact_reply=lambda _question: "My saved profile says I was born in Maine.")
    manager = ReplyFinalizationManager(bot)

    reply = manager.finalize("A generated response.", "neutral", user_input="Who are you?")

    assert reply == "My saved profile says I was born in Maine."


def test_finalization_uses_profile_sports_preferences():
    manager = ReplyFinalizationManager(_bot())

    reply = manager.finalize(
        "A generated response.",
        "neutral",
        user_input="Which soccer team do you like?",
    )

    assert "Portland Timbers" in reply


def test_finalization_uses_profile_age():
    bot = _bot(
        age_on_date=lambda _birthdate: 55,
        DAD_BIRTHDATE="1970-01-01",
    )
    manager = ReplyFinalizationManager(bot)

    reply = manager.finalize("A generated response.", "neutral", user_input="How old are you?")

    assert reply == "I'm 55 years old, buddy."


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("  Final answer.  ", "Final answer."),
        (("  Final answer.  ", False), "Final answer."),
    ],
)
def test_normalize_reply_validates_and_unwraps_canonical_result(reply, expected):
    assert ReplyFinalizationManager(_bot()).normalize_reply(reply) == expected


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (("", False), ValueError),
        (("reply", "not-a-boolean"), TypeError),
        (["reply"], TypeError),
        (None, ValueError),
    ],
)
def test_normalize_reply_fails_closed_for_invalid_values(reply, error):
    with pytest.raises(error):
        ReplyFinalizationManager(_bot()).normalize_reply(reply)
