import pytest

pytestmark = pytest.mark.unit


def test_naturalize_memory_summary_rewrites_first_person_statements(bot):
    assert bot.naturalize_memory_summary("i'm stressed about work") == "Tony is stressed about work."
    assert bot.naturalize_memory_summary("i have been saving more money") == "Tony has been saving more money."
    assert bot.naturalize_memory_summary("trying to get back to the gym") == "Tony is trying to get back to the gym."


def test_naturalize_memory_summary_adds_prefix_for_non_tony_summary(bot):
    assert bot.naturalize_memory_summary("Feels off today") == "Tony shared that feels off today."


def test_normalize_memory_text_collapses_case_and_whitespace(bot):
    normalized = bot.normalize_memory_text("  Tony   HAS Been Saving Money  ")

    assert normalized == "tony has been saving money"


def test_normalize_mood_maps_alias_phrases(bot):
    assert bot.normalize_mood("burned out") == "tired"
    assert bot.normalize_mood("I feel anxious lately") == "stressed"
    assert bot.normalize_mood("curious") == "neutral"
