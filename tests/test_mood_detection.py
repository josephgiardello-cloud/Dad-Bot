import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("response_text", "expected"),
    [
        ("Mood: positive\nReason: Clear excitement and pride.", "positive"),
        ("dominant mood: anxious", "stressed"),
        ("The mood is burned out because he sounds drained.", "tired"),
        ("He sounds frustrated and fed up.", "frustrated"),
        ("No clear label here.", None),
    ],
)
def test_extract_mood_label_handles_formats_and_aliases(bot, response_text, expected):
    assert bot.mood_manager.extract_label(response_text) == expected


@pytest.mark.parametrize(
    ("model_reply", "expected"),
    [
        ("Mood: positive\nReason: Pride and excitement.", "positive"),
        ("Mood: stressed\nReason: Overwhelm and pressure.", "stressed"),
        ("Mood: sad\nReason: Quiet emotional pain.", "sad"),
        ("Mood: frustrated\nReason: Immediate irritation.", "frustrated"),
        ("Mood: tired\nReason: Clear fatigue.", "tired"),
        ("Mood: neutral\nReason: Mostly factual and reflective.", "neutral"),
    ],
)
def test_detect_mood_uses_mocked_ollama_reply(bot, mocker, model_reply, expected):
    mocker.patch.object(bot, "call_ollama_chat", return_value={"message": {"content": model_reply}})

    mood = bot.mood_manager.detect("test input")

    assert mood == expected


def test_detect_mood_defaults_to_neutral_on_invalid_response(bot, mocker):
    mocker.patch.object(bot, "call_ollama_chat", return_value={"message": {"content": "I cannot tell."}})

    assert bot.mood_manager.detect("ambiguous") == "neutral"


def test_detect_mood_defaults_to_neutral_on_runtime_error(bot, mocker):
    mocker.patch.object(bot, "call_ollama_chat", side_effect=RuntimeError("ollama unavailable"))

    assert bot.mood_manager.detect("overwhelmed") == "neutral"


def test_detect_mood_fastpath_skips_ollama_for_obvious_signal(bot, mocker):
    mocked = mocker.patch.object(bot, "call_ollama_chat")

    detected = bot.mood_manager.detect("I am completely overwhelmed and really anxious about everything.")

    assert detected == "stressed"
    mocked.assert_not_called()


def test_detect_mood_reuses_recent_cached_result(bot, mocker):
    mocked = mocker.patch.object(
        bot,
        "call_ollama_chat",
        return_value={"message": {"content": "Mood: neutral\nReason: Mostly factual and reflective."}},
    )

    first = bot.mood_manager.detect("I guess I'm here again.")
    second = bot.mood_manager.detect("I guess I'm here again.")

    assert first == "neutral"
    assert second == "neutral"
    mocked.assert_called_once()


def test_detect_mood_returns_neutral_without_model_in_light_mode(bot, mocker):
    bot.LIGHT_MODE = True
    mocked = mocker.patch.object(bot, "call_ollama_chat")

    detected = bot.mood_manager.detect("I am completely overwhelmed and really anxious about everything.")

    assert detected == "neutral"
    mocked.assert_not_called()
