from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from dadbot.ui import oral_history
from dadbot.ui.media import service, stt

pytestmark = pytest.mark.unit


def test_reviewed_transcript_is_saved_through_memory_catalog(bot):
    transcript = "Grandma taught us to make soup together every Sunday."

    saved = oral_history.save_reviewed_transcript(bot, transcript)

    assert saved is not None
    assert saved["summary"] == transcript
    assert saved["category"] == "oral_history"
    evidence = saved["oral_history_evidence"]
    assert evidence["text"] == transcript
    assert evidence["kind"] == "reviewed_transcript"
    assert evidence["sha256"] == hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    assert any(
        memory["summary"] == transcript and memory["category"] == "oral_history"
        for memory in bot.memory_catalog()
    )
    persisted = json.loads(Path(bot.memory_path).read_text(encoding="utf-8"))
    saved_record = next(memory for memory in persisted["memories"] if memory["summary"] == transcript)
    assert saved_record["category"] == "oral_history"
    assert saved_record["oral_history_evidence"] == evidence


def test_empty_reviewed_transcript_does_not_create_memory(bot):
    before = bot.memory_catalog()

    saved = oral_history.save_reviewed_transcript(bot, " \n ")

    assert saved is None
    assert bot.memory_catalog() == before


def test_storage_failure_is_not_reported_as_a_saved_story(bot, monkeypatch):
    monkeypatch.setattr(bot.memory_manager._storage, "_write_json_memory_store_unlocked", lambda _store: None)
    with pytest.raises(RuntimeError, match="durable storage"):
        oral_history.save_reviewed_transcript(bot, "A story that could not be saved.")


def test_distinct_short_and_financial_stories_survive_catalog_cleanup(bot):
    stories = ["Yes.", "I saved money for a bicycle.", "My budget paid for a family trip."]

    for story in stories:
        assert oral_history.save_reviewed_transcript(bot, story) is not None

    catalog = bot.memory_catalog()
    assert {memory["summary"] for memory in catalog if memory["category"] == "oral_history"} == set(stories)
    persisted = json.loads(Path(bot.memory_path).read_text(encoding="utf-8"))
    assert {memory["summary"] for memory in persisted["memories"] if memory["category"] == "oral_history"} == set(stories)


def test_saving_existing_story_preserves_the_reviewed_capitalization(bot):
    assert oral_history.save_reviewed_transcript(bot, "My story.") is not None
    assert oral_history.save_reviewed_transcript(bot, "MY STORY.") is not None

    stories = [memory["summary"] for memory in bot.memory_catalog() if memory["category"] == "oral_history"]
    assert stories == ["MY STORY."]
    persisted = json.loads(Path(bot.memory_path).read_text(encoding="utf-8"))
    assert any(memory["summary"] == "MY STORY." for memory in persisted["memories"])


def test_local_transcriber_returns_editable_text_and_cleans_temporary_audio(tmp_path, monkeypatch):
    audio_path = tmp_path / "oral-history.wav"
    model_calls = []

    class Model:
        def transcribe(self, path, **kwargs):
            model_calls.append((path, kwargs))
            return (
                [SimpleNamespace(text=" We gathered at the lake."), SimpleNamespace(text=" Every summer.")],
                SimpleNamespace(language="en"),
            )

    monkeypatch.setattr(stt, "create_temp_file_path", lambda **_kwargs: str(audio_path))
    monkeypatch.setattr(stt, "load_faster_whisper_model", lambda _model: Model())

    transcript, error = stt.transcribe_audio_bytes(
        b"audio bytes",
        backend="faster_whisper",
        model_name="base",
        language="en",
    )

    assert error == ""
    assert transcript == "We gathered at the lake. Every summer."
    assert model_calls == [(str(audio_path), {"language": "en", "vad_filter": True, "beam_size": 1})]
    assert not audio_path.exists()


def test_transcription_updates_existing_draft_and_requires_new_review(monkeypatch):
    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.changed = None

        def expander(self, *_args, **_kwargs):
            return nullcontext()

        def spinner(self, *_args, **_kwargs):
            return nullcontext()

        def caption(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def audio_input(self, *_args, **_kwargs):
            return BytesIO(b"recording")

        def file_uploader(self, *_args, **_kwargs):
            return None

        def button(self, label, **_kwargs):
            return label == "Transcribe locally"

        def text_area(self, _label, *, key, on_change, **_kwargs):
            self.changed = on_change
            return self.session_state.get(key, "")

        def checkbox(self, _label, *, key):
            return self.session_state.get(key, False)

    clip_id = hashlib.sha256(b"recording").hexdigest()
    draft_key = f"oral-history-transcript:{clip_id}"
    reviewed_key = f"oral-history-reviewed:{clip_id}"
    fake = FakeStreamlit()
    fake.session_state.update({draft_key: "An earlier draft", reviewed_key: True})
    monkeypatch.setattr(service, "st", fake)
    monkeypatch.setattr(service, "voice_preferences", lambda: {})
    monkeypatch.setattr(service, "local_stt_backend_status", lambda _voice: (True, "faster_whisper", "ready"))
    monkeypatch.setattr(stt, "transcribe_audio_bytes", lambda *_args, **_kwargs: ("The new transcript.", ""))

    service.MediaService().render_oral_history_capture(None)

    assert fake.session_state[draft_key] == "The new transcript."
    assert fake.session_state[reviewed_key] is False
    fake.session_state[reviewed_key] = True
    fake.changed()
    assert fake.session_state[reviewed_key] is False


def test_voice_page_saves_reviewed_transcript_with_evidence(bot, monkeypatch):
    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.messages = []

        def expander(self, *_args, **_kwargs):
            return nullcontext()

        def caption(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def audio_input(self, *_args, **_kwargs):
            return BytesIO(b"recording")

        def file_uploader(self, *_args, **_kwargs):
            return None

        def button(self, label, **kwargs):
            return label == "Save transcript as memory" and not kwargs.get("disabled", False)

        def text_area(self, *_args, **_kwargs):
            return "The family gathered for supper."

        def checkbox(self, *_args, **_kwargs):
            return True

        def success(self, message):
            self.messages.append(message)

    fake = FakeStreamlit()
    monkeypatch.setattr(service, "st", fake)
    monkeypatch.setattr(service, "voice_preferences", lambda: {})
    monkeypatch.setattr(service, "local_stt_backend_status", lambda _voice: (True, "faster_whisper", "ready"))

    service.MediaService().render_oral_history_capture(bot)

    saved = next(memory for memory in bot.memory_catalog() if memory["summary"] == "The family gathered for supper.")
    assert saved["oral_history_evidence"]["text"] == saved["summary"]
    assert fake.messages == ["Reviewed oral history saved to Dad's memory."]
