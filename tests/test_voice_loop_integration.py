from __future__ import annotations

import io
import wave
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dadbot.ui import interaction_controller, state_manager
from dadbot.ui import helpers
from dadbot.ui.media import service
from dadbot.managers.tts import TTSManager

pytestmark = pytest.mark.unit


def _wav_chunk(value: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes((value.to_bytes(2, "little", signed=True)) * 4)
    return output.getvalue()


class FakeController:
    def __init__(self) -> None:
        self.voice_config = {
            "tts_enabled": True,
            "muted": False,
            "tts_autoplay": True,
        }
        self.events: list[str] = []

    def begin_speaking(self) -> None:
        self.events.append("speaking")

    def complete_turn(self) -> None:
        self.events.append("complete")

    def mark_error(self, _message: str) -> None:
        self.events.append("error")


def test_voice_transcript_uses_runtime_and_local_piper_chunks(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    audio_elements: list[tuple[bytes, dict]] = []
    markdown: list[str] = []
    controller = FakeController()

    monkeypatch.setattr(service.st, "spinner", lambda _label: nullcontext())
    monkeypatch.setattr(service.st, "markdown", lambda text: markdown.append(text))
    monkeypatch.setattr(
        service.st,
        "audio",
        lambda audio, **kwargs: audio_elements.append((audio, kwargs)),
    )
    monkeypatch.setattr(service.st, "warning", lambda _message: None)
    monkeypatch.setattr(service.st, "error", lambda _message: None)
    monkeypatch.setattr(
        interaction_controller,
        "get_chat_event_api",
        lambda: SimpleNamespace(active_thread_id="voice-session"),
    )

    def process_prompt_via_runtime(*, thread_id, prompt, attachments):
        calls.append(("turn", f"{thread_id}:{prompt}:{attachments}"))
        return {"reply": "First sentence. Second sentence."}

    monkeypatch.setattr(interaction_controller, "process_prompt_via_runtime", process_prompt_via_runtime)

    def synthesize_piper_audio(_manager, text):
        calls.append(("piper", text))
        return _wav_chunk(len(text)), ""

    monkeypatch.setattr(service.TTSManager, "synthesize_piper_audio", synthesize_piper_audio)

    reply = service.MediaService()._process_voice_transcript(object(), controller, "Tell me a story.")

    assert reply == "First sentence. Second sentence."
    assert calls == [
        ("turn", "voice-session:Tell me a story.:[]"),
        ("piper", "First sentence."),
        ("piper", "Second sentence."),
    ]
    assert markdown == ["**Dad:** First sentence. Second sentence."]
    assert len(audio_elements) == 1
    with wave.open(io.BytesIO(audio_elements[0][0]), "rb") as audio:
        assert audio.getnframes() == 8
    assert audio_elements[0][1] == {"format": "audio/wav", "autoplay": True}
    assert controller.events == ["speaking", "complete"]


def test_voice_transcript_respects_muted_tts_and_reports_runtime_failure(monkeypatch) -> None:
    messages: list[str] = []
    controller = FakeController()
    controller.voice_config["muted"] = True
    monkeypatch.setattr(service.st, "spinner", lambda _label: nullcontext())
    monkeypatch.setattr(service.st, "error", lambda message: messages.append(str(message)))
    monkeypatch.setattr(service.st, "warning", lambda _message: None)
    monkeypatch.setattr(service.st, "markdown", lambda _text: None)
    monkeypatch.setattr(
        interaction_controller,
        "get_chat_event_api",
        lambda: SimpleNamespace(active_thread_id="voice-session"),
    )
    rejections: list[str] = []
    monkeypatch.setattr(
        state_manager,
        "record_runtime_rejection",
        lambda _exc, *, action: rejections.append(action),
    )

    def fail_turn(**_kwargs):
        raise RuntimeError("runtime unavailable")

    monkeypatch.setattr(interaction_controller, "process_prompt_via_runtime", fail_turn)
    reply = service.MediaService()._process_voice_transcript(object(), controller, "Hello.")

    assert reply == ""
    assert messages == ["Conversation turn failed: runtime unavailable"]
    assert rejections == ["voice_turn"]
    assert controller.events == ["error"]


def test_piper_resolves_executable_beside_active_python(monkeypatch, tmp_path) -> None:
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python_exe = scripts / "python.exe"
    piper_exe = scripts / "piper.exe"
    model_path = tmp_path / "voice.onnx"
    python_exe.touch()
    piper_exe.touch()
    model_path.touch()
    monkeypatch.setattr("dadbot.managers.tts.sys.executable", str(python_exe))
    monkeypatch.setattr("dadbot.managers.tts.shutil.which", lambda _name: None)
    manager = TTSManager(
        SimpleNamespace(PROFILE={"voice": {"tts_piper_model_path": str(model_path)}}),
    )

    assert manager.piper_tts_available() is True
    assert manager._piper_executable() == str(piper_exe)


def test_piper_status_uses_active_environment_executable(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    monkeypatch.setattr(helpers, "resolve_piper_executable", lambda: "piper.exe")

    ready, backend, _message = helpers.local_tts_backend_status(
        {"tts_backend": "piper", "tts_piper_model_path": str(model_path)},
    )

    assert ready is True
    assert backend == "piper"
