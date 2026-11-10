from __future__ import annotations

import io
import threading
import wave

import pytest

from dadbot.speech.voice_loop import VoiceTurnLoop, concatenate_wav_chunks

pytestmark = pytest.mark.unit


def _wav_chunk(value: int, *, frames: int = 4, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes((value.to_bytes(2, "little", signed=True)) * frames)
    return output.getvalue()


def test_voice_loop_runs_transcription_turn_then_sentence_tts_chunks() -> None:
    calls: list[tuple[str, str]] = []
    emitted: list[tuple[bytes, str]] = []
    loop = VoiceTurnLoop(
        transcribe=lambda audio: (calls.append(("stt", audio.decode())) or "Tell me a story.", ""),
        execute_turn=lambda transcript: calls.append(("turn", transcript)) or "First sentence. Second sentence!",
        synthesize_chunk=lambda text: (
            calls.append(("tts", text)) or _wav_chunk(len(text)),
            "",
            "audio/wav",
        ),
    )

    result = loop.run(b"captured audio", audio_sink=lambda audio, mime: emitted.append((audio, mime)))

    assert calls == [
        ("stt", "captured audio"),
        ("turn", "Tell me a story."),
        ("tts", "First sentence."),
        ("tts", "Second sentence!"),
    ]
    assert result.transcript == "Tell me a story."
    assert result.response == "First sentence. Second sentence!"
    assert result.interrupted is False
    assert len(result.audio_chunks) == 2
    assert emitted == list(result.audio_chunks)


def test_voice_loop_stops_synthesizing_after_barge_in() -> None:
    cancelled = threading.Event()
    synthesized: list[str] = []
    loop = VoiceTurnLoop(
        transcribe=lambda _audio: ("hello", ""),
        execute_turn=lambda _transcript: "First sentence. Second sentence.",
        synthesize_chunk=lambda text: (
            synthesized.append(text) or _wav_chunk(1),
            "",
            "audio/wav",
        ),
    )

    result = loop.run(
        b"audio",
        cancelled=cancelled,
        audio_sink=lambda _audio, _mime: cancelled.set(),
    )

    assert synthesized == ["First sentence."]
    assert result.interrupted is True
    assert len(result.audio_chunks) == 1


def test_voice_loop_reports_barge_in_after_last_chunk_is_emitted() -> None:
    cancelled = threading.Event()
    loop = VoiceTurnLoop(
        execute_turn=lambda _transcript: "One sentence.",
        synthesize_chunk=lambda _text: (_wav_chunk(1), "", "audio/wav"),
    )

    result = loop.run_transcript(
        "hello",
        cancelled=cancelled,
        audio_sink=lambda _audio, _mime: cancelled.set(),
    )

    assert result.interrupted is True
    assert len(result.audio_chunks) == 1


def test_voice_loop_stops_on_transcription_failure() -> None:
    executed: list[str] = []
    loop = VoiceTurnLoop(
        transcribe=lambda _audio: ("", "local model unavailable"),
        execute_turn=lambda transcript: executed.append(transcript) or "reply",
        synthesize_chunk=lambda _text: (_wav_chunk(1), "", "audio/wav"),
    )

    result = loop.run(b"audio")

    assert result.error == "local model unavailable"
    assert result.transcript == ""
    assert executed == []


def test_voice_loop_surfaces_audio_sink_failures() -> None:
    loop = VoiceTurnLoop(
        execute_turn=lambda _transcript: "A short reply.",
        synthesize_chunk=lambda _text: (_wav_chunk(1), "", "audio/wav"),
    )

    def fail_audio_sink(_audio: bytes, _mime: str) -> None:
        raise RuntimeError("player stopped")

    result = loop.run_transcript("hello", audio_sink=fail_audio_sink)

    assert result.response == "A short reply."
    assert result.error == "Audio playback failed: player stopped"


def test_concatenate_wav_chunks_joins_pcm_and_rejects_mismatched_formats() -> None:
    joined = concatenate_wav_chunks([_wav_chunk(1), _wav_chunk(2, frames=2)])

    with wave.open(io.BytesIO(joined), "rb") as source:
        assert source.getnframes() == 6
        assert source.readframes(6) == (1).to_bytes(2, "little", signed=True) * 4 + (
            2
        ).to_bytes(2, "little", signed=True) * 2

    with pytest.raises(ValueError, match="incompatible WAV formats"):
        concatenate_wav_chunks([_wav_chunk(1), _wav_chunk(2, sample_rate=22050)])
    with pytest.raises(ValueError, match="valid WAV audio"):
        concatenate_wav_chunks([b"not a wav file"])
