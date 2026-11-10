from __future__ import annotations

import io
import re
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass

Transcribe = Callable[[bytes], tuple[str, str]]
ExecuteTurn = Callable[[str], str]
SynthesizeChunk = Callable[[str], tuple[bytes | None, str, str]]
AudioSink = Callable[[bytes, str], None]


@dataclass(frozen=True)
class VoiceTurnResult:
    transcript: str
    response: str
    audio_chunks: tuple[tuple[bytes, str], ...] = ()
    interrupted: bool = False
    error: str = ""


def _sentence_chunks(text: str) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", normalized) if chunk.strip()]


class VoiceTurnLoop:
    """Coordinate a captured utterance, conversational turn, and chunked TTS."""

    def __init__(
        self,
        *,
        transcribe: Transcribe | None = None,
        execute_turn: ExecuteTurn,
        synthesize_chunk: SynthesizeChunk | None = None,
    ) -> None:
        self._transcribe = transcribe
        self._execute_turn = execute_turn
        self._synthesize_chunk = synthesize_chunk

    def run(
        self,
        audio: bytes,
        *,
        cancelled: threading.Event | None = None,
        audio_sink: AudioSink | None = None,
    ) -> VoiceTurnResult:
        if not audio:
            return VoiceTurnResult("", "", error="No audio data received.")
        if self._transcribe is None:
            return VoiceTurnResult("", "", error="No transcription provider is configured.")
        try:
            transcript, error = self._transcribe(audio)
        except Exception as exc:
            return VoiceTurnResult("", "", error=f"Transcription failed: {exc}")
        if error:
            return VoiceTurnResult("", "", error=str(error))
        return self.run_transcript(
            transcript,
            cancelled=cancelled,
            audio_sink=audio_sink,
        )

    def run_transcript(
        self,
        transcript: str,
        *,
        cancelled: threading.Event | None = None,
        audio_sink: AudioSink | None = None,
    ) -> VoiceTurnResult:
        normalized_transcript = str(transcript or "").strip()
        if not normalized_transcript:
            return VoiceTurnResult("", "", error="No transcript captured.")
        if cancelled is not None and cancelled.is_set():
            return VoiceTurnResult(normalized_transcript, "", interrupted=True)

        try:
            response = str(self._execute_turn(normalized_transcript) or "").strip()
        except Exception as exc:
            return VoiceTurnResult(normalized_transcript, "", error=f"Conversation turn failed: {exc}")

        if cancelled is not None and cancelled.is_set():
            return VoiceTurnResult(normalized_transcript, response, interrupted=True)
        if self._synthesize_chunk is None:
            return VoiceTurnResult(normalized_transcript, response)

        chunks: list[tuple[bytes, str]] = []
        for sentence in _sentence_chunks(response):
            if cancelled is not None and cancelled.is_set():
                return VoiceTurnResult(
                    normalized_transcript,
                    response,
                    tuple(chunks),
                    interrupted=True,
                )
            try:
                audio, error, audio_format = self._synthesize_chunk(sentence)
            except Exception as exc:
                return VoiceTurnResult(
                    normalized_transcript,
                    response,
                    tuple(chunks),
                    error=f"Speech synthesis failed: {exc}",
                )
            if error or not audio:
                return VoiceTurnResult(
                    normalized_transcript,
                    response,
                    tuple(chunks),
                    error=str(error or "Speech synthesis returned no audio."),
                )
            chunk = (bytes(audio), str(audio_format or "audio/wav"))
            chunks.append(chunk)
            if audio_sink is not None:
                try:
                    audio_sink(*chunk)
                except Exception as exc:
                    return VoiceTurnResult(
                        normalized_transcript,
                        response,
                        tuple(chunks),
                        error=f"Audio playback failed: {exc}",
                    )

        if cancelled is not None and cancelled.is_set():
            return VoiceTurnResult(
                normalized_transcript,
                response,
                tuple(chunks),
                interrupted=True,
            )
        return VoiceTurnResult(normalized_transcript, response, tuple(chunks))


def concatenate_wav_chunks(chunks: list[bytes] | tuple[bytes, ...]) -> bytes:
    """Join compatible PCM WAV chunks for players that accept one audio asset."""
    if not chunks:
        return b""
    parameters: tuple[int, int, int, str, str] | None = None
    frames: list[bytes] = []
    try:
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk), "rb") as source:
                current = source.getparams()
                current_format = (
                    current.nchannels,
                    current.sampwidth,
                    current.framerate,
                    current.comptype,
                    current.compname,
                )
                if parameters is None:
                    parameters = current_format
                elif current_format != parameters:
                    raise ValueError("Speech chunks have incompatible WAV formats.")
                frames.append(source.readframes(source.getnframes()))
    except (EOFError, wave.Error) as exc:
        raise ValueError("Speech chunks must be valid WAV audio.") from exc

    if parameters is None:
        return b""
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(parameters[0])
        target.setsampwidth(parameters[1])
        target.setframerate(parameters[2])
        target.setcomptype(parameters[3], parameters[4])
        for frame_data in frames:
            target.writeframesraw(frame_data)
    return output.getvalue()
