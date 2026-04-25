from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None

try:
    import whisper as openai_whisper  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    openai_whisper = None


@st.cache_resource(show_spinner=False)
def load_faster_whisper_model(model_name):
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is unavailable")
    return WhisperModel(model_name, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner=False)
def load_openai_whisper_model(model_name):
    if openai_whisper is None:
        raise RuntimeError("openai-whisper is unavailable")
    return openai_whisper.load_model(model_name)


def transcribe_audio_bytes(audio_bytes, *, backend, model_name: str = "base", language: str = "en"):
    if not audio_bytes:
        return "", "No audio data received."

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name

        if backend == "faster_whisper":
            model = load_faster_whisper_model(model_name)
            segments, _info = model.transcribe(temp_path, language=language or None, vad_filter=True, beam_size=1)
            transcript = " ".join((segment.text or "").strip() for segment in segments).strip()
            return transcript, ""

        if backend == "openai_whisper":
            model = load_openai_whisper_model(model_name)
            result = model.transcribe(temp_path, language=language or None, fp16=False)
            transcript = str((result or {}).get("text") or "").strip()
            return transcript, ""

        return "", f"Unsupported STT backend: {backend}"
    except Exception as exc:
        return "", f"Transcription failed: {exc}"
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
