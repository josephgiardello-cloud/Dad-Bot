from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from contextlib import suppress
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path

from dadbot.utils import create_temp_file_path, safe_unlink

try:
    import pyttsx3  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

try:
    import edge_tts  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    edge_tts = None

try:
    import streamlit as st  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional runtime dependency
    st = None


def edge_tts_available() -> bool:
    return edge_tts is not None


def _elevenlabs_credentials(voice_cfg: dict | None = None) -> tuple[str, str]:
    config = voice_cfg if isinstance(voice_cfg, dict) else {}
    secret_api_key = ""
    if st is not None:
        with suppress(Exception):
            secret_api_key = str(st.secrets.get("DADBOT_ELEVENLABS_API_KEY", "") or "").strip()
    api_key = str(
        config.get("elevenlabs_api_key")
        or os.environ.get("DADBOT_ELEVENLABS_API_KEY")
        or secret_api_key
        or ""
    ).strip()
    voice_id = str(
        config.get("elevenlabs_voice_id")
        or os.environ.get("DADBOT_ELEVENLABS_VOICE_ID")
        or ""
    ).strip()
    return api_key, voice_id


def elevenlabs_available(voice_cfg: dict | None = None) -> bool:
    api_key, voice_id = _elevenlabs_credentials(voice_cfg)
    return bool(api_key and voice_id)


def synthesize_elevenlabs_audio(reply_text, *, voice_cfg: dict | None = None):
    text = str(reply_text or "").strip()
    if not text:
        return None, ""

    config = voice_cfg if isinstance(voice_cfg, dict) else {}
    api_key, voice_id = _elevenlabs_credentials(config)
    if not api_key or not voice_id:
        return None, "ElevenLabs requires API key and voice ID."

    model_id = str(config.get("elevenlabs_model_id") or "eleven_turbo_v2_5").strip() or "eleven_turbo_v2_5"
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(config.get("elevenlabs_stability", 0.45) or 0.45),
            "similarity_boost": float(config.get("elevenlabs_similarity_boost", 0.80) or 0.80),
            "style": float(config.get("elevenlabs_style", 0.15) or 0.15),
            "use_speaker_boost": bool(config.get("elevenlabs_speaker_boost", True)),
        },
    }
    request = Request(
        url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=mp3_44100_128",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            audio_bytes = response.read()
        if not audio_bytes:
            return None, "ElevenLabs returned empty audio."
        return audio_bytes, ""
    except HTTPError as exc:
        body = ""
        with suppress(Exception):
            body = exc.read().decode("utf-8", errors="ignore").strip()
        detail = body or str(exc.reason or exc)
        return None, f"ElevenLabs failed: HTTP {exc.code} {detail}"
    except URLError as exc:
        return None, f"ElevenLabs failed: {exc.reason}"
    except Exception as exc:
        return None, f"ElevenLabs failed: {exc}"


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def synthesize_piper_audio(
    text: str,
    *,
    model_path: str,
    rate: float = 1.0,
) -> tuple[bytes | None, str]:
    _ = rate
    piper_exe = shutil.which("piper")
    if not piper_exe:
        return (
            None,
            "Piper executable not found. Install from https://github.com/rhasspy/piper",
        )
    model_path = str(model_path or "").strip()
    if not model_path or not Path(model_path).exists():
        return None, (
            "Piper model file not configured or not found. "
            "Download a .onnx model from https://rhasspy.github.io/piper-samples/ "
            "and set the path in Preferences -> Voice -> Piper model path."
        )
    temp_path = None
    try:
        temp_path = create_temp_file_path(suffix=".wav", prefix="dadbot_tts_")
        result = subprocess.run(
            [piper_exe, "--model", model_path, "--output_file", temp_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=45,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:200]
            return None, f"Piper failed (exit {result.returncode}): {stderr}"
        audio_bytes = Path(temp_path).read_bytes()
        return audio_bytes, ""
    except subprocess.TimeoutExpired:
        return None, "Piper timed out after 45 seconds."
    except Exception as exc:
        return None, f"Piper error: {exc}"
    finally:
        if temp_path:
            safe_unlink(temp_path)


def resolve_tts_voice_id(engine, voice_profile):
    profile = str(voice_profile or "warm_dad").strip().lower()
    profile_hints = {
        "warm_dad": ("david", "mark", "male", "guy", "father", "adult"),
        "deep_dad": ("david", "mark", "male", "deep", "baritone", "adult"),
        "gentle_dad": ("gentle", "calm", "soothing", "male", "david", "adult"),
    }
    hints = profile_hints.get(profile, profile_hints["warm_dad"])
    voices = list(engine.getProperty("voices") or [])

    scored = []
    for voice in voices:
        name = str(getattr(voice, "name", "") or "").lower()
        voice_id = str(getattr(voice, "id", "") or "").lower()
        blob = f"{name} {voice_id}"
        score = sum(1 for hint in hints if hint in blob)
        scored.append((score, getattr(voice, "id", "")))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_id = scored[0][1] if scored and scored[0][1] else ""
    return best_id


def resolve_edge_voice_name(voice_profile: str) -> str:
    profile = str(voice_profile or "warm_dad").strip().lower()
    mapping = {
        "warm_dad": "en-US-GuyNeural",
        "deep_dad": "en-US-ChristopherNeural",
        "gentle_dad": "en-US-EricNeural",
    }
    return mapping.get(profile, "en-US-GuyNeural")


def synthesize_edge_tts_audio(
    reply_text,
    *,
    voice_profile: str = "warm_dad",
    rate_delta: int = 0,
    pacing: int = 50,
):
    if edge_tts is None:
        return None, "edge-tts is not installed."

    text = str(reply_text or "").strip()
    if not text:
        return None, ""

    pace_delta = int((max(0, min(100, int(pacing or 50))) - 50) * 0.6)
    rate_percent = max(-50, min(50, int(rate_delta or 0) + pace_delta))
    rate = f"{rate_percent:+d}%"
    edge_voice = resolve_edge_voice_name(voice_profile)

    temp_path = None
    try:
        temp_path = create_temp_file_path(suffix=".mp3", prefix="dadbot_tts_")

        async def _synthesize() -> None:
            communicator = edge_tts.Communicate(text=text, voice=edge_voice, rate=rate)
            await communicator.save(temp_path)

        _run_async(_synthesize())
        audio_bytes = Path(temp_path).read_bytes()
        return audio_bytes, ""
    except Exception as exc:
        return None, f"edge-tts failed: {exc}"
    finally:
        if temp_path:
            safe_unlink(temp_path)


def synthesize_tts_audio(
    reply_text,
    *,
    voice_profile: str = "warm_dad",
    rate_delta: int = 0,
    pacing: int = 50,
):
    if pyttsx3 is None:
        return None, "pyttsx3 is not installed."
    text = str(reply_text or "").strip()
    if not text:
        return None, ""

    temp_path = None
    try:
        temp_path = create_temp_file_path(suffix=".wav", prefix="dadbot_tts_")

        engine = pyttsx3.init()
        voice_id = resolve_tts_voice_id(engine, voice_profile)
        if voice_id:
            engine.setProperty("voice", voice_id)

        default_rate = int(engine.getProperty("rate") or 180)
        pace_delta = int((max(0, min(100, int(pacing or 50))) - 50) * 0.6)
        engine.setProperty(
            "rate",
            max(120, min(240, default_rate + int(rate_delta or 0) + pace_delta)),
        )
        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        audio_bytes = Path(temp_path).read_bytes()
        return audio_bytes, ""
    except Exception as exc:
        return None, f"TTS failed: {exc}"
    finally:
        if temp_path:
            safe_unlink(temp_path)
