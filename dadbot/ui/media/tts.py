from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import pyttsx3  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None


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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name
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
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name

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
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
