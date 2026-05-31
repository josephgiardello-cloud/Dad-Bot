"""Text-to-speech routing: Piper (neural) with pyttsx3 fallback."""

from __future__ import annotations

import json
import os
from contextlib import suppress
import shutil
import subprocess
import sys
import tempfile
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path

from dadbot.utils import create_temp_file_path, safe_unlink

try:
    import streamlit as st  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional at runtime outside Streamlit
    st = None

try:
    import edge_tts  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    edge_tts = None


class TTSManager:
    """Owns TTS backend selection and audio synthesis."""

    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def _voice_profile(bot: object) -> str:
        profile = getattr(bot, "PROFILE", {}) or {}
        voice = profile.get("voice", {}) if isinstance(profile, dict) else {}
        return str(voice.get("voice_profile") or "warm_dad").strip().lower()

    @staticmethod
    def _resolve_edge_voice_name(voice_profile: str) -> str:
        mapping = {
            "warm_dad": "en-US-GuyNeural",
            "deep_dad": "en-US-ChristopherNeural",
            "gentle_dad": "en-US-EricNeural",
        }
        return mapping.get(str(voice_profile or "warm_dad"), "en-US-GuyNeural")

    def _voice_config(self) -> dict:
        profile = getattr(self.bot, "PROFILE", {}) or {}
        if not isinstance(profile, dict):
            return {}
        voice = profile.get("voice", {})
        return voice if isinstance(voice, dict) else {}

    def _edge_voice_name(self) -> str:
        voice_cfg = self._voice_config()
        explicit = str(voice_cfg.get("edge_tts_voice") or "").strip()
        if explicit:
            return explicit
        return self._resolve_edge_voice_name(self._voice_profile(self.bot))

    def _apply_pronunciation_overrides(self, text: str) -> str:
        voice_cfg = self._voice_config()
        overrides = voice_cfg.get("pronunciation_overrides", {})
        if not isinstance(overrides, dict):
            return text

        rendered = str(text or "")
        for source, replacement in overrides.items():
            source_word = str(source or "").strip()
            replacement_word = str(replacement or "").strip()
            if not source_word or not replacement_word:
                continue
            rendered = re.sub(rf"\\b{re.escape(source_word)}\\b", replacement_word, rendered, flags=re.IGNORECASE)
        return rendered

    def edge_tts_available(self) -> bool:
        return edge_tts is not None

    def _elevenlabs_credentials(self) -> tuple[str, str]:
        voice_cfg = self._voice_config()
        secret_api_key = ""
        if st is not None:
            with suppress(Exception):
                secret_api_key = str(st.secrets.get("DADBOT_ELEVENLABS_API_KEY", "") or "").strip()
        api_key = str(
            voice_cfg.get("elevenlabs_api_key")
            or os.environ.get("DADBOT_ELEVENLABS_API_KEY")
            or secret_api_key
            or ""
        ).strip()
        voice_id = str(voice_cfg.get("elevenlabs_voice_id") or os.environ.get("DADBOT_ELEVENLABS_VOICE_ID") or "").strip()
        return api_key, voice_id

    def elevenlabs_available(self) -> bool:
        api_key, voice_id = self._elevenlabs_credentials()
        return bool(api_key and voice_id)

    def synthesize_elevenlabs_audio(self, text: str) -> tuple[bytes | None, str]:
        text = str(text or "").strip()
        if not text:
            return None, "Empty text"

        api_key, voice_id = self._elevenlabs_credentials()
        if not api_key or not voice_id:
            return None, "ElevenLabs requires API key and voice ID"

        voice_cfg = self._voice_config()
        model_id = str(voice_cfg.get("elevenlabs_model_id") or "eleven_turbo_v2_5").strip() or "eleven_turbo_v2_5"
        rendered_text = self._apply_pronunciation_overrides(text)
        payload = {
            "text": rendered_text,
            "model_id": model_id,
            "voice_settings": {
                "stability": float(voice_cfg.get("elevenlabs_stability", 0.45) or 0.45),
                "similarity_boost": float(voice_cfg.get("elevenlabs_similarity_boost", 0.80) or 0.80),
                "style": float(voice_cfg.get("elevenlabs_style", 0.15) or 0.15),
                "use_speaker_boost": bool(voice_cfg.get("elevenlabs_speaker_boost", True)),
            },
        }

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=mp3_44100_128"
        req = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as response:
                audio_bytes = response.read()
            if not audio_bytes:
                return None, "ElevenLabs returned empty audio"
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

    def synthesize_edge_tts_audio(self, text: str) -> tuple[bytes | None, str]:
        edge_tts_exe = shutil.which("edge-tts")
        if edge_tts_exe:
            edge_tts_cmd = [edge_tts_exe]
        elif edge_tts is not None:
            edge_tts_cmd = [sys.executable, "-m", "edge_tts"]
        else:
            return None, "edge-tts is not installed"
        text = str(text or "").strip()
        if not text:
            return None, "Empty text"
        voice_cfg = self._voice_config()
        voice_name = self._edge_voice_name()
        rate = str(voice_cfg.get("edge_tts_rate") or "+0%").strip() or "+0%"
        pitch = str(voice_cfg.get("edge_tts_pitch") or "+0Hz").strip() or "+0Hz"
        volume = str(voice_cfg.get("edge_tts_volume") or "+0%").strip() or "+0%"
        rendered_text = self._apply_pronunciation_overrides(text)
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix="dadbot_tts_")
            temp_path = temp_file.name
            temp_file.close()

            result = subprocess.run(
                edge_tts_cmd
                + [
                    "--voice",
                    voice_name,
                    f"--rate={rate}",
                    f"--pitch={pitch}",
                    f"--volume={volume}",
                    "--text",
                    rendered_text,
                    "--write-media",
                    temp_path,
                ],
                capture_output=True,
                timeout=25,
                text=True,
            )
            if result.returncode != 0:
                stderr = str(result.stderr or "").strip()
                return None, f"edge-tts failed: {stderr or ('exit ' + str(result.returncode))}"

            audio_bytes = Path(temp_path).read_bytes()
            return audio_bytes, ""
        except Exception as exc:
            return None, f"edge-tts failed: {exc}"
        finally:
            if temp_file is not None:
                safe_unlink(temp_file.name)

    @staticmethod
    def _resolve_pyttsx3_voice_id(engine, voice_profile: str) -> str:
        profile_hints = {
            "warm_dad": ("david", "mark", "male", "guy", "father", "adult"),
            "deep_dad": ("david", "mark", "male", "deep", "baritone", "adult"),
            "gentle_dad": ("gentle", "calm", "soothing", "male", "david", "adult"),
        }
        hints = profile_hints.get(voice_profile, profile_hints["warm_dad"])
        voices = list(engine.getProperty("voices") or [])
        scored: list[tuple[int, str]] = []
        for voice in voices:
            name = str(getattr(voice, "name", "") or "").lower()
            voice_id = str(getattr(voice, "id", "") or "")
            blob = f"{name} {voice_id.lower()}"
            score = sum(1 for hint in hints if hint in blob)
            scored.append((score, voice_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored and scored[0][1] else ""

    def piper_tts_available(self) -> bool:
        """Quick check: piper binary exists on PATH and model file is configured."""
        if not shutil.which("piper"):
            return False
        model_path = self.bot.PROFILE.get("voice", {}).get("piper_model_path", "")
        return bool(model_path and Path(model_path).is_file())

    def synthesize_piper_audio(
        self,
        text: str,
        model_path: str | None = None,
    ) -> tuple[bytes | None, str]:
        """Synthesize speech with Piper TTS. Returns (audio_bytes, error_message)."""
        text = str(text or "").strip()
        if not text:
            return None, "Empty text"
        if model_path is None:
            model_path = self.bot.PROFILE.get("voice", {}).get("piper_model_path", "")
        if not model_path or not Path(model_path).is_file():
            return None, f"Piper model not found: {model_path}"
        tmp_path = create_temp_file_path(suffix=".wav", prefix="dadbot_tts_")
        try:
            cmd = ["piper", "--model", str(model_path), "--output_file", tmp_path]
            result = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=15,
            )
            if result.returncode != 0:
                return None, f"Piper failed: {result.stderr.strip()}"
            audio_bytes = Path(tmp_path).read_bytes()
            return audio_bytes, ""
        except Exception as exc:
            return None, f"Piper TTS error: {exc}"
        finally:
            safe_unlink(tmp_path)

    def get_tts_audio(self, text: str) -> tuple[bytes | None, str, str]:
        """Central TTS router. Returns (audio_bytes, error_message, mime_type).
        Routes elevenlabs/piper/edge-tts/pyttsx3 depending on backend configuration.
        """
        text = str(text or "").strip()
        if not text:
            return None, "Empty text", ""
        voice_profile = self._voice_profile(self.bot)
        backend = str(self.bot.PROFILE.get("voice", {}).get("tts_backend", "auto") or "auto").strip().lower()

        if backend == "piper" and self.piper_tts_available():
            audio_bytes, error = self.synthesize_piper_audio(text)
            return audio_bytes, error, "audio/wav"

        if backend in {"elevenlabs", "voice_clone", "11labs"}:
            audio_bytes, error = self.synthesize_elevenlabs_audio(text)
            return audio_bytes, error, "audio/mpeg"

        if backend in {"edge_tts", "edge-tts"}:
            audio_bytes, error = self.synthesize_edge_tts_audio(text)
            return audio_bytes, error, "audio/mpeg"

        if backend == "auto":
            if self.elevenlabs_available():
                audio_bytes, error = self.synthesize_elevenlabs_audio(text)
                if audio_bytes:
                    return audio_bytes, "", "audio/mpeg"
            if self.piper_tts_available():
                audio_bytes, error = self.synthesize_piper_audio(text)
                if audio_bytes:
                    return audio_bytes, "", "audio/wav"
            if self.edge_tts_available():
                audio_bytes, error = self.synthesize_edge_tts_audio(text)
                if audio_bytes:
                    return audio_bytes, "", "audio/mpeg"

        tmp_path = create_temp_file_path(suffix=".wav", prefix="dadbot_tts_")
        try:
            import pyttsx3

            engine = pyttsx3.init()
            voice_id = self._resolve_pyttsx3_voice_id(engine, voice_profile)
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()
            audio_bytes = Path(tmp_path).read_bytes()
            return audio_bytes, "", "audio/wav"
        except Exception as exc:
            return None, f"pyttsx3 TTS error: {exc}", ""
        finally:
            safe_unlink(tmp_path)
