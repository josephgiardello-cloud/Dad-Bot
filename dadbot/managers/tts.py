"""Text-to-speech routing: Piper (neural) with pyttsx3 fallback."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class TTSManager:
    """Owns TTS backend selection and audio synthesis."""

    def __init__(self, bot):
        self.bot = bot

    def piper_tts_available(self) -> bool:
        """Quick check: piper binary exists on PATH and model file is configured."""
        if not shutil.which("piper"):
            return False
        model_path = self.bot.PROFILE.get("voice", {}).get("piper_model_path", "")
        return bool(model_path and Path(model_path).is_file())

    def synthesize_piper_audio(self, text: str, model_path: str | None = None) -> tuple[bytes | None, str]:
        """Synthesize speech with Piper TTS. Returns (audio_bytes, error_message)."""
        text = str(text or "").strip()
        if not text:
            return None, "Empty text"
        if model_path is None:
            model_path = self.bot.PROFILE.get("voice", {}).get("piper_model_path", "")
        if not model_path or not Path(model_path).is_file():
            return None, f"Piper model not found: {model_path}"
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            cmd = ["piper", "--model", str(model_path), "--output_file", tmp_path]
            result = subprocess.run(cmd, input=text, text=True, capture_output=True, timeout=15)
            if result.returncode != 0:
                return None, f"Piper failed: {result.stderr.strip()}"
            audio_bytes = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)
            return audio_bytes, ""
        except Exception as exc:
            return None, f"Piper TTS error: {exc}"

    def get_tts_audio(self, text: str) -> tuple[bytes | None, str]:
        """Central TTS router. Returns (audio_bytes, error_message).
        Routes to Piper when configured; falls back to pyttsx3 WAV export."""
        text = str(text or "").strip()
        if not text:
            return None, "Empty text"
        backend = self.bot.PROFILE.get("voice", {}).get("tts_backend", "pyttsx3")
        if backend == "piper" and self.piper_tts_available():
            return self.synthesize_piper_audio(text)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()
            audio_bytes = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)
            return audio_bytes, ""
        except Exception as exc:
            return None, f"pyttsx3 TTS error: {exc}"
