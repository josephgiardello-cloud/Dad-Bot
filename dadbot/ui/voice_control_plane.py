from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

VOICE_MODES = {"push_to_talk", "always_listening", "ambient"}
VOICE_STATES = {
    "IDLE",
    "LISTENING",
    "RECORDING",
    "TRANSCRIBING",
    "THINKING",
    "SPEAKING",
    "ERROR",
}

VOICE_TRANSITION_EVENTS = {
    "ENABLE",
    "DISABLE",
    "MODE_CHANGED",
    "RESUME_PASSIVE",
    "BEGIN_RECORDING",
    "BEGIN_TRANSCRIBING",
    "BEGIN_THINKING",
    "BEGIN_SPEAKING",
    "COMPLETE_TURN",
    "CLEAR_ERROR",
    "CANCEL",
    "ERROR",
}

# Formal guarded transition constraints (P3).
_TRANSITION_GUARDS: dict[str, set[str]] = {
    "ENABLE": {"IDLE", "LISTENING", "ERROR"},
    "DISABLE": set(VOICE_STATES),
    "MODE_CHANGED": set(VOICE_STATES),
    "RESUME_PASSIVE": set(VOICE_STATES),
    "BEGIN_RECORDING": {"IDLE", "LISTENING"},
    "BEGIN_TRANSCRIBING": {"RECORDING"},
    "BEGIN_THINKING": {"TRANSCRIBING"},
    "BEGIN_SPEAKING": {"THINKING", "IDLE", "LISTENING"},
    "COMPLETE_TURN": {
        "RECORDING",
        "TRANSCRIBING",
        "THINKING",
        "SPEAKING",
        "LISTENING",
        "IDLE",
    },
    "CLEAR_ERROR": {"ERROR"},
    "CANCEL": {
        "RECORDING",
        "TRANSCRIBING",
        "THINKING",
        "SPEAKING",
        "LISTENING",
        "IDLE",
    },
    "ERROR": set(VOICE_STATES),
}


class VoiceSessionController:
    """Single source of truth for voice mode, lifecycle, and runtime status."""

    def __init__(
        self,
        voice_config: dict[str, Any],
        *,
        runtime_state: dict[str, Any] | None = None,
        ledger_emitter: Callable[[str, dict[str, Any]], None] | None = None,
        session_id_provider: Callable[[], str] | None = None,
        trace_id_provider: Callable[[], str] | None = None,
    ) -> None:
        self.voice_config = voice_config if isinstance(voice_config, dict) else {}
        self.runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
        self._ledger_emitter = ledger_emitter
        self._session_id_provider = session_id_provider
        self._trace_id_provider = trace_id_provider
        self._ensure_config_defaults()
        self._ensure_runtime_defaults()
        self._resume_passive_state(record_event=False)

    def _ensure_config_defaults(self) -> None:
        defaults = {
            "enabled": False,
            "mode": "push_to_talk",
            "auto_send_always_listening": True,
            "wake_word_required": False,
            "wake_word_phrase": "hey dad",
            "stt_enabled": True,
            "stt_backend": "auto",
            "stt_model": "base",
            "stt_language": "en",
            "tts_enabled": True,
            "tts_autoplay": False,
            "tts_voice": "warm_dad",
            "tts_backend": "pyttsx3",
            "tts_piper_model_path": "",
            "tts_rate": 0,
            "warmth": 70,
            "dad_joke_frequency": 35,
            "pacing": 50,
            "muted": False,
            "mic_preference": "default",
            "last_used_device": "default",
            "auto_listen_allowed": False,
            "last_mode": "push_to_talk",
            "interruptions_enabled": True,
            "barge_in_enabled": True,
            "barge_in_min_audio_bytes": 4000,
            "allow_tts_cancel": True,
            "priority_override_enabled": True,
        }
        for key, value in defaults.items():
            self.voice_config.setdefault(key, value)

        mode = str(self.voice_config.get("mode") or "push_to_talk").strip().lower()
        if mode not in VOICE_MODES:
            mode = "push_to_talk"
            self.voice_config["mode"] = mode
        self.voice_config["last_mode"] = mode
        self.voice_config["auto_listen_allowed"] = mode in {
            "always_listening",
            "ambient",
        }

    def _ensure_runtime_defaults(self) -> None:
        self.runtime_state.setdefault("state", "IDLE")
        self.runtime_state.setdefault(
            "mode",
            str(self.voice_config.get("mode") or "push_to_talk"),
        )
        self.runtime_state.setdefault("recording", False)
        self.runtime_state.setdefault("transcribing", False)
        self.runtime_state.setdefault("thinking", False)
        self.runtime_state.setdefault("speaking", False)
        self.runtime_state.setdefault("mic_available", False)
        self.runtime_state.setdefault(
            "muted",
            bool(self.voice_config.get("muted", False)),
        )
        self.runtime_state.setdefault("last_error", "")
        self.runtime_state.setdefault("last_transcript", "")
        self.runtime_state.setdefault("last_event", "")
        self.runtime_state.setdefault("events", [])
        self.runtime_state.setdefault("transition_version", 0)
        self.runtime_state.setdefault("cancel_requested", False)
        self.runtime_state.setdefault("cancel_reason", "")
        self.runtime_state.setdefault("active_turn_id", "")
        self.runtime_state.setdefault("priority_override", "normal")
        self.runtime_state.setdefault("known_devices", ["default"])

    def _record_event(self, name: str, detail: dict[str, Any] | None = None) -> None:
        event = {
            "timestamp": time.time(),
            "event": str(name or "event"),
            "state": str(self.runtime_state.get("state") or "IDLE"),
            "mode": str(self.voice_config.get("mode") or "push_to_talk"),
            "detail": dict(detail or {}),
            "transition_version": int(
                self.runtime_state.get("transition_version") or 0,
            ),
        }
        events = self.runtime_state.setdefault("events", [])
        events.append(event)
        if len(events) > 60:
            del events[:-60]
        self.runtime_state["last_event"] = str(name or "")
        self._emit_ledger_event(
            "VOICE_EVENT",
            {
                "event": event["event"],
                "state": event["state"],
                "mode": event["mode"],
                "detail": event["detail"],
                "transition_version": event["transition_version"],
                "timestamp": event["timestamp"],
            },
        )

    def _emit_ledger_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not callable(self._ledger_emitter):
            return
        enriched = dict(payload or {})
        if callable(self._session_id_provider):
            try:
                enriched.setdefault(
                    "session_id",
                    str(self._session_id_provider() or "default"),
                )
            except Exception:
                enriched.setdefault("session_id", "default")
        if callable(self._trace_id_provider):
            try:
                trace_id = str(self._trace_id_provider() or "").strip()
            except Exception:
                trace_id = ""
            if trace_id:
                enriched.setdefault("trace_id", trace_id)
        try:
            self._ledger_emitter(str(event_type or "VOICE_EVENT"), enriched)
        except Exception:
            # Voice control must never fail hard because telemetry emission failed.
            return

    def _guard_allows(self, event_name: str, current_state: str) -> bool:
        allowed = _TRANSITION_GUARDS.get(str(event_name or "").strip().upper())
        if allowed is None:
            return False
        return str(current_state or "IDLE").strip().upper() in allowed

    def _passive_target_state(self) -> str:
        enabled = bool(self.voice_config.get("enabled", False))
        mode = str(self.voice_config.get("mode") or "push_to_talk").strip().lower()
        if enabled and mode in {"always_listening", "ambient"}:
            return "LISTENING"
        return "IDLE"

    def _resolve_target_state(
        self,
        event_name: str,
        *,
        fallback_state: str | None = None,
    ) -> str:
        event_name = str(event_name or "").strip().upper()
        if event_name in {
            "ENABLE",
            "MODE_CHANGED",
            "RESUME_PASSIVE",
            "COMPLETE_TURN",
            "CLEAR_ERROR",
            "CANCEL",
        }:
            return self._passive_target_state()
        if event_name == "BEGIN_RECORDING":
            return "RECORDING"
        if event_name == "BEGIN_TRANSCRIBING":
            return "TRANSCRIBING"
        if event_name == "BEGIN_THINKING":
            return "THINKING"
        if event_name == "BEGIN_SPEAKING":
            return "SPEAKING"
        if event_name == "DISABLE":
            return "IDLE"
        if event_name == "ERROR":
            return "ERROR"
        return str(fallback_state or self.runtime_state.get("state") or "IDLE").strip().upper()

    def _set_state(self, state: str, *, reason: str = "", error: str = "") -> None:
        normalized = str(state or "IDLE").strip().upper()
        if normalized not in VOICE_STATES:
            normalized = "ERROR"
            error = error or f"Unsupported voice state: {state!r}"

        previous = str(self.runtime_state.get("state") or "IDLE").strip().upper()
        self.runtime_state["state"] = normalized
        self.runtime_state["recording"] = normalized == "RECORDING"
        self.runtime_state["transcribing"] = normalized == "TRANSCRIBING"
        self.runtime_state["thinking"] = normalized == "THINKING"
        self.runtime_state["speaking"] = normalized == "SPEAKING"
        self.runtime_state["last_error"] = str(error or "")
        if previous != normalized:
            self.runtime_state["transition_version"] = int(self.runtime_state.get("transition_version") or 0) + 1
            self._emit_ledger_event(
                "VOICE_STATE_TRANSITION",
                {
                    "from_state": previous,
                    "to_state": normalized,
                    "reason": str(reason or ""),
                    "error": str(error or ""),
                    "mode": str(self.voice_config.get("mode") or "push_to_talk"),
                    "transition_version": int(
                        self.runtime_state.get("transition_version") or 0,
                    ),
                },
            )

    def _transition(
        self,
        event_name: str,
        *,
        detail: dict[str, Any] | None = None,
        force: bool = False,
        error: str = "",
    ) -> bool:
        normalized_event = str(event_name or "").strip().upper()
        if normalized_event not in VOICE_TRANSITION_EVENTS:
            self.mark_error(f"Unsupported transition event: {event_name!r}")
            return False

        current_state = str(self.runtime_state.get("state") or "IDLE").strip().upper()
        if not force and not self._guard_allows(normalized_event, current_state):
            self.mark_error(
                f"Invalid voice transition: event={normalized_event} from state={current_state}",
            )
            return False

        target_state = self._resolve_target_state(normalized_event)
        self._set_state(target_state, reason=normalized_event.lower(), error=error)
        self._record_event(normalized_event.lower(), detail)
        return True

    def _resume_passive_state(self, *, record_event: bool = True) -> None:
        enabled = bool(self.voice_config.get("enabled", False))
        mode = str(self.voice_config.get("mode") or "push_to_talk").strip().lower()
        self._set_state(self._passive_target_state(), reason="resume_passive")
        if record_event:
            self._record_event(
                "passive_state_resumed",
                {"enabled": enabled, "mode": mode},
            )

    def set_mode(self, mode: str) -> bool:
        normalized = str(mode or "push_to_talk").strip().lower()
        if normalized not in VOICE_MODES:
            self.mark_error(f"Unsupported voice mode: {mode!r}")
            return False
        self.voice_config["mode"] = normalized
        self.voice_config["last_mode"] = normalized
        self.voice_config["auto_listen_allowed"] = normalized in {
            "always_listening",
            "ambient",
        }
        self.runtime_state["mode"] = normalized
        self._transition("MODE_CHANGED", detail={"mode": normalized}, force=True)
        return True

    def set_enabled(self, enabled: bool) -> None:
        self.voice_config["enabled"] = bool(enabled)
        if bool(enabled):
            self._transition("ENABLE", detail={"enabled": True}, force=True)
            return
        self._transition("DISABLE", detail={"enabled": False}, force=True)

    def set_muted(self, muted: bool) -> None:
        value = bool(muted)
        self.voice_config["muted"] = value
        self.runtime_state["muted"] = value
        self._record_event("mute_changed", {"muted": value})

    def set_device(self, device_id: str, *, label: str = "") -> None:
        resolved = str(device_id or "default").strip() or "default"
        self.voice_config["last_used_device"] = resolved
        known = list(self.runtime_state.get("known_devices") or ["default"])
        if resolved not in known:
            known.append(resolved)
        if "default" not in known:
            known.insert(0, "default")
        self.runtime_state["known_devices"] = known[:12]
        self._record_event(
            "device_selected",
            {"device_id": resolved, "label": str(label or "")},
        )

    def mark_mic_available(self, available: bool) -> None:
        self.runtime_state["mic_available"] = bool(available)

    def begin_recording(self) -> None:
        self._transition("BEGIN_RECORDING")

    def begin_transcribing(self) -> None:
        self._transition("BEGIN_TRANSCRIBING")

    def begin_thinking(self) -> None:
        self._transition("BEGIN_THINKING")

    def begin_speaking(self) -> None:
        self.runtime_state["cancel_requested"] = False
        self.runtime_state["cancel_reason"] = ""
        self._transition("BEGIN_SPEAKING")

    def mark_error(self, message: str) -> None:
        self._transition(
            "ERROR",
            detail={"message": str(message or "")},
            force=True,
            error=str(message or "Unknown voice error"),
        )

    def clear_error(self) -> None:
        self.runtime_state["last_error"] = ""
        self._transition("CLEAR_ERROR", force=True)

    def complete_turn(self) -> None:
        self._transition("COMPLETE_TURN", force=True)

    # Interruptibility model (P2).
    def start_turn(self, *, turn_id: str, priority: str = "normal") -> None:
        self.runtime_state["active_turn_id"] = str(turn_id or "")
        self.runtime_state["priority_override"] = str(priority or "normal")
        self.runtime_state["cancel_requested"] = False
        self.runtime_state["cancel_reason"] = ""
        self._record_event(
            "turn_started",
            {
                "turn_id": self.runtime_state["active_turn_id"],
                "priority": str(priority or "normal"),
            },
        )

    def request_cancel(
        self,
        *,
        reason: str = "user_interrupt",
        priority: str = "normal",
    ) -> bool:
        self.runtime_state["cancel_requested"] = True
        self.runtime_state["cancel_reason"] = str(reason or "user_interrupt")
        self.runtime_state["priority_override"] = str(priority or "normal")
        cancelled = self._transition(
            "CANCEL",
            detail={
                "reason": self.runtime_state["cancel_reason"],
                "priority": self.runtime_state["priority_override"],
            },
            force=True,
        )
        return bool(cancelled)

    def consume_cancel(self) -> dict[str, Any]:
        payload = {
            "cancel_requested": bool(self.runtime_state.get("cancel_requested", False)),
            "cancel_reason": str(self.runtime_state.get("cancel_reason") or ""),
            "priority_override": str(
                self.runtime_state.get("priority_override") or "normal",
            ),
        }
        self.runtime_state["cancel_requested"] = False
        self.runtime_state["cancel_reason"] = ""
        return payload

    def can_barge_in(self, audio_bytes: bytes) -> bool:
        if not bool(self.voice_config.get("interruptions_enabled", True)):
            return False
        if not bool(self.voice_config.get("barge_in_enabled", True)):
            return False
        min_bytes = max(1, int(self.voice_config.get("barge_in_min_audio_bytes") or 1))
        if len(audio_bytes or b"") < min_bytes:
            return False
        return str(self.runtime_state.get("state") or "IDLE") in {
            "THINKING",
            "SPEAKING",
        }

    def apply_priority_override(self, priority: str, *, reason: str = "manual") -> bool:
        if not bool(self.voice_config.get("priority_override_enabled", True)):
            return False
        normalized = str(priority or "normal").strip().lower() or "normal"
        self.runtime_state["priority_override"] = normalized
        self._record_event(
            "priority_override",
            {"priority": normalized, "reason": str(reason or "")},
        )
        return True

    def process_audio_capture(
        self,
        audio_bytes: bytes,
        *,
        transcribe_fn: Callable[[bytes], tuple[str, str]],
        min_audio_bytes: int = 3500,
    ) -> tuple[str, str]:
        if not bool(self.voice_config.get("enabled", False)):
            return "", "Voice controls are disabled."
        if not audio_bytes:
            return "", "No audio data received."
        if len(audio_bytes) < max(1, int(min_audio_bytes or 1)):
            self._transition("COMPLETE_TURN", force=True)
            return "", ""

        if self.can_barge_in(audio_bytes):
            self.request_cancel(reason="barge_in_audio", priority="high")

        self.mark_mic_available(True)
        self.voice_config["last_used_device"] = str(
            self.voice_config.get("mic_preference") or "default",
        )

        self.begin_recording()
        self.begin_transcribing()

        try:
            transcript_text, transcript_error = transcribe_fn(audio_bytes)
        except Exception as exc:
            self.mark_error(f"Transcription failed: {exc}")
            return "", str(
                self.runtime_state.get("last_error") or "Transcription failed",
            )

        if transcript_error:
            self.mark_error(str(transcript_error))
            return "", str(transcript_error)

        transcript = str(transcript_text or "").strip()
        if not transcript:
            self._transition("COMPLETE_TURN", force=True)
            return "", "No transcript captured."

        if bool(self.voice_config.get("wake_word_required", False)):
            wake_phrase = str(self.voice_config.get("wake_word_phrase") or "hey dad").strip().lower()
            if wake_phrase and not transcript.lower().startswith(wake_phrase):
                self._transition("COMPLETE_TURN", force=True)
                self._record_event("wake_phrase_missing", {"wake_phrase": wake_phrase})
                return "", ""
            if wake_phrase:
                transcript = transcript[len(wake_phrase) :].strip(" ,.!?-:")
                if not transcript:
                    self._transition("COMPLETE_TURN", force=True)
                    return "", "Wake phrase heard. Keep talking after it."

        self.runtime_state["last_transcript"] = transcript
        self.begin_thinking()
        self._record_event("transcript_ready", {"chars": len(transcript)})
        return transcript, ""

    def status_text(self) -> str:
        labels = {
            "IDLE": "Idle",
            "LISTENING": "Listening...",
            "RECORDING": "Recording...",
            "TRANSCRIBING": "Processing audio...",
            "THINKING": "Thinking...",
            "SPEAKING": "Speaking...",
            "ERROR": "Error",
        }
        return labels.get(str(self.runtime_state.get("state") or "IDLE"), "Idle")

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": str(self.voice_config.get("mode") or "push_to_talk"),
            "state": str(self.runtime_state.get("state") or "IDLE"),
            "status": self.status_text(),
            "recording": bool(self.runtime_state.get("recording", False)),
            "transcribing": bool(self.runtime_state.get("transcribing", False)),
            "thinking": bool(self.runtime_state.get("thinking", False)),
            "speaking": bool(self.runtime_state.get("speaking", False)),
            "muted": bool(self.runtime_state.get("muted", False)),
            "mic_available": bool(self.runtime_state.get("mic_available", False)),
            "last_error": str(self.runtime_state.get("last_error") or ""),
            "last_event": str(self.runtime_state.get("last_event") or ""),
            "last_transcript": str(self.runtime_state.get("last_transcript") or ""),
            "last_used_device": str(
                self.voice_config.get("last_used_device") or "default",
            ),
            "known_devices": list(
                self.runtime_state.get("known_devices") or ["default"],
            ),
            "transition_version": int(
                self.runtime_state.get("transition_version") or 0,
            ),
            "cancel_requested": bool(self.runtime_state.get("cancel_requested", False)),
            "cancel_reason": str(self.runtime_state.get("cancel_reason") or ""),
            "active_turn_id": str(self.runtime_state.get("active_turn_id") or ""),
            "priority_override": str(
                self.runtime_state.get("priority_override") or "normal",
            ),
            "safety_flags": {
                "auto_listen_allowed": bool(
                    self.voice_config.get("auto_listen_allowed", False),
                ),
                "wake_word_required": bool(
                    self.voice_config.get("wake_word_required", False),
                ),
                "interruptions_enabled": bool(
                    self.voice_config.get("interruptions_enabled", True),
                ),
                "barge_in_enabled": bool(
                    self.voice_config.get("barge_in_enabled", True),
                ),
                "allow_tts_cancel": bool(
                    self.voice_config.get("allow_tts_cancel", True),
                ),
            },
            "events": list(self.runtime_state.get("events") or []),
        }
